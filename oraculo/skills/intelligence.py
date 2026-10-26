"""Skills de inteligência (13-18)."""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date

import numpy as np

from ..core import agent_ids as A
from ..core.types import NewsItem, Series
from ._math import norm_cdf
from .base import SkillError, SkillRun, skill
from .knowledge import (
    CONFIRMATION_MARKERS,
    EVENT_SECTOR_MAP,
    KNOWLEDGE_VERSION,
    NEGATIONS,
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
    RUMOR_MARKERS,
    source_tier,
)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def _shingles(text: str, k: int = 3) -> set[str]:
    w = _norm(text).split()
    return {" ".join(w[i : i + k]) for i in range(max(1, len(w) - k + 1))}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def classify_reliability(item: NewsItem) -> str:
    text = f"{item.title} {item.body}".lower()
    if any(m in text for m in RUMOR_MARKERS):
        return "rumor"
    tier = source_tier(item.source, item.url)
    if tier == 1 or (tier == 2 and any(m in text for m in CONFIRMATION_MARKERS)):
        return "confirmada"
    return "não verificada"


@skill(
    id="global_news_monitor",
    name="Monitoramento de notícias",
    version="1.0.0",
    category="inteligencia",
    objective="Filtrar notícias com procedência, eliminar duplicidades, identificar novidades e separar rumores de fatos confirmados.",
    authorized_agents=[A.NOTICIAS, A.GEOPOLITICA, A.ORQUESTRADOR, A.CIIO],
    inputs={
        "items": "lista de NewsItem coletados de fontes reais (RSS, APIs licenciadas, CVM)",
        "seen_fingerprints": "impressões digitais já vistas (memória analítica), opcional",
        "dedup_threshold": "similaridade Jaccard para duplicidade (padrão 0.6)",
    },
    required_inputs=["items"],
    tools=["NewsItem feeds", "knowledge.source_tier"],
    procedure=[
        "Descartar itens sem fonte ou horário (regra obrigatória 2).",
        "Agrupar duplicatas por similaridade de trigramas do título.",
        "Classificar cada grupo: confirmada / não verificada / rumor; contar fontes independentes.",
        "Marcar como novidade o que não consta das impressões digitais já vistas.",
    ],
    validation=["Nenhuma notícia sem fonte na saída."],
    output_schema={"grupos": "lista de notícias únicas", "novidades": "int", "descartadas": "int"},
    failure_handling="Itens inválidos são descartados e contabilizados, nunca corrigidos por suposição.",
)
def global_news_monitor(
    run: SkillRun, items: list[NewsItem], seen_fingerprints: list[str] | None = None, dedup_threshold: float = 0.6
) -> dict:
    seen = set(seen_fingerprints or [])
    groups: list[dict] = []
    for it in sorted(items, key=lambda x: x.published_at):
        sh = _shingles(it.title)
        for g in groups:
            if _jaccard(sh, g["_sh"]) >= dedup_threshold:
                g["itens"].append(it)
                break
        else:
            groups.append({"_sh": sh, "itens": [it]})
    out_groups = []
    for g in groups:
        first: NewsItem = g["itens"][0]
        fp = " ".join(sorted(g["_sh"]))[:200]
        sources = sorted({i.source for i in g["itens"]})
        statuses = [classify_reliability(i) for i in g["itens"]]
        tier2_sources = {i.source for i in g["itens"] if source_tier(i.source, i.url) == 2}
        if "confirmada" in statuses or len(tier2_sources) >= 2:  # oficial, ou 2 veículos profissionais independentes
            status = "confirmada"
        elif all(s == "rumor" for s in statuses):
            status = "rumor"
        else:
            status = "não verificada"
        out_groups.append(
            {
                "titulo": first.title,
                "primeira_publicacao": first.published_at.isoformat(),
                "fontes": sources,
                "n_fontes_independentes": len(sources),
                "melhor_nivel_fonte": min(source_tier(i.source, i.url) for i in g["itens"]),
                "status": status,
                "tickers": sorted({t for i in g["itens"] for t in i.tickers}),
                "url": first.url,
                "fingerprint": fp,
                "novidade": fp not in seen,
                "sintetica": any(i.synthetic for i in g["itens"]),
            }
        )
    return {
        "grupos": out_groups,
        "total_recebido": len(items),
        "unicas": len(out_groups),
        "duplicadas": len(items) - len(out_groups),
        "novidades": sum(g["novidade"] for g in out_groups),
        "rumores": sum(g["status"] == "rumor" for g in out_groups),
    }


def score_sentiment(text: str) -> tuple[float, int]:
    """Retorna (escore em [-1, 1], nº de termos encontrados)."""
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    pos = neg = 0
    for terms, sign in ((POSITIVE_TERMS, 1), (NEGATIVE_TERMS, -1)):
        for term in terms:
            for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", t):
                before = t[: m.start()].split()[-3:]
                s = -sign if any(w in NEGATIONS for w in before) else sign
                if s > 0:
                    pos += 1
                else:
                    neg += 1
    n = pos + neg
    return ((pos - neg) / n if n else 0.0), n


@skill(
    id="news_sentiment_analysis",
    name="Análise de sentimento",
    version="1.0.0",
    category="inteligencia",
    objective="Medir o tom das notícias (léxico financeiro PT/EN com negação) por item e agregado, com nível de confiança.",
    authorized_agents=[A.NOTICIAS, A.QUANT, A.GEOPOLITICA, A.CIIO],
    inputs={"items": "lista de NewsItem", "weight_by_tier": "ponderar por confiabilidade da fonte (padrão True)"},
    required_inputs=["items"],
    tools=["knowledge.POSITIVE_TERMS/NEGATIVE_TERMS"],
    procedure=[
        "Contar termos positivos/negativos, invertendo sob negação nas 3 palavras anteriores.",
        "Escore por item = (pos-neg)/(pos+neg).",
        "Agregado ponderado por nível da fonte (1:1.0, 2:0.7, 3:0.3) e penalidade para rumores (0.3).",
        "Confiança cresce com nº de termos e de itens.",
    ],
    validation=["Escores em [-1, 1]."],
    output_schema={"escore_agregado": "float", "confianca": "float", "itens": "list"},
    failure_handling="Sem termos reconhecidos => escore 0 com confiança 0 (não inferido).",
)
def news_sentiment_analysis(run: SkillRun, items: list[NewsItem], weight_by_tier: bool = True) -> dict:
    if not items:
        raise SkillError("Nenhuma notícia para analisar.")
    rows, num, den, terms_total = [], 0.0, 0.0, 0
    tier_w = {1: 1.0, 2: 0.7, 3: 0.3}
    for it in items:
        s, n = score_sentiment(f"{it.title}. {it.body}")
        rel = classify_reliability(it)
        w = (tier_w[source_tier(it.source, it.url)] if weight_by_tier else 1.0) * (0.3 if rel == "rumor" else 1.0)
        w *= min(n, 5) / 5
        num += s * w
        den += w
        terms_total += n
        rows.append({"titulo": it.title, "escore": s, "termos": n, "peso": w, "status": rel})
    agg = num / den if den else 0.0
    conf = min(1.0, terms_total / 20) * min(1.0, len(items) / 5)
    run.warn("Sentimento léxico não captura ironia nem contexto; usar como evidência auxiliar.", degrade=False)
    return {"escore_agregado": agg, "confianca": conf, "n_itens": len(items), "itens": rows}


@news_sentiment_analysis.validator
def _sent_range(out: dict) -> list[str]:
    return [] if -1 <= out["escore_agregado"] <= 1 else ["escore fora de [-1,1]"]


@skill(
    id="event_impact_mapping",
    name="Mapeamento de impacto de eventos",
    version="1.0.0",
    category="inteligencia",
    objective="Mapear setores e empresas potencialmente afetados por um evento, combinando canais de transmissão e sensibilidades medidas.",
    authorized_agents=[A.NOTICIAS, A.GEOPOLITICA, A.ECON_BRASIL, A.ECON_GLOBAL, A.COMMODITIES, A.SETORIAL, A.CIIO, A.RISCO],
    inputs={
        "event_type": f"um de: {', '.join(sorted(EVENT_SECTOR_MAP))}",
        "companies": "dict[ticker, {'setor': str, 'betas': dict[fator, beta] opcional}]",
        "factor": "fator associado ao evento para usar betas medidos (opcional, ex.: 'petroleo')",
        "shock_pct": "magnitude do choque no fator, em % (opcional)",
    },
    required_inputs=["event_type", "companies"],
    tools=["knowledge.EVENT_SECTOR_MAP", "betas de commodity_exposure/fx_impact_analysis"],
    procedure=[
        "Buscar canais setoriais do evento.",
        "Pontuar empresas por direção x intensidade do setor.",
        "Se houver beta medido para o fator e choque, estimar impacto = beta x choque e priorizar sobre a hipótese setorial.",
        "Ordenar por magnitude absoluta.",
    ],
    validation=["Tipo de evento conhecido."],
    output_schema={"empresas": "lista ordenada", "setores": "lista", "versao_conhecimento": "str"},
    failure_handling="Evento desconhecido => falha explícita (sem extrapolação).",
)
def event_impact_mapping(
    run: SkillRun, event_type: str, companies: dict, factor: str | None = None, shock_pct: float | None = None
) -> dict:
    if event_type not in EVENT_SECTOR_MAP:
        raise SkillError(f"Evento '{event_type}' não mapeado. Conhecidos: {sorted(EVENT_SECTOR_MAP)}")
    sectors = {s: (d, i, ch) for s, d, i, ch in EVENT_SECTOR_MAP[event_type]}
    rows = []
    for t, info in companies.items():
        sec = info.get("setor")
        row = {"ticker": t, "setor": sec, "direcao_hipotese": 0, "intensidade": 0, "canal": None, "impacto_estimado_pct": None, "base": "sem exposição mapeada"}
        if sec in sectors:
            d, i, ch = sectors[sec]
            row.update({"direcao_hipotese": d, "intensidade": i, "canal": ch, "base": "hipótese setorial"})
        beta = (info.get("betas") or {}).get(factor) if factor else None
        if beta is not None and shock_pct is not None:
            row["impacto_estimado_pct"] = beta * shock_pct
            row["base"] = "beta medido"
        rows.append(row)
    rows.sort(key=lambda r: -(abs(r["impacto_estimado_pct"]) if r["impacto_estimado_pct"] is not None else r["intensidade"] / 10))
    return {
        "evento": event_type,
        "setores": [{"setor": s, "direcao": d, "intensidade": i, "canal": c} for s, (d, i, c) in sectors.items()],
        "empresas": rows,
        "versao_conhecimento": KNOWLEDGE_VERSION,
        "ressalva": "Canais setoriais são hipóteses estruturais; confirmar com dados históricos (historical_event_comparison).",
    }


@skill(
    id="geopolitical_risk_scoring",
    name="Avaliação de risco geopolítico",
    version="1.0.0",
    category="inteligencia",
    objective="Compor um índice 0-100 de risco geopolítico relevante para o Brasil a partir de eventos com fonte.",
    authorized_agents=[A.GEOPOLITICA, A.RISCO, A.CIIO],
    inputs={
        "events": "lista de dict: nome, severidade (1-5), probabilidade (0-1), exposicao_brasil (0-1), data (ISO), fonte",
        "half_life_days": "meia-vida da relevância (padrão 30)",
    },
    required_inputs=["events"],
    tools=["numpy"],
    procedure=[
        "Rejeitar eventos sem fonte.",
        "Escore individual = severidade/5 x probabilidade x (0,5+0,5 x exposição) x decaimento temporal.",
        "Índice = 100 x (1 - Π(1 - escore_i)).",
    ],
    validation=["Índice em [0, 100]."],
    output_schema={"indice": "float", "eventos": "lista ordenada", "rejeitados": "int"},
    failure_handling="Eventos inválidos são rejeitados e contados.",
)
def geopolitical_risk_scoring(run: SkillRun, events: list[dict], half_life_days: float = 30) -> dict:
    today = run.ctx.now.date()
    rows, rejected = [], 0
    for e in events:
        if not e.get("fonte"):
            rejected += 1
            continue
        try:
            sev, prob, exp_ = float(e["severidade"]), float(e["probabilidade"]), float(e.get("exposicao_brasil", 0.5))
            d = date.fromisoformat(e["data"])
        except (KeyError, ValueError):
            rejected += 1
            continue
        if not (1 <= sev <= 5 and 0 <= prob <= 1 and 0 <= exp_ <= 1):
            rejected += 1
            continue
        age = max((today - d).days, 0)
        decay = 0.5 ** (age / half_life_days)
        s = sev / 5 * prob * (0.5 + 0.5 * exp_) * decay
        rows.append({"nome": e["nome"], "escore": s, "idade_dias": age, "fonte": e["fonte"]})
    if rejected:
        run.warn(f"{rejected} evento(s) rejeitado(s) por falta de fonte ou campos inválidos.")
    idx = 100 * (1 - float(np.prod([1 - r["escore"] for r in rows]))) if rows else 0.0
    rows.sort(key=lambda r: -r["escore"])
    level = "baixo" if idx < 25 else ("moderado" if idx < 50 else ("elevado" if idx < 75 else "extremo"))
    return {"indice": idx, "nivel": level, "eventos": rows, "rejeitados": rejected}


@skill(
    id="seasonality_detection",
    name="Detecção de sazonalidade",
    version="1.0.0",
    category="inteligencia",
    objective="Testar padrões de calendário e exigir confirmação estatística fora da amostra.",
    authorized_agents=[A.SAZONAL, A.QUANT, A.AUDITOR],
    inputs={
        "prices": "Series diária",
        "by": "'month' (mês do ano) ou 'weekday' (dia da semana)",
        "oos_fraction": "fração final reservada para teste fora da amostra (padrão 0.4)",
        "alpha": "nível de significância antes da correção de Bonferroni (padrão 0.05)",
    },
    required_inputs=["prices"],
    tools=["numpy"],
    procedure=[
        "Calcular retornos por período (mensal composto ou diário por dia da semana).",
        "Dividir cronologicamente em dentro (IS) e fora da amostra (OOS).",
        "Para cada grupo: t = (média_grupo - média_geral) / (dp_grupo/√n) no IS.",
        "Significância com correção de Bonferroni pelo nº de grupos.",
        "Padrão confirmado somente se significativo no IS e com mesmo sinal e t ≥ 1,65 no OOS.",
    ],
    validation=["Ao menos 3 anos de dados para 'month'."],
    output_schema={"grupos": "lista", "padroes_confirmados": "lista", "padroes_rejeitados_oos": "lista"},
    failure_handling="Amostra insuficiente => falha explícita.",
)
def seasonality_detection(
    run: SkillRun, prices: Series, by: str = "month", oos_fraction: float = 0.4, alpha: float = 0.05
) -> dict:
    dates, v = prices.dates, prices.values
    if by == "month":
        month_end: dict[tuple[int, int], float] = {}
        for d, x in zip(dates, v):
            month_end[(d.year, d.month)] = x
        keys = sorted(month_end)
        if len(keys) < 37:
            raise SkillError("São necessários ao menos 3 anos de dados mensais.")
        rets = [(keys[i], month_end[keys[i]] / month_end[keys[i - 1]] - 1) for i in range(1, len(keys))]
        labels = {m: f"mês {m:02d}" for m in range(1, 13)}
        grp = [(k[1], r) for k, r in rets]
    elif by == "weekday":
        r = v[1:] / v[:-1] - 1
        grp = [(dates[i + 1].weekday(), float(r[i])) for i in range(len(r))]
        labels = {i: ["seg", "ter", "qua", "qui", "sex", "sáb", "dom"][i] for i in range(7)}
        if len(grp) < 500:
            raise SkillError("São necessários ao menos 500 retornos diários.")
    else:
        raise SkillError("Parâmetro 'by' deve ser 'month' ou 'weekday'.")

    cut = int(len(grp) * (1 - oos_fraction))
    is_, oos = grp[:cut], grp[cut:]
    groups = sorted({g for g, _ in grp})
    m_tests = len(groups)
    # valor crítico bicaudal com Bonferroni (aproximação normal; busca numérica)
    target = 1 - alpha / (2 * m_tests)
    crit = 0.0
    while norm_cdf(crit) < target:
        crit += 0.001

    def stats(sample: list, g: int) -> dict | None:
        xs = np.array([r for k, r in sample if k == g])
        allm = float(np.mean([r for _, r in sample]))
        if len(xs) < 3 or xs.std(ddof=1) == 0:
            return None
        t = (xs.mean() - allm) / (xs.std(ddof=1) / math.sqrt(len(xs)))
        return {"n": len(xs), "media": float(xs.mean()), "excesso": float(xs.mean() - allm), "t": float(t), "taxa_alta": float((xs > 0).mean())}

    rows, confirmed, rejected = [], [], []
    for g in groups:
        si, so = stats(is_, g), stats(oos, g)
        sig_is = si is not None and abs(si["t"]) >= crit
        conf = bool(sig_is and so is not None and np.sign(so["excesso"]) == np.sign(si["excesso"]) and abs(so["t"]) >= 1.65)
        rows.append({"grupo": labels[g], "dentro_amostra": si, "fora_amostra": so, "significativo_is": sig_is, "confirmado_oos": conf})
        if conf:
            confirmed.append(labels[g])
        elif sig_is:
            rejected.append(labels[g])
    if rejected:
        run.warn(f"Padrões significativos só dentro da amostra (provável sobreajuste): {', '.join(rejected)}", degrade=False)
    return {
        "agrupamento": by,
        "valor_critico_bonferroni": crit,
        "n_is": len(is_),
        "n_oos": len(oos),
        "grupos": rows,
        "padroes_confirmados": confirmed,
        "padroes_rejeitados_oos": rejected,
    }


@skill(
    id="historical_event_comparison",
    name="Comparação com eventos históricos",
    version="1.0.0",
    category="inteligencia",
    objective="Estudo de evento: retornos (e retornos anormais) após eventos históricos semelhantes, e posição do evento atual nessa distribuição.",
    authorized_agents=[A.NOTICIAS, A.GEOPOLITICA, A.SAZONAL, A.QUANT, A.ECON_BRASIL, A.ECON_GLOBAL, A.RISCO, A.CIIO],
    inputs={
        "prices": "Series do ativo",
        "event_dates": "lista de datas ISO de eventos históricos comparáveis",
        "benchmark": "Series de referência para retorno anormal (opcional)",
        "windows": "janelas em pregões (padrão [1, 5, 21, 63])",
        "current_event_date": "data do evento atual (opcional)",
    },
    required_inputs=["prices", "event_dates"],
    tools=["numpy"],
    procedure=[
        "Localizar o pregão de cada evento (primeiro pregão >= data).",
        "Retorno acumulado em cada janela; anormal = ativo - benchmark.",
        "Resumo: média, mediana, % positivos, pior e melhor caso.",
        "Para o evento atual: retorno realizado até agora e percentil histórico na janela correspondente.",
    ],
    validation=["Ao menos 3 eventos históricos com janela completa."],
    output_schema={"janelas": "dict", "eventos": "lista", "evento_atual": "dict | None"},
    failure_handling="Eventos sem janela completa são ignorados com aviso.",
)
def historical_event_comparison(
    run: SkillRun,
    prices: Series,
    event_dates: list[str],
    benchmark: Series | None = None,
    windows: list[int] | None = None,
    current_event_date: str | None = None,
) -> dict:
    windows = windows or [1, 5, 21, 63]
    pd_, pv = prices.dates, prices.values
    bmap = dict(zip(benchmark.dates, benchmark.values)) if benchmark is not None else None

    def locate(d: date) -> int | None:
        for i, x in enumerate(pd_):
            if x >= d:
                return i
        return None

    def window_ret(i0: int, w: int) -> tuple[float, float | None] | None:
        if i0 - 1 < 0 or i0 + w - 1 >= len(pv):
            return None
        base, end = pv[i0 - 1], pv[i0 + w - 1]
        r = end / base - 1
        ab = None
        if bmap is not None:
            b0, b1 = bmap.get(pd_[i0 - 1]), bmap.get(pd_[i0 + w - 1])
            if b0 and b1:
                ab = r - (b1 / b0 - 1)
        return r, ab

    events = []
    for ds in event_dates:
        i0 = locate(date.fromisoformat(ds))
        if i0 is None:
            run.warn(f"Evento {ds} fora do histórico.", degrade=False)
            continue
        events.append({"data": ds, "pregao": pd_[i0].isoformat(), "retornos": {w: window_ret(i0, w) for w in windows}})
    summary = {}
    for w in windows:
        rs = [e["retornos"][w][0] for e in events if e["retornos"][w] is not None]
        ab = [e["retornos"][w][1] for e in events if e["retornos"][w] is not None and e["retornos"][w][1] is not None]
        if len(rs) < 3:
            summary[w] = None
            continue
        a = np.array(rs)
        summary[w] = {
            "n": len(a),
            "media": float(a.mean()),
            "mediana": float(np.median(a)),
            "pct_positivos": float((a > 0).mean()),
            "pior": float(a.min()),
            "melhor": float(a.max()),
            "anormal_media": float(np.mean(ab)) if ab else None,
            "_dist": a,
        }
    if all(v is None for v in summary.values()):
        raise SkillError("Menos de 3 eventos com janelas completas.")
    current = None
    if current_event_date:
        i0 = locate(date.fromisoformat(current_event_date))
        if i0 is not None and i0 > 0:
            elapsed = len(pv) - i0
            w = max([x for x in windows if x <= elapsed], default=None)
            if w and summary.get(w):
                r = pv[i0 + w - 1] / pv[i0 - 1] - 1
                current = {"janela": w, "retorno": float(r), "percentil_historico": float((summary[w]["_dist"] < r).mean() * 100)}
    for v in summary.values():
        if v:
            v.pop("_dist")
    for e in events:
        e["retornos"] = {w: (None if x is None else {"retorno": x[0], "anormal": x[1]}) for w, x in e["retornos"].items()}
    return {"janelas": summary, "eventos": events, "evento_atual": current}

