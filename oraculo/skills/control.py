"""Skills de controle (25-32)."""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any, Callable

import numpy as np

from ..core import agent_ids as A
from ..core.types import Series
from ._math import align, brier, max_drawdown, pct_returns, reliability
from .base import SkillError, SkillRun, skill
from .knowledge import RUMOR_MARKERS, source_tier


def _perf(r: np.ndarray, periods: int = 252) -> dict:
    if len(r) == 0:
        return {}
    eq = np.cumprod(1 + r)
    years = len(r) / periods
    vol = float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else 0.0
    cagr = float(eq[-1] ** (1 / years) - 1) if years > 0 and eq[-1] > 0 else -1.0
    return {
        "retorno_total": float(eq[-1] - 1),
        "cagr": cagr,
        "volatilidade": vol,
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(periods)) if r.std(ddof=1) > 0 else 0.0,
        "max_drawdown": max_drawdown(np.concatenate([[1.0], eq])),
    }


@skill(
    id="backtesting_engine",
    name="Testes históricos",
    version="1.0.0",
    category="controle",
    objective="Simular uma regra de posição com defasagem de execução e custos, comparando com benchmark.",
    authorized_agents=[A.QUANT, A.AUDITOR, A.CARTEIRA, A.SAZONAL],
    inputs={
        "prices": "Series do ativo",
        "positions": "lista de posições em [-1, 1] decididas no fechamento de cada data",
        "cost_bps": "custo por unidade de giro em bps (corretagem+emolumentos+slippage), padrão 10",
        "benchmark": "Series de referência (opcional; padrão: comprar e manter o ativo)",
    },
    required_inputs=["prices", "positions"],
    tools=["numpy"],
    procedure=[
        "Executar a posição decidida em t somente no retorno t→t+1 (sem antecipação).",
        "Custo = |Δposição| x custo_bps.",
        "Métricas: retorno, CAGR, volatilidade, Sharpe, drawdown, giro, taxa de acerto, vs. benchmark.",
    ],
    validation=["Posições em [-1, 1].", "Mesmo tamanho dos preços."],
    output_schema={"estrategia": "dict", "benchmark": "dict", "giro_anual": "float", "custos_totais": "float"},
    failure_handling="Falha com menos de 60 pregões.",
)
def backtesting_engine(run: SkillRun, prices: Series, positions: list[float], cost_bps: float = 10.0, benchmark: Series | None = None) -> dict:
    p = prices.values
    pos = np.nan_to_num(np.asarray(positions, float))
    if len(pos) != len(p):
        raise SkillError("Posições e preços com tamanhos diferentes.")
    if (np.abs(pos) > 1 + 1e-9).any():
        raise SkillError("Posições devem estar em [-1, 1].")
    if len(p) < 60:
        raise SkillError("São necessários ao menos 60 pregões.")
    r = pct_returns(p)
    held = pos[:-1]  # posição decidida em t aplicada ao retorno t→t+1
    turnover = np.abs(np.diff(np.concatenate([[0.0], held])))
    costs = turnover * cost_bps / 1e4
    strat = held * r - costs
    if benchmark is not None:
        bmap = dict(zip(benchmark.dates, benchmark.values))
        b = np.array([bmap.get(d, np.nan) for d in prices.dates])
        bench_r = pct_returns(b)
        bench_r = bench_r[np.isfinite(bench_r)]
        bench_name = benchmark.name
    else:
        bench_r, bench_name = r, "comprar e manter"
    active = held != 0
    return {
        "estrategia": _perf(strat),
        "benchmark": {"nome": bench_name, **_perf(bench_r)},
        "excesso_retorno_total": _perf(strat).get("retorno_total", 0) - _perf(bench_r).get("retorno_total", 0),
        "giro_anual": float(turnover.sum() / len(r) * 252),
        "custos_totais": float(costs.sum()),
        "taxa_acerto": float((np.sign(held[active]) == np.sign(r[active])).mean()) if active.any() else None,
        "exposicao_media": float(np.abs(held).mean()),
        "n_pregoes": int(len(r)),
        "execucao": "posição decidida no fechamento de t aplicada ao retorno de t para t+1",
    }


@skill(
    id="walk_forward_validation",
    name="Validação cronológica",
    version="1.0.0",
    category="controle",
    objective="Avaliar modelos fora da amostra em janelas cronológicas com expurgo (purge) para evitar vazamento.",
    authorized_agents=[A.QUANT, A.AUDITOR],
    inputs={
        "X": "matriz de atributos (n x k)",
        "y": "rótulos binários (NaN quando desconhecidos)",
        "model_factory": "função sem argumentos que retorna objeto com fit(X, y) e predict_proba(X)",
        "horizon": "horizonte do rótulo em observações (define o expurgo)",
        "min_train": "tamanho mínimo do treino (padrão 252)",
        "step": "tamanho de cada bloco de teste (padrão 63)",
        "feature_available_at": "índice em que cada linha de atributos fica disponível (opcional, para checagem de vazamento)",
    },
    required_inputs=["X", "y", "model_factory", "horizon"],
    tools=["numpy"],
    procedure=[
        "Para cada bloco de teste [s, s+step): treinar com linhas i tais que i + horizonte < s (expurgo).",
        "Prever o bloco e acumular previsões fora da amostra.",
        "Métricas por bloco e agregadas: Brier, acerto direcional, comparação com taxa-base do treino.",
        "Checar vazamento: disponibilidade dos atributos ≤ índice da previsão.",
    ],
    validation=["Ao menos 3 blocos de teste."],
    output_schema={"blocos": "list", "agregado": "dict", "vazamento_detectado": "bool", "previsoes_oos": "list"},
    failure_handling="Dados insuficientes para 3 blocos => falha.",
    heavy=True,
)
def walk_forward_validation(
    run: SkillRun,
    X: Any,
    y: Any,
    model_factory: Callable[[], Any],
    horizon: int,
    min_train: int = 252,
    step: int = 63,
    feature_available_at: Any = None,
) -> dict:
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    valid = ~np.isnan(X).any(axis=1)
    leak = False
    if feature_available_at is not None:
        avail = np.asarray(feature_available_at)
        leak = bool((avail > np.arange(n)).any())
        if leak:
            run.warn("VAZAMENTO: atributos disponíveis após o momento da previsão.")
    folds, all_p, all_y, all_base, oos = [], [], [], [], []
    start = min_train
    while start < n:
        end = min(start + step, n)
        tr = np.arange(n)[valid & ~np.isnan(y) & (np.arange(n) + horizon < start)]
        te = np.arange(start, end)
        te = te[valid[te] & ~np.isnan(y[te])]
        if len(tr) >= min_train // 2 and len(te) > 0:
            m = model_factory().fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            base = float(y[tr].mean())
            folds.append(
                {
                    "inicio": int(start),
                    "fim": int(end),
                    "n_treino": int(len(tr)),
                    "n_teste": int(len(te)),
                    "brier": brier(p, y[te]),
                    "brier_taxa_base": brier(np.full(len(te), base), y[te]),
                    "acerto": float(((p > 0.5) == (y[te] == 1)).mean()),
                }
            )
            all_p.extend(p.tolist())
            all_y.extend(y[te].tolist())
            all_base.extend([base] * len(te))
            oos.extend({"i": int(i), "prob": float(pp), "y": float(yy)} for i, pp, yy in zip(te, p, y[te]))
        start = end
    if len(folds) < 3:
        raise SkillError(f"Apenas {len(folds)} bloco(s) de teste; mínimo 3.")
    P, Y, B = np.array(all_p), np.array(all_y), np.array(all_base)
    table, ece = reliability(P, Y)
    agg = {
        "n_previsoes": int(len(P)),
        "brier": brier(P, Y),
        "brier_taxa_base": brier(B, Y),
        "habilidade_brier": 1 - brier(P, Y) / brier(B, Y) if brier(B, Y) > 0 else 0.0,
        "acerto": float(((P > 0.5) == (Y == 1)).mean()),
        "acerto_sempre_alta": float(Y.mean()),
        "ece": ece,
        "blocos_melhores_que_base": int(sum(f["brier"] < f["brier_taxa_base"] for f in folds)),
        "n_blocos": len(folds),
    }
    return {"blocos": folds, "agregado": agg, "calibracao": table, "vazamento_detectado": leak, "previsoes_oos": oos, "expurgo": horizon}


@skill(
    id="portfolio_risk_analysis",
    name="Risco de carteira",
    version="1.0.0",
    category="controle",
    objective="Medir concentração, volatilidade, VaR/CVaR, correlações, contribuição de risco, drawdown e liquidez da carteira.",
    authorized_agents=[A.RISCO, A.CARTEIRA, A.CIIO],
    inputs={
        "positions": "dict[ticker, valor em R$]",
        "prices": "dict[ticker, Series]",
        "sectors": "dict[ticker, setor] (opcional)",
        "adv": "dict[ticker, volume financeiro médio diário em R$] (opcional)",
        "participation": "fração máxima do volume diário negociável (padrão 0.2)",
    },
    required_inputs=["positions", "prices"],
    tools=["numpy"],
    procedure=[
        "Pesos, HHI e número efetivo de ativos.",
        "Retornos alinhados: volatilidade anual, VaR/CVaR histórico 95% de 1 dia, drawdown da carteira.",
        "Matriz de correlação, pares mais correlacionados e contribuição marginal ao risco.",
        "Concentração setorial e dias para liquidar cada posição.",
    ],
    validation=["Pesos somam 1.", "Contribuições de risco somam ~100%."],
    output_schema={"pesos": "dict", "concentracao": "dict", "risco": "dict", "correlacoes": "dict", "liquidez": "dict", "alertas": "list"},
    failure_handling="Ativos sem série são excluídos das métricas de risco e listados.",
)
def portfolio_risk_analysis(
    run: SkillRun,
    positions: dict,
    prices: dict,
    sectors: dict | None = None,
    adv: dict | None = None,
    participation: float = 0.2,
) -> dict:
    total = float(sum(positions.values()))
    if total <= 0:
        raise SkillError("Carteira sem valor positivo.")
    w_all = {t: v / total for t, v in positions.items()}
    hhi = float(sum(w**2 for w in w_all.values()))
    alerts = [f"Concentração: {t} representa {w:.0%} da carteira" for t, w in w_all.items() if w > 0.25]
    out: dict = {
        "valor_total": total,
        "pesos": w_all,
        "concentracao": {"hhi": hhi, "n_efetivo": 1 / hhi, "maior_peso": max(w_all.values())},
    }
    if sectors:
        sw: dict[str, float] = {}
        for t, w in w_all.items():
            sw[sectors.get(t, "não classificado")] = sw.get(sectors.get(t, "não classificado"), 0) + w
        out["concentracao"]["setores"] = sw
        alerts += [f"Concentração setorial: {s} = {w:.0%}" for s, w in sw.items() if w > 0.40]
    tickers = [t for t in positions if t in prices]
    missing = [t for t in positions if t not in prices]
    if missing:
        run.warn(f"Sem série de preços: {', '.join(missing)} (fora das métricas de risco).")
    if len(tickers) >= 1:
        _, arrs = align(*[prices[t] for t in tickers])
        if len(arrs[0]) < 60:
            raise SkillError("Histórico comum inferior a 60 pregões.")
        R = np.column_stack([pct_returns(a) for a in arrs])
        w = np.array([positions[t] for t in tickers], float)
        w = w / w.sum()
        port = R @ w
        cov = np.cov(R, rowvar=False) if len(tickers) > 1 else np.array([[R[:, 0].var(ddof=1)]])
        pvar = float(w @ cov @ w)
        mrc = cov @ w / np.sqrt(pvar) if pvar > 0 else np.zeros(len(w))
        rc = w * mrc / np.sqrt(pvar) if pvar > 0 else np.zeros(len(w))
        q = np.percentile(port, 5)
        vols = np.sqrt(np.diag(cov))
        out["risco"] = {
            "volatilidade_anual": float(np.sqrt(pvar * 252)),
            "var95_1d_pct": float(-q),
            "cvar95_1d_pct": float(-port[port <= q].mean()),
            "var95_1d_rs": float(-q * total),
            "max_drawdown_historico": max_drawdown(np.cumprod(1 + port)),
            "razao_diversificacao": float((w @ vols) / np.sqrt(pvar)) if pvar > 0 else None,
            "contribuicao_risco": dict(zip(tickers, rc.tolist())),
            "n_pregoes": int(len(port)),
        }
        if len(tickers) > 1:
            C = np.corrcoef(R, rowvar=False)
            pairs = sorted(
                ((tickers[i], tickers[j], float(C[i, j])) for i in range(len(tickers)) for j in range(i + 1, len(tickers))),
                key=lambda x: -x[2],
            )
            out["correlacoes"] = {
                "matriz": {tickers[i]: dict(zip(tickers, C[i].tolist())) for i in range(len(tickers))},
                "pares_mais_correlacionados": [{"a": a, "b": b, "corr": c} for a, b, c in pairs[:5]],
                "correlacao_media": float(C[np.triu_indices(len(tickers), 1)].mean()),
            }
            alerts += [f"Correlação elevada: {a}/{b} = {c:.2f}" for a, b, c in pairs if c > 0.8]
    if adv:
        liq = {}
        for t, v in positions.items():
            if adv.get(t):
                days = v / (adv[t] * participation)
                liq[t] = days
                if days > 5:
                    alerts.append(f"Liquidez: {t} levaria {days:.1f} pregões para liquidar a {participation:.0%} do volume")
        out["liquidez"] = {"dias_para_liquidar": liq, "participacao_maxima": participation}
    out["alertas"] = alerts
    return out


@portfolio_risk_analysis.validator
def _weights_sum(out: dict) -> list[str]:
    s = sum(out["pesos"].values())
    errs = [] if abs(s - 1) < 1e-6 else [f"pesos somam {s}"]
    rc = out.get("risco", {}).get("contribuicao_risco")
    if rc and abs(sum(rc.values()) - 1) > 1e-3:
        errs.append("contribuições de risco não somam 100%")
    return errs


@skill(
    id="prediction_audit",
    name="Auditoria das previsões",
    version="1.0.0",
    category="controle",
    objective="Avaliar o desempenho real das previsões registradas: acerto, erro, calibração, retorno líquido, por horizonte e vs. benchmarks.",
    authorized_agents=[A.AUDITOR, A.CIIO, A.QUANT],
    inputs={
        "predictions": "lista de dict: id, ticker, made_at, horizon, prob_up, expected_return (opcional), model_version, "
        "realized_return (None se pendente), benchmark_return (opcional)",
        "cost_bps": "custo de ida e volta em bps (padrão 20)",
        "long_only": "simulação apenas comprada (padrão False)",
    },
    required_inputs=["predictions"],
    tools=["numpy"],
    procedure=[
        "Separar previsões vencidas e pendentes.",
        "Acerto direcional, erro médio absoluto do retorno esperado, Brier, ECE.",
        "Retorno hipotético: posição = sinal(prob-0,5) (ou só compra), menos custos.",
        "Quebrar por horizonte e por versão de modelo; comparar com 'sempre alta' e benchmark.",
        "Listar as falhas mais relevantes (erros com maior confiança).",
    ],
    validation=["Contagens consistentes."],
    output_schema={"total": "int", "vencidas": "int", "acerto_direcional": "float", "brier": "float", "por_horizonte": "dict", "piores_falhas": "list"},
    failure_handling="Sem previsões vencidas => retorna contagens e aviso, sem métricas.",
)
def prediction_audit(run: SkillRun, predictions: list[dict], cost_bps: float = 20.0, long_only: bool = False) -> dict:
    done = [p for p in predictions if p.get("realized_return") is not None]
    out: dict = {"total": len(predictions), "vencidas": len(done), "pendentes": len(predictions) - len(done)}
    if not done:
        run.warn("Nenhuma previsão vencida para auditar.", degrade=False)
        return out

    def metrics(rows: list[dict]) -> dict:
        P = np.array([r["prob_up"] for r in rows], float)
        R = np.array([r["realized_return"] for r in rows], float)
        Y = (R > 0).astype(float)
        pos = np.where(P > 0.5, 1.0, 0.0 if long_only else -1.0)
        net = pos * R - np.abs(pos) * cost_bps / 1e4
        m = {
            "n": len(rows),
            "acerto_direcional": float(((P > 0.5) == (Y == 1)).mean()),
            "acerto_sempre_alta": float(Y.mean()),
            "brier": brier(P, Y),
            "brier_taxa_base": brier(np.full(len(Y), Y.mean()), Y),
            "retorno_hipotetico_medio_liquido": float(net.mean()),
            "retorno_hipotetico_acumulado_liquido": float(np.prod(1 + net) - 1),
        }
        er = [(r["expected_return"], r["realized_return"]) for r in rows if r.get("expected_return") is not None]
        m["erro_medio_absoluto"] = float(np.mean([abs(a - b) for a, b in er])) if er else None
        bm = [r["benchmark_return"] for r in rows if r.get("benchmark_return") is not None]
        m["retorno_medio_benchmark"] = float(np.mean(bm)) if bm else None
        return m

    out.update(metrics(done))
    P = np.array([r["prob_up"] for r in done])
    Y = np.array([r["realized_return"] > 0 for r in done], float)
    out["calibracao"], out["ece"] = reliability(P, Y)
    by_h: dict[str, list] = {}
    by_m: dict[str, list] = {}
    for r in done:
        by_h.setdefault(str(r.get("horizon")), []).append(r)
        by_m.setdefault(str(r.get("model_version")), []).append(r)
    out["por_horizonte"] = {h: metrics(rs) for h, rs in by_h.items()}
    out["por_modelo"] = {m: metrics(rs) for m, rs in by_m.items()}
    wrong = [r for r in done if (r["prob_up"] > 0.5) != (r["realized_return"] > 0)]
    wrong.sort(key=lambda r: -abs(r["prob_up"] - 0.5))
    out["piores_falhas"] = [
        {k: r.get(k) for k in ("id", "ticker", "made_at", "horizon", "prob_up", "realized_return", "model_version")} for r in wrong[:5]
    ]
    if len(done) < 30:
        run.warn(f"Apenas {len(done)} previsões vencidas: métricas com alta incerteza amostral.")
    return out


@skill(
    id="data_quality_validation",
    name="Qualidade dos dados",
    version="1.0.0",
    category="controle",
    objective="Detectar faltas, duplicidades, datas fora de ordem, valores inválidos, outliers e atrasos nas séries.",
    authorized_agents=list(A.ALL),
    inputs={"series": "lista de Series", "positive_only": "valores devem ser positivos (preços), padrão True", "outlier_z": "limiar robusto (padrão 8)"},
    required_inputs=["series"],
    tools=["numpy"],
    procedure=[
        "Checar procedência, tamanho e ordem cronológica.",
        "Duplicidades de data, NaN/infinito, valores não positivos.",
        "Outliers por z robusto (MAD) nos retornos.",
        "Lacunas > 7 dias corridos em séries diárias e atraso da última observação.",
    ],
    validation=["Um relatório por série."],
    output_schema={"series": "lista de relatórios", "aprovadas": "int", "reprovadas": "int"},
    failure_handling="Nunca corrige dados; apenas reporta. Série com erro crítico é reprovada.",
)
def data_quality_validation(run: SkillRun, series: list[Series], positive_only: bool = True, outlier_z: float = 8.0) -> dict:
    reports = []
    for s in series:
        crit, warn = [], []
        v = s.values
        if len(v) == 0:
            crit.append("série vazia")
        if any(s.dates[i] >= s.dates[i + 1] for i in range(len(s.dates) - 1)):
            dup = len(s.dates) - len(set(s.dates))
            crit.append(f"datas fora de ordem ou duplicadas ({dup} duplicadas)")
        if not np.isfinite(v).all():
            crit.append(f"{int((~np.isfinite(v)).sum())} valores NaN/infinitos")
        if positive_only and (v[np.isfinite(v)] <= 0).any():
            crit.append("valores não positivos")
        if len(v) > 10 and np.isfinite(v).all() and (v > 0).all():
            r = np.diff(np.log(v))
            mad = np.median(np.abs(r - np.median(r))) * 1.4826
            if mad > 0:
                z = np.abs(r - np.median(r)) / mad
                idx = np.where(z > outlier_z)[0]
                if len(idx):
                    warn.append(f"{len(idx)} variação(ões) extrema(s), ex.: {s.dates[idx[0] + 1].isoformat()}")
        if s.frequency.value == "daily" and len(s.dates) > 1:
            gaps = [(s.dates[i], s.dates[i + 1]) for i in range(len(s.dates) - 1) if (s.dates[i + 1] - s.dates[i]) > timedelta(days=7)]
            if gaps:
                warn.append(f"{len(gaps)} lacuna(s) > 7 dias, ex.: {gaps[0][0]} → {gaps[0][1]}")
        stale = s.is_stale(run.ctx.now)
        if stale:
            warn.append(f"dado atrasado (última observação {s.last_date})")
        if s.synthetic:
            warn.append("dado sintético")
        reports.append(
            {"serie": s.name, "fonte": s.source, "n": len(s), "ultima_data": s.last_date.isoformat() if s.last_date else None,
             "atrasada": stale, "criticos": crit, "avisos": warn, "aprovada": not crit}
        )
    return {"series": reports, "aprovadas": sum(r["aprovada"] for r in reports), "reprovadas": sum(not r["aprovada"] for r in reports)}


@skill(
    id="source_verification",
    name="Verificação de fontes",
    version="1.0.0",
    category="controle",
    objective="Classificar cada afirmação como confirmada, parcialmente confirmada, não verificada ou rumor, segundo fontes independentes.",
    authorized_agents=[A.NOTICIAS, A.COMPLIANCE, A.AUDITOR, A.CIIO, A.GEOPOLITICA],
    inputs={"claims": "lista de dict: afirmacao, fontes: [{nome, url, publicado_em, texto (opcional)}]"},
    required_inputs=["claims"],
    tools=["knowledge.source_tier"],
    procedure=[
        "Classificar o nível de cada fonte (1 oficial, 2 imprensa profissional, 3 outras).",
        "Contar fontes independentes (domínios distintos).",
        "Confirmada: ≥1 oficial ou ≥2 de nível 2. Parcial: 1 de nível 2. Rumor: marcadores de rumor sem fonte oficial.",
    ],
    validation=["Toda afirmação recebe um veredito."],
    output_schema={"afirmacoes": "lista com veredito, fontes e justificativa"},
    failure_handling="Afirmação sem fontes => 'não verificada'.",
)
def source_verification(run: SkillRun, claims: list[dict]) -> dict:
    from urllib.parse import urlparse

    rows = []
    for c in claims:
        srcs = c.get("fontes") or []
        tiers, domains, rumor = [], set(), False
        for s in srcs:
            tiers.append(source_tier(s.get("nome", ""), s.get("url")))
            domains.add(urlparse(s.get("url") or "").netloc or s.get("nome", "").lower())
            txt = (s.get("texto") or "").lower() + " " + c.get("afirmacao", "").lower()
            rumor = rumor or any(m in txt for m in RUMOR_MARKERS)
        n1, n2 = tiers.count(1), tiers.count(2)
        if n1 >= 1:
            verdict, why = "confirmada", "fonte oficial"
        elif rumor:
            verdict, why = "rumor", "linguagem de rumor sem confirmação oficial"
        elif n2 >= 2 and len(domains) >= 2:
            verdict, why = "confirmada", f"{n2} veículos profissionais independentes"
        elif n2 == 1:
            verdict, why = "parcialmente confirmada", "apenas um veículo profissional"
        else:
            verdict, why = "não verificada", "sem fontes confiáveis"
        rows.append({"afirmacao": c.get("afirmacao"), "veredito": verdict, "justificativa": why, "niveis_fontes": tiers, "fontes_independentes": len(domains)})
    return {"afirmacoes": rows}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    edges = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref)
    b = np.histogram(cur, edges)[0] / len(cur)
    a, b = np.clip(a, 1e-4, None), np.clip(b, 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())


@skill(
    id="model_drift_detection",
    name="Identificação de deterioração dos modelos",
    version="1.0.0",
    category="controle",
    objective="Detectar mudança na distribuição dos atributos (PSI) e piora do desempenho recente dos modelos.",
    authorized_agents=[A.AUDITOR, A.QUANT, A.ORQUESTRADOR],
    inputs={
        "reference_features": "matriz de atributos do período de validação",
        "recent_features": "matriz de atributos recentes",
        "feature_names": "nomes das colunas",
        "reference_brier": "Brier na validação (opcional)",
        "recent_probs": "probabilidades recentes (opcional)",
        "recent_outcomes": "resultados recentes (opcional)",
    },
    required_inputs=["reference_features", "recent_features"],
    tools=["numpy (PSI)"],
    procedure=[
        "PSI por atributo com 10 faixas de quantis da referência (<0,1 estável; 0,1-0,25 moderado; >0,25 relevante).",
        "Brier recente vs. referência; piora > 10% sinaliza deterioração.",
        "Recomendar revalidação quando houver deriva relevante.",
    ],
    validation=["PSI não negativo."],
    output_schema={"psi": "dict", "deriva_relevante": "list", "piora_desempenho": "bool | None", "recomendacao": "str"},
    failure_handling="Sem resultados recentes, avalia apenas a deriva de atributos.",
)
def model_drift_detection(
    run: SkillRun,
    reference_features: Any,
    recent_features: Any,
    feature_names: list[str] | None = None,
    reference_brier: float | None = None,
    recent_probs: list[float] | None = None,
    recent_outcomes: list[int] | None = None,
) -> dict:
    R, C = np.asarray(reference_features, float), np.asarray(recent_features, float)
    if R.ndim == 1:
        R, C = R[:, None], C[:, None]
    names = feature_names or [f"x{i}" for i in range(R.shape[1])]
    scores = {n: psi(R[:, i], C[:, i]) for i, n in enumerate(names)}
    relevant = [n for n, s in scores.items() if s > 0.25]
    moderate = [n for n, s in scores.items() if 0.1 < s <= 0.25]
    degraded = None
    recent_brier = None
    if reference_brier is not None and recent_probs is not None and recent_outcomes is not None and len(recent_probs) >= 20:
        recent_brier = brier(np.array(recent_probs), np.array(recent_outcomes))
        degraded = recent_brier > reference_brier * 1.10
    rec = (
        "Suspender uso em produção e revalidar."
        if (degraded or len(relevant) >= 2)
        else ("Revalidar no próximo ciclo." if relevant or moderate else "Sem ação necessária.")
    )
    return {"psi": scores, "deriva_relevante": relevant, "deriva_moderada": moderate, "brier_recente": recent_brier,
            "brier_referencia": reference_brier, "piora_desempenho": degraded, "recomendacao": rec}


# --- Compliance -----------------------------------------------------------------

DISCLAIMER = (
    "Este conteúdo tem caráter informativo e educacional, não constitui recomendação de compra ou venda "
    "de valores mobiliários nem garantia de resultados. Previsões são probabilísticas e podem falhar. "
    "Rentabilidade passada não garante rentabilidade futura."
)
ORDER_PATTERNS = (
    r"\bcompre\b", r"\bvenda já\b", r"\bvenda agora\b", r"\bcompre já\b", r"\bcompre agora\b", r"\bzere\b",
    r"\baproveite antes\b", r"\bnão perca\b", r"\bbuy now\b", r"\bsell now\b",
)
CERTAINTY_PATTERNS = (r"\bgarantid[oa]s?\b", r"\bcerteza\b", r"\bsem risco\b", r"\bvai subir\b", r"\bvai cair\b", r"\bimposs[ií]vel perder\b", r"\blucro certo\b")
PII_PATTERNS = {"cpf": r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", "telefone": r"\(\d{2}\)\s?\d{4,5}-\d{4}"}


@skill(
    id="compliance_review",
    name="Revisão regulatória",
    version="1.0.0",
    category="controle",
    objective="Bloquear relatórios que configurem ordem de compra/venda, promessa de resultado, falta de fontes, dados sintéticos não sinalizados ou dados pessoais.",
    authorized_agents=[A.COMPLIANCE, A.CIIO, A.COMUNICADOR],
    inputs={
        "text": "texto do relatório",
        "sources_count": "número de fontes citadas",
        "uses_synthetic": "se o relatório usa dados sintéticos",
        "stale_inputs": "se há dados atrasados",
        "licensed_sources": "lista de fontes cujo direito de uso foi confirmado (opcional)",
        "sources": "lista de nomes de fontes usadas (opcional)",
    },
    required_inputs=["text"],
    tools=["regex"],
    procedure=[
        "Verificar aviso legal (caráter informativo, sem recomendação, sem garantia).",
        "Bloquear linguagem de ordem ('compre', 'venda agora') e de certeza ('garantido', 'vai subir').",
        "Exigir citação de fontes; exigir aviso de dado sintético/atrasado quando aplicável.",
        "Detectar dados pessoais (CPF, e-mail, telefone) - LGPD (Lei 13.709/2018).",
        "Sinalizar fontes sem confirmação de direito de uso.",
        "Referências: Resolução CVM 20/2021 (análise de valores mobiliários) e Resolução CVM 19/2021 (consultoria).",
    ],
    validation=["Veredito aprovado/reprovado com lista de pendências."],
    output_schema={"aprovado": "bool", "pendencias": "list[{gravidade, regra, detalhe}]", "aviso_legal": "str"},
    failure_handling="Na dúvida, reprova e descreve a pendência.",
)
def compliance_review(
    run: SkillRun,
    text: str,
    sources_count: int = 0,
    uses_synthetic: bool = False,
    stale_inputs: bool = False,
    licensed_sources: list[str] | None = None,
    sources: list[str] | None = None,
) -> dict:
    low = text.lower()
    issues = []

    def add(sev: str, rule: str, detail: str) -> None:
        issues.append({"gravidade": sev, "regra": rule, "detalhe": detail})

    if "não constitui recomendação" not in low:
        add("bloqueante", "aviso_legal", "Aviso legal ausente.")
    for pat in ORDER_PATTERNS:
        m = re.search(pat, low)
        if m:
            add("bloqueante", "ordem_de_negociacao", f"Linguagem de ordem: '{m.group(0)}'.")
    for pat in CERTAINTY_PATTERNS:
        for m in re.finditer(pat, low):
            ctx = low[max(0, m.start() - 25) : m.start()]
            if "não" in ctx or "nem" in ctx or "sem garantia" in ctx:
                continue
            add("bloqueante", "promessa_de_resultado", f"Linguagem de certeza: '{m.group(0)}'.")
    if sources_count == 0:
        add("bloqueante", "fontes", "Nenhuma fonte citada.")
    if uses_synthetic and "sintétic" not in low:
        add("bloqueante", "dado_sintetico", "Relatório usa dados sintéticos sem sinalização.")
    if stale_inputs and "atrasad" not in low:
        add("alta", "dado_atrasado", "Relatório usa dados atrasados sem sinalização.")
    for name, pat in PII_PATTERNS.items():
        if re.search(pat, text):
            add("bloqueante", "lgpd", f"Possível dado pessoal ({name}) no relatório.")
    if sources and licensed_sources is not None:
        unl = [s for s in sources if not any(l.lower() in s.lower() for l in licensed_sources)]
        if unl:
            add("media", "direito_de_uso", f"Direito de uso não confirmado: {', '.join(sorted(set(unl)))[:300]}")
    return {
        "aprovado": not any(i["gravidade"] in ("bloqueante", "alta") for i in issues),
        "pendencias": issues,
        "aviso_legal": DISCLAIMER,
        "referencias": ["Resolução CVM 20/2021", "Resolução CVM 19/2021", "Lei 13.709/2018 (LGPD)"],
    }

