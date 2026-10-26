"""Lógica analítica dos agentes especialistas.

Cada função recebe o dossiê da análise, executa as skills que o agente tem
permissão de usar e devolve um `AgentView` com posição, confiança, evidências
e riscos. Quando faltam dados, a posição é "evidência insuficiente" -
nunca uma suposição.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core import agent_ids as A
from ..core.types import AgentView, NewsItem, Series, Stance
from ..skills.base import SkillContext, SkillResult, run_skill
from ..skills.knowledge import EVENT_SECTOR_MAP


@dataclass
class Dossier:
    """Evidências compartilhadas de uma análise (somente leitura para os especialistas)."""

    ticker: str
    ctx: SkillContext
    prices: Series | None = None
    volume: Series | None = None
    benchmark: Series | None = None
    macro: dict[str, Series] = field(default_factory=dict)
    global_macro: dict[str, Series] = field(default_factory=dict)
    commodities: dict[str, Series] = field(default_factory=dict)
    news: list[NewsItem] = field(default_factory=list)
    geo_events: list[dict] = field(default_factory=list)
    financials: dict | None = None
    financials_source: str | None = None
    valuation_inputs: dict | None = None
    peers: dict | None = None
    sector: str | None = None
    horizon: int = 5
    results: dict[str, SkillResult] = field(default_factory=dict)
    collection_errors: list[str] = field(default_factory=list)

    def run(self, skill_id: str, agent_id: str, key: str | None = None, **inputs: Any) -> SkillResult:
        r = run_skill(skill_id, agent_id, self.ctx, **inputs)
        self.results[key or f"{agent_id}:{skill_id}"] = r
        return r


def _insufficient(agent_id: str, why: str) -> AgentView:
    return AgentView(agent_id, Stance.INSUFFICIENT, 0.0, why, warnings=[why])


def _view(agent_id: str, score: float, conf: float, summary: str, evidence: list[str], risks: list[str], results: list[SkillResult]) -> AgentView:
    """score em [-1, 1] -> posição; |score| < 0,2 => neutra."""
    stance = Stance.POSITIVE if score >= 0.2 else (Stance.NEGATIVE if score <= -0.2 else Stance.NEUTRAL)
    v = AgentView(agent_id, stance, float(np.clip(conf, 0, 1)), summary, evidence, risks)
    for r in results:
        v.sources.extend(r.sources)
        v.skill_results.append(r.run_id)
        v.warnings.extend(r.warnings)
    return v


def _failed(agent_id: str, r: SkillResult) -> AgentView:
    v = _insufficient(agent_id, f"{r.skill_id} sem resultado: {r.error}")
    v.skill_results.append(r.run_id)
    return v


# --- Economia e mercados ------------------------------------------------------


def economista_brasil(d: Dossier) -> AgentView:
    ag = A.ECON_BRASIL
    if not d.macro:
        return _insufficient(ag, "Indicadores macro brasileiros indisponíveis.")
    mon = d.run("brazil_macro_monitor", ag, indicators=d.macro)
    if not mon.ok:
        return _failed(ag, mon)
    results = [mon]
    evidence = [f"{i['serie']}: {i['ultimo']:.4g} em {i['data']}" for i in mon.output["indicadores"]]
    jr = mon.output.get("juro_real_ex_post")
    if jr is not None:
        evidence.append(f"Juro real ex-post: {jr:.2f}% a.a.")
    phase = None
    if "selic_meta" in d.macro:
        cb = d.run("central_bank_analysis", ag, policy_rate=d.macro["selic_meta"], inflation_12m=d.macro.get("ipca_12m"), inflation_target=3.0)
        if cb.ok:
            results.append(cb)
            phase = cb.output["fase"]
            evidence.append(f"Copom: fase de {phase}; última decisão {cb.output['ultima_decisao_bps']:+.0f} bps")
    if not d.sector:
        return _view(ag, 0.0, 0.3, "Diagnóstico macro sem setor da empresa para mapear impacto.", evidence, [], results)
    event = None
    if phase == "aperto" or (jr is not None and jr > 7):
        event = "alta_juros_brasil"
    elif phase == "afrouxamento":
        event = "queda_juros_brasil"
    score, risks = 0.0, []
    if event:
        m = d.run("event_impact_mapping", ag, event_type=event, companies={d.ticker: {"setor": d.sector}})
        if m.ok:
            results.append(m)
            row = m.output["empresas"][0]
            score = row["direcao_hipotese"] * row["intensidade"] / 3
            if row["canal"]:
                evidence.append(f"Canal ({event}): {row['canal']} [hipótese setorial]")
                if score < 0:
                    risks.append(f"Ambiente de juros: {row['canal']}")
    summary = f"Juros {'em ' + phase if phase else 'sem fase identificada'}; impacto estimado para o setor {d.sector}: " + (
        "favorável" if score > 0 else "desfavorável" if score < 0 else "neutro"
    )
    return _view(ag, score, 0.45, summary, evidence, risks, results)


def economista_global(d: Dossier) -> AgentView:
    ag = A.ECON_GLOBAL
    if not d.global_macro:
        return _insufficient(ag, "Nenhuma série macro global configurada (requer fonte licenciada/CSV).")
    r = d.run("global_macro_monitor", ag, indicators=d.global_macro)
    if not r.ok:
        return _failed(ag, r)
    n = r.output["n_materiais"]
    return _view(ag, 0.0, 0.3, f"{n} mudança(s) material(is) em indicadores globais.", r.output["alertas"] or ["Sem mudanças materiais"], r.output["alertas"], [r])


def geopolitica(d: Dossier) -> AgentView:
    ag = A.GEOPOLITICA
    if not d.geo_events:
        return _insufficient(ag, "Nenhum evento geopolítico com fonte informado.")
    r = d.run("geopolitical_risk_scoring", ag, events=d.geo_events)
    if not r.ok:
        return _failed(ag, r)
    idx = r.output["indice"]
    score = -min(idx / 75, 1.0) if idx >= 25 else 0.0
    ev = [f"Índice geopolítico {idx:.0f}/100 ({r.output['nivel']})"] + [f"{e['nome']} ({e['fonte']})" for e in r.output["eventos"][:3]]
    return _view(ag, score, 0.35, f"Risco geopolítico {r.output['nivel']}.", ev, ev[1:] if score < 0 else [], [r])


def commodities(d: Dossier) -> AgentView:
    ag = A.COMMODITIES
    if not d.commodities or d.prices is None:
        return _insufficient(ag, "Séries de commodities não configuradas.")
    results, ev, risks = [], [], []
    for name, s in d.commodities.items():
        r = d.run("commodity_exposure", ag, key=f"{ag}:commodity_exposure:{name}", stock_prices=d.prices, commodity_prices=s)
        if r.ok:
            results.append(r)
            b, t = r.output["beta"], r.output["t_stat"]
            ev.append(f"Beta a {name}: {b:.2f} (t={t:.1f}, R²={r.output['r2']:.2f})")
            if t and abs(t) > 2 and abs(b) > 0.2:
                risks.append(f"Sensibilidade relevante a {name} (beta {b:.2f})")
    if not results:
        return _insufficient(ag, "Nenhuma sensibilidade a commodities estimável.")
    return _view(ag, 0.0, 0.4, "Exposições a commodities medidas (sem direção projetada para as commodities).", ev, risks, results)


# --- Análise de ativos --------------------------------------------------------


def fundamentalista(d: Dossier) -> AgentView:
    ag = A.FUNDAMENTALISTA
    if not d.financials or not d.financials_source:
        return _insufficient(ag, "Demonstrações financeiras não fornecidas (CVM/RI).")
    r = d.run("fundamental_analysis", ag, financials=d.financials, financials_source=d.financials_source)
    if not r.ok:
        return _failed(ag, r)
    q = r.output["pontuacao_qualidade"]
    ind = r.output["indicadores"]
    ev = [f"Pontuação de qualidade {q}/6"] + [
        f"{k}: {v:.2%}" if k.startswith(("margem", "ro", "cresc")) else f"{k}: {v:.2f}"
        for k, v in ind.items() if v is not None and k in ("roe", "roic", "margem_ebitda", "divida_liquida_ebitda", "crescimento_receita")
    ]
    score = (q - 3) / 3
    results = [r]
    if d.valuation_inputs:
        price = d.prices.last_value if d.prices is not None else None
        v = d.run("valuation_engine", ag, price=price, **d.valuation_inputs)
        if v.ok:
            results.append(v)
            up = v.output.get("upside_dcf")
            if up is not None:
                ev.append(f"DCF: R$ {v.output['valor_por_acao_dcf']:.2f}/ação ({up:+.0%} vs. preço)")
                score = float(np.clip(score + np.clip(up, -0.5, 0.5), -1, 1))
    return _view(ag, score, 0.55, f"Qualidade fundamentalista {q}/6.", ev, r.output["alertas"], results)


def tecnico(d: Dossier) -> AgentView:
    ag = A.TECNICO
    if d.prices is None:
        return _insufficient(ag, "Cotações indisponíveis.")
    r = d.run("technical_analysis", ag, prices=d.prices, volume=d.volume)
    if not r.ok:
        return _failed(ag, r)
    o = r.output
    score = {"alta": 0.5, "baixa": -0.5}.get(o["tendencia"], 0.0)
    risks = []
    if o["rsi14"] > 70:
        score -= 0.2
        risks.append(f"RSI {o['rsi14']:.0f}: sobrecompra")
    elif o["rsi14"] < 30:
        score += 0.2
        risks.append(f"RSI {o['rsi14']:.0f}: sobrevenda (tendência pode persistir)")
    if o["drawdown_atual"] < -0.2:
        risks.append(f"Ação {o['drawdown_atual']:.0%} abaixo do topo do período")
    ev = [
        f"Tendência: {o['tendencia']} (preço {o['preco']:.2f} vs. MMS50 {o['mms50']:.2f})",
        f"RSI14 {o['rsi14']:.0f}; MACD hist. {o['macd']['histograma']:+.3f}",
        f"Volatilidade 21d anualizada {o['volatilidade_21d_anual']:.0%}",
        f"Suportes {', '.join(f'{x:.2f}' for x in o['suportes']) or 'n/d'}; resistências {', '.join(f'{x:.2f}' for x in o['resistencias']) or 'n/d'}",
    ]
    return _view(ag, score, 0.4, f"Leitura técnica: tendência de {o['tendencia']}.", ev, risks, [r])


def setorial(d: Dossier) -> AgentView:
    ag = A.SETORIAL
    if not d.peers:
        return _insufficient(ag, "Dados de pares não fornecidos.")
    r = d.run("sector_comparison", ag, target=d.ticker, peers=d.peers["dados"], data_source=d.peers["fonte"], higher_is_better=d.peers.get("maior_melhor"))
    if not r.ok:
        return _failed(ag, r)
    s = r.output["pontuacao_composta"]
    ev = [f"Pontuação relativa {s:.0f}/100 entre {r.output['n_empresas']} empresas"] + [
        f"{m}: percentil {x['percentil_ajustado']:.0f}" for m, x in r.output["metricas"].items()
    ]
    return _view(ag, (s - 50) / 50, 0.45, f"Posição relativa no setor: {s:.0f}/100.", ev, [], [r])


def fluxo(d: Dossier) -> AgentView:
    ag = A.FLUXO
    if d.prices is None or d.volume is None:
        return _insufficient(ag, "Volume indisponível.")
    r = d.run("technical_analysis", ag, prices=d.prices, volume=d.volume)
    if not r.ok:
        return _failed(ag, r)
    from ..skills._math import align

    _, (p, v) = align(d.prices, d.volume)
    adv = float((p[-21:] * v[-21:]).mean())
    vr = r.output["volume_relativo_20d"]
    ev = [f"Volume financeiro médio 21d: R$ {adv/1e6:,.1f} mi", f"Volume relativo (último/média 20d): {vr:.2f}x" if vr else "Volume relativo n/d",
          "Fluxo estrangeiro e derivativos: não avaliados (exigem dados licenciados da B3)."]
    risks = [f"Liquidez baixa (R$ {adv/1e6:.1f} mi/dia)"] if adv < 5e6 else []
    score = 0.0 if not risks else -0.3
    return _view(ag, score, 0.3, "Liquidez e volume avaliados.", ev, risks, [r])


def sazonal(d: Dossier) -> AgentView:
    ag = A.SAZONAL
    if d.prices is None:
        return _insufficient(ag, "Cotações indisponíveis.")
    r = d.run("seasonality_detection", ag, prices=d.prices, by="month")
    if not r.ok:
        return _insufficient(ag, f"Sazonalidade não testável: {r.error}")
    month = d.ctx.now.month
    label = f"mês {month:02d}"
    ev = [f"Padrões confirmados fora da amostra: {', '.join(r.output['padroes_confirmados']) or 'nenhum'}"]
    if r.output["padroes_rejeitados_oos"]:
        ev.append(f"Rejeitados fora da amostra: {', '.join(r.output['padroes_rejeitados_oos'])}")
    score = 0.0
    if label in r.output["padroes_confirmados"]:
        g = next(x for x in r.output["grupos"] if x["grupo"] == label)
        score = 0.3 if g["fora_amostra"]["excesso"] > 0 else -0.3
    return _view(ag, score, 0.25, "Sazonalidade testada com correção de Bonferroni e validação fora da amostra.", ev, [], [r])


def noticias(d: Dossier) -> AgentView:
    ag = A.NOTICIAS
    if not d.news:
        return _insufficient(ag, "Nenhuma notícia com fonte coletada para o ativo.")
    mon = d.run("global_news_monitor", ag, items=d.news)
    if not mon.ok:
        return _failed(ag, mon)
    sent = d.run("news_sentiment_analysis", ag, items=d.news)
    results = [mon] + ([sent] if sent.ok else [])
    ev = [f"{mon.output['unicas']} notícias únicas ({mon.output['duplicadas']} duplicadas, {mon.output['rumores']} rumores)"]
    ev += [f"[{g['status']}] {g['titulo']} - {', '.join(g['fontes'])}" for g in mon.output["grupos"][:5]]
    score, conf = 0.0, 0.2
    if sent.ok:
        score, conf = sent.output["escore_agregado"], 0.2 + 0.4 * sent.output["confianca"]
        ev.append(f"Sentimento agregado {score:+.2f} (confiança {sent.output['confianca']:.0%})")
    risks = [f"Rumor em circulação: {g['titulo']}" for g in mon.output["grupos"] if g["status"] == "rumor"]
    return _view(ag, score, conf, "Notícias verificadas e classificadas.", ev, risks, results)


# --- Inteligência preditiva ---------------------------------------------------


def quant(d: Dossier, production_model: dict | None) -> tuple[AgentView, dict]:
    """Etapa 3. Retorna a visão e o pacote quantitativo (que terá hash de integridade)."""
    ag = A.QUANT
    if d.prices is None:
        return _insufficient(ag, "Cotações indisponíveis."), {}
    params = {}
    if production_model:
        params = {"model_params": production_model["params"], "model_version": f"{production_model['model_id']}@{production_model['version']}", "model_status": "producao"}
    fc = d.run("price_direction_forecast", ag, prices=d.prices, horizon=d.horizon, **params)
    dist = d.run("return_distribution_forecast", ag, prices=d.prices, horizon=d.horizon)
    vol = d.run("volatility_forecast", ag, prices=d.prices, horizon=21)
    results = [r for r in (fc, dist, vol) if r.ok]
    package = {
        "previsao": fc.output if fc.ok else None,
        "distribuicao": dist.output if dist.ok else None,
        "volatilidade": vol.output if vol.ok else None,
        "erros": [r.error for r in (fc, dist, vol) if not r.ok],
    }
    if not fc.ok:
        return _failed(ag, fc), package
    o = fc.output
    validated = o["status_modelo"] == "producao"
    edge = o["prob_alta"] - o["taxa_base"]
    ev = [f"P(alta em {d.horizon} pregões) = {o['prob_alta']:.0%} vs. taxa-base {o['taxa_base']:.0%} [modelo {o['modelo']}, {o['status_modelo']}]"]
    if dist.ok:
        q = dist.output["quantis"]
        ev.append(f"Distribuição {d.horizon}d: p05 {q['p05']:+.1%}, mediana {q['p50']:+.1%}, p95 {q['p95']:+.1%}")
    if vol.ok:
        ev.append(f"Vol. prevista (GARCH, 21d): {vol.output['vol_garch_horizonte']:.0%} a.a.")
    score = float(np.clip(edge * 5, -1, 1)) if validated else 0.0
    conf = 0.5 if validated else 0.1
    risks = [] if validated else ["Modelo preditivo sem validação/aprovação: probabilidade apenas informativa, excluída da síntese."]
    return _view(ag, score, conf, "Previsões probabilísticas calculadas.", ev, risks, results), package


# --- Risco e validação --------------------------------------------------------


def risco(d: Dossier, quant_pkg: dict) -> AgentView:
    ag = A.RISCO
    if d.prices is None:
        return _insufficient(ag, "Cotações indisponíveis.")
    r = d.run("portfolio_risk_analysis", ag, key=f"{ag}:risco_ativo", positions={d.ticker: 1.0}, prices={d.ticker: d.prices})
    if not r.ok:
        return _failed(ag, r)
    rk = r.output["risco"]
    ev = [f"Volatilidade histórica {rk['volatilidade_anual']:.0%} a.a.", f"VaR 95% 1 dia {rk['var95_1d_pct']:.1%}; CVaR {rk['cvar95_1d_pct']:.1%}",
          f"Máximo drawdown no período {rk['max_drawdown_historico']:.0%}"]
    risks = []
    if rk["volatilidade_anual"] > 0.45:
        risks.append(f"Volatilidade elevada ({rk['volatilidade_anual']:.0%} a.a.)")
    if rk["max_drawdown_historico"] < -0.4:
        risks.append(f"Histórico de quedas profundas ({rk['max_drawdown_historico']:.0%})")
    v = (quant_pkg or {}).get("volatilidade")
    if v and v["vol_garch_horizonte"] > v["vol_longo_prazo"] * 1.25:
        risks.append("Volatilidade prevista acima da média de longo prazo")
    score = -0.2 * len(risks)
    return _view(ag, score, 0.5, f"{len(risks)} risco(s) relevante(s) identificados.", ev, risks or ["Nenhum risco extremo identificado nas métricas históricas (não elimina riscos de cauda)."], [r])


def auditor(d: Dossier, quant_pkg: dict, quant_digest: str | None, digest_now: str | None) -> AgentView:
    ag = A.AUDITOR
    series = [s for s in (d.prices, d.volume, d.benchmark, *d.macro.values()) if s is not None]
    results, ev, risks = [], [], []
    rejected_series = 0
    if series:
        q = d.run("data_quality_validation", ag, series=series)
        results.append(q)
        if q.ok:
            rejected_series = q.output["reprovadas"]
            ev.append(f"Qualidade de dados: {q.output['aprovadas']} séries aprovadas, {q.output['reprovadas']} reprovadas")
            for s in q.output["series"]:
                risks += [f"{s['serie']}: {c}" for c in s["criticos"]]
                risks += [f"{s['serie']}: {w}" for w in s["avisos"] if "atrasado" in w or "sintético" in w]
    fc = (quant_pkg or {}).get("previsao")
    if fc:
        if fc["status_modelo"] != "producao":
            risks.append(f"Previsão gerada por modelo '{fc['status_modelo']}' ({fc['modelo']}): não pode sustentar conclusões.")
        else:
            ev.append(f"Modelo em produção: {fc['modelo']}")
    if quant_digest and digest_now and quant_digest != digest_now:
        risks.append("INTEGRIDADE: resultado quantitativo alterado após a modelagem.")
    elif quant_digest:
        ev.append(f"Integridade do resultado quantitativo verificada (hash {quant_digest}).")
    blocking = rejected_series > 0 or any("INTEGRIDADE" in x for x in risks)
    v = _view(ag, 0.0, 0.6, "Auditoria de dados, modelo e integridade concluída.", ev, risks, results)
    if blocking:
        v.stance = Stance.INSUFFICIENT
        v.summary = "Auditoria reprovou dados ou integridade: conclusões bloqueadas."
    return v


def event_types() -> list[str]:
    return sorted(EVENT_SECTOR_MAP)
