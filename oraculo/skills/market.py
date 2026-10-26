"""Skills de mercado (7-12)."""
from __future__ import annotations

import numpy as np

from ..core import agent_ids as A
from ..core.types import Series
from ._math import align, ema, max_drawdown, ols, pct_returns, rsi, sma
from .base import SkillError, SkillRun, finite, skill


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b in (None, 0):
        return None
    return a / b


@skill(
    id="fundamental_analysis",
    name="Análise fundamentalista",
    version="1.1.0",
    category="mercado",
    objective="Calcular rentabilidade, margens, endividamento, crescimento e qualidade do lucro a partir das demonstrações.",
    authorized_agents=[A.FUNDAMENTALISTA, A.SETORIAL, A.CARTEIRA, A.CIIO],
    inputs={
        "financials": "dict com receita, ebitda, ebit, lucro_liquido, patrimonio_liquido, ativo_total, divida_bruta, caixa, "
        "fco (fluxo de caixa operacional), capex, despesa_financeira, aliquota_ir, receita_anterior, lucro_anterior",
        "financials_source": "fonte (ex.: CVM/ITR, release de RI) e data de referência",
        "market_cap": "valor de mercado em R$ (preço x ações em circulação), opcional, para múltiplos",
        "financial_institution": "True para bancos/seguradoras: critérios de alavancagem e caixa não se aplicam",
    },
    required_inputs=["financials", "financials_source"],
    tools=["numpy"],
    procedure=[
        "Calcular margens (bruta/EBITDA/líquida), ROE, ROIC, dívida líquida/EBITDA e cobertura de juros.",
        "Crescimento de receita e lucro vs. período anterior.",
        "Conversão de caixa (FCO/lucro) e accruals ((lucro-FCO)/ativo).",
        "Pontuar qualidade 0-6 com regras explícitas (proporcional aos critérios aplicáveis ao tipo de empresa).",
        "Com valor de mercado: P/L, P/VP, EV/EBITDA e rendimento do fluxo de caixa livre.",
    ],
    validation=["Receita positiva.", "Indicadores finitos."],
    output_schema={"indicadores": "dict", "pontuacao_qualidade": "int (0-6)", "alertas": "list"},
    failure_handling="Indicadores sem dados ficam None; não há preenchimento por estimativa.",
)
def fundamental_analysis(
    run: SkillRun, financials: dict, financials_source: str, market_cap: float | None = None, financial_institution: bool = False
) -> dict:
    from ..core.types import SourceRef, utcnow

    run.cite(SourceRef(financials_source, utcnow()))
    f = financials
    if not f.get("receita") or f["receita"] <= 0:
        raise SkillError("Receita ausente ou não positiva.")
    net_debt = (f.get("divida_bruta") or 0) - (f.get("caixa") or 0)
    tax = f.get("aliquota_ir", 0.34)
    invested = (f.get("patrimonio_liquido") or 0) + net_debt
    ind = {
        "margem_ebitda": _div(f.get("ebitda"), f["receita"]),
        "margem_ebit": _div(f.get("ebit"), f["receita"]),
        "margem_liquida": _div(f.get("lucro_liquido"), f["receita"]),
        "roe": _div(f.get("lucro_liquido"), f.get("patrimonio_liquido")),
        "roic": _div(f["ebit"] * (1 - tax), invested) if f.get("ebit") is not None and invested > 0 else None,
        "divida_liquida": net_debt,
        "divida_liquida_ebitda": _div(net_debt, f.get("ebitda")),
        "cobertura_juros": _div(f.get("ebit"), f.get("despesa_financeira")),
        "crescimento_receita": _div(f["receita"] - f["receita_anterior"], f["receita_anterior"]) if f.get("receita_anterior") else None,
        "crescimento_lucro": (
            _div(f["lucro_liquido"] - f["lucro_anterior"], abs(f["lucro_anterior"]))
            if f.get("lucro_anterior") and f.get("lucro_liquido") is not None
            else None
        ),
        "fluxo_caixa_livre": (f["fco"] - f.get("capex", 0)) if f.get("fco") is not None else None,
        "conversao_caixa": _div(f.get("fco"), f.get("lucro_liquido")),
        "accruals": (
            _div(f["lucro_liquido"] - f["fco"], f.get("ativo_total"))
            if f.get("fco") is not None and f.get("lucro_liquido") is not None
            else None
        ),
    }
    checks = {
        "lucro_positivo": (f.get("lucro_liquido") or 0) > 0,
        "fco_positivo": (f.get("fco") or 0) > 0,
        "fco_maior_que_lucro": (ind["conversao_caixa"] or 0) > 1,
        "roe_acima_12pct": (ind["roe"] or 0) > 0.12,
        "alavancagem_controlada": ind["divida_liquida_ebitda"] is not None and ind["divida_liquida_ebitda"] < 2.5,
        "receita_crescendo": (ind["crescimento_receita"] or 0) > 0,
    }
    if financial_institution:
        # balanço de instituição financeira: dívida, EBITDA e fluxo de caixa operacional não são comparáveis
        for k in ("fco_positivo", "fco_maior_que_lucro", "alavancagem_controlada"):
            checks.pop(k)
        for k in ("margem_ebitda", "margem_ebit", "roic", "divida_liquida", "divida_liquida_ebitda", "cobertura_juros",
                  "fluxo_caixa_livre", "conversao_caixa", "accruals"):
            ind[k] = None
    multiples = {}
    if market_cap:
        ev = market_cap + (ind["divida_liquida"] or 0)
        multiples = {
            "valor_mercado": market_cap,
            "p_l": _div(market_cap, f.get("lucro_liquido")) if (f.get("lucro_liquido") or 0) > 0 else None,
            "p_vp": _div(market_cap, f.get("patrimonio_liquido")),
            "ev_ebitda": _div(ev, f.get("ebitda")) if not financial_institution and (f.get("ebitda") or 0) > 0 else None,
            "rendimento_fcl": _div(ind["fluxo_caixa_livre"], market_cap),
        }
    alerts = []
    if ind["divida_liquida_ebitda"] is not None and ind["divida_liquida_ebitda"] > 3.5:
        alerts.append(f"Alavancagem elevada: dívida líquida/EBITDA = {ind['divida_liquida_ebitda']:.2f}x")
    if ind["cobertura_juros"] is not None and ind["cobertura_juros"] < 1.5:
        alerts.append(f"Cobertura de juros baixa: {ind['cobertura_juros']:.2f}x")
    if ind["accruals"] is not None and ind["accruals"] > 0.10:
        alerts.append("Accruals elevados: lucro pouco suportado por caixa.")
    missing = [k for k, v in ind.items() if v is None]
    if missing:
        run.warn(f"Indicadores sem dados ou não aplicáveis: {', '.join(missing)}", degrade=False)
    score = round(6 * sum(checks.values()) / len(checks))
    return {"indicadores": ind, "criterios": checks, "pontuacao_qualidade": int(score), "alertas": alerts, "multiplos": multiples,
            "instituicao_financeira": financial_institution}


@skill(
    id="valuation_engine",
    name="Avaliação de empresas",
    version="1.0.0",
    category="mercado",
    objective="Estimar valor por ação via DCF (FCFF) e múltiplos comparáveis, com tabela de sensibilidade.",
    authorized_agents=[A.FUNDAMENTALISTA, A.SETORIAL, A.CARTEIRA],
    inputs={
        "fcff_projection": "lista de FCFF projetados (R$)",
        "wacc": "custo médio ponderado de capital (decimal)",
        "terminal_growth": "crescimento na perpetuidade (decimal)",
        "net_debt": "dívida líquida (R$)",
        "shares": "número de ações",
        "price": "preço atual (opcional)",
        "peer_multiples": "dict com listas de múltiplos dos pares, ex.: {'ev_ebitda': [...], 'pl': [...]}",
        "company_metrics": "dict com ebitda e lucro_liquido para aplicar múltiplos",
        "assumptions_source": "origem das premissas",
    },
    required_inputs=["fcff_projection", "wacc", "terminal_growth", "net_debt", "shares", "assumptions_source"],
    tools=["numpy"],
    procedure=[
        "Descontar FCFF ao WACC e somar valor terminal (Gordon).",
        "Equity = EV - dívida líquida; valor por ação = equity/ações.",
        "Sensibilidade WACC ±1pp x g ±0,5pp.",
        "Múltiplos: mediana dos pares aplicada às métricas da empresa.",
    ],
    validation=["WACC > g.", "Ações > 0.", "Valor terminal não pode passar de 85% do EV sem aviso."],
    output_schema={"valor_por_acao_dcf": "float", "upside_dcf": "float | None", "sensibilidade": "matriz", "multiplos": "dict"},
    failure_handling="Falha se WACC <= g. Múltiplos omitidos sem pares.",
)
def valuation_engine(
    run: SkillRun,
    fcff_projection: list[float],
    wacc: float,
    terminal_growth: float,
    net_debt: float,
    shares: float,
    assumptions_source: str,
    price: float | None = None,
    peer_multiples: dict | None = None,
    company_metrics: dict | None = None,
) -> dict:
    from ..core.types import SourceRef, utcnow

    run.cite(SourceRef(assumptions_source, utcnow()))
    if wacc <= terminal_growth:
        raise SkillError("WACC deve ser maior que o crescimento na perpetuidade.")
    if shares <= 0:
        raise SkillError("Número de ações deve ser positivo.")
    if not fcff_projection:
        raise SkillError("Projeção de FCFF vazia.")

    def dcf(w: float, g: float) -> tuple[float, float]:
        cf = np.asarray(fcff_projection, float)
        t = np.arange(1, len(cf) + 1)
        pv = float((cf / (1 + w) ** t).sum())
        tv = cf[-1] * (1 + g) / (w - g) / (1 + w) ** len(cf)
        ev = pv + tv
        return (ev - net_debt) / shares, tv / ev if ev else np.nan

    per_share, tv_share = dcf(wacc, terminal_growth)
    if tv_share > 0.85:
        run.warn(f"Valor terminal representa {tv_share:.0%} do EV: alta sensibilidade às premissas.")
    grid_w = [wacc - 0.01, wacc, wacc + 0.01]
    grid_g = [terminal_growth - 0.005, terminal_growth, terminal_growth + 0.005]
    sens = [[(dcf(w, g)[0] if w > g else None) for g in grid_g] for w in grid_w]
    out = {
        "valor_por_acao_dcf": per_share,
        "participacao_valor_terminal": tv_share,
        "upside_dcf": (per_share / price - 1) if price else None,
        "sensibilidade": {"wacc": grid_w, "g": grid_g, "valores": sens},
        "multiplos": {},
    }
    if peer_multiples and company_metrics:
        mult = {}
        if peer_multiples.get("ev_ebitda") and company_metrics.get("ebitda"):
            m = float(np.median(peer_multiples["ev_ebitda"]))
            mult["ev_ebitda_mediana_pares"] = m
            mult["valor_por_acao_ev_ebitda"] = (m * company_metrics["ebitda"] - net_debt) / shares
        if peer_multiples.get("pl") and company_metrics.get("lucro_liquido"):
            m = float(np.median(peer_multiples["pl"]))
            mult["pl_mediana_pares"] = m
            mult["valor_por_acao_pl"] = m * company_metrics["lucro_liquido"] / shares
        out["multiplos"] = mult
    return out


valuation_engine.validator(finite("valor_por_acao_dcf"))


@skill(
    id="technical_analysis",
    name="Indicadores técnicos",
    version="1.0.0",
    category="mercado",
    objective="Calcular tendência, médias, RSI, MACD, Bandas de Bollinger, volatilidade, volume e suportes/resistências.",
    authorized_agents=[A.TECNICO, A.FLUXO, A.QUANT, A.RISCO],
    inputs={"prices": "Series de fechamento ajustado", "volume": "Series de volume (opcional)"},
    required_inputs=["prices"],
    tools=["numpy"],
    procedure=[
        "MMS 20/50/200, MME 12/26, MACD e sinal (9).",
        "RSI 14, Bandas de Bollinger (20, 2σ), volatilidade realizada 21d anualizada.",
        "Tendência: preço vs. MMS50 e MMS200 e inclinação da MMS50.",
        "Suportes/resistências: mínimos/máximos locais (janela 10) nos últimos 120 pregões, agrupados a 1,5%.",
        "Volume relativo: último / média 20.",
    ],
    validation=["Ao menos 60 preços.", "RSI entre 0 e 100."],
    output_schema={"tendencia": "str", "rsi14": "float", "macd": "dict", "suportes": "list", "resistencias": "list"},
    failure_handling="MMS200 omitida com menos de 200 preços (aviso).",
)
def technical_analysis(run: SkillRun, prices: Series, volume: Series | None = None) -> dict:
    p = prices.values
    if len(p) < 60:
        raise SkillError("São necessários ao menos 60 preços.")
    s20, s50, s200 = sma(p, 20), sma(p, 50), sma(p, 200)
    macd_line = ema(p, 12) - ema(p, 26)
    signal = ema(macd_line, 9)
    r = rsi(p)
    rets = pct_returns(p)
    bb_mid, bb_sd = s20[-1], p[-20:].std(ddof=1)
    last = float(p[-1])
    slope50 = float((s50[-1] / s50[-11] - 1)) if not np.isnan(s50[-11]) else 0.0
    have200 = not np.isnan(s200[-1])
    if not have200:
        run.warn("Menos de 200 preços: MMS200 indisponível.", degrade=False)
    if last > s50[-1] and (not have200 or s50[-1] > s200[-1]) and slope50 > 0:
        trend = "alta"
    elif last < s50[-1] and (not have200 or s50[-1] < s200[-1]) and slope50 < 0:
        trend = "baixa"
    else:
        trend = "lateral/indefinida"

    window = p[-120:]
    lows, highs = [], []
    for i in range(10, len(window) - 10):
        seg = window[i - 10 : i + 11]
        if window[i] == seg.min():
            lows.append(float(window[i]))
        if window[i] == seg.max():
            highs.append(float(window[i]))

    def cluster(levels: list[float]) -> list[float]:
        out: list[list[float]] = []
        for x in sorted(levels):
            if out and abs(x / np.mean(out[-1]) - 1) < 0.015:
                out[-1].append(x)
            else:
                out.append([x])
        return [float(np.mean(c)) for c in out]

    supports = [x for x in cluster(lows) if x < last][-3:]
    resistances = [x for x in cluster(highs) if x > last][:3]
    out = {
        "preco": last,
        "data": prices.last_date.isoformat(),
        "mms20": float(s20[-1]),
        "mms50": float(s50[-1]),
        "mms200": float(s200[-1]) if have200 else None,
        "tendencia": trend,
        "inclinacao_mms50_10d": slope50,
        "rsi14": float(r[-1]),
        "macd": {"linha": float(macd_line[-1]), "sinal": float(signal[-1]), "histograma": float(macd_line[-1] - signal[-1])},
        "bollinger": {"superior": float(bb_mid + 2 * bb_sd), "media": float(bb_mid), "inferior": float(bb_mid - 2 * bb_sd)},
        "volatilidade_21d_anual": float(rets[-21:].std(ddof=1) * np.sqrt(252)),
        "retorno_21d": float(p[-1] / p[-22] - 1),
        "retorno_63d": float(p[-1] / p[-64] - 1) if len(p) > 64 else None,
        "drawdown_atual": float(last / p.max() - 1),
        "max_drawdown": max_drawdown(p),
        "suportes": supports,
        "resistencias": resistances,
        "volume_relativo_20d": None,
    }
    if volume is not None and len(volume) >= 21:
        v = volume.values
        out["volume_relativo_20d"] = float(v[-1] / v[-21:-1].mean())
    return out


@technical_analysis.validator
def _rsi_range(out: dict) -> list[str]:
    return [] if 0 <= out["rsi14"] <= 100 else ["RSI fora de [0,100]"]


@skill(
    id="sector_comparison",
    name="Comparação setorial",
    version="1.0.0",
    category="mercado",
    objective="Posicionar uma empresa frente aos pares por percentil e z-score em cada métrica.",
    authorized_agents=[A.SETORIAL, A.FUNDAMENTALISTA, A.CARTEIRA],
    inputs={
        "target": "ticker da empresa avaliada",
        "peers": "dict[ticker, dict[métrica, valor]] incluindo o alvo",
        "higher_is_better": "dict[métrica, bool]",
        "data_source": "fonte dos dados",
    },
    required_inputs=["target", "peers", "data_source"],
    tools=["numpy"],
    procedure=[
        "Para cada métrica: mediana, percentil do alvo e z-score.",
        "Inverter o sinal das métricas em que menor é melhor.",
        "Pontuação composta = média dos percentis ajustados.",
    ],
    validation=["Ao menos 3 empresas no grupo.", "Alvo presente."],
    output_schema={"metricas": "dict", "pontuacao_composta": "float (0-100)"},
    failure_handling="Métricas com menos de 3 valores são ignoradas com aviso.",
)
def sector_comparison(run: SkillRun, target: str, peers: dict, data_source: str, higher_is_better: dict | None = None) -> dict:
    from ..core.types import SourceRef, utcnow

    run.cite(SourceRef(data_source, utcnow()))
    if target not in peers:
        raise SkillError("Alvo ausente na lista de pares.")
    if len(peers) < 3:
        raise SkillError("São necessárias ao menos 3 empresas.")
    hib = higher_is_better or {}
    metrics = sorted({m for d in peers.values() for m in d})
    res, scores = {}, []
    for m in metrics:
        vals = {t: d[m] for t, d in peers.items() if d.get(m) is not None}
        if len(vals) < 3 or target not in vals:
            run.warn(f"Métrica '{m}' ignorada (dados insuficientes).", degrade=False)
            continue
        arr = np.array(list(vals.values()), float)
        x = vals[target]
        pct = float((arr < x).mean() * 100 + (arr == x).mean() * 50)
        good = hib.get(m, True)
        adj = pct if good else 100 - pct
        sd = arr.std(ddof=1)
        res[m] = {
            "valor": x,
            "mediana": float(np.median(arr)),
            "percentil": pct,
            "percentil_ajustado": adj,
            "z": float((x - arr.mean()) / sd) if sd > 0 else 0.0,
            "maior_e_melhor": good,
        }
        scores.append(adj)
    if not scores:
        raise SkillError("Nenhuma métrica comparável.")
    return {"alvo": target, "n_empresas": len(peers), "metricas": res, "pontuacao_composta": float(np.mean(scores))}


@skill(
    id="earnings_analysis",
    name="Interpretação de resultados trimestrais",
    version="1.0.0",
    category="mercado",
    objective="Medir surpresa vs. consenso, crescimento anual, variação de margens e qualidade do resultado.",
    authorized_agents=[A.FUNDAMENTALISTA, A.SETORIAL, A.NOTICIAS],
    inputs={
        "actual": "dict trimestre atual (receita, ebitda, lucro_liquido, fco, ativo_total)",
        "year_ago": "dict mesmo trimestre do ano anterior",
        "consensus": "dict de consenso (opcional; fonte licenciada)",
        "source": "fonte do resultado (ex.: release de RI / CVM ITR) e data",
    },
    required_inputs=["actual", "year_ago", "source"],
    tools=["numpy"],
    procedure=[
        "Crescimento a/a de receita, EBITDA e lucro.",
        "Variação de margem EBITDA e líquida em p.p.",
        "Surpresa = (realizado/consenso - 1) quando houver consenso.",
        "Qualidade: accruals e conversão de caixa.",
    ],
    validation=["Receita positiva nos dois períodos."],
    output_schema={"crescimento": "dict", "margens": "dict", "surpresa": "dict | None", "qualidade": "dict"},
    failure_handling="Sem consenso, a surpresa fica None (não é estimada).",
)
def earnings_analysis(run: SkillRun, actual: dict, year_ago: dict, source: str, consensus: dict | None = None) -> dict:
    from ..core.types import SourceRef, utcnow

    run.cite(SourceRef(source, utcnow()))
    if not actual.get("receita") or not year_ago.get("receita"):
        raise SkillError("Receita ausente.")
    growth = {
        k: _div(actual[k] - year_ago[k], abs(year_ago[k]))
        for k in ("receita", "ebitda", "lucro_liquido")
        if actual.get(k) is not None and year_ago.get(k)
    }
    margins = {}
    for k in ("ebitda", "lucro_liquido"):
        if actual.get(k) is not None and year_ago.get(k) is not None:
            margins[f"margem_{k}_pp"] = (actual[k] / actual["receita"] - year_ago[k] / year_ago["receita"]) * 100
    surprise = None
    if consensus:
        surprise = {k: _div(actual[k], consensus[k]) - 1 for k in consensus if actual.get(k) is not None and consensus.get(k)}
    else:
        run.warn("Sem consenso de mercado: surpresa não calculada.", degrade=False)
    quality = {
        "conversao_caixa": _div(actual.get("fco"), actual.get("lucro_liquido")),
        "accruals": (
            _div(actual["lucro_liquido"] - actual["fco"], actual.get("ativo_total"))
            if actual.get("fco") is not None and actual.get("lucro_liquido") is not None
            else None
        ),
    }
    return {"crescimento": growth, "margens": margins, "surpresa": surprise, "qualidade": quality}


@skill(
    id="commodity_exposure",
    name="Sensibilidade a commodities",
    version="1.0.0",
    category="mercado",
    objective="Estimar beta, correlação e estabilidade da relação entre a ação e uma commodity.",
    authorized_agents=[A.COMMODITIES, A.SETORIAL, A.RISCO, A.CARTEIRA, A.QUANT],
    inputs={"stock_prices": "Series da ação", "commodity_prices": "Series da commodity (em R$ ou US$)", "window": "janela da estabilidade (padrão 126)"},
    required_inputs=["stock_prices", "commodity_prices"],
    tools=["numpy (OLS)"],
    procedure=[
        "Alinhar datas e calcular retornos diários.",
        "OLS: retorno_ação = a + b * retorno_commodity.",
        "Beta em janelas móveis para medir estabilidade (desvio dos betas).",
    ],
    validation=["Ao menos 120 datas em comum."],
    output_schema={"beta": "float", "erro_padrao": "float", "correlacao": "float", "r2": "float", "beta_movel": "dict"},
    failure_handling="Falha se houver menos de 120 datas em comum.",
)
def commodity_exposure(run: SkillRun, stock_prices: Series, commodity_prices: Series, window: int = 126) -> dict:
    _, (p, c) = align(stock_prices, commodity_prices)
    if len(p) < 120:
        raise SkillError(f"Apenas {len(p)} datas em comum; mínimo 120.")
    rs, rc = pct_returns(p), pct_returns(c)
    res = ols(rs, rc)
    betas = [ols(rs[i - window : i], rc[i - window : i]).coef[1] for i in range(window, len(rs) + 1, 21)]
    stab = float(np.std(betas)) if len(betas) > 1 else None
    if stab is not None and abs(res.coef[1]) > 0 and stab > abs(res.coef[1]):
        run.warn("Beta instável ao longo do tempo: relação pouco confiável.", degrade=False)
    return {
        "beta": float(res.coef[1]),
        "erro_padrao": float(res.stderr[1]),
        "t_stat": float(res.coef[1] / res.stderr[1]) if res.stderr[1] > 0 else None,
        "correlacao": float(np.corrcoef(rs, rc)[0, 1]),
        "r2": res.r2,
        "n_obs": res.n,
        "beta_movel": {"ultimo": float(betas[-1]) if betas else None, "desvio": stab, "n_janelas": len(betas)},
    }
