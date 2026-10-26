"""Skills macroeconômicas (1-6)."""
from __future__ import annotations

import numpy as np

from ..core import agent_ids as A
from ..core.types import Series
from ._math import align, ols, pct_returns
from .base import SkillError, SkillRun, finite, skill


def _indicator_snapshot(run: SkillRun, s: Series, z_threshold: float) -> dict:
    v = s.values
    if len(v) < 3:
        raise SkillError(f"Série '{s.name}' com menos de 3 observações.")
    last = float(v[-1])
    lookback = v[-min(len(v), 252):]
    diffs = np.diff(lookback)
    last_change = float(v[-1] - v[-2])
    sd = float(diffs.std(ddof=1)) if len(diffs) > 2 else 0.0
    z = last_change / sd if sd > 0 else 0.0
    level_z = float((last - lookback.mean()) / lookback.std(ddof=1)) if lookback.std(ddof=1) > 0 else 0.0
    return {
        "serie": s.name,
        "ultimo": last,
        "data": s.last_date.isoformat(),
        "unidade": s.unit,
        "variacao_ultima": last_change,
        "z_variacao": z,
        "z_nivel_1a": level_z,
        "mudanca_material": abs(z) >= z_threshold,
        "atrasado": s.is_stale(run.ctx.now),
        "fonte": s.source,
    }


def _monitor(run: SkillRun, indicators: dict[str, Series], z_threshold: float) -> dict:
    snaps, alerts, failures = [], [], []
    for key, s in indicators.items():
        try:
            snap = _indicator_snapshot(run, s, z_threshold)
        except SkillError as exc:
            failures.append(f"{key}: {exc}")
            continue
        snaps.append(snap)
        if snap["mudanca_material"]:
            alerts.append(f"{key}: variação de {snap['variacao_ultima']:+.4g} (z={snap['z_variacao']:+.2f})")
    if not snaps:
        raise SkillError("Nenhum indicador válido para monitorar.")
    for f in failures:
        run.warn(f)
    return {"indicadores": snaps, "alertas": alerts, "n_materiais": len(alerts)}


@skill(
    id="global_macro_monitor",
    name="Monitoramento econômico global",
    version="1.0.0",
    category="macro",
    objective="Acompanhar indicadores globais (juros, inflação, atividade, câmbio, liquidez) e sinalizar mudanças materiais.",
    authorized_agents=[A.ECON_GLOBAL, A.CIIO, A.ORQUESTRADOR, A.ECON_BRASIL],
    inputs={"indicators": "dict[str, Series] de indicadores globais", "z_threshold": "limiar de materialidade (padrão 2.0)"},
    required_inputs=["indicators"],
    tools=["DataHub", "numpy"],
    procedure=[
        "Validar procedência e atraso de cada série.",
        "Calcular nível, última variação e z-score da variação vs. 1 ano.",
        "Marcar mudança material quando |z| >= limiar.",
    ],
    validation=["Ao menos um indicador válido.", "Valores finitos."],
    output_schema={"indicadores": "lista de snapshots", "alertas": "lista de textos", "n_materiais": "int"},
    failure_handling="Indicadores inválidos são reportados como aviso; falha somente se nenhum for válido.",
)
def global_macro_monitor(run: SkillRun, indicators: dict[str, Series], z_threshold: float = 2.0) -> dict:
    return _monitor(run, indicators, z_threshold)


@skill(
    id="brazil_macro_monitor",
    name="Indicadores brasileiros",
    version="1.0.0",
    category="macro",
    objective="Monitorar Selic, IPCA, câmbio e atividade no Brasil, incluindo juro real ex-post.",
    authorized_agents=[A.ECON_BRASIL, A.CIIO, A.ORQUESTRADOR, A.ECON_GLOBAL],
    inputs={
        "indicators": "dict[str, Series] (ex.: selic_meta, ipca_12m, usdbrl_ptax)",
        "z_threshold": "limiar de materialidade",
    },
    required_inputs=["indicators"],
    tools=["BCB SGS", "numpy"],
    procedure=[
        "Executar o monitor padrão de indicadores.",
        "Se houver selic_meta e ipca_12m, calcular juro real ex-post = (1+selic)/(1+ipca)-1.",
    ],
    validation=["Ao menos um indicador válido."],
    output_schema={"indicadores": "lista", "alertas": "lista", "juro_real_ex_post": "float | None"},
    failure_handling="Juro real omitido se faltar Selic ou IPCA 12m.",
)
def brazil_macro_monitor(run: SkillRun, indicators: dict[str, Series], z_threshold: float = 2.0) -> dict:
    out = _monitor(run, indicators, z_threshold)
    selic, ipca = indicators.get("selic_meta"), indicators.get("ipca_12m")
    out["juro_real_ex_post"] = (
        ((1 + selic.last_value / 100) / (1 + ipca.last_value / 100) - 1) * 100 if selic and ipca else None
    )
    return out


@skill(
    id="central_bank_analysis",
    name="Decisões de bancos centrais",
    version="1.0.0",
    category="macro",
    objective="Identificar a última decisão de política monetária, a fase do ciclo e o aperto/afrouxamento acumulado.",
    authorized_agents=[A.ECON_BRASIL, A.ECON_GLOBAL, A.CIIO],
    inputs={
        "policy_rate": "Series da taxa básica (ex.: selic_meta)",
        "inflation_12m": "Series de inflação 12m (opcional)",
        "inflation_target": "meta de inflação em % (opcional)",
    },
    required_inputs=["policy_rate"],
    tools=["numpy"],
    procedure=[
        "Detectar mudanças discretas na taxa (decisões).",
        "Classificar a fase: aperto, afrouxamento ou manutenção.",
        "Somar a variação acumulada desde a inversão de direção.",
        "Calcular juro real e distância da inflação para a meta, se disponíveis.",
    ],
    validation=["Taxa finita.", "Pelo menos uma observação."],
    output_schema={"taxa_atual": "float", "ultima_decisao_bps": "float", "fase": "str", "ciclo_acumulado_bps": "float"},
    failure_handling="Sem decisões no histórico => fase 'manutenção' com aviso.",
)
def central_bank_analysis(
    run: SkillRun, policy_rate: Series, inflation_12m: Series | None = None, inflation_target: float | None = None
) -> dict:
    v, d = policy_rate.values, policy_rate.dates
    if len(v) < 2:
        raise SkillError("Histórico insuficiente da taxa básica.")
    changes = [(d[i], float(v[i] - v[i - 1])) for i in range(1, len(v)) if abs(v[i] - v[i - 1]) > 1e-9]
    out: dict = {"taxa_atual": float(v[-1]), "data": d[-1].isoformat(), "n_decisoes_alteracao": len(changes)}
    if not changes:
        run.warn("Nenhuma alteração da taxa no período analisado.", degrade=False)
        out.update({"ultima_decisao_bps": 0.0, "fase": "manutenção", "ciclo_acumulado_bps": 0.0})
    else:
        last_date, last_chg = changes[-1]
        direction = np.sign(last_chg)
        cum = 0.0
        for _, c in reversed(changes):
            if np.sign(c) != direction:
                break
            cum += c
        days_since = (d[-1] - last_date).days
        fase = ("aperto" if direction > 0 else "afrouxamento") if days_since < 120 else "manutenção"
        out.update(
            {
                "ultima_decisao_data": last_date.isoformat(),
                "ultima_decisao_bps": round(last_chg * 100, 1),
                "dias_desde_ultima_alteracao": days_since,
                "fase": fase,
                "ciclo_acumulado_bps": round(cum * 100, 1),
            }
        )
    if inflation_12m is not None and len(inflation_12m):
        infl = inflation_12m.last_value
        out["inflacao_12m"] = infl
        out["juro_real_ex_post"] = ((1 + v[-1] / 100) / (1 + infl / 100) - 1) * 100
        if inflation_target is not None:
            out["desvio_da_meta_pp"] = infl - inflation_target
    return out


central_bank_analysis.validator(finite("taxa_atual", "ultima_decisao_bps"))


@skill(
    id="yield_curve_analysis",
    name="Análise de curvas de juros",
    version="1.0.0",
    category="macro",
    objective="Medir inclinação, curvatura, inversões e decompor a variação da curva (nível, inclinação, curvatura).",
    authorized_agents=[A.ECON_BRASIL, A.ECON_GLOBAL, A.QUANT, A.CIIO],
    inputs={
        "curve": "dict[prazo_em_anos, taxa_%] atual (ex.: curva DI de fonte licenciada)",
        "previous_curve": "dict no mesmo formato (opcional)",
        "curve_source": "fonte e horário da curva (obrigatório)",
    },
    required_inputs=["curve", "curve_source"],
    tools=["numpy"],
    procedure=[
        "Ordenar vértices e interpolar linearmente 0.25a, 2a, 5a e 10a quando cobertos.",
        "Inclinação = 10a-2a (ou maior-menor prazo). Curvatura = 2*5a - 2a - 10a.",
        "Detectar inversões entre vértices consecutivos.",
        "Se houver curva anterior: deslocamento paralelo médio, variação de inclinação e curvatura.",
    ],
    validation=["Pelo menos 3 vértices.", "Prazos positivos."],
    output_schema={"inclinacao_pp": "float", "curvatura_pp": "float", "invertida": "bool", "variacao": "dict | None"},
    failure_handling="Vértices fora da cobertura não são extrapolados; métricas ficam None.",
)
def yield_curve_analysis(run: SkillRun, curve: dict, curve_source: str, previous_curve: dict | None = None) -> dict:
    from ..core.types import SourceRef, utcnow

    run.cite(SourceRef(source=curve_source, retrieved_at=utcnow()))

    def prep(c: dict) -> tuple[np.ndarray, np.ndarray]:
        items = sorted((float(k), float(v)) for k, v in c.items())
        if len(items) < 3:
            raise SkillError("A curva precisa de ao menos 3 vértices.")
        t = np.array([i[0] for i in items])
        if (t <= 0).any():
            raise SkillError("Prazos devem ser positivos.")
        return t, np.array([i[1] for i in items])

    def metrics(t: np.ndarray, r: np.ndarray) -> dict:
        def at(x: float) -> float | None:
            return float(np.interp(x, t, r)) if t[0] <= x <= t[-1] else None

        r2, r5, r10 = at(2), at(5), at(10)
        slope = (r10 - r2) if (r10 is not None and r2 is not None) else float(r[-1] - r[0])
        curv = (2 * r5 - r2 - r10) if None not in (r2, r5, r10) else None
        return {"r2a": r2, "r5a": r5, "r10a": r10, "inclinacao_pp": slope, "curvatura_pp": curv}

    t, r = prep(curve)
    m = metrics(t, r)
    inversions = [f"{t[i]:g}a > {t[i+1]:g}a" for i in range(len(t) - 1) if r[i] > r[i + 1] + 1e-9]
    out = {**m, "invertida": m["inclinacao_pp"] < 0, "inversoes_locais": inversions, "vertices": len(t), "variacao": None}
    if previous_curve:
        tp, rp = prep(previous_curve)
        grid = t[(t >= tp[0]) & (t <= tp[-1])]
        delta = np.interp(grid, t, r) - np.interp(grid, tp, rp)
        mp = metrics(tp, rp)
        out["variacao"] = {
            "deslocamento_paralelo_pp": float(delta.mean()),
            "variacao_inclinacao_pp": m["inclinacao_pp"] - mp["inclinacao_pp"],
            "variacao_curvatura_pp": (
                m["curvatura_pp"] - mp["curvatura_pp"] if m["curvatura_pp"] is not None and mp["curvatura_pp"] is not None else None
            ),
        }
    return out


@skill(
    id="fx_impact_analysis",
    name="Impacto cambial nas empresas",
    version="1.0.0",
    category="macro",
    objective="Estimar o efeito de um choque cambial sobre resultado operacional e a sensibilidade histórica da ação ao dólar.",
    authorized_agents=[A.ECON_BRASIL, A.FUNDAMENTALISTA, A.SETORIAL, A.CARTEIRA, A.RISCO],
    inputs={
        "fx_shock_pct": "variação do USD/BRL em % (ex.: +10)",
        "exposure": "dict: receita_usd_pct, custo_usd_pct, receita, custos, divida_usd, hedge_pct",
        "stock_prices": "Series da ação (opcional, para beta empírico)",
        "fx_series": "Series USD/BRL (opcional)",
    },
    required_inputs=["fx_shock_pct", "exposure"],
    tools=["numpy (OLS)"],
    procedure=[
        "Impacto operacional = receita*receita_usd% * choque - custos*custo_usd% * choque.",
        "Impacto financeiro = -divida_usd*(1-hedge%)*choque.",
        "Se houver séries: beta da ação ao câmbio por OLS em retornos diários alinhados.",
    ],
    validation=["Percentuais entre 0 e 100."],
    output_schema={"impacto_operacional": "float", "impacto_financeiro": "float", "beta_cambio": "float | None"},
    failure_handling="Beta empírico omitido se as séries tiverem menos de 60 datas em comum.",
)
def fx_impact_analysis(
    run: SkillRun,
    fx_shock_pct: float,
    exposure: dict,
    stock_prices: Series | None = None,
    fx_series: Series | None = None,
) -> dict:
    shock = fx_shock_pct / 100
    for k in ("receita_usd_pct", "custo_usd_pct", "hedge_pct"):
        if not 0 <= exposure.get(k, 0) <= 100:
            raise SkillError(f"{k} deve estar entre 0 e 100.")
    rev, cost = exposure.get("receita", 0.0), exposure.get("custos", 0.0)
    op = rev * exposure.get("receita_usd_pct", 0) / 100 * shock - cost * exposure.get("custo_usd_pct", 0) / 100 * shock
    fin = -exposure.get("divida_usd", 0.0) * (1 - exposure.get("hedge_pct", 0) / 100) * shock
    out = {"choque_pct": fx_shock_pct, "impacto_operacional": op, "impacto_financeiro": fin, "impacto_total": op + fin}
    out["beta_cambio"] = None
    if stock_prices is not None and fx_series is not None:
        _, (p, fx) = align(stock_prices, fx_series)
        if len(p) >= 60:
            res = ols(pct_returns(p), pct_returns(fx))
            out.update(
                {
                    "beta_cambio": float(res.coef[1]),
                    "beta_erro_padrao": float(res.stderr[1]),
                    "r2": res.r2,
                    "n_obs": res.n,
                    "retorno_implicito_acao_pct": float(res.coef[1] * fx_shock_pct),
                }
            )
        else:
            run.warn("Menos de 60 datas em comum: beta cambial não estimado.")
    return out


@skill(
    id="economic_regime_detection",
    name="Identificação de regimes econômicos",
    version="1.0.0",
    category="macro",
    objective="Classificar o regime macro (crescimento x inflação) e o regime de volatilidade do mercado.",
    authorized_agents=[A.ECON_GLOBAL, A.ECON_BRASIL, A.QUANT, A.CIIO, A.CARTEIRA],
    inputs={
        "growth": "Series de atividade (ex.: IBC-Br, PMI)",
        "inflation": "Series de inflação (ex.: IPCA 12m)",
        "market_prices": "Series de um índice (opcional, regime de volatilidade)",
        "momentum_window": "observações para o momentum (padrão 3)",
    },
    required_inputs=["growth", "inflation"],
    tools=["numpy"],
    procedure=[
        "Momentum = variação da série na janela.",
        "Quadrantes: crescimento↑/inflação↓ = expansão desinflacionária; ↑/↑ = reflação; ↓/↑ = estagflação; ↓/↓ = desaceleração desinflacionária.",
        "Volatilidade: vol 21d anualizada vs. percentis do histórico (baixa < p33, alta > p67).",
    ],
    validation=["Janela menor que o tamanho das séries."],
    output_schema={"regime_macro": "str", "momentum_crescimento": "float", "momentum_inflacao": "float", "regime_volatilidade": "str | None"},
    failure_handling="Regime de volatilidade omitido com menos de 126 preços.",
)
def economic_regime_detection(
    run: SkillRun, growth: Series, inflation: Series, market_prices: Series | None = None, momentum_window: int = 3
) -> dict:
    if len(growth) <= momentum_window or len(inflation) <= momentum_window:
        raise SkillError("Séries curtas demais para a janela de momentum.")
    g = float(growth.values[-1] - growth.values[-1 - momentum_window])
    i = float(inflation.values[-1] - inflation.values[-1 - momentum_window])
    regimes = {
        (True, False): "expansão desinflacionária",
        (True, True): "reflação",
        (False, True): "estagflação",
        (False, False): "desaceleração desinflacionária",
    }
    out = {"regime_macro": regimes[(g > 0, i > 0)], "momentum_crescimento": g, "momentum_inflacao": i, "regime_volatilidade": None}
    if market_prices is not None:
        r = pct_returns(market_prices.values)
        if len(r) >= 126:
            rolling = np.array([r[k - 21 : k].std(ddof=1) for k in range(21, len(r) + 1)]) * np.sqrt(252)
            cur = float(rolling[-1])
            p33, p67 = np.percentile(rolling, [33, 67])
            out["volatilidade_21d_anual"] = cur
            out["percentil_volatilidade"] = float((rolling < cur).mean() * 100)
            out["regime_volatilidade"] = "baixa" if cur < p33 else ("alta" if cur > p67 else "normal")
        else:
            run.warn("Menos de 126 preços: regime de volatilidade não calculado.", degrade=False)
    return out
