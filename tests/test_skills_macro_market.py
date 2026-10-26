from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from oraculo.core import agent_ids as A
from oraculo.core.types import Frequency
from oraculo.skills.base import SkillStatus, run_skill

from .conftest import gbm, make_series


# --- Macro -------------------------------------------------------------------------

def test_macro_monitor_flags_material_change():
    rng = np.random.default_rng(0)
    v = 10 + np.cumsum(rng.normal(0, 0.01, 300))
    v[-1] = v[-2] + 0.5  # salto de ~50 desvios
    r = run_skill("global_macro_monitor", A.ECON_GLOBAL, indicators={"x": make_series(v, "x")})
    assert r.ok and r.output["n_materiais"] == 1
    assert r.output["indicadores"][0]["mudanca_material"]


def test_brazil_monitor_real_rate():
    selic = make_series(np.full(50, 15.0), "selic_meta")
    ipca = make_series(np.full(50, 5.0), "ipca_12m")
    r = run_skill("brazil_macro_monitor", A.ECON_BRASIL, indicators={"selic_meta": selic, "ipca_12m": ipca})
    assert abs(r.output["juro_real_ex_post"] - ((1.15 / 1.05 - 1) * 100)) < 1e-9


def test_stale_series_is_flagged_and_degrades():
    old = make_series(np.linspace(10, 11, 50), "selic_meta", end=date.today() - timedelta(days=40))
    r = run_skill("brazil_macro_monitor", A.ECON_BRASIL, indicators={"selic_meta": old})
    assert r.status == SkillStatus.DEGRADED
    assert any("ATRASADO" in w for w in r.warnings)


def test_central_bank_tightening_cycle():
    v = [10.0] * 30 + [10.5] * 30 + [11.0] * 30 + [11.75] * 10
    r = run_skill("central_bank_analysis", A.ECON_BRASIL, policy_rate=make_series(v, "selic_meta"))
    assert r.output["fase"] == "aperto"
    assert r.output["ultima_decisao_bps"] == 75.0
    assert r.output["ciclo_acumulado_bps"] == 175.0


def test_yield_curve_inversion_and_shift():
    cur = {0.5: 15.0, 2: 14.0, 5: 13.0, 10: 12.5}
    prev = {0.5: 14.0, 2: 13.0, 5: 12.0, 10: 11.5}
    r = run_skill("yield_curve_analysis", A.ECON_BRASIL, curve=cur, previous_curve=prev, curve_source="Teste DI")
    assert r.output["invertida"] is True
    assert abs(r.output["variacao"]["deslocamento_paralelo_pp"] - 1.0) < 1e-9
    assert run_skill("yield_curve_analysis", A.ECON_BRASIL, curve={1: 1, 2: 2}, curve_source="x").status == SkillStatus.FAILED


def test_fx_impact_and_empirical_beta():
    rng = np.random.default_rng(3)
    fx_r = rng.normal(0, 0.008, 400)
    stock_r = 0.8 * fx_r + rng.normal(0, 0.002, 400)
    fx = make_series(5 * np.cumprod(1 + fx_r), "fx")
    st = make_series(20 * np.cumprod(1 + stock_r), "st")
    exp = {"receita": 100, "custos": 60, "receita_usd_pct": 80, "custo_usd_pct": 20, "divida_usd": 50, "hedge_pct": 50}
    r = run_skill("fx_impact_analysis", A.ECON_BRASIL, fx_shock_pct=10, exposure=exp, stock_prices=st, fx_series=fx)
    assert abs(r.output["impacto_operacional"] - (100 * 0.8 * 0.1 - 60 * 0.2 * 0.1)) < 1e-9
    assert abs(r.output["impacto_financeiro"] - (-50 * 0.5 * 0.1)) < 1e-9
    assert abs(r.output["beta_cambio"] - 0.8) < 0.05


def test_regime_detection_quadrants():
    up = make_series(np.linspace(100, 110, 12), "g", freq=Frequency.MONTHLY)
    down = make_series(np.linspace(6, 4, 12), "i", freq=Frequency.MONTHLY)
    r = run_skill("economic_regime_detection", A.ECON_BRASIL, growth=up, inflation=down, market_prices=make_series(gbm(300)))
    assert r.output["regime_macro"] == "expansão desinflacionária"
    assert r.output["regime_volatilidade"] in ("baixa", "normal", "alta")


# --- Mercado -----------------------------------------------------------------------

FIN = {"receita": 1000, "ebitda": 300, "ebit": 250, "lucro_liquido": 150, "patrimonio_liquido": 1000, "ativo_total": 2000,
       "divida_bruta": 500, "caixa": 200, "fco": 200, "capex": 80, "despesa_financeira": 50, "receita_anterior": 900, "lucro_anterior": 120}


def test_fundamental_ratios():
    r = run_skill("fundamental_analysis", A.FUNDAMENTALISTA, financials=FIN, financials_source="CVM DFP 2025")
    ind = r.output["indicadores"]
    assert abs(ind["margem_ebitda"] - 0.3) < 1e-12
    assert abs(ind["roe"] - 0.15) < 1e-12
    assert abs(ind["divida_liquida_ebitda"] - 1.0) < 1e-12
    assert r.output["pontuacao_qualidade"] == 6


def test_valuation_dcf_known_value():
    r = run_skill("valuation_engine", A.FUNDAMENTALISTA, fcff_projection=[100.0], wacc=0.10, terminal_growth=0.0, net_debt=0.0,
                  shares=1.0, assumptions_source="teste", price=1000.0)
    # PV = 100/1.1 + (100/0.10)/1.1 = 1000
    assert abs(r.output["valor_por_acao_dcf"] - 1000.0) < 1e-6
    assert abs(r.output["upside_dcf"]) < 1e-9
    bad = run_skill("valuation_engine", A.FUNDAMENTALISTA, fcff_projection=[1.0], wacc=0.03, terminal_growth=0.05, net_debt=0,
                    shares=1, assumptions_source="t")
    assert bad.status == SkillStatus.FAILED and "WACC" in bad.error


def test_technical_uptrend_and_rsi_bounds():
    p = make_series(np.linspace(10, 30, 300) * (1 + 0.002 * np.sin(np.arange(300))))
    r = run_skill("technical_analysis", A.TECNICO, prices=p)
    assert r.output["tendencia"] == "alta"
    assert 0 <= r.output["rsi14"] <= 100
    assert run_skill("technical_analysis", A.TECNICO, prices=make_series(np.ones(30))).status == SkillStatus.FAILED


def test_sector_comparison_percentiles():
    peers = {"A": {"roe": 0.30, "divida": 1.0}, "B": {"roe": 0.10, "divida": 3.0}, "C": {"roe": 0.20, "divida": 2.0}}
    r = run_skill("sector_comparison", A.SETORIAL, target="A", peers=peers, data_source="teste", higher_is_better={"divida": False})
    assert r.output["metricas"]["roe"]["percentil"] > 50
    assert r.output["metricas"]["divida"]["percentil_ajustado"] > 50


def test_earnings_growth_and_no_invented_consensus():
    r = run_skill("earnings_analysis", A.FUNDAMENTALISTA, actual={"receita": 110, "lucro_liquido": 12, "fco": 15, "ativo_total": 100},
                  year_ago={"receita": 100, "lucro_liquido": 10}, source="Release 3T26")
    assert abs(r.output["crescimento"]["receita"] - 0.10) < 1e-12
    assert r.output["surpresa"] is None


def test_commodity_exposure_recovers_beta():
    rng = np.random.default_rng(5)
    c = rng.normal(0, 0.02, 500)
    s = 1.5 * c + rng.normal(0, 0.005, 500)
    r = run_skill("commodity_exposure", A.COMMODITIES, stock_prices=make_series(10 * np.cumprod(1 + s)), commodity_prices=make_series(80 * np.cumprod(1 + c), "oil"))
    assert abs(r.output["beta"] - 1.5) < 0.05
    assert r.output["t_stat"] > 10
