from __future__ import annotations

from datetime import date

import numpy as np

from oraculo.core import agent_ids as A
from oraculo.core.types import NewsItem, utcnow
from oraculo.skills.base import SkillStatus, run_skill
from oraculo.skills.predictive import direction_features, training_set

from .conftest import gbm, make_series, news


# --- Inteligência --------------------------------------------------------------------

def test_news_dedup_and_rumor_classification():
    items = [
        news("Petrobras anuncia novo plano de investimentos para 2027", tickers=["PETR4"]),
        news("Petrobras anuncia novo plano de investimentos para 2027 e 2028", source="Valor", url="https://valor.globo.com/x"),
        news("Vale estaria negociando venda de ativo, segundo fontes", source="Blog X", url="https://blogx.com/a"),
    ]
    r = run_skill("global_news_monitor", A.NOTICIAS, items=items)
    assert r.output["unicas"] == 2 and r.output["duplicadas"] == 1
    statuses = {g["titulo"][:9]: g["status"] for g in r.output["grupos"]}
    assert statuses["Petrobras"] == "confirmada"
    assert statuses["Vale esta"] == "rumor"
    seen = [g["fingerprint"] for g in r.output["grupos"]]
    r2 = run_skill("global_news_monitor", A.NOTICIAS, items=items, seen_fingerprints=seen)
    assert r2.output["novidades"] == 0


def test_news_without_source_is_rejected():
    bad = NewsItem(title="Sem fonte", source="", published_at=utcnow(), retrieved_at=utcnow())
    r = run_skill("global_news_monitor", A.NOTICIAS, items=[bad])
    assert r.status == SkillStatus.FAILED


def test_sentiment_with_negation():
    pos = run_skill("news_sentiment_analysis", A.NOTICIAS, items=[news("Lucro recorde e crescimento forte, acima do esperado")])
    neg = run_skill("news_sentiment_analysis", A.NOTICIAS, items=[news("Empresa não teve lucro e registra queda e prejuízo")])
    assert pos.output["escore_agregado"] > 0.5
    assert neg.output["escore_agregado"] < -0.5


def test_event_impact_mapping():
    r = run_skill("event_impact_mapping", A.NOTICIAS, event_type="alta_petroleo",
                  companies={"PETR4": {"setor": "petroleo_gas"}, "AZUL4": {"setor": "aviacao", "betas": {"petroleo": -0.6}}, "X": {"setor": "bancos"}},
                  factor="petroleo", shock_pct=20)
    rows = {e["ticker"]: e for e in r.output["empresas"]}
    assert rows["PETR4"]["direcao_hipotese"] == 1
    assert abs(rows["AZUL4"]["impacto_estimado_pct"] + 12) < 1e-9 and rows["AZUL4"]["base"] == "beta medido"
    assert rows["X"]["direcao_hipotese"] == 0
    assert run_skill("event_impact_mapping", A.NOTICIAS, event_type="inventado", companies={}).status == SkillStatus.FAILED


def test_geopolitical_scoring_rejects_unsourced():
    today = date.today().isoformat()
    ev = [{"nome": "Conflito", "severidade": 5, "probabilidade": 1, "exposicao_brasil": 1, "data": today, "fonte": "Reuters"},
          {"nome": "Boato", "severidade": 5, "probabilidade": 1, "data": today}]
    r = run_skill("geopolitical_risk_scoring", A.GEOPOLITICA, events=ev)
    assert r.output["rejeitados"] == 1
    assert 99 <= r.output["indice"] <= 100


def test_seasonality_requires_out_of_sample_confirmation():
    # retornos mensais com efeito forte e persistente em dezembro
    rng = np.random.default_rng(2)
    n = 252 * 12
    p = make_series(np.ones(n))
    rets = rng.normal(0, 0.01, n)
    for i, d in enumerate(p.dates):
        if d.month == 12:
            rets[i] += 0.01
    s = make_series(20 * np.cumprod(1 + rets))
    r = run_skill("seasonality_detection", A.SAZONAL, prices=s, by="month")
    assert "mês 12" in r.output["padroes_confirmados"]
    rw = run_skill("seasonality_detection", A.SAZONAL, prices=make_series(gbm(n, mu=0, seed=9)), by="month")
    assert rw.output["padroes_confirmados"] == []


def test_historical_event_comparison():
    p = make_series(gbm(600, seed=4))
    dates = [p.dates[i].isoformat() for i in (100, 200, 300, 400)]
    r = run_skill("historical_event_comparison", A.NOTICIAS, prices=p, event_dates=dates, windows=[1, 5, 21],
                  current_event_date=p.dates[-10].isoformat())
    assert r.output["janelas"][5]["n"] == 4
    assert r.output["evento_atual"]["janela"] == 5


# --- Preditivas ------------------------------------------------------------------------

def test_direction_features_have_no_lookahead():
    p = gbm(500, seed=7)
    X1, _ = direction_features(p)
    p2 = p.copy()
    p2[400:] *= 3  # muda o futuro
    X2, _ = direction_features(p2)
    assert np.allclose(X1[:400], X2[:400], equal_nan=True)


def test_training_set_only_uses_known_labels():
    p = gbm(500, seed=7)
    X, y = training_set(p, horizon=5, upto=300)
    assert len(y) == len(range(63, 296))  # i + 5 <= 300


def test_price_direction_forecast_flags_unvalidated(prices):
    r = run_skill("price_direction_forecast", A.QUANT, prices=prices, horizon=5)
    assert r.ok and 0 <= r.output["prob_alta"] <= 1
    assert r.output["status_modelo"] == "experimental" and r.output["calibrada"] is False
    assert any("sem validação" in w for w in r.warnings)


def test_return_distribution_and_volatility(prices):
    d = run_skill("return_distribution_forecast", A.QUANT, prices=prices, horizon=10)
    q = list(d.output["quantis"].values())
    assert q == sorted(q)
    v = run_skill("volatility_forecast", A.RISCO, prices=prices)
    assert v.ok and v.output["persistencia"] < 1
    assert 0.1 < v.output["vol_garch_horizonte"] < 0.5  # sigma diária 1,5% ~ 24% a.a.


def test_scenario_simulation_recovers_linear_exposure():
    rng = np.random.default_rng(8)
    n = 800
    f = rng.normal(0, 0.01, n)
    s = 2.0 * f + rng.normal(0, 0.002, n)
    fac = make_series(100 * np.cumprod(1 + f), "fac")
    st = make_series(10 * np.cumprod(1 + s), "st")
    r = run_skill("scenario_simulation", A.CARTEIRA, positions={"ST": 1000.0}, scenario={"fac": -10}, stock_prices={"ST": st}, factor_series={"fac": fac})
    assert abs(r.output["posicoes"][0]["impacto_pct"] + 20) < 1.5
    lo, hi = r.output["carteira"]["intervalo_90_rs"]
    assert lo < r.output["carteira"]["impacto_rs"] < hi
    assert r.output["hipoteses"] and r.output["limitacoes"]


def test_scenario_target_level_conversion():
    rng = np.random.default_rng(1)
    fx = make_series(5 * np.cumprod(1 + rng.normal(0, 0.005, 600)), "fx")
    st = make_series(10 * np.cumprod(1 + rng.normal(0, 0.01, 600)), "st")
    r = run_skill("scenario_simulation", A.CARTEIRA, positions={"ST": 1.0}, scenario={"fx": {"alvo": 6.5, "atual": 5.0}},
                  stock_prices={"ST": st}, factor_series={"fx": fx})
    assert abs(r.output["choques_aplicados"]["fx"] - 30.0) < 1e-9


def test_model_ensemble_weights_by_brier():
    r = run_skill("model_ensemble", A.QUANT, forecasts=[{"modelo": "a", "prob": 0.8, "brier_historico": 0.1}, {"modelo": "b", "prob": 0.4, "brier_historico": 0.3}])
    assert r.output["pesos"]["a"] > r.output["pesos"]["b"]
    assert r.output["discordancia_alta"] is True


def test_calibration_metrics():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(int)
    r = run_skill("prediction_calibration", A.AUDITOR, probs=p.tolist(), outcomes=y.tolist(), new_prob=0.7)
    assert r.output["ece"] < 0.05 and r.output["metodo"] == "isotonica"
    assert abs(r.output["prob_calibrada"] - 0.7) < 0.1
    overconf = np.where(p > 0.5, 0.99, 0.01)
    r2 = run_skill("prediction_calibration", A.AUDITOR, probs=overconf.tolist(), outcomes=y.tolist())
    assert r2.output["ece"] > 0.2
    assert run_skill("prediction_calibration", A.AUDITOR, probs=[0.5] * 10, outcomes=[1] * 10).status == SkillStatus.FAILED



def test_scenario_flags_shock_outside_history():
    rng = np.random.default_rng(1)
    fx = make_series(5 * np.cumprod(1 + rng.normal(0, 0.003, 600)), "fx")
    st = make_series(10 * np.cumprod(1 + rng.normal(0, 0.01, 600)), "st")
    r = run_skill("scenario_simulation", A.CARTEIRA, positions={"ST": 1.0}, scenario={"fx": 30}, stock_prices={"ST": st}, factor_series={"fx": fx})
    assert "fx" in r.output["choques_fora_do_historico"]
    assert any("FORA DO HISTÓRICO" in w for w in r.warnings)
