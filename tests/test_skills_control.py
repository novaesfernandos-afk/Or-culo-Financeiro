from __future__ import annotations

import numpy as np

from oraculo.core import agent_ids as A
from oraculo.core.types import Series, utcnow
from oraculo.skills.base import SkillStatus, run_skill
from oraculo.skills.control import DISCLAIMER
from oraculo.skills.predictive import DirectionModel

from .conftest import gbm, make_series


def test_backtest_execution_lag_and_costs():
    p = make_series(np.array([10, 11, 12.1, 13.31] + [13.31] * 60))
    r = run_skill("backtesting_engine", A.QUANT, prices=p, positions=[1.0] * len(p), cost_bps=0)
    assert abs(r.output["estrategia"]["retorno_total"] - (13.31 / 10 - 1)) < 1e-9
    # sinal decidido no último dia não gera retorno (não há t+1)
    pos = [0.0] * (len(p) - 1) + [1.0]
    r2 = run_skill("backtesting_engine", A.QUANT, prices=p, positions=pos, cost_bps=10)
    assert r2.output["estrategia"]["retorno_total"] == 0
    with_cost = run_skill("backtesting_engine", A.QUANT, prices=p, positions=[1.0] * len(p), cost_bps=100)
    assert with_cost.output["custos_totais"] > 0
    assert run_skill("backtesting_engine", A.QUANT, prices=p, positions=[2.0] * len(p)).status == SkillStatus.FAILED


def test_walk_forward_purge_and_leak_detection():
    rng = np.random.default_rng(0)
    n = 900
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(0, 0.5, n) > 0).astype(float)
    r = run_skill("walk_forward_validation", A.QUANT, X=X, y=y, model_factory=lambda: DirectionModel(5), horizon=5)
    assert r.ok and r.output["agregado"]["habilidade_brier"] > 0.2
    for f in r.output["blocos"]:
        assert f["n_treino"] <= f["inicio"] - 5
    leak = run_skill("walk_forward_validation", A.AUDITOR, X=X, y=y, model_factory=lambda: DirectionModel(5), horizon=5,
                     feature_available_at=np.arange(n) + 1)
    assert leak.output["vazamento_detectado"] is True


def test_portfolio_risk():
    a, b, c = make_series(gbm(400, seed=1), "a"), make_series(gbm(400, seed=2), "b"), make_series(gbm(400, seed=1) * 1.01, "c")
    r = run_skill("portfolio_risk_analysis", A.RISCO, positions={"A": 60, "B": 30, "C": 10}, prices={"A": a, "B": b, "C": c},
                  sectors={"A": "x", "B": "y", "C": "x"}, adv={"A": 1.0, "B": 1e9})
    o = r.output
    assert r.status in (SkillStatus.OK, SkillStatus.DEGRADED)
    assert abs(sum(o["risco"]["contribuicao_risco"].values()) - 1) < 1e-6
    assert any("Concentração: A" in x for x in o["alertas"])
    assert any("Correlação elevada" in x for x in o["alertas"])
    assert any("Liquidez: A" in x for x in o["alertas"])
    assert o["risco"]["cvar95_1d_pct"] >= o["risco"]["var95_1d_pct"]


def test_prediction_audit_metrics():
    preds = [{"id": str(i), "ticker": "X", "made_at": "2026-01-01", "horizon": 5 if i % 2 else 21, "prob_up": 0.7,
              "realized_return": 0.01 if i < 7 else -0.01, "model_version": "m@1"} for i in range(10)]
    preds.append({"id": "p", "ticker": "X", "made_at": "2026-01-01", "horizon": 5, "prob_up": 0.6, "realized_return": None, "model_version": "m@1"})
    r = run_skill("prediction_audit", A.AUDITOR, predictions=preds, cost_bps=0)
    o = r.output
    assert o["total"] == 11 and o["vencidas"] == 10 and o["pendentes"] == 1
    assert abs(o["acerto_direcional"] - 0.7) < 1e-12
    assert set(o["por_horizonte"]) == {"5", "21"}
    assert len(o["piores_falhas"]) == 3


def test_data_quality_detects_problems():
    good = make_series(gbm(100))
    bad = make_series(gbm(100))
    bad.values[10] = -1
    bad.dates[20] = bad.dates[19]
    r = run_skill("data_quality_validation", A.AUDITOR, series=[good, bad])
    assert r.output["aprovadas"] == 1 and r.output["reprovadas"] == 1
    assert run_skill("data_quality_validation", A.AUDITOR, series=[good]).output["series"][0]["criticos"] == []


def test_source_verification():
    claims = [
        {"afirmacao": "Copom eleva Selic", "fontes": [{"nome": "Banco Central", "url": "https://www.bcb.gov.br/x"}]},
        {"afirmacao": "Fusão em estudo", "fontes": [{"nome": "Blog", "url": "https://blog.com", "texto": "segundo fontes, estaria em estudo"}]},
        {"afirmacao": "Dividendo extra", "fontes": [{"nome": "Reuters", "url": "https://reuters.com/a"}, {"nome": "Valor", "url": "https://valor.globo.com/b"}]},
        {"afirmacao": "Algo", "fontes": []},
    ]
    v = [x["veredito"] for x in run_skill("source_verification", A.NOTICIAS, claims=claims).output["afirmacoes"]]
    assert v == ["confirmada", "rumor", "confirmada", "não verificada"]


def test_model_drift_detection():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, (1000, 2))
    cur = np.column_stack([rng.normal(2, 1, 300), rng.normal(0, 1, 300)])
    r = run_skill("model_drift_detection", A.AUDITOR, reference_features=ref, recent_features=cur, feature_names=["a", "b"],
                  reference_brier=0.2, recent_probs=[0.9] * 50, recent_outcomes=[0] * 50)
    assert r.output["deriva_relevante"] == ["a"]
    assert r.output["piora_desempenho"] is True
    assert "Suspender" in r.output["recomendacao"]


def test_compliance_review_blocks_and_approves():
    ok = run_skill("compliance_review", A.COMPLIANCE, text=f"Análise com riscos e cenários.\n{DISCLAIMER}", sources_count=2)
    assert ok.output["aprovado"] is True
    bad = run_skill("compliance_review", A.COMPLIANCE, text="Compre já! Lucro garantido. Contato: 123.456.789-00", sources_count=0, uses_synthetic=True)
    rules = {p["regra"] for p in bad.output["pendencias"]}
    assert bad.output["aprovado"] is False
    assert {"aviso_legal", "ordem_de_negociacao", "promessa_de_resultado", "fontes", "dado_sintetico", "lgpd"} <= rules


def test_series_without_source_is_impossible():
    import pytest

    with pytest.raises(ValueError):
        Series("x", [], [], source="", retrieved_at=utcnow())
