"""Um teste (ou mais) por regra obrigatória de funcionamento (seção 6)."""
from __future__ import annotations

import re
import time

import numpy as np
import pytest

from oraculo.core import agent_ids as A
from oraculo.core.costs import CostMonitor, throttle
from oraculo.core.rules import MANDATORY_RULES, RULE_ENFORCEMENT, check_series
from oraculo.core.types import AgentView, Stance
from oraculo.data.hub import DataHub
from oraculo.data.news import DEFAULT_FEEDS
from oraculo.data.providers.base import ProviderError
from oraculo.data.providers.synthetic import SyntheticProvider
from oraculo.memory import MemoryStore
from oraculo.modes import run_mode
from oraculo.orchestration.pipeline import DIRECTION_MODEL_ID, deep_analysis
from oraculo.skills._math import isotonic_fit
from oraculo.skills.base import run_skill
from oraculo.skills.predictive import DirectionModel, interval_coverage, training_set

from .conftest import make_series


def test_twelve_rules_declared_and_mapped():
    assert len(MANDATORY_RULES) == 12 and set(RULE_ENFORCEMENT) == set(range(1, 13))


def test_rule_01_never_invents_data():
    with pytest.raises(ProviderError, match="Nenhum valor será estimado"):
        DataHub().get("price:PETR4")


def test_rule_02_source_and_time_recorded(prices):
    r = run_skill("technical_analysis", A.TECNICO, prices=prices)
    assert r.sources[0].source and r.sources[0].retrieved_at and r.sources[0].as_of


def test_rule_03_stale_data_identified():
    from datetime import date, timedelta

    old = make_series(np.linspace(10, 11, 100), end=date.today() - timedelta(days=30))
    assert any("ATRASADO" in w for w in check_series(old))


def _production_model(mem: MemoryStore, prices, h: int = 5) -> str:
    """Registra, valida (Auditor), aprova (humano) e promove um modelo com mapa de calibração."""
    X, y = training_set(prices.values, h)
    m = DirectionModel(h).fit(X, y)
    p = m.params()
    th, vals = isotonic_fit(m.predict_proba(X), y)
    p["calibracao"] = {"metodo": "isotonica", "limiares": th.tolist(), "valores": vals.tolist(), "n_oos": len(y)}
    mid = DIRECTION_MODEL_ID.format(h=h)
    mem.register_model(mid, "1", A.QUANT, p)
    mem.attach_validation(mid, "1", A.AUDITOR, {"agregado": {"habilidade_brier": 0.05, "blocos_melhores_que_base": 7, "n_blocos": 8,
                                                             "n_previsoes": 1000, "ece": 0.02, "acerto": 0.56, "acerto_sempre_alta": 0.52},
                                                "vazamento_detectado": False})
    mem.approve_model(mid, "1", "user:responsavel")
    mem.promote_model(mid, "1", "user:responsavel")
    return mid


def test_rule_04_no_uncalibrated_probability_in_reports(svc):
    r = deep_analysis("PETR4", svc.hub, svc.memory)
    md = r["relatorio_markdown"]
    assert r["quant"]["previsao"] is None
    assert not re.search(r"P\(alta.*\d+%", md) and "Probabilidade calibrada" not in md
    for s in r["relatorio"]["cenarios"][:3]:
        if not r["relatorio"]["cenarios"][3]["calibrada"]:
            assert s["probabilidade"] is None


def test_rule_04_calibrated_production_model_is_shown_and_logged(svc):
    prices = svc.hub.get("price:PETR4")
    _production_model(svc.memory, prices)
    r = deep_analysis("PETR4", svc.hub, svc.memory)
    assert r["quant"]["previsao"]["calibrada"] is True and r["quant"]["previsao"]["status_modelo"] == "producao"
    assert "Probabilidade calibrada de alta" in r["relatorio_markdown"]
    assert r["etapas"]["7_registro"]["previsao_id"]


def test_rule_04_interval_coverage_measures_calibration():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 1500)
    cov = interval_coverage(r, horizon=5)
    assert cov["n"] >= 20 and abs(cov["cobertura_90"] - 0.9) < 0.1 and cov["calibrada"] is True


def test_rule_05_no_future_information(prices):
    X, y = training_set(prices.values, horizon=5, upto=400)
    assert len(y) == len(range(63, 396))
    p = make_series(np.r_[np.full(60, 10.0), np.full(10, 20.0)])
    out = run_skill("backtesting_engine", A.QUANT, prices=p, positions=[0.0] * 59 + [1.0] * 11, cost_bps=0).output
    assert out["estrategia"]["retorno_total"] == pytest.approx(1.0)  # posição de t só captura t->t+1


def test_rule_06_opinions_are_not_probabilities(svc):
    d = AgentView(A.TECNICO, Stance.POSITIVE, 0.42, "x").to_dict()
    assert "confidence" not in d and d["peso_qualitativo"] == "média"
    md = deep_analysis("PETR4", svc.hub, svc.memory)["relatorio_markdown"]
    assert not re.search(r"confiança \d+%", md)
    assert "não probabilidades estatísticas" in md


@pytest.mark.parametrize("text", ["Retorno garantido de 20%", "Lucro certo!", "Ação vai subir amanhã", "Investimento sem risco"])
def test_rule_07_no_profit_promises(text):
    from oraculo.skills.control import DISCLAIMER

    r = run_skill("compliance_review", A.COMPLIANCE, text=f"{text}\n{DISCLAIMER}", sources_count=1)
    assert r.output["aprovado"] is False


def test_rule_08_experimental_separated_from_production(svc):
    run_mode("pesquisa", svc, {"acao": "treinar_validar", "tickers": ["PETR4"]})
    assert all(a["environment"] == "producao" for a in svc.memory.analyses())
    assert svc.memory.analyses(environment="experimental")
    svc.memory.log_prediction("X", "2026-01-01", 5, 0.6, "m@1", "experimental", "d", "r", synthetic=True)
    out = run_mode("auditoria", svc, {})["auditoria"]["output"]
    assert out["total"] == 0  # auditoria padrão considera só produção


def test_rule_09_source_access_limits():
    assert "google" not in DEFAULT_FEEDS.split(",")  # uso pessoal/não comercial: desligado por padrão
    t0 = time.monotonic()
    for _ in range(3):
        throttle("https://brapi.dev/api/quote/X")
    assert time.monotonic() - t0 >= 0.5  # intervalo mínimo de 0,3 s entre chamadas


def test_rule_10_no_trading_routes(svc):
    from oraculo.api.app import create_app

    paths = [r.path.lower() for r in create_app(svc).routes]
    assert not any(re.search(r"order|ordem|trade|negocia|corretora|broker", p) for p in paths)


def test_rule_11_costs_monitored_and_budget_enforced():
    mem = MemoryStore()
    mon = CostMonitor(budgets={"SINTÉTICO (gerador interno)": 1}, memory=mem)
    import oraculo.data.hub as hub_mod

    orig = hub_mod.COSTS
    hub_mod.COSTS = mon
    try:
        hub = DataHub([SyntheticProvider()])
        hub.get("price:AAA")
        with pytest.raises(ProviderError, match="Orçamento diário"):
            hub.get("price:BBB")
    finally:
        hub_mod.COSTS = orig
    assert mon.today()["chamadas_dados"] == 1
    assert mem.usage()[0]["calls"] == 1


def test_rule_12_insufficient_data_is_displayed(svc):
    from oraculo.api.dashboard import build

    r = deep_analysis("PETR4", svc.hub, svc.memory)
    assert r["sintese"]["frentes_sem_evidencia"]
    assert "Sem evidência suficiente" in r["relatorio_markdown"]
    d = build(svc)
    assert any("Nenhum modelo preditivo em produção" in x for x in d["dados_insuficientes"])
