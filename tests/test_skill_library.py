"""Testes estruturais da biblioteca: as 32 skills existem, têm especificação completa e respeitam permissões."""
from __future__ import annotations

import pytest

from oraculo.core import agent_ids as A
from oraculo.skills import REGISTRY
from oraculo.skills.base import SkillStatus, run_skill

EXPECTED = [
    "global_macro_monitor", "brazil_macro_monitor", "central_bank_analysis", "yield_curve_analysis", "fx_impact_analysis",
    "economic_regime_detection", "fundamental_analysis", "valuation_engine", "technical_analysis", "sector_comparison",
    "earnings_analysis", "commodity_exposure", "global_news_monitor", "news_sentiment_analysis", "event_impact_mapping",
    "geopolitical_risk_scoring", "seasonality_detection", "historical_event_comparison", "price_direction_forecast",
    "return_distribution_forecast", "volatility_forecast", "scenario_simulation", "model_ensemble", "prediction_calibration",
    "backtesting_engine", "walk_forward_validation", "portfolio_risk_analysis", "prediction_audit", "data_quality_validation",
    "source_verification", "model_drift_detection", "compliance_review",
]


def test_all_32_skills_registered():
    assert sorted(REGISTRY) == sorted(EXPECTED)
    assert len(REGISTRY) == 32


@pytest.mark.parametrize("skill_id", EXPECTED)
def test_spec_complete(skill_id):
    spec = REGISTRY[skill_id].spec
    for field in ("id", "name", "version", "category", "objective", "failure_handling"):
        assert getattr(spec, field), f"{skill_id}.{field} vazio"
    for field in ("authorized_agents", "tools", "procedure", "validation"):
        assert len(getattr(spec, field)) > 0, f"{skill_id}.{field} vazio"
    assert spec.inputs and spec.output_schema
    assert set(spec.required_inputs) <= set(spec.inputs)
    assert set(spec.authorized_agents) <= set(A.ALL)
    assert spec.version.count(".") == 2


def test_unauthorized_agent_is_denied(prices):
    r = run_skill("price_direction_forecast", A.COMUNICADOR, prices=prices)
    assert r.status == SkillStatus.DENIED
    assert "não autorizado" in r.error


def test_missing_required_input_fails():
    r = run_skill("technical_analysis", A.TECNICO)
    assert r.status == SkillStatus.FAILED
    assert "prices" in r.error


def test_result_has_provenance_and_digests(prices):
    r = run_skill("technical_analysis", A.TECNICO, prices=prices)
    assert r.ok
    assert r.sources and r.sources[0].source == "Fixture de teste"
    assert len(r.input_digest) == 16 and len(r.output_digest) == 16
    d = r.to_dict()
    assert d["skill_id"] == "technical_analysis" and d["version"] == "1.0.0"


def test_unexpected_exception_is_captured(prices):
    # uma série com valores não numéricos vira NaN e a skill deve falhar de forma controlada, sem exceção
    r = run_skill("valuation_engine", A.FUNDAMENTALISTA, fcff_projection=[1, 2], wacc="x", terminal_growth=0.03, net_debt=0,
                  shares=1, assumptions_source="teste")
    assert r.status == SkillStatus.FAILED
