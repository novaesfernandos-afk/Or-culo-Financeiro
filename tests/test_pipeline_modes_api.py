"""Fluxo de 7 etapas, os 8 modos, memória do usuário e API."""
from __future__ import annotations

import numpy as np
import pytest

from oraculo.agents.registry import AGENTS, consistency_report
from oraculo.core.types import Series
from oraculo.data.hub import DataHub
from oraculo.memory import MemoryStore
from oraculo.modes import MODES, run_mode
from oraculo.modes.opportunity import price_factors
from oraculo.modes.scenario import parse_question
from oraculo.orchestration.pipeline import deep_analysis

from .conftest import gbm


def test_18_agents_consistent_with_skills():
    assert len(AGENTS) == 18
    assert consistency_report() == []
    for a in AGENTS.values():
        assert a.mission and a.tools and a.skills and a.data_sources and a.deliverables and a.quality_criteria
        assert any("inventar" in r for r in a.restrictions)
    assert "alterar resultados dos modelos" in " ".join(AGENTS["A01"].restrictions)


def test_eight_modes():
    assert sorted(m.id for m in MODES.values()) == list(range(1, 9))


def test_deep_analysis_seven_stages(svc):
    r = run_mode("analise_profunda", svc, {"ticker": "petr4", "setor": "petroleo_gas"})
    assert set(r["etapas"]) == {"1_coleta", "2_analise_paralela", "3_modelagem", "4_contestacao", "5_sintese", "6_comunicacao", "7_registro"}
    md = r["relatorio_markdown"]
    for section in ("Tese favorável", "Tese contrária", "Principais indicadores", "Catalisadores", "Riscos",
                    "Cenários probabilísticos", "Condições que invalidariam a tese", "Fontes e horários"):
        assert f"## {section}" in md
    assert "DADOS SINTÉTICOS" in md and "não constitui recomendação" in md
    assert r["etapas"]["6_comunicacao"]["publicado"] is True
    assert r["etapas"]["7_registro"]["previsao_id"]
    assert len(r["comite"]["membros"]) == 3
    assert r["sintese"]["previsao_usada_na_sintese"] is False  # modelo não validado não entra na síntese
    assert svc.memory.get_analysis(r["analise_id"])["subject"] == "PETR4"


def test_deep_analysis_without_data_never_invents():
    r = deep_analysis("PETR4", DataHub(), MemoryStore())
    assert r["conclusao"] == "evidência insuficiente"
    assert "Cotações indisponíveis" in r["erro"]


def test_all_modes_run_in_demo(svc):
    svc.memory.watch("PETR4")
    for t, v in (("PETR4", 50.0), ("VALE3", 30.0), ("ITUB4", 20.0)):
        svc.memory.set_position(t, v)
    assert "alertas" in run_mode("monitoramento", svc, {})
    assert run_mode("cacador_oportunidades", svc, {"universo": ["PETR4", "VALE3", "ITUB4"], "liquidez_minima_brl": 0})["ranking"]
    assert run_mode("analise_evento", svc, {"tipo_evento": "alta_petroleo", "data_evento": "2026-06-01", "tickers": ["PETR4"]})["reacao_observada"]
    assert run_mode("simulacao_cenarios", svc, {"pergunta": "E se o dólar chegasse a R$ 6,50?"})["resultado"]["status"] in ("ok", "degradado")
    assert run_mode("analise_carteira", svc, {})["risco"]["status"] in ("ok", "degradado")
    rd = run_mode("pesquisa", svc, {"acao": "treinar_validar", "tickers": ["PETR4", "VALE3"]})
    assert rd["status"] in ("rejeitado", "validado") and rd["certificado_pelo_auditor"] is False  # sintético nunca vai à produção
    assert run_mode("pesquisa", svc, {"acao": "comparar_estrategias", "ticker": "PETR4"})["estrategias"]
    assert "auditoria" in run_mode("auditoria", svc, {})


def test_monitoring_heavy_models_respect_cooldown(svc, monkeypatch):
    import oraculo.modes.monitoring as mon

    monkeypatch.setattr(mon, "PRICE_Z_MATERIAL", 0.0)  # tudo é material
    r1 = mon.run_cycle(svc, {"tickers": ["PETR4"]})
    r2 = mon.run_cycle(svc, {"tickers": ["PETR4"]})
    assert r1["modelos_pesados_executados"] == ["PETR4"]
    assert r2["modelos_pesados_executados"] == []


def test_opportunity_factors_ignore_price_level():
    p = gbm(400, seed=3)
    assert price_factors(p) == pytest.approx(price_factors(p * 0.01))


@pytest.mark.parametrize("q,expected", [
    ("O que aconteceria com minha carteira se a Selic subisse 2 pontos percentuais?", {"selic": 2.0}),
    ("E se o petróleo caísse 20%?", {"petroleo": -20.0}),
    ("E se o dólar chegasse a R$ 6,50?", {"dolar": {"alvo": 6.5}}),
    ("E se os juros caíssem 1,5 p.p.?", {"selic": -1.5}),
])
def test_scenario_question_parser(q, expected):
    assert parse_question(q) == expected


def test_scenario_parser_refuses_ambiguous():
    with pytest.raises(ValueError):
        parse_question("E se acontecer algo?")


def test_user_memory():
    m = MemoryStore()
    m.watch("wege3")
    m.set_pref("tema", "escuro")
    m.add_alert("WEGE3", {"tipo": "preco_acima", "valor": 50})
    m.add_goal("Aposentadoria em 20 anos")
    m.set_position("WEGE3", 1000, "bens_capital")
    assert m.watchlist() == ["WEGE3"] and m.prefs()["tema"] == "escuro"
    assert m.alerts()[0]["condition"]["valor"] == 50 and m.goals() and m.portfolio()[0]["value"] == 1000


def test_prediction_resolution(svc):
    s = svc.hub.get("price:PETR4")
    ref = s.dates[-20]
    pid = svc.memory.log_prediction("PETR4", ref.isoformat(), 5, 0.6, "m@1", "producao", "x", "teste", synthetic=True)
    assert svc.memory.resolve_due_predictions(lambda t: s) == 1
    p = next(p for p in svc.memory.predictions(include_synthetic=True) if p["id"] == pid)
    i = s.dates.index(ref)
    assert p["realized_return"] == pytest.approx(s.values[i + 5] / s.values[i] - 1)


def test_api_endpoints(svc):
    from fastapi.testclient import TestClient

    from oraculo.api.app import create_app

    c = TestClient(create_app(svc))
    assert c.get("/").status_code == 200
    assert len(c.get("/api/agents").json()) == 18
    assert len(c.get("/api/skills").json()) == 32
    assert len(c.get("/api/modes").json()) == 8
    r = c.post("/api/committee", json={"ticker": "VALE3"})
    assert r.status_code == 200 and len(r.json()["comite"]["membros"]) == 3
    assert c.post("/api/modes/simulacao_cenarios", json={"pergunta": "E se nada?"}).status_code == 400
    assert c.post("/api/modes/pesquisa", json={"acao": "aprovar", "modelo_id": "x", "versao": "1", "responsavel": "A01"}).status_code == 400
    assert c.post("/api/user/watchlist", json={"ticker": "ABEV3"}).json() == ["ABEV3"]


def test_series_type_is_numpy():
    s = Series("x", [], [], source="s", retrieved_at=__import__("oraculo.core.types", fromlist=["utcnow"]).utcnow())
    assert isinstance(s.values, np.ndarray)
