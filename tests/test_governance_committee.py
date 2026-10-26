"""Governança de modelos, Comitê de Investimentos, síntese do CIIO e Orquestrador."""
from __future__ import annotations

import threading
import time

import pytest

from oraculo.agents import committee as cm
from oraculo.agents.chief import synthesize
from oraculo.core import agent_ids as A
from oraculo.core.types import AgentView, Stance
from oraculo.data.providers.base import ProviderError
from oraculo.memory import GovernanceError, MemoryStore
from oraculo.orchestration.orchestrator import CycleError, Orchestrator, Task
from oraculo.skills.base import run_skill

from .conftest import make_series

GOOD_VALIDATION = {"agregado": {"habilidade_brier": 0.05, "blocos_melhores_que_base": 6, "n_blocos": 8, "n_previsoes": 1000, "ece": 0.03,
                                 "acerto": 0.56, "acerto_sempre_alta": 0.52},
                   "vazamento_detectado": False}


# --- Governança --------------------------------------------------------------------

def test_model_lifecycle_requires_validation_and_human_approval():
    m = MemoryStore()
    m.register_model("mod", "1", A.QUANT, {"w": [1]})
    with pytest.raises(GovernanceError):
        m.promote_model("mod", "1", "user:ana")  # experimental não vai à produção
    with pytest.raises(GovernanceError):
        m.attach_validation("mod", "1", A.QUANT, GOOD_VALIDATION)  # só o Auditor certifica
    assert m.attach_validation("mod", "1", A.AUDITOR, GOOD_VALIDATION) is True
    with pytest.raises(GovernanceError):
        m.approve_model("mod", "1", A.CIIO)  # agentes não aprovam
    with pytest.raises(GovernanceError):
        m.update_model_params("mod", "1", A.QUANT, {"w": [2]})  # validado é imutável
    m.approve_model("mod", "1", "user:ana")
    m.promote_model("mod", "1", "user:ana")
    assert m.production_model("mod")["version"] == "1"
    with pytest.raises(GovernanceError):
        m.register_model("mod", "1", A.QUANT, {})  # versões são imutáveis


def test_author_cannot_approve_and_bad_validation_rejected():
    m = MemoryStore()
    m.register_model("mod", "1", "user:ana", {})
    m.attach_validation("mod", "1", A.AUDITOR, GOOD_VALIDATION)
    with pytest.raises(GovernanceError):
        m.approve_model("mod", "1", "user:ana")
    m.register_model("mod", "2", A.QUANT, {})
    bad = {"agregado": {**GOOD_VALIDATION["agregado"], "habilidade_brier": -0.01}, "vazamento_detectado": False}
    assert m.attach_validation("mod", "2", A.AUDITOR, bad) is False
    assert m.get_model("mod", "2")["status"] == "rejeitado"


def test_promotion_archives_previous_production():
    m = MemoryStore()
    for v in ("1", "2"):
        m.register_model("mod", v, A.QUANT, {})
        m.attach_validation("mod", v, A.AUDITOR, GOOD_VALIDATION)
        m.approve_model("mod", v, "user:bia")
        m.promote_model("mod", v, "user:bia")
    assert m.production_model("mod")["version"] == "2"
    assert m.get_model("mod", "1")["status"] == "arquivado"
    actions = [e["action"] for e in m.model_history("mod")]
    assert actions.count("producao") == 2 and "arquivado" in actions


# --- Comitê ------------------------------------------------------------------------

def _tech_result(trend="alta", rsi=75.0, vol=0.2, synthetic=False):
    import numpy as np

    p = make_series(np.linspace(10, 20, 300), synthetic=synthetic)
    r = run_skill("technical_analysis", A.TECNICO, prices=p)
    r.output.update({"tendencia": trend, "rsi14": rsi, "volatilidade_21d_anual": vol})
    return r


def test_committee_members_are_isolated_and_auditor_rechecks():
    results = {"tec": _tech_result()}
    ev1, ev2 = cm.build_evidence(results), cm.build_evidence(results)
    ev1["tec"].output["tendencia"] = "baixa"  # adulteração em uma cópia
    assert ev2["tec"].output["tendencia"] == "alta"
    bull = cm.otimista(cm.build_evidence(results))
    bull.arguments[0].claimed_value = "inventado"
    audit = cm.auditor(cm.build_evidence(results), [bull], allow_synthetic=False)
    tampered = next(r for r in audit["revisoes"] if r["argumento"].startswith("Tendência"))
    assert tampered["veredito"] == "rejeitado" and "difere" in tampered["observacoes"][0]


def test_committee_inconclusive_and_insufficient():
    out = cm.run_committee({"tec": _tech_result(rsi=75, vol=0.6)})
    # favoráveis: tendência (1.0) + macd? ; contrários: sobrecompra (0.5) + vol alta (1.0)
    assert out["conclusao"] in ("inconclusivo", "evidência insuficiente", "inclinação favorável", "inclinação desfavorável")
    few = cm.run_committee({"tec": _tech_result(rsi=50, vol=0.2)}, min_accepted=5)
    assert few["conclusao"] == "evidência insuficiente"


def test_committee_rejects_synthetic_unless_demo():
    res = {"tec": _tech_result(synthetic=True)}
    strict = cm.run_committee(res, allow_synthetic=False)
    assert all(r["veredito"] == "rejeitado" for r in strict["revisoes_auditor"])
    demo = cm.run_committee(res, allow_synthetic=True)
    assert any(r["peso"] > 0 for r in demo["revisoes_auditor"])


def test_committee_rejects_unvalidated_model_claims(prices):
    fc = run_skill("price_direction_forecast", A.QUANT, prices=prices)
    fc.output["vantagem_sobre_base"] = 0.2  # mesmo com vantagem alta, modelo não validado é rejeitado
    out = cm.run_committee({"fc": fc})
    rev = [r for r in out["revisoes_auditor"] if r["tema"] == "modelo preditivo"]
    assert rev and all(r["veredito"] == "rejeitado" for r in rev)


# --- CIIO --------------------------------------------------------------------------

def _v(aid, stance, conf=0.5):
    return AgentView(aid, stance, conf, "resumo")


def test_chief_requires_minimum_evidence_and_preserves_contradictions():
    views = {A.TECNICO: _v(A.TECNICO, Stance.POSITIVE), A.RISCO: _v(A.RISCO, Stance.NEGATIVE)}
    s = synthesize(views, {"conclusao": "inclinação favorável"}, {}, True)
    assert s["conclusao"] == "evidência insuficiente"
    assert s["contradicoes"] and set(s["contradicoes"][0]["agentes"]) == {A.TECNICO, A.RISCO}


def test_chief_does_not_force_consensus_and_blocks_on_integrity():
    views = {a: _v(a, Stance.NEGATIVE) for a in (A.TECNICO, A.RISCO, A.FUNDAMENTALISTA)}
    s = synthesize(views, {"conclusao": "inclinação favorável"}, {}, True)
    assert s["conclusao"] == "inconclusivo"
    assert synthesize(views, {"conclusao": "inclinação desfavorável"}, {}, True)["conclusao"] == "inclinação desfavorável"
    assert synthesize(views, {"conclusao": "inclinação desfavorável"}, {}, False)["conclusao"] == "bloqueado"


# --- Orquestrador ------------------------------------------------------------------

def test_orchestrator_parallel_dependencies_retries_and_blocking():
    calls = {"flaky": 0}
    started = {}
    lock = threading.Lock()

    def slow(name):
        def f(_):
            with lock:
                started[name] = time.perf_counter()
            time.sleep(0.2)
            return name
        return f

    def flaky(_):
        calls["flaky"] += 1
        if calls["flaky"] < 2:
            raise ProviderError("instável")
        return "ok"

    def boom(_):
        raise RuntimeError("falha")

    o = Orchestrator(backoff_base=0.01)
    res = o.run([
        Task("a", "A02", slow("a")), Task("b", "A02", slow("b")),
        Task("c", "A02", lambda d: d["a"] + d["b"], deps=("a", "b")),
        Task("f", "A02", flaky, retries=2),
        Task("x", "A02", boom, retries=3), Task("y", "A02", lambda d: 1, deps=("x",)),
    ])
    assert res["c"] == "ab" and res["f"] == "ok"
    assert abs(started["a"] - started["b"]) < 0.15  # paralelo
    rec = {r["id"]: r for r in o.log()}
    assert rec["f"]["attempts"] == 2
    assert rec["x"]["status"] == "falhou" and rec["x"]["attempts"] == 1  # erro não transitório não é repetido
    assert rec["y"]["status"] == "bloqueada"
    with pytest.raises(CycleError):
        Orchestrator().run([Task("p", "A02", lambda d: 1, deps=("q",)), Task("q", "A02", lambda d: 1, deps=("p",))])


def test_committee_requires_independent_evidence_families():
    # só argumentos derivados de preço: não basta para inclinação, mesmo com balanço favorável
    out = cm.run_committee({"tec": _tech_result(trend="alta", rsi=50, vol=0.2)}, min_accepted=1)
    assert out["conclusao"] == "inconclusivo"
    assert "única família" in out["texto_conclusao"]
    fund = run_skill("fundamental_analysis", A.FUNDAMENTALISTA, financials={
        "receita": 1000, "ebitda": 300, "ebit": 250, "lucro_liquido": 150, "patrimonio_liquido": 1000, "ativo_total": 2000,
        "divida_bruta": 500, "caixa": 200, "fco": 200, "despesa_financeira": 50, "receita_anterior": 900}, financials_source="CVM")
    out2 = cm.run_committee({"tec": _tech_result(trend="alta", rsi=50, vol=0.2), "fund": fund}, min_accepted=1)
    assert out2["conclusao"] == "inclinação favorável"
    assert out2["familias_validadas"]["favoravel"] == ["fundamentos", "preços"]


def test_permanent_provider_errors_are_not_retried():
    calls = {"n": 0}

    def unauthorized(_):
        calls["n"] += 1
        raise ProviderError("HTTP 401", permanent=True)

    o = Orchestrator(backoff_base=0.01)
    o.run([Task("x", "A02", unauthorized, retries=3)])
    assert calls["n"] == 1 and o.records["x"].status == "falhou"


def test_validation_gate_rejects_model_no_better_than_always_up():
    m = MemoryStore()
    m.register_model("mod", "1", A.QUANT, {})
    weak = {"agregado": {**GOOD_VALIDATION["agregado"], "acerto": 0.519, "acerto_sempre_alta": 0.52}, "vazamento_detectado": False}
    assert m.attach_validation("mod", "1", A.AUDITOR, weak) is False
