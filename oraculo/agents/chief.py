"""Agente 01 - Chief Investment Intelligence Officer: síntese e controle de evidência."""
from __future__ import annotations

from ..core import agent_ids as A
from ..core.types import AgentView, Stance

# Agentes de controle/comunicação não contam como "frentes de evidência" independentes.
NON_EVIDENCE = {A.CIIO, A.ORQUESTRADOR, A.AUDITOR, A.COMPLIANCE, A.COMUNICADOR}
MIN_EVIDENCE_FRONTS = 3
CONTRADICTION_MIN_CONF = 0.35


def _sign(s: Stance) -> int:
    return {Stance.POSITIVE: 1, Stance.NEGATIVE: -1}.get(s, 0)


def synthesize(views: dict[str, AgentView], committee: dict | None, quant_pkg: dict, integrity_ok: bool) -> dict:
    fronts = [v for k, v in views.items() if k not in NON_EVIDENCE and v.stance != Stance.INSUFFICIENT]
    insufficient = [k for k, v in views.items() if k not in NON_EVIDENCE and v.stance == Stance.INSUFFICIENT]
    num = sum(_sign(v.stance) * v.confidence for v in fronts)
    den = sum(v.confidence for v in fronts)
    aggregate = num / den if den else 0.0

    contradictions = []
    for a in fronts:
        for b in fronts:
            if a.agent_id < b.agent_id and _sign(a.stance) * _sign(b.stance) == -1 and min(a.confidence, b.confidence) >= CONTRADICTION_MIN_CONF:
                contradictions.append({"agentes": [a.agent_id, b.agent_id], "posicoes": [a.stance.value, b.stance.value],
                                       "resumos": [a.summary, b.summary]})
    revisions = []
    for k, v in views.items():
        stale = [w for w in v.warnings if "ATRASADO" in w]
        if stale:
            revisions.append({"agente": k, "motivo": "dados atrasados", "detalhe": stale[:3]})
    auditor = views.get(A.AUDITOR)
    blocked = (auditor is not None and auditor.stance == Stance.INSUFFICIENT) or not integrity_ok

    committee_concl = (committee or {}).get("conclusao")
    if blocked:
        final = "bloqueado"
        reason = "Auditoria reprovou dados ou detectou violação de integridade."
    elif len(fronts) < MIN_EVIDENCE_FRONTS:
        final = "evidência insuficiente"
        reason = f"Apenas {len(fronts)} frente(s) com evidência; mínimo {MIN_EVIDENCE_FRONTS}."
    elif committee_concl in (None, "evidência insuficiente", "inconclusivo"):
        final = committee_concl or "inconclusivo"
        reason = "Comitê de Investimentos não encontrou inclinação sustentada pelas evidências validadas."
    else:
        committee_sign = 1 if committee_concl == "inclinação favorável" else -1
        agg_sign = 1 if aggregate > 0.1 else (-1 if aggregate < -0.1 else 0)
        if agg_sign == committee_sign:
            final = committee_concl
            reason = "Comitê e síntese dos especialistas apontam na mesma direção."
        else:
            final = "inconclusivo"
            reason = "Comitê e síntese dos especialistas divergem; divergência preservada, sem consenso forçado."
    return {
        "conclusao": final,
        "justificativa": reason,
        "indice_agregado_especialistas": aggregate,
        "nota_indice": "Média ponderada das posições qualitativas dos especialistas (-1 a +1); não é probabilidade.",
        "frentes_com_evidencia": [v.agent_id for v in fronts],
        "frentes_sem_evidencia": insufficient,
        "contradicoes": contradictions,
        "revisoes_solicitadas": revisions,
        "conclusao_comite": committee_concl,
        "integridade_quantitativa": "verificada" if integrity_ok else "VIOLADA",
        "previsao_usada_na_sintese": bool(quant_pkg.get("previsao") and quant_pkg["previsao"]["status_modelo"] == "producao"
                                          and quant_pkg["previsao"].get("calibrada")),
    }
