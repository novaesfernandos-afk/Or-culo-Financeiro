"""Modo 8 - Auditoria do desempenho histórico das previsões."""
from __future__ import annotations

from ..core import agent_ids as A
from ..skills.base import SkillContext, run_skill
from .base import Services


def run(svc: Services, params: dict) -> dict:
    include_synth = bool(params.get("incluir_sinteticos", svc.demo))
    resolved = svc.memory.resolve_due_predictions(lambda t: svc.hub.try_get(f"price:{t}"))
    preds = svc.memory.predictions(include_synthetic=include_synth)
    only_prod = bool(params.get("somente_producao", True))  # regra 8: produção separada de experimentos
    if only_prod:
        preds = [p for p in preds if p["model_status"] == "producao"]
    rows = [
        {"id": p["id"], "ticker": p["ticker"], "made_at": p["ref_date"], "horizon": p["horizon"], "prob_up": p["prob_up"],
         "expected_return": p["expected_return"], "model_version": p["model_version"], "realized_return": p["realized_return"],
         "benchmark_return": p["benchmark_return"]}
        for p in preds
    ]
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    r = run_skill("prediction_audit", A.AUDITOR, ctx, predictions=rows, cost_bps=float(params.get("custo_bps", 20)))
    return {
        "modo": "auditoria",
        "previsoes_resolvidas_agora": resolved,
        "filtros": {"incluir_sinteticos": include_synth, "somente_producao": only_prod},
        "auditoria": r.to_dict(),
        "modelos": svc.memory.models(),
        "nota": "Previsões de modelos não validados são auditadas separadamente por versão (ver 'por_modelo').",
    }
