"""Modo 4 - Análise de evento: antes x depois e empresas potencialmente afetadas."""
from __future__ import annotations

from datetime import date

import numpy as np

from ..core import agent_ids as A
from ..skills.base import SkillContext, run_skill
from .base import DEFAULT_UNIVERSE, Services


def before_after(dates: list[date], values: np.ndarray, event: date, before: int, after: int) -> dict | None:
    idx = next((i for i, d in enumerate(dates) if d >= event), None)
    if idx is None or idx < before + 1:
        return None
    r = np.diff(np.log(values))
    pre = r[idx - 1 - before : idx - 1]
    post = r[idx - 1 : idx - 1 + after]
    if len(post) == 0:
        return None
    return {
        "retorno_antes": float(np.expm1(pre.sum())),
        "retorno_depois": float(np.expm1(post.sum())),
        "vol_antes": float(pre.std(ddof=1) * np.sqrt(252)) if len(pre) > 1 else None,
        "vol_depois": float(post.std(ddof=1) * np.sqrt(252)) if len(post) > 1 else None,
        "pregoes_depois_disponiveis": int(len(post)),
    }


def run(svc: Services, params: dict) -> dict:
    etype = params["tipo_evento"]
    ev_date = date.fromisoformat(params["data_evento"])
    tickers = [t.upper() for t in (params.get("tickers") or svc.memory.watchlist() or DEFAULT_UNIVERSE)]
    sectors = params.get("setores") or {}
    before, after = int(params.get("janela_antes", 21)), int(params.get("janela_depois", 5))
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    data, errors = svc.fetch_many([f"price:{t}" for t in tickers] + ["price:^BVSP"])
    bench = data.get("price:^BVSP")
    bench_ba = before_after(bench.dates, bench.values, ev_date, before, after) if bench else None

    rows = []
    for t in tickers:
        s = data.get(f"price:{t}")
        if s is None:
            continue
        ba = before_after(s.dates, s.values, ev_date, before, after)
        if ba is None:
            rows.append({"ticker": t, "status": "sem dados suficientes ao redor do evento"})
            continue
        if bench_ba:
            ba["retorno_anormal_depois"] = ba["retorno_depois"] - bench_ba["retorno_depois"]
        rows.append({"ticker": t, "setor": sectors.get(t), **ba})
    mapping = run_skill("event_impact_mapping", A.NOTICIAS, ctx, event_type=etype,
                        companies={t: {"setor": sectors.get(t)} for t in tickers})
    hist = None
    if params.get("datas_eventos_historicos") and params.get("ticker_referencia"):
        ref = data.get(f"price:{params['ticker_referencia'].upper()}")
        if ref is not None:
            hist = run_skill("historical_event_comparison", A.NOTICIAS, ctx, prices=ref, event_dates=params["datas_eventos_historicos"],
                             benchmark=bench, current_event_date=params["data_evento"]).to_dict()
    measured = [r for r in rows if "retorno_depois" in r]
    measured.sort(key=lambda r: -abs(r.get("retorno_anormal_depois", r["retorno_depois"])))
    return {
        "modo": "analise_evento",
        "evento": {"tipo": etype, "data": ev_date.isoformat(), "janela_antes": before, "janela_depois": after},
        "ibovespa": bench_ba,
        "reacao_observada": measured,
        "sem_dados": [r for r in rows if "retorno_depois" not in r],
        "mapa_de_impacto": mapping.to_dict(),
        "comparacao_historica": hist,
        "ressalvas": [
            "Reação observada inclui outros fatores do período, não apenas o evento.",
            "Mapa de impacto setorial é hipótese estrutural; prevalece a reação medida quando houver dados.",
        ],
        "falhas_coleta": errors,
    }
