"""Modo 6 - Análise de carteira. Não executa ordens de compra ou venda."""
from __future__ import annotations

import numpy as np

from ..core import agent_ids as A
from ..skills._math import align, pct_returns
from ..skills.base import SkillContext, run_skill
from .base import DEFAULT_UNIVERSE, Services, adv_brl


def run(svc: Services, params: dict) -> dict:
    stored = svc.memory.portfolio()
    positions = params.get("posicoes") or {p["ticker"]: p["value"] for p in stored}
    sectors = params.get("setores") or {p["ticker"]: p["sector"] for p in stored if p.get("sector")}
    if not positions:
        raise ValueError("Carteira vazia: cadastre posições ou informe 'posicoes'.")
    positions = {t.upper(): float(v) for t, v in positions.items()}
    candidates = [t for t in (params.get("candidatos") or DEFAULT_UNIVERSE) if t not in positions]
    keys = [f"{k}:{t}" for t in positions for k in ("price", "volume")] + [f"price:{t}" for t in candidates]
    data, errors = svc.fetch_many(keys)
    prices = {t: data[f"price:{t}"] for t in positions if f"price:{t}" in data}
    adv = {t: adv_brl(data[f"price:{t}"], data[f"volume:{t}"]) for t in positions if f"price:{t}" in data and f"volume:{t}" in data}
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    risk = run_skill("portfolio_risk_analysis", A.CARTEIRA, ctx, positions=positions, prices=prices, sectors=sectors or None, adv=adv or None)

    alternatives = []
    if risk.ok and prices:
        _, arrs = align(*prices.values())
        w = np.array([positions[t] for t in prices], float)
        w /= w.sum()
        port_dates, _ = align(*prices.values())
        port_r = np.column_stack([pct_returns(a) for a in arrs]) @ w
        pmap = dict(zip(port_dates[1:], port_r))
        for t in candidates:
            s = data.get(f"price:{t}")
            if s is None:
                continue
            common = [d for d in s.dates[1:] if d in pmap]
            if len(common) < 120:
                continue
            idx = {d: i for i, d in enumerate(s.dates)}
            cr = np.array([s.values[idx[d]] / s.values[idx[d] - 1] - 1 for d in common])
            pr = np.array([pmap[d] for d in common])
            alternatives.append({"ticker": t, "correlacao_com_carteira": float(np.corrcoef(cr, pr)[0, 1]),
                                 "volatilidade_anual": float(cr.std(ddof=1) * np.sqrt(252))})
        alternatives.sort(key=lambda x: x["correlacao_com_carteira"])
    return {
        "modo": "analise_carteira",
        "risco": risk.to_dict(),
        "alternativas_diversificacao": alternatives[:5],
        "nota": "Alternativas com menor correlação histórica são sugestões para estudo (ex.: Análise Profunda), não ordens. "
                "O sistema não executa compras ou vendas.",
        "falhas_coleta": errors,
    }
