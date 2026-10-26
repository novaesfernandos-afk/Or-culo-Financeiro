"""Modo 3 - Caçador de oportunidades.

Ranking explicável por fatores invariantes à escala de preço: o preço nominal
da ação nunca entra no cálculo (preço baixo não é oportunidade).
"""
from __future__ import annotations

import numpy as np

from ..core import agent_ids as A
from ..data.providers.base import ProviderError
from ..skills.base import SkillContext, run_skill
from .base import DEFAULT_UNIVERSE, Services, adv_brl, zscore

FACTOR_DOC = {
    "momentum_12_1": "Retorno de 12 meses excluindo o último mês (tendência persistente).",
    "baixa_volatilidade": "Volatilidade de 63 pregões com sinal invertido (menor risco pontua mais).",
    "tendencia": "Distância do preço à MMS200, limitada a ±30%.",
    "resiliencia": "Drawdown atual em relação ao máximo de 252 pregões (menor queda pontua mais).",
    "qualidade": "Pontuação fundamentalista 0-6 (somente se demonstrações forem fornecidas).",
    "valuation": "Upside do DCF (somente se premissas forem fornecidas).",
    "rendimento_lucro": "Lucro dos últimos 12 meses / valor de mercado (inverso do P/L), do último DFP na CVM.",
}
DEFAULT_WEIGHTS = {"momentum_12_1": 1.0, "baixa_volatilidade": 1.0, "tendencia": 0.5, "resiliencia": 0.5, "qualidade": 1.0, "valuation": 1.0, "rendimento_lucro": 1.0}


def price_factors(p: np.ndarray) -> dict[str, float]:
    """Fatores a partir de preços. Invariantes a multiplicar todos os preços por uma constante."""
    r = np.diff(np.log(p))
    s200 = p[-200:].mean()
    return {
        "momentum_12_1": float(p[-22] / p[-253] - 1),
        "baixa_volatilidade": float(-r[-63:].std(ddof=1) * np.sqrt(252)),
        "tendencia": float(np.clip(p[-1] / s200 - 1, -0.3, 0.3)),
        "resiliencia": float(p[-1] / p[-252:].max() - 1),
    }


def run(svc: Services, params: dict) -> dict:
    universe = [t.upper() for t in (params.get("universo") or DEFAULT_UNIVERSE)]
    min_adv = float(params.get("liquidez_minima_brl", 10e6))
    weights = {**DEFAULT_WEIGHTS, **(params.get("pesos") or {})}
    fundamentals = params.get("fundamentos") or {}  # {ticker: {"financials":..., "financials_source":...}}
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    keys = [f"{k}:{t}" for t in universe for k in ("price", "volume")]
    data, errors = svc.fetch_many(keys)

    eligible, excluded = [], []
    for t in universe:
        p, v = data.get(f"price:{t}"), data.get(f"volume:{t}")
        if p is None or v is None:
            excluded.append({"ticker": t, "motivo": "dados indisponíveis"})
            continue
        q = run_skill("data_quality_validation", A.QUANT, ctx, series=[p])
        if not q.ok or q.output["reprovadas"]:
            excluded.append({"ticker": t, "motivo": "reprovado na qualidade de dados"})
            continue
        if len(p) < 260:
            excluded.append({"ticker": t, "motivo": f"histórico curto ({len(p)} pregões)"})
            continue
        if p.is_stale():
            excluded.append({"ticker": t, "motivo": f"dado atrasado (última data {p.last_date})"})
            continue
        adv = adv_brl(p, v)
        if adv < min_adv:
            excluded.append({"ticker": t, "motivo": f"liquidez R$ {adv/1e6:.1f} mi/dia abaixo do mínimo"})
            continue
        f = price_factors(p.values)
        fund_source = None
        if t not in fundamentals and svc.cvm is not None and params.get("usar_fundamentos_cvm", True):
            try:
                fa = svc.cvm.annual_financials(t)
                mcap = p.last_value * fa["acoes_em_circulacao"] if fa["acoes_em_circulacao"] else None
                fr = run_skill("fundamental_analysis", A.FUNDAMENTALISTA, ctx, financials=fa["financials"],
                               financials_source=fa["financials_source"], market_cap=mcap,
                               financial_institution=fa["instituicao_financeira"])
                if fr.ok:
                    f["qualidade"] = float(fr.output["pontuacao_qualidade"])
                    pl = (fr.output.get("multiplos") or {}).get("p_l")
                    if pl:
                        f["rendimento_lucro"] = 1 / pl
                    fund_source = fa["financials_source"]
            except ProviderError as exc:
                errors.append(f"CVM {t}: {exc}")
        if t in fundamentals:
            fr = run_skill("fundamental_analysis", A.FUNDAMENTALISTA, ctx, **fundamentals[t])
            if fr.ok:
                f["qualidade"] = float(fr.output["pontuacao_qualidade"])
            if fundamentals[t].get("upside_dcf") is not None:
                f["valuation"] = float(fundamentals[t]["upside_dcf"])
        eligible.append({"ticker": t, "liquidez_brl": adv, "fatores": f, "sintetico": p.synthetic, "data": p.last_date.isoformat(),
                         "fonte_fundamentos": fund_source})

    if not eligible:
        return {"modo": "cacador_oportunidades", "ranking": [], "excluidos": excluded, "falhas_coleta": errors}
    names = [k for k in DEFAULT_WEIGHTS if any(k in e["fatores"] for e in eligible)]
    Z = {k: zscore(np.array([e["fatores"].get(k, np.nan) for e in eligible])) for k in names}
    for i, e in enumerate(eligible):
        contrib = {k: (float(Z[k][i]) * weights[k] if np.isfinite(Z[k][i]) else None) for k in names}
        used = [k for k in names if contrib[k] is not None]
        wsum = sum(weights[k] for k in used)
        e["pontuacao"] = sum(contrib[k] for k in used) / wsum if wsum else 0.0
        e["contribuicoes"] = contrib
        top = sorted(((k, c) for k, c in contrib.items() if c is not None), key=lambda x: -x[1])
        e["explicacao"] = (
            f"Pontos fortes: {', '.join(k for k, c in top[:2] if c > 0) or 'nenhum'}; "
            f"pontos fracos: {', '.join(k for k, c in top[::-1][:2] if c < 0) or 'nenhum'}."
        )
    eligible.sort(key=lambda e: -e["pontuacao"])
    for i, e in enumerate(eligible, 1):
        e["posicao"] = i
    return {
        "modo": "cacador_oportunidades",
        "ranking": eligible,
        "excluidos": excluded,
        "fatores": {k: FACTOR_DOC[k] for k in names},
        "pesos": {k: weights[k] for k in names},
        "metodologia": "Z-scores transversais por fator, média ponderada. O preço nominal não é usado: preço baixo não é sinal de oportunidade.",
        "ressalva": "Ranking é ponto de partida para Análise Profunda, não recomendação.",
        "dados_sinteticos": any(e["sintetico"] for e in eligible),
        "falhas_coleta": errors,
    }
