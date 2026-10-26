"""Painel principal: visão consolidada com fonte, horário e atraso de cada dado (regras 2, 3 e 12)."""
from __future__ import annotations

from ..core.costs import COSTS
from ..core.types import Series, utcnow
from ..modes.base import Services, valued_portfolio
from ..orchestration.pipeline import MACRO_KEYS

MACRO_LABELS = {"selic_meta": "Selic (meta)", "ipca_12m": "IPCA 12 meses", "usdbrl_ptax": "Dólar (PTAX venda)"}


def _point(s: Series) -> dict:
    prev = float(s.values[-2]) if len(s) > 1 else None
    return {
        "valor": s.last_value,
        "variacao": (s.last_value / prev - 1) if prev else None,
        "data": s.last_date.isoformat(),
        "unidade": s.unit,
        "fonte": s.source,
        "coletado_em": s.retrieved_at.isoformat(),
        "atrasado": s.is_stale(),
        "sintetico": s.synthetic,
    }


def build(svc: Services) -> dict:
    mem = svc.memory
    portfolio = mem.portfolio()
    tickers = sorted(set(mem.watchlist()) | {p["ticker"] for p in portfolio})
    keys = [f"macro:{k}" for k in MACRO_KEYS] + [f"price:{t}" for t in tickers]
    data, errors = svc.fetch_many(keys) if keys else ({}, [])
    insufficient: list[str] = []

    macro = []
    for k in MACRO_KEYS:
        s = data.get(f"macro:{k}")
        if s is None:
            insufficient.append(f"{MACRO_LABELS[k]}: indisponível")
            macro.append({"indicador": MACRO_LABELS[k], "disponivel": False})
        else:
            macro.append({"indicador": MACRO_LABELS[k], "disponivel": True, **_point(s)})

    assets = []
    for t in tickers:
        s = data.get(f"price:{t}")
        if s is None:
            insufficient.append(f"{t}: cotação indisponível")
            assets.append({"ticker": t, "disponivel": False})
        else:
            assets.append({"ticker": t, "disponivel": True, **_point(s)})

    valued, val_errors, _ = valued_portfolio(svc)
    errors += val_errors
    total = sum(p["valor_usado"] for p in valued)
    cost = sum(p["custo"] for p in valued)
    positions = [{"ticker": p["ticker"], "valor": p["valor_usado"], "peso": p["peso"], "setor": p["setor"],
                  "resultado_pct": p["resultado_pct"], "base_valor": p["base_valor"]} for p in valued]
    mk = [p for p in valued if p["base_valor"] == "mercado"]
    mk_value, mk_cost = sum(p["valor_mercado"] for p in mk), sum(p["custo"] for p in mk)
    at_cost = [p["ticker"] for p in valued if p["base_valor"] != "mercado"]
    if at_cost:
        insufficient.append(f"Sem cotação, valor a custo: {', '.join(at_cost)}")

    models = mem.models()
    by_status: dict[str, int] = {}
    for m in models:
        by_status[m["status"]] = by_status.get(m["status"], 0) + 1
    production = [f"{m['model_id']}@{m['version']}" for m in models if m["status"] == "producao"]
    if not production:
        insufficient.append("Nenhum modelo preditivo em produção: previsões direcionais não são emitidas.")

    health: dict[str, dict] = {}
    for rec in svc.hub.log:
        h = health.setdefault(rec.provider, {"fonte": rec.provider, "chamadas": 0, "falhas": 0, "ultimo_erro": None})
        h["chamadas"] += 1
        if not rec.ok:
            h["falhas"] += 1
            h["ultimo_erro"] = rec.error
    if svc.news is not None and svc.news.errors:
        health["Notícias (RSS)"] = {"fonte": "Notícias (RSS)", "chamadas": None, "falhas": len(svc.news.errors), "ultimo_erro": svc.news.errors[-1]}

    return {
        "gerado_em": utcnow().isoformat(),
        "modo_demonstracao": svc.demo,
        "macro": macro,
        "ativos": assets,
        "carteira": {
            "valor_total": total,
            "custo_total": cost,
            # resultado só das posições com cotação: posições a custo não têm resultado conhecido
            "resultado_rs": (mk_value - mk_cost) if mk else None,
            "resultado_pct": (mk_value / mk_cost - 1) if mk and mk_cost else None,
            "n_com_cotacao": len(mk),
            "posicoes": positions,
        },
        "alertas": mem.alert_log(10),
        "analises_recentes": mem.analyses(limit=5),
        "modelos": {"por_status": by_status, "em_producao": production},
        "custos_hoje": COSTS.today(),
        "saude_fontes": list(health.values()),
        "dados_insuficientes": insufficient,
        "falhas_coleta": errors,
        "fontes_ativas": {
            "cotacoes": [p.name for p in svc.hub.providers],
            "cvm": svc.cvm is not None,
            "noticias": svc.news.licenses() if svc.news else [],
        },
    }
