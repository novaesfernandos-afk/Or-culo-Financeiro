"""Modo 1 - Monitoramento contínuo.

Ciclos leves (indicadores, preços, notícias) a cada execução. Modelos pesados
só rodam para um ativo quando há mudança material, e no máximo uma vez por
janela de resfriamento (padrão 6 h).
"""
from __future__ import annotations

import time

import numpy as np

from ..core import agent_ids as A
from ..orchestration.pipeline import MACRO_KEYS, news_from_dicts
from ..skills.base import SkillContext, run_skill
from .base import Services

HEAVY_COOLDOWN_S = 6 * 3600
PRICE_Z_MATERIAL = 2.5


def evaluate_user_alert(cond: dict, last: float, prev: float) -> str | None:
    t, v = cond.get("tipo"), cond.get("valor")
    if t == "preco_acima" and last > v:
        return f"preço {last:.2f} acima de {v}"
    if t == "preco_abaixo" and last < v:
        return f"preço {last:.2f} abaixo de {v}"
    if t == "variacao_diaria_pct" and prev and abs(last / prev - 1) * 100 >= v:
        return f"variação diária de {(last / prev - 1):+.1%} (limite {v}%)"
    return None


def run_cycle(svc: Services, params: dict | None = None) -> dict:
    p = params or {}
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    tickers = [t.upper() for t in (p.get("tickers") or svc.memory.watchlist())]
    keys = [f"macro:{k}" for k in MACRO_KEYS] + [f"price:{t}" for t in tickers]
    data, errors = svc.fetch_many(keys)
    alerts: list[dict] = []
    heavy_runs: list[str] = []

    macro = {k.split(":", 1)[1]: v for k, v in data.items() if k.startswith("macro:")}
    macro_out = None
    if macro:
        r = run_skill("brazil_macro_monitor", A.ORQUESTRADOR, ctx, indicators=macro)
        macro_out = r.to_dict()
        if r.ok:
            for a in r.output["alertas"]:
                alerts.append({"tipo": "macro", "mensagem": f"Mudança material: {a}"})
            for i in r.output["indicadores"]:
                if i["atrasado"]:
                    alerts.append({"tipo": "dado_atrasado", "mensagem": f"{i['serie']} atrasado (última data {i['data']})"})

    cooldown = svc.state.setdefault("heavy_last_run", {})
    price_status = []
    for t in tickers:
        s = data.get(f"price:{t}")
        if s is None or len(s) < 62:
            price_status.append({"ticker": t, "status": "sem dados"})
            continue
        r = np.diff(np.log(s.values[-61:]))
        z = float(r[-1] / r[:-1].std(ddof=1)) if r[:-1].std(ddof=1) > 0 else 0.0
        material = abs(z) >= PRICE_Z_MATERIAL
        row = {"ticker": t, "preco": s.last_value, "data": s.last_date.isoformat(), "z_retorno": z, "material": material,
               "atrasado": s.is_stale(), "fonte": s.source}
        if material:
            alerts.append({"tipo": "preco", "mensagem": f"{t}: movimento de {np.expm1(r[-1]):+.1%} (z={z:+.1f})"})
            if time.time() - cooldown.get(t, 0) > HEAVY_COOLDOWN_S:
                vf = run_skill("volatility_forecast", A.RISCO, ctx, prices=s, horizon=21)
                row["previsao_volatilidade"] = vf.output if vf.ok else vf.error
                cooldown[t] = time.time()
                heavy_runs.append(t)
            else:
                row["previsao_volatilidade"] = "resfriamento ativo: modelo pesado não reexecutado"
        for al in svc.memory.alerts():
            if al["ticker"] == t:
                msg = evaluate_user_alert(al["condition"], s.values[-1], s.values[-2])
                if msg:
                    alerts.append({"tipo": "alerta_usuario", "mensagem": f"{t}: {msg}", "alerta_id": al["id"]})
        price_status.append(row)

    news_out = None
    if p.get("noticias"):
        items = news_from_dicts(p["noticias"])
        seen = svc.state.setdefault("news_seen", [])
        r = run_skill("global_news_monitor", A.NOTICIAS, ctx, items=items, seen_fingerprints=seen)
        news_out = r.to_dict()
        if r.ok:
            for g in r.output["grupos"]:
                seen.append(g["fingerprint"])
                if g["novidade"] and g["status"] == "confirmada" and (not g["tickers"] or set(g["tickers"]) & set(tickers)):
                    alerts.append({"tipo": "noticia", "mensagem": f"[{g['status']}] {g['titulo']} ({', '.join(g['fontes'])})"})

    for a in alerts:
        svc.memory.log_alert(a["tipo"], a["mensagem"], a)
    return {
        "modo": "monitoramento",
        "ativos": price_status,
        "macro": macro_out,
        "noticias": news_out,
        "alertas": alerts,
        "modelos_pesados_executados": heavy_runs,
        "falhas_coleta": errors,
        "custo": svc.hub.usage(),
        "politica": f"Modelos pesados só sob mudança material (|z| >= {PRICE_Z_MATERIAL}) e no máximo 1x a cada {HEAVY_COOLDOWN_S // 3600} h por ativo.",
    }


def run(svc: Services, params: dict) -> dict:
    action = params.get("acao", "ciclo")
    if action == "iniciar":
        interval = float(params.get("intervalo_segundos", 900))
        svc.scheduler.every("monitoramento", interval, lambda: svc.state.__setitem__("ultimo_ciclo", run_cycle(svc, params)))
        svc.scheduler.start()
        return {"modo": "monitoramento", "status": "agendado", "intervalo_segundos": interval}
    if action == "parar":
        svc.scheduler.stop()
        return {"modo": "monitoramento", "status": "parado"}
    if action == "status":
        return {"modo": "monitoramento", "agendador": svc.scheduler.status(), "ultimo_ciclo": svc.state.get("ultimo_ciclo"),
                "alertas_recentes": svc.memory.alert_log(50)}
    return run_cycle(svc, params)
