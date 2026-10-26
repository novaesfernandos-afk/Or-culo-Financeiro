"""Modo 7 - Pesquisa e desenvolvimento (ambiente separado da produção).

Ciclo de vida de um modelo:
experimental (A11 cria) -> validado/rejeitado (A15 certifica por critérios fixos)
-> aprovado (responsável humano, diferente do autor) -> produção (responsável humano).
Nenhum modelo experimental substitui o de produção sem validação e aprovação.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from ..core import agent_ids as A
from ..core.types import utcnow
from ..orchestration.pipeline import DIRECTION_MODEL_ID
from ..skills._math import isotonic_fit, sma
from ..skills.base import SkillContext, run_skill
from ..skills.predictive import DirectionModel, direction_features, direction_labels
from .base import Services


def build_panel(series: dict, horizon: int) -> tuple[np.ndarray, np.ndarray, list]:
    rows = []
    for t, s in series.items():
        X, valid = direction_features(s.values)
        y = direction_labels(s.values, horizon)
        for i in np.where(valid)[0]:
            rows.append((s.dates[i], t, X[i], y[i]))
    rows.sort(key=lambda r: (r[0], r[1]))
    if not rows:
        return np.empty((0, 0)), np.empty(0), []
    return np.array([r[2] for r in rows]), np.array([r[3] for r in rows]), [(r[0], r[1]) for r in rows]


def train_validate(svc: Services, params: dict) -> dict:
    tickers = [t.upper() for t in params["tickers"]]
    h = int(params.get("horizonte", 5))
    l2 = float(params.get("l2", 5.0))
    data, errors = svc.fetch_many([f"price:{t}" for t in tickers])
    series = {t: data[f"price:{t}"] for t in tickers if f"price:{t}" in data}
    if not series:
        raise ValueError("Nenhuma série de preços disponível para treinar.")
    synthetic = any(s.synthetic for s in series.values())
    X, y, idx = build_panel(series, h)
    n = len(series)
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    wf = run_skill("walk_forward_validation", A.QUANT, ctx, X=X, y=y, model_factory=lambda: DirectionModel(h, l2),
                   horizon=h * n, min_train=int(params.get("min_treino", 252)) * n, step=63 * n)
    if not wf.ok:
        return {"modo": "pesquisa", "erro": wf.error, "falhas_coleta": errors}
    lab = ~np.isnan(y)
    model = DirectionModel(h, l2).fit(X[lab], y[lab])
    oos = wf.output["previsoes_oos"]
    th, vals = isotonic_fit(np.array([o["prob"] for o in oos]), np.array([o["y"] for o in oos]))
    model_id = DIRECTION_MODEL_ID.format(h=h)
    p = model.params()
    p["tickers"] = sorted(series)
    p["dados_sinteticos"] = synthetic
    p["calibracao"] = {"metodo": "isotonica", "limiares": th.tolist(), "valores": vals.tolist(), "n_oos": len(oos)}
    version = utcnow().strftime("%Y%m%d%H%M%S") + "-" + hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:6]
    svc.memory.register_model(model_id, version, A.QUANT, p, f"Logística L2={l2}, h={h}, {n} ativo(s)")
    validation = {k: v for k, v in wf.output.items() if k != "previsoes_oos"}
    if synthetic:
        validation["vazamento_detectado"] = validation["vazamento_detectado"]
    passed = svc.memory.attach_validation(model_id, version, A.AUDITOR, validation)
    if synthetic and passed:
        svc.memory.reject_model(model_id, version, A.AUDITOR, "Treinado com dados sintéticos: não elegível para produção.")
        passed = False
    out = {
        "modo": "pesquisa",
        "ambiente": "experimental",
        "modelo": f"{model_id}@{version}",
        "status": svc.memory.get_model(model_id, version)["status"],
        "certificado_pelo_auditor": passed,
        "validacao": validation["agregado"],
        "blocos": validation["blocos"],
        "proximo_passo": "Aprovação humana (acao='aprovar', responsavel='user:<nome>')" if passed else "Modelo não elegível para produção.",
        "dados_sinteticos": synthetic,
        "falhas_coleta": errors,
    }
    svc.memory.save_analysis("pesquisa", ",".join(sorted(series)), {"tickers": tickers, "horizonte": h, "l2": l2},
                             {k: v for k, v in out.items() if k != "blocos"}, [A.QUANT, A.AUDITOR], [], out["status"], environment="experimental")
    return out


def compare_strategies(svc: Services, params: dict) -> dict:
    t = params["ticker"].upper()
    cost = float(params.get("custo_bps", 10))
    s = svc.hub.get(f"price:{t}")
    p = s.values
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    fast, slow = int(params.get("mm_rapida", 50)), int(params.get("mm_lenta", 200))
    f, sl = sma(p, fast), sma(p, slow)
    strategies = {
        "comprar_e_manter": np.ones(len(p)),
        f"cruzamento_mm_{fast}_{slow}": np.where(np.isnan(sl), 0.0, (f > sl).astype(float)),
        "momentum_63d": np.concatenate([np.zeros(63), (p[63:] > p[:-63]).astype(float)]),
    }
    prod = svc.memory.production_model(DIRECTION_MODEL_ID.format(h=int(params.get("horizonte", 5))))
    if prod:
        m = DirectionModel.from_params(prod["params"])
        X, valid = direction_features(p)
        pr = np.full(len(p), 0.5)
        pr[valid] = m.predict_proba(X[valid])
        strategies[f"modelo_producao_{prod['version']}"] = (pr > 0.5).astype(float)
    out = {}
    for name, pos in strategies.items():
        r = run_skill("backtesting_engine", A.QUANT, ctx, prices=s, positions=pos.tolist(), cost_bps=cost)
        out[name] = r.output if r.ok else {"erro": r.error}
    return {
        "modo": "pesquisa",
        "ticker": t,
        "estrategias": out,
        "ressalvas": ["Backtest dentro da amostra para regras simples; o modelo de produção pode ter sido treinado neste mesmo período.",
                      "Custos aproximados; não inclui impostos."],
        "dados_sinteticos": s.synthetic,
    }


def run(svc: Services, params: dict) -> dict:
    action = params.get("acao", "listar")
    mem = svc.memory
    if action == "treinar_validar":
        return train_validate(svc, params)
    if action == "comparar_estrategias":
        return compare_strategies(svc, params)
    if action == "aprovar":
        mem.approve_model(params["modelo_id"], params["versao"], params["responsavel"], params.get("nota", ""))
    elif action == "promover":
        mem.promote_model(params["modelo_id"], params["versao"], params["responsavel"])
    elif action == "rejeitar":
        mem.reject_model(params["modelo_id"], params["versao"], params["responsavel"], params.get("nota", ""))
    elif action == "historico":
        return {"modo": "pesquisa", "historico": mem.model_history(params["modelo_id"])}
    elif action != "listar":
        raise ValueError(f"Ação desconhecida: {action}")
    return {"modo": "pesquisa", "modelos": mem.models()}
