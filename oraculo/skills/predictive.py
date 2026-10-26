"""Skills preditivas (19-24).

Todas as previsões são probabilísticas e acompanhadas de incerteza. Nenhuma
previsão usa informação posterior à data de referência (sem vazamento).
"""
from __future__ import annotations

import numpy as np

from ..core import agent_ids as A
from ..core.types import Series
from ._math import (
    align,
    brier,
    ewma_vol,
    isotonic_apply,
    isotonic_fit,
    logistic_fit,
    logistic_predict,
    ols,
    pct_returns,
    reliability,
    rsi,
    sma,
)
from .base import SkillError, SkillRun, finite, prob_in_unit, skill

FEATURES = ("mom_5", "mom_21", "mom_63", "vol_21", "rsi_14", "dist_mms50")


def direction_features(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de atributos por data (somente dados até a própria data) e máscara de validade.

    Atributos são invariantes à escala de preço: preço baixo não é sinal de oportunidade.
    """
    p = np.asarray(prices, float)
    n = len(p)
    X = np.full((n, len(FEATURES)), np.nan)
    r = np.concatenate([[np.nan], p[1:] / p[:-1] - 1])
    s50 = sma(p, 50)
    rs = rsi(p)
    for i in range(n):
        if i < 63:
            continue
        X[i, 0] = p[i] / p[i - 5] - 1
        X[i, 1] = p[i] / p[i - 21] - 1
        X[i, 2] = p[i] / p[i - 63] - 1
        X[i, 3] = np.nanstd(r[i - 20 : i + 1], ddof=1) * np.sqrt(252)
        X[i, 4] = (rs[i] - 50) / 50 if not np.isnan(rs[i]) else np.nan
        X[i, 5] = p[i] / s50[i] - 1 if not np.isnan(s50[i]) else np.nan
    valid = ~np.isnan(X).any(axis=1)
    return X, valid


def direction_labels(prices: np.ndarray, horizon: int) -> np.ndarray:
    p = np.asarray(prices, float)
    y = np.full(len(p), np.nan)
    y[: len(p) - horizon] = (p[horizon:] > p[: len(p) - horizon]).astype(float)
    return y


class DirectionModel:
    """Regressão logística padronizada sobre atributos de momentum/volatilidade."""

    name = "logit_momentum"

    def __init__(self, horizon: int = 5, l2: float = 5.0) -> None:
        self.horizon = horizon
        self.l2 = l2
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectionModel":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.w = logistic_fit((X - self.mu) / self.sd, y, self.l2)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise SkillError("Modelo não treinado.")
        return logistic_predict(self.w, (X - self.mu) / self.sd)

    def params(self) -> dict:
        return {"horizon": self.horizon, "l2": self.l2, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(), "features": list(FEATURES)}

    @classmethod
    def from_params(cls, params: dict) -> "DirectionModel":
        m = cls(params["horizon"], params.get("l2", 5.0))
        m.mu, m.sd, m.w = (np.array(params[k]) for k in ("mu", "sd", "w"))
        return m


def training_set(prices: np.ndarray, horizon: int, upto: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Amostra de treino cujo rótulo já é conhecido no índice `upto` (sem vazamento)."""
    X, valid = direction_features(prices)
    y = direction_labels(prices, horizon)
    upto = len(prices) - 1 if upto is None else upto
    idx = np.arange(len(prices))
    m = valid & ~np.isnan(y) & (idx + horizon <= upto)
    return X[m], y[m]


@skill(
    id="price_direction_forecast",
    name="Previsão probabilística de direção",
    version="1.1.0",
    category="preditiva",
    objective="Estimar a probabilidade de alta do ativo no horizonte informado, usando modelo versionado.",
    authorized_agents=[A.QUANT],
    inputs={
        "prices": "Series de preços ajustados",
        "horizon": "horizonte em pregões (padrão 5)",
        "model_params": "parâmetros de modelo aprovado na memória de modelos (opcional)",
        "model_version": "identificador da versão aprovada",
        "model_status": "status no registro (produção/experimental)",
    },
    required_inputs=["prices"],
    tools=["DirectionModel", "numpy"],
    procedure=[
        "Calcular atributos invariantes à escala (momentum, volatilidade, RSI, distância da MMS50).",
        "Usar modelo de produção com mapa de calibração (isotônica fora da amostra); na ausência, ajuste local marcado como experimental.",
        "Comparar com a taxa-base histórica de altas.",
    ],
    validation=["Probabilidades em [0, 1]."],
    output_schema={"prob_alta": "float", "taxa_base": "float", "modelo": "str", "status_modelo": "str"},
    failure_handling="Menos de 250 observações válidas => falha.",
    heavy=True,
)
def price_direction_forecast(
    run: SkillRun,
    prices: Series,
    horizon: int = 5,
    model_params: dict | None = None,
    model_version: str | None = None,
    model_status: str | None = None,
) -> dict:
    p = prices.values
    X, valid = direction_features(p)
    if not valid[-1]:
        raise SkillError("Histórico insuficiente para calcular os atributos atuais.")
    Xtr, ytr = training_set(p, horizon)
    if len(ytr) < 250:
        raise SkillError(f"Apenas {len(ytr)} observações rotuladas; mínimo 250.")
    if model_params:
        model = DirectionModel.from_params(model_params)
        if model.horizon != horizon:
            raise SkillError("Horizonte do modelo aprovado difere do solicitado.")
        version, status = model_version or "desconhecida", model_status or "desconhecido"
    else:
        model = DirectionModel(horizon).fit(Xtr, ytr)
        version, status = f"{DirectionModel.name}_h{horizon}/ajuste-local", "experimental"
        run.warn("Modelo experimental ajustado localmente, sem validação fora da amostra nem calibração: uso restrito ao Modo 7.")
    raw = float(model.predict_proba(X[-1:])[0])
    calib = (model_params or {}).get("calibracao")
    prob = isotonic_apply(np.array(calib["limiares"]), np.array(calib["valores"]), raw) if calib else raw
    if not calib:
        run.warn("Probabilidade sem mapa de calibração (regra 4): não pode ser apresentada como probabilidade.", degrade=False)
    return {
        "prob_alta": prob,
        "prob_bruta": raw,
        "calibrada": bool(calib),
        "horizonte_pregoes": horizon,
        "taxa_base": float(ytr.mean()),
        "vantagem_sobre_base": prob - float(ytr.mean()),
        "modelo": version,
        "status_modelo": status,
        "atributos_atuais": dict(zip(FEATURES, X[-1].tolist())),
        "n_treino": int(len(ytr)),
        "data_referencia": prices.last_date.isoformat(),
    }


price_direction_forecast.validator(prob_in_unit("prob_alta", "taxa_base"))


def _fhs(r: np.ndarray, horizon: int, n_sims: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Simulação histórica filtrada usando apenas os retornos `r` (passado). Retorna log-retornos acumulados."""
    sig = ewma_vol(r)
    z = (r / sig)[20:]
    z = (z - z.mean()) / z.std()
    sigma_now = float(np.sqrt(0.94 * sig[-1] ** 2 + 0.06 * r[-1] ** 2))
    return (float(r.mean()) + sigma_now * rng.choice(z, size=(n_sims, horizon))).sum(axis=1), sigma_now


def interval_coverage(r: np.ndarray, horizon: int, max_points: int = 40, min_history: int = 250, n_sims: int = 1000, seed: int = 3) -> dict:
    """Cobertura real das faixas em datas passadas, sem usar informação posterior a cada data."""
    rng = np.random.default_rng(seed)
    ends = list(range(len(r) - horizon, min_history - 1, -horizon))[:max_points]
    hit90 = hit50 = 0
    for t in ends:
        sims, _ = _fhs(r[:t], horizon, n_sims, rng)
        p05, p25, p75, p95 = np.percentile(sims, [5, 25, 75, 95])
        realized = r[t : t + horizon].sum()
        hit90 += p05 <= realized <= p95
        hit50 += p25 <= realized <= p75
    n = len(ends)
    if n < 20:
        return {"n": n, "cobertura_90": None, "cobertura_50": None, "calibrada": False, "motivo": "menos de 20 janelas passadas"}
    c90, c50 = hit90 / n, hit50 / n
    ok = abs(c90 - 0.90) <= 0.07 and abs(c50 - 0.50) <= 0.12
    return {"n": n, "cobertura_90": c90, "cobertura_50": c50, "calibrada": bool(ok),
            "motivo": "cobertura dentro da tolerância" if ok else "cobertura fora da tolerância"}


@skill(
    id="return_distribution_forecast",
    name="Estimativa da distribuição de retornos",
    version="1.1.0",
    category="preditiva",
    objective="Estimar quantis do retorno acumulado no horizonte por simulação histórica filtrada (bootstrap de resíduos padronizados).",
    authorized_agents=[A.QUANT, A.RISCO, A.CARTEIRA],
    inputs={"prices": "Series", "horizon": "pregões (padrão 21)", "n_sims": "simulações (padrão 5000)", "seed": "semente"},
    required_inputs=["prices"],
    tools=["numpy", "EWMA"],
    procedure=[
        "Volatilidade EWMA (λ=0,94); resíduos padronizados z = r/σ.",
        "Simular trajetórias com σ atual e z reamostrados com reposição.",
        "Reportar quantis 5/25/50/75/95 e média.",
        "Calibração (regra 4): em até 40 datas passadas não sobrepostas, refazer a simulação só com dados anteriores "
        "e medir a cobertura real das faixas p05-p95 (nominal 90%) e p25-p75 (nominal 50%).",
        "Faixas só recebem rótulo de probabilidade se a cobertura medida estiver dentro da tolerância (±7 p.p. e ±12 p.p.).",
    ],
    validation=["Quantis monotônicos."],
    output_schema={"quantis": "dict", "media": "float", "calibracao": "dict (cobertura_90, cobertura_50, calibrada)"},
    failure_handling="Menos de 250 retornos => falha.",
    heavy=True,
)
def return_distribution_forecast(run: SkillRun, prices: Series, horizon: int = 21, n_sims: int = 5000, seed: int = 7) -> dict:
    r = np.diff(np.log(prices.values))
    if len(r) < 250:
        raise SkillError("São necessários ao menos 250 retornos.")
    rng = np.random.default_rng(seed)
    sims, sigma_now = _fhs(r, horizon, n_sims, rng)
    ret = np.exp(sims) - 1
    q = np.percentile(ret, [5, 25, 50, 75, 95])
    cal = interval_coverage(r, horizon)
    if not cal["calibrada"]:
        run.warn(f"Faixas sem calibração confirmada ({cal['motivo']}): exibir como faixas, não como probabilidades.", degrade=False)
    return {
        "horizonte_pregoes": horizon,
        "quantis": {"p05": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]},
        "media": float(ret.mean()),
        "fracao_simulacoes_positivas": float((ret > 0).mean()),
        "calibracao": cal,
        "vol_diaria_atual": sigma_now,
        "n_simulacoes": n_sims,
    }


@return_distribution_forecast.validator
def _monotone(out: dict) -> list[str]:
    q = list(out["quantis"].values())
    return [] if all(q[i] <= q[i + 1] for i in range(len(q) - 1)) else ["quantis não monotônicos"]


def garch_fit(r: np.ndarray) -> dict:
    """GARCH(1,1) com variance targeting, ajustado por busca em grade da verossimilhança."""
    r = r - r.mean()
    var_l = r.var()
    A_, B_ = np.meshgrid(np.arange(0.01, 0.30, 0.01), np.arange(0.50, 0.99, 0.01))
    ok = (A_ + B_) < 0.999
    a_grid, b_grid = A_[ok], B_[ok]
    w_grid = var_l * (1 - a_grid - b_grid)
    h = np.full(len(a_grid), var_l)
    ll = np.zeros(len(a_grid))
    for t in range(len(r)):  # todas as combinações da grade em paralelo
        ll += np.log(h) + r[t] ** 2 / h
        h = w_grid + a_grid * r[t] ** 2 + b_grid * h
    k = int(np.argmin(ll))
    best = (-0.5 * ll[k], float(a_grid[k]), float(b_grid[k]))
    _, a, b = best
    w = var_l * (1 - a - b)
    h = var_l
    for t in range(len(r)):
        h = w + a * r[t] ** 2 + b * h
    return {"alpha": float(a), "beta": float(b), "omega": float(w), "var_longo_prazo": float(var_l), "var_proxima": float(h), "loglik": float(best[0])}


@skill(
    id="volatility_forecast",
    name="Previsão de volatilidade",
    version="1.0.0",
    category="preditiva",
    objective="Prever a volatilidade por EWMA e GARCH(1,1) para o horizonte informado.",
    authorized_agents=[A.QUANT, A.RISCO, A.TECNICO, A.CARTEIRA],
    inputs={"prices": "Series", "horizon": "pregões (padrão 21)", "max_obs": "máximo de observações no ajuste GARCH (padrão 750)"},
    required_inputs=["prices"],
    tools=["numpy"],
    procedure=[
        "Vol realizada 21d e EWMA (λ=0,94).",
        "GARCH(1,1) com variance targeting; previsão média no horizonte pela reversão à média.",
        "Anualizar por √252.",
    ],
    validation=["Volatilidades positivas e finitas.", "α+β < 1."],
    output_schema={"vol_realizada_21d": "float", "vol_ewma": "float", "vol_garch_horizonte": "float", "garch": "dict"},
    failure_handling="Menos de 250 retornos => falha.",
    heavy=True,
)
def volatility_forecast(run: SkillRun, prices: Series, horizon: int = 21, max_obs: int = 750) -> dict:
    r = np.diff(np.log(prices.values))[-max_obs:]
    if len(r) < 250:
        raise SkillError("São necessários ao menos 250 retornos.")
    g = garch_fit(r)
    persist = g["alpha"] + g["beta"]
    path = [g["var_longo_prazo"] + persist ** (k) * (g["var_proxima"] - g["var_longo_prazo"]) for k in range(horizon)]
    ew = ewma_vol(r)
    ann = np.sqrt(252)
    return {
        "horizonte_pregoes": horizon,
        "vol_realizada_21d": float(r[-21:].std(ddof=1) * ann),
        "vol_ewma": float(np.sqrt(0.94 * ew[-1] ** 2 + 0.06 * r[-1] ** 2) * ann),
        "vol_garch_horizonte": float(np.sqrt(np.mean(path)) * ann),
        "vol_longo_prazo": float(np.sqrt(g["var_longo_prazo"]) * ann),
        "persistencia": persist,
        "meia_vida_choques_pregoes": float(np.log(0.5) / np.log(persist)) if 0 < persist < 1 else None,
        "garch": g,
    }


volatility_forecast.validator(finite("vol_ewma", "vol_garch_horizonte"))


@skill(
    id="scenario_simulation",
    name="Simulação de cenários",
    version="1.0.0",
    category="preditiva",
    objective="Estimar o efeito condicional de choques (Selic, petróleo, câmbio...) sobre posições, explicitando hipóteses e incertezas.",
    authorized_agents=[A.QUANT, A.CARTEIRA, A.RISCO, A.ECON_BRASIL, A.CIIO],
    inputs={
        "positions": "dict[ticker, valor em R$]",
        "scenario": "dict[fator, choque]; choque em % (preços) ou p.p. (taxas), ou {'alvo': x, 'atual': y}",
        "stock_prices": "dict[ticker, Series] para estimar sensibilidades",
        "factor_series": "dict[fator, Series]",
        "factor_kind": "dict[fator, 'retorno'|'nivel'] (taxas: 'nivel')",
        "n_sims": "simulações para incerteza dos betas (padrão 4000)",
    },
    required_inputs=["positions", "scenario", "stock_prices", "factor_series"],
    tools=["numpy (OLS multivariada)"],
    procedure=[
        "Converter alvos de nível em choques (ex.: dólar a R$ 6,50 a partir da cotação atual).",
        "Estimar betas semanais de cada ação aos fatores do cenário (OLS multivariada, 5 pregões).",
        "Impacto pontual = Σ beta x choque; intervalo por simulação dos betas ~ N(beta, erro padrão).",
        "Listar hipóteses e limitações.",
    ],
    validation=["Choques finitos.", "Ao menos uma posição com sensibilidade estimada."],
    output_schema={"posicoes": "lista", "carteira": "dict", "hipoteses": "list", "limitacoes": "list"},
    failure_handling="Posições sem dados suficientes ficam fora e são listadas.",
    heavy=True,
)
def scenario_simulation(
    run: SkillRun,
    positions: dict,
    scenario: dict,
    stock_prices: dict,
    factor_series: dict,
    factor_kind: dict | None = None,
    n_sims: int = 4000,
    seed: int = 11,
) -> dict:
    kinds = factor_kind or {}
    shocks: dict[str, float] = {}
    for f, sh in scenario.items():
        if f not in factor_series:
            raise SkillError(f"Fator '{f}' sem série histórica para estimar sensibilidade.")
        kind = kinds.get(f, "retorno")
        if isinstance(sh, dict):
            cur = sh.get("atual", factor_series[f].last_value)
            tgt = sh["alvo"]
            shocks[f] = (tgt / cur - 1) * 100 if kind == "retorno" else (tgt - cur)
        else:
            shocks[f] = float(sh)
    factors = list(shocks)
    rng = np.random.default_rng(seed)
    rows, skipped, sims_total = [], [], np.zeros(n_sims)
    out_of_range: dict[str, float] = {}
    total_value = float(sum(positions.values()))
    step = 5  # retornos semanais reduzem ruído de assincronia
    for t, value in positions.items():
        s = stock_prices.get(t)
        if s is None:
            skipped.append(f"{t}: sem série de preços")
            continue
        _, arrs = align(s, *[factor_series[f] for f in factors])
        if len(arrs[0]) < 60 * step // 2:
            skipped.append(f"{t}: histórico comum insuficiente")
            continue
        sampled = [a[::step] for a in arrs]
        y = pct_returns(sampled[0]) * 100
        X = []
        for f, a in zip(factors, sampled[1:]):
            X.append(pct_returns(a) * 100 if kinds.get(f, "retorno") == "retorno" else np.diff(a))
        X = np.column_stack(X)
        try:
            res = ols(y, X)
        except SkillError as exc:
            skipped.append(f"{t}: {exc}")
            continue
        b, se = res.coef[1:], res.stderr[1:]
        sh = np.array([shocks[f] for f in factors])
        for j, f in enumerate(factors):
            max_move = float(np.abs(X[:, j]).max())
            if max_move > 0 and abs(shocks[f]) > 2 * max_move and f not in out_of_range:
                out_of_range[f] = abs(shocks[f]) / max_move
        point = float(b @ sh)
        draws = rng.normal(b, se, size=(n_sims, len(b))) @ sh
        pnl = value * draws / 100
        sims_total += pnl
        rows.append(
            {
                "ticker": t,
                "valor": value,
                "betas": dict(zip(factors, b.tolist())),
                "erros_padrao": dict(zip(factors, se.tolist())),
                "r2": res.r2,
                "impacto_pct": point,
                "impacto_rs": value * point / 100,
                "intervalo_90_rs": [float(np.percentile(pnl, 5)), float(np.percentile(pnl, 95))],
            }
        )
    if not rows:
        raise SkillError("Nenhuma posição com sensibilidade estimável. " + "; ".join(skipped))
    for s in skipped:
        run.warn(s)
    for f, k in out_of_range.items():
        run.warn(
            f"FORA DO HISTÓRICO: choque em '{f}' é {k:.1f}x a maior variação semanal observada; "
            "extrapolação linear de alta incerteza - o intervalo não captura esse risco de modelo."
        )
    pt = sum(r["impacto_rs"] for r in rows)
    return {
        "choques_aplicados": shocks,
        "posicoes": rows,
        "fora_da_simulacao": skipped,
        "choques_fora_do_historico": out_of_range,
        "carteira": {
            "valor_total": total_value,
            "impacto_rs": pt,
            "impacto_pct": pt / total_value * 100 if total_value else None,
            "intervalo_90_rs": [float(np.percentile(sims_total, 5)), float(np.percentile(sims_total, 95))],
        },
        "hipoteses": [
            "Relação linear entre fatores e ações, estimada com retornos semanais históricos.",
            "Choques aplicados de forma instantânea; demais fatores constantes (ceteris paribus).",
            "Taxas de juros tratadas por variação em p.p.; preços por variação percentual.",
        ],
        "limitacoes": [
            "Choques grandes podem estar fora do intervalo histórico; efeitos não lineares não são capturados.",
            "Intervalo considera apenas incerteza dos betas, não o risco idiossincrático nem mudanças de correlação.",
            "Correlações históricas podem não se repetir em crises.",
        ],
    }


@skill(
    id="model_ensemble",
    name="Combinação de modelos",
    version="1.0.0",
    category="preditiva",
    objective="Combinar previsões probabilísticas ponderando pelo desempenho histórico (Brier) e medir a discordância entre modelos.",
    authorized_agents=[A.QUANT, A.CIIO],
    inputs={"forecasts": "lista de dict: modelo, prob, brier_historico (opcional)"},
    required_inputs=["forecasts"],
    tools=["numpy"],
    procedure=[
        "Peso ∝ 1/Brier histórico (pesos iguais se ausente).",
        "Probabilidade combinada = média ponderada.",
        "Discordância = amplitude e desvio padrão das probabilidades; alerta se amplitude > 0,25.",
    ],
    validation=["Probabilidade combinada em [0, 1]."],
    output_schema={"prob_combinada": "float", "pesos": "dict", "amplitude": "float", "discordancia_alta": "bool"},
    failure_handling="Previsões inválidas são descartadas; falha se nenhuma restar.",
)
def model_ensemble(run: SkillRun, forecasts: list[dict]) -> dict:
    valid = [f for f in forecasts if f.get("prob") is not None and 0 <= f["prob"] <= 1]
    if len(valid) < len(forecasts):
        run.warn(f"{len(forecasts) - len(valid)} previsão(ões) inválida(s) descartada(s).")
    if not valid:
        raise SkillError("Nenhuma previsão válida.")
    w = np.array([1 / f["brier_historico"] if f.get("brier_historico") else 1.0 for f in valid])
    if not all(f.get("brier_historico") for f in valid):
        w = np.ones(len(valid))
    w = w / w.sum()
    p = np.array([f["prob"] for f in valid])
    amp = float(p.max() - p.min())
    return {
        "prob_combinada": float(w @ p),
        "pesos": {f["modelo"]: float(x) for f, x in zip(valid, w)},
        "amplitude": amp,
        "desvio": float(p.std()),
        "discordancia_alta": amp > 0.25,
    }


model_ensemble.validator(prob_in_unit("prob_combinada"))


@skill(
    id="prediction_calibration",
    name="Calibração de probabilidades",
    version="1.0.0",
    category="preditiva",
    objective="Medir a calibração histórica e corrigir novas probabilidades (isotônica ou Platt).",
    authorized_agents=[A.QUANT, A.AUDITOR],
    inputs={"probs": "probabilidades históricas", "outcomes": "resultados 0/1", "new_prob": "probabilidade a calibrar (opcional)"},
    required_inputs=["probs", "outcomes"],
    tools=["numpy", "PAV isotônico", "Platt"],
    procedure=[
        "Tabela de confiabilidade (10 faixas), Brier e ECE.",
        "n ≥ 200: regressão isotônica; senão: Platt (logística sobre o logit).",
        "Aplicar o mapa à nova probabilidade.",
    ],
    validation=["Probabilidades em [0, 1].", "Resultados binários."],
    output_schema={"brier": "float", "ece": "float", "tabela": "list", "prob_calibrada": "float | None"},
    failure_handling="Menos de 30 pares => falha (calibração não confiável).",
)
def prediction_calibration(run: SkillRun, probs: list[float], outcomes: list[int], new_prob: float | None = None) -> dict:
    p, y = np.asarray(probs, float), np.asarray(outcomes, float)
    if len(p) != len(y):
        raise SkillError("probs e outcomes com tamanhos diferentes.")
    if len(p) < 30:
        raise SkillError("São necessários ao menos 30 pares para calibrar.")
    if ((p < 0) | (p > 1)).any() or not set(np.unique(y)) <= {0.0, 1.0}:
        raise SkillError("Probabilidades fora de [0,1] ou resultados não binários.")
    table, ece = reliability(p, y)
    out = {"n": len(p), "brier": brier(p, y), "brier_referencia": brier(np.full(len(y), y.mean()), y), "ece": ece, "tabela": table}
    if len(p) >= 200:
        th, vals = isotonic_fit(p, y)
        method = "isotonica"
        mapper = lambda x: isotonic_apply(th, vals, x)  # noqa: E731
    else:
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        w = logistic_fit(lg[:, None], y, l2=0.1)
        method = "platt"
        mapper = lambda x: float(logistic_predict(w, np.array([[np.log(max(min(x, 1 - 1e-6), 1e-6) / (1 - max(min(x, 1 - 1e-6), 1e-6)))]]))[0])  # noqa: E731
    out["metodo"] = method
    out["prob_calibrada"] = mapper(new_prob) if new_prob is not None else None
    return out


prediction_calibration.validator(prob_in_unit("prob_calibrada", "brier"))
