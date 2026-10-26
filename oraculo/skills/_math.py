"""Rotinas numéricas compartilhadas (apenas numpy)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import erf, sqrt

import numpy as np

from ..core.types import Series
from .base import SkillError


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def align(*series: Series) -> tuple[list[date], list[np.ndarray]]:
    """Alinha séries pelas datas em comum."""
    common = set(series[0].dates)
    for s in series[1:]:
        common &= set(s.dates)
    dates = sorted(common)
    out = []
    for s in series:
        idx = {d: i for i, d in enumerate(s.dates)}
        out.append(np.array([s.values[idx[d]] for d in dates], dtype=float))
    return dates, out


def pct_returns(v: np.ndarray) -> np.ndarray:
    return v[1:] / v[:-1] - 1.0


def log_returns(v: np.ndarray) -> np.ndarray:
    return np.diff(np.log(v))


@dataclass
class OLSResult:
    coef: np.ndarray
    stderr: np.ndarray
    r2: float
    n: int
    resid_std: float


def ols(y: np.ndarray, X: np.ndarray, add_const: bool = True) -> OLSResult:
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if add_const:
        X = np.column_stack([np.ones(len(X)), X])
    n, k = X.shape
    if n <= k + 2:
        raise SkillError(f"Amostra insuficiente para regressão (n={n}, k={k}).")
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = n - k
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError as exc:
        raise SkillError("Matriz singular na regressão.") from exc
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    return OLSResult(coef, np.sqrt(np.diag(cov)), r2, n, float(np.sqrt(s2)))


def sma(v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        c = np.cumsum(np.insert(v, 0, 0.0))
        out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def ema(v: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(len(v))
    a = 2 / (n + 1)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1]
    return out


def rsi(v: np.ndarray, n: int = 14) -> np.ndarray:
    d = np.diff(v)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    out = np.full(len(v), np.nan)
    if len(d) < n:
        return out
    au, ad = up[:n].mean(), dn[:n].mean()
    for i in range(n, len(d) + 1):
        if i > n:
            au = (au * (n - 1) + up[i - 1]) / n
            ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100.0 if ad == 0 else 100 - 100 / (1 + au / ad)
    return out


def max_drawdown(values: np.ndarray) -> float:
    peak = np.maximum.accumulate(values)
    return float((values / peak - 1).min())


def ewma_vol(r: np.ndarray, lam: float = 0.94) -> np.ndarray:
    var = np.empty(len(r))
    var[0] = r[: min(20, len(r))].var() if len(r) > 1 else r[0] ** 2
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i - 1] ** 2
    return np.sqrt(var)


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Regressão logística com L2 via Newton-Raphson (intercepto não penalizado)."""
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    reg = np.eye(X.shape[1]) * l2
    reg[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + reg @ w
        H = (X * (p * (1 - p))[:, None]).T @ X + reg
        step = np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logistic_predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[list[dict], float]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.sum() == 0:
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        rows.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()), "prob_media": conf, "freq_observada": acc})
        ece += m.sum() / len(p) * abs(conf - acc)
    return rows, float(ece)


def isotonic_fit(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators. Retorna (limiares de p, valores calibrados)."""
    order = np.argsort(p)
    xs, ys = np.asarray(p, float)[order], np.asarray(y, float)[order]
    blocks = [[ys[i], 1.0, xs[i], xs[i]] for i in range(len(ys))]  # média, peso, x_min, x_max
    merged: list[list[float]] = []
    for b in blocks:
        merged.append(b)
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            b2 = merged.pop()
            b1 = merged.pop()
            w = b1[1] + b2[1]
            merged.append([(b1[0] * b1[1] + b2[0] * b2[1]) / w, w, b1[2], b2[3]])
    thresholds = np.array([(m[2] + m[3]) / 2 for m in merged])
    values = np.array([m[0] for m in merged])
    return thresholds, values


def isotonic_apply(thresholds: np.ndarray, values: np.ndarray, p: float) -> float:
    return float(np.interp(p, thresholds, values))
