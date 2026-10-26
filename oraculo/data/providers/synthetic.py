"""Gerador de dados SINTÉTICOS para testes e demonstração.

Toda série produzida aqui é marcada `synthetic=True`. Relatórios que usam
esses dados exibem o aviso "DADO SINTÉTICO" e a revisão de compliance os
sinaliza. Nunca devem ser apresentados como cotações ou indicadores reais.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np

from ...core.types import Frequency, Series, utcnow
from .base import DataProvider


def business_days(end: date, n: int) -> list[date]:
    out: list[date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out[::-1]


def _seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


class SyntheticProvider(DataProvider):
    name = "SINTÉTICO (gerador interno)"
    prefixes = ("price:", "volume:", "high:", "low:", "macro:", "factor:")
    cost_per_call = 0.0

    def __init__(self, n_days: int = 800, end: date | None = None) -> None:
        self.n_days = n_days
        self.end = end or date.today()

    def fetch(self, key: str, start: date | None = None, end: date | None = None) -> Series:
        rng = np.random.default_rng(_seed(key))
        kind, name = key.split(":", 1)
        dates = business_days(self.end, self.n_days)
        if kind in ("price", "high", "low"):
            base_rng = np.random.default_rng(_seed("price:" + name))
            mu, sigma = base_rng.uniform(-0.0002, 0.0006), base_rng.uniform(0.012, 0.03)
            r = base_rng.normal(mu, sigma, self.n_days)
            values = 20 * np.exp(np.cumsum(r))
            if kind == "high":
                values = values * (1 + np.abs(rng.normal(0, 0.006, self.n_days)))
            elif kind == "low":
                values = values * (1 - np.abs(rng.normal(0, 0.006, self.n_days)))
        elif kind == "volume":
            values = rng.lognormal(15, 0.4, self.n_days)
        elif name.startswith("selic"):
            # degraus de 0,25 p.p. a cada ~45 dias úteis, como decisões de política monetária
            steps = np.zeros(self.n_days)
            steps[::45] = rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5], size=len(steps[::45]))
            values = np.clip(10 + np.cumsum(steps), 2, 20)
        elif name.startswith("ipca"):
            values = np.clip(4.5 + np.cumsum(rng.normal(0, 0.03, self.n_days)), 0, 15)
        elif name.startswith("usdbrl"):
            values = 5.0 * np.exp(np.cumsum(rng.normal(0, 0.008, self.n_days)))
        else:
            values = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, self.n_days)))
        s = Series(
            name=key,
            dates=dates,
            values=values,
            source=self.name,
            retrieved_at=utcnow(),
            frequency=Frequency.DAILY,
            synthetic=True,
        )
        if start or end:
            s = s.until(end or self.end)
            if start:
                keep = [i for i, d in enumerate(s.dates) if d >= start]
                s.dates = [s.dates[i] for i in keep]
                s.values = s.values[keep]
        return s
