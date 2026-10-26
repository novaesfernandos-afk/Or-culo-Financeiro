"""Infraestrutura comum dos modos de trabalho."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.types import Series
from ..data.hub import DataHub, default_hub
from ..memory.store import MemoryStore
from ..orchestration.orchestrator import Orchestrator, Scheduler, Task
from ..skills._math import align

# Universo padrão (configurável). Sem BRAPI_TOKEN, a brapi pode limitar alguns ativos.
DEFAULT_UNIVERSE = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3",
    "SUZB3", "PRIO3", "RADL3", "EQTL3", "GGBR4", "LREN3", "MGLU3", "RENT3",
]

# Fatores para simulação de cenários: chave no DataHub e natureza do choque.
FACTORS: dict[str, dict[str, str]] = {
    "selic": {"key": "macro:selic_meta", "kind": "nivel", "unit": "p.p."},
    "dolar": {"key": "macro:usdbrl_ptax", "kind": "retorno", "unit": "%"},
    "ibovespa": {"key": "price:^BVSP", "kind": "retorno", "unit": "%"},
    "petroleo": {"key": "factor:petroleo", "kind": "retorno", "unit": "%"},
    "minerio": {"key": "factor:minerio", "kind": "retorno", "unit": "%"},
}


@dataclass
class Services:
    hub: DataHub
    memory: MemoryStore
    scheduler: Scheduler = field(default_factory=Scheduler)
    demo: bool = False
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, db_path: str | None = None, demo: bool | None = None) -> "Services":
        demo = demo if demo is not None else os.environ.get("ORACULO_DEMO") == "1"
        db = db_path or os.environ.get("ORACULO_DB", "data/oraculo.db")
        return cls(hub=default_hub(demo), memory=MemoryStore(db if not demo else ":memory:"), demo=demo)

    def fetch_many(self, keys: list[str], retries: int = 1) -> tuple[dict[str, Series], list[str]]:
        orch = Orchestrator(hub=self.hub)
        res = orch.run([Task(k, "A02", (lambda key: lambda _d: self.hub.get(key))(k), retries=retries, optional=True) for k in keys])
        errors = [f"{r['id']}: {r['error']}" for r in orch.log() if r["status"] != "ok"]
        return {k: v for k, v in res.items() if v is not None}, errors


@dataclass(frozen=True)
class ModeSpec:
    id: int
    slug: str
    name: str
    objective: str
    agents: tuple[str, ...]
    run: Callable[[Services, dict], dict]

    def to_dict(self) -> dict:
        return {"id": self.id, "slug": self.slug, "name": self.name, "objective": self.objective, "agents": list(self.agents)}


def adv_brl(prices: Series, volume: Series, n: int = 21) -> float:
    _, (p, v) = align(prices, volume)
    return float((p[-n:] * v[-n:]).mean()) if len(p) else 0.0


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if m.sum() >= 2 and x[m].std() > 0:
        out[m] = (x[m] - x[m].mean()) / x[m].std()
    elif m.sum() >= 1:
        out[m] = 0.0
    return out
