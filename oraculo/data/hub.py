"""Hub de dados: roteia chaves para provedores, faz cache e contabiliza custos."""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date

from ..core.costs import COSTS, BudgetExceeded
from ..core.types import Series
from .providers.base import DataProvider, ProviderError
from .providers.bcb import BCBProvider
from .providers.brapi import BrapiProvider
from .providers.csv_provider import CSVProvider
from .providers.synthetic import SyntheticProvider


@dataclass
class FetchRecord:
    key: str
    provider: str
    ok: bool
    seconds: float
    cost: float
    error: str | None = None


@dataclass
class DataHub:
    providers: list[DataProvider] = field(default_factory=list)
    cache_ttl_seconds: float = 900.0
    log: list[FetchRecord] = field(default_factory=list)
    _cache: dict[str, tuple[float, Series]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, provider: DataProvider) -> None:
        self.providers.append(provider)

    def get(self, key: str, start: date | None = None, end: date | None = None) -> Series:
        cache_key = f"{key}|{start}|{end}"
        with self._lock:
            hit = self._cache.get(cache_key)
            if hit and time.time() - hit[0] < self.cache_ttl_seconds:
                return hit[1]
        errors = []
        all_permanent = True
        for p in self.providers:
            if not p.supports(key):
                continue
            t0 = time.time()
            try:
                COSTS.check(p.name)
            except BudgetExceeded as exc:
                errors.append(f"{p.name}: {exc}")
                continue
            try:
                s = p.fetch(key, start, end)
            except ProviderError as exc:
                COSTS.record("dados", p.name, time.time() - t0, p.cost_per_call, ok=False)
                self._record(FetchRecord(key, p.name, False, time.time() - t0, p.cost_per_call, str(exc)))
                errors.append(f"{p.name}: {exc}")
                all_permanent = all_permanent and exc.permanent
                continue
            self._record(FetchRecord(key, p.name, True, time.time() - t0, p.cost_per_call))
            COSTS.record("dados", p.name, time.time() - t0, p.cost_per_call)
            with self._lock:
                self._cache[cache_key] = (time.time(), s)
            return s
        detail = "; ".join(errors) if errors else "nenhum provedor configurado para esta chave"
        raise ProviderError(f"Dado indisponível para '{key}' ({detail}). Nenhum valor será estimado no lugar.",
                            permanent=not errors or all_permanent)

    def try_get(self, key: str, start: date | None = None, end: date | None = None) -> Series | None:
        try:
            return self.get(key, start, end)
        except ProviderError:
            return None

    def _record(self, rec: FetchRecord) -> None:
        with self._lock:
            self.log.append(rec)

    def usage(self) -> dict:
        with self._lock:
            return {
                "calls": len(self.log),
                "failures": sum(1 for r in self.log if not r.ok),
                "cost_units": round(sum(r.cost for r in self.log), 2),
                "seconds": round(sum(r.seconds for r in self.log), 3),
            }


def default_hub(demo: bool | None = None) -> DataHub:
    """Hub com fontes reais. Em modo demonstração (ORACULO_DEMO=1) usa somente dados sintéticos."""
    demo = demo if demo is not None else os.environ.get("ORACULO_DEMO") == "1"
    hub = DataHub()
    if demo:
        hub.add(SyntheticProvider())
        return hub
    hub.add(CSVProvider())
    hub.add(BCBProvider())
    hub.add(BrapiProvider())
    return hub
