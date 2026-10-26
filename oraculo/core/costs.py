"""Monitoramento de custos de APIs e de processamento (regra 11) e limites de acesso às fontes (regra 9).

- Cada chamada a provedor de dados e cada execução de skill é contabilizada por dia.
- Orçamentos diários por provedor bloqueiam novas chamadas quando excedidos (sem estimar dados no lugar).
- Intervalo mínimo entre requisições ao mesmo servidor evita sobrecarregar as fontes.

Orçamentos: ORACULO_ORCAMENTO='{"brapi.dev": 1000}' (chamadas/dia). O Oráculo não usa modelos de
linguagem; o custo de "inferência" é o tempo de CPU das skills e modelos.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

DEFAULT_BUDGETS = {"brapi.dev": 1000}
MIN_INTERVAL_S = {"brapi.dev": 0.3, "api.bcb.gov.br": 0.2, "dados.cvm.gov.br": 1.0, "news.google.com": 1.0}
DEFAULT_INTERVAL_S = 0.2


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class CostMonitor:
    budgets: dict[str, float] = field(default_factory=lambda: {**DEFAULT_BUDGETS, **json.loads(os.environ.get("ORACULO_ORCAMENTO", "{}"))})
    memory: object | None = None  # MemoryStore, para persistir o consumo
    _day: str = field(default_factory=lambda: date.today().isoformat())
    _usage: dict[tuple[str, str], dict] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _roll(self) -> None:
        today = date.today().isoformat()
        if today != self._day:
            self._day, self._usage = today, {}

    def check(self, provider: str) -> None:
        with self._lock:
            self._roll()
            limit = self.budgets.get(provider)
            used = self._usage.get(("dados", provider), {}).get("chamadas", 0)
            if limit is not None and used >= limit:
                raise BudgetExceeded(f"Orçamento diário de {provider} esgotado ({used}/{limit:g} chamadas). Nenhum dado será estimado no lugar.")

    def record(self, kind: str, name: str, seconds: float, units: float = 0.0, ok: bool = True) -> None:
        with self._lock:
            self._roll()
            u = self._usage.setdefault((kind, name), {"chamadas": 0, "falhas": 0, "unidades": 0.0, "segundos": 0.0})
            u["chamadas"] += 1
            u["falhas"] += 0 if ok else 1
            u["unidades"] += units
            u["segundos"] += seconds
        if self.memory is not None:
            self.memory.add_usage(self._day, kind, name, seconds, units, ok)

    def today(self) -> dict:
        with self._lock:
            self._roll()
            rows = [{"tipo": k, "nome": n, **v, "orcamento": self.budgets.get(n) if k == "dados" else None} for (k, n), v in self._usage.items()]
        rows.sort(key=lambda r: (r["tipo"], -r["chamadas"]))
        return {
            "dia": self._day,
            "itens": rows,
            "chamadas_dados": sum(r["chamadas"] for r in rows if r["tipo"] == "dados"),
            "segundos_processamento": round(sum(r["segundos"] for r in rows if r["tipo"] == "inferencia"), 3),
            "orcamentos": self.budgets,
        }


COSTS = CostMonitor()

_last_call: dict[str, float] = {}
_rate_lock = threading.Lock()


def throttle(url: str) -> None:
    """Respeita um intervalo mínimo entre requisições ao mesmo servidor."""
    host = urlparse(url).netloc
    interval = next((v for k, v in MIN_INTERVAL_S.items() if host.endswith(k)), DEFAULT_INTERVAL_S)
    with _rate_lock:
        wait = _last_call.get(host, 0) + interval - time.monotonic()
        _last_call[host] = time.monotonic() + max(wait, 0)
    if wait > 0:
        time.sleep(wait)
