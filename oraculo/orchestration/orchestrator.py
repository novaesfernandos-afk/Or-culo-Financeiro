"""Orquestrador Operacional (Agente 02): execução de tarefas em DAG.

- Executa em paralelo as tarefas cujas dependências já terminaram.
- Repete tarefas com falha transitória (espera exponencial, limite de tentativas).
- Tarefas cujas dependências falharam são marcadas como 'bloqueada', nunca executadas com dados faltantes.
- Registra duração, tentativas e custo de dados de cada tarefa.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.types import utcnow
from ..data.providers.base import ProviderError


@dataclass
class Task:
    id: str
    agent_id: str
    fn: Callable[[dict[str, Any]], Any]  # recebe os resultados das dependências
    deps: tuple[str, ...] = ()
    retries: int = 1
    retry_on: tuple[type[BaseException], ...] = (ProviderError, TimeoutError, ConnectionError)
    optional: bool = False  # falha não bloqueia dependentes opcionais
    description: str = ""


@dataclass
class TaskRecord:
    id: str
    agent_id: str
    status: str = "pendente"  # pendente | executando | ok | falhou | bloqueada
    attempts: int = 0
    started_at: str | None = None
    seconds: float = 0.0
    error: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class CycleError(ValueError):
    pass


@dataclass
class Orchestrator:
    max_workers: int = 8
    backoff_base: float = 0.5
    hub: Any = None
    records: dict[str, TaskRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, tasks: list[Task]) -> dict[str, Any]:
        by_id = {t.id: t for t in tasks}
        if len(by_id) != len(tasks):
            raise ValueError("IDs de tarefa duplicados.")
        for t in tasks:
            for d in t.deps:
                if d not in by_id:
                    raise ValueError(f"Tarefa {t.id} depende de {d}, inexistente.")
        self._check_cycles(by_id)
        self.records = {t.id: TaskRecord(t.id, t.agent_id, description=t.description) for t in tasks}
        results: dict[str, Any] = {}
        usage_before = self.hub.usage() if self.hub else None
        t_start = time.perf_counter()
        pending = set(by_id)
        running: dict[Future, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while pending or running:
                for tid in sorted(pending):
                    t = by_id[tid]
                    states = [self.records[d].status for d in t.deps]
                    if any(s in ("falhou", "bloqueada") and not by_id[d].optional for s, d in zip(states, t.deps)):
                        self.records[tid].status = "bloqueada"
                        failed = [d for d, s in zip(t.deps, states) if s in ("falhou", "bloqueada")]
                        self.records[tid].error = f"dependência sem resultado: {', '.join(failed)}"
                        pending.discard(tid)
                    elif all(s in ("ok", "falhou", "bloqueada") for s in states):
                        dep_results = {d: results.get(d) for d in t.deps}
                        running[pool.submit(self._run_one, t, dep_results)] = tid
                        pending.discard(tid)
                if not running:
                    continue
                done, _ = wait(list(running), return_when=FIRST_COMPLETED)
                for f in done:
                    tid = running.pop(f)
                    ok, value = f.result()
                    if ok:
                        results[tid] = value
        self.summary = {
            "tarefas": len(tasks),
            "ok": sum(r.status == "ok" for r in self.records.values()),
            "falhas": sum(r.status == "falhou" for r in self.records.values()),
            "bloqueadas": sum(r.status == "bloqueada" for r in self.records.values()),
            "segundos": round(time.perf_counter() - t_start, 3),
        }
        if self.hub and usage_before:
            after = self.hub.usage()
            self.summary["custo_dados"] = {k: round(after[k] - usage_before[k], 3) for k in after}
        return results

    def _run_one(self, t: Task, deps: dict[str, Any]) -> tuple[bool, Any]:
        rec = self.records[t.id]
        rec.status = "executando"
        rec.started_at = utcnow().isoformat()
        t0 = time.perf_counter()
        for attempt in range(1, t.retries + 2):
            rec.attempts = attempt
            try:
                value = t.fn(deps)
                rec.status = "ok"
                rec.seconds = round(time.perf_counter() - t0, 4)
                return True, value
            except t.retry_on as exc:
                rec.error = f"{type(exc).__name__}: {exc}"
                if attempt <= t.retries and not getattr(exc, "permanent", False):
                    time.sleep(self.backoff_base * 2 ** (attempt - 1))
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - falha não transitória: sem repetição
                rec.error = f"{type(exc).__name__}: {exc}"
                break
        rec.status = "falhou"
        rec.seconds = round(time.perf_counter() - t0, 4)
        return False, None

    @staticmethod
    def _check_cycles(by_id: dict[str, Task]) -> None:
        state: dict[str, int] = {}

        def visit(n: str) -> None:
            if state.get(n) == 1:
                raise CycleError(f"Ciclo de dependências envolvendo {n}")
            if state.get(n) == 2:
                return
            state[n] = 1
            for d in by_id[n].deps:
                visit(d)
            state[n] = 2

        for n in by_id:
            visit(n)

    def log(self) -> list[dict]:
        return [r.to_dict() for r in self.records.values()]


class Scheduler:
    """Agendador simples de rotinas periódicas (usado pelo Modo 1)."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def every(self, name: str, seconds: float, fn: Callable[[], Any]) -> None:
        self.jobs[name] = {"seconds": seconds, "fn": fn, "next": time.time(), "last_error": None, "runs": 0}

    def tick(self) -> list[str]:
        ran = []
        now = time.time()
        for name, j in self.jobs.items():
            if now >= j["next"]:
                try:
                    j["fn"]()
                    j["last_error"] = None
                except Exception as exc:  # noqa: BLE001
                    j["last_error"] = repr(exc)
                j["runs"] += 1
                j["next"] = now + j["seconds"]
                ran.append(name)
        return ran

    def start(self, poll: float = 5.0) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                self.tick()
                self._stop.wait(poll)

        self._thread = threading.Thread(target=loop, daemon=True, name="oraculo-scheduler")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        return {n: {k: v for k, v in j.items() if k != "fn"} for n, j in self.jobs.items()}
