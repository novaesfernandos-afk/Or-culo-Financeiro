"""Framework da biblioteca de skills.

Cada skill é uma função Python executável e testável, registrada com uma
especificação versionada (`SkillSpec`) que contém: identificador, objetivo,
agentes autorizados, entradas, ferramentas, procedimento, critérios de
validação, formato de saída e tratamento de falhas.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..core.costs import COSTS
from ..core.rules import check_news, check_series, ProvenanceError
from ..core.types import NewsItem, Series, SourceRef, to_jsonable, utcnow


class SkillStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degradado"  # executou com ressalvas (dados atrasados, amostra pequena...)
    INVALID = "invalido"  # executou, mas a saída falhou nos critérios de validação
    FAILED = "falhou"
    DENIED = "negado"  # agente sem permissão


class SkillError(RuntimeError):
    """Falha esperada de domínio (dados insuficientes, parâmetros inválidos)."""


class PermissionDenied(PermissionError):
    pass


@dataclass(frozen=True)
class SkillSpec:
    id: str
    name: str
    version: str
    category: str
    objective: str
    authorized_agents: tuple[str, ...]
    inputs: dict[str, str]
    required_inputs: tuple[str, ...]
    tools: tuple[str, ...]
    procedure: tuple[str, ...]
    validation: tuple[str, ...]
    output_schema: dict[str, str]
    failure_handling: str
    heavy: bool = False  # modelos pesados: o Modo 1 só os executa sob mudança material

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "objective": self.objective,
            "authorized_agents": list(self.authorized_agents),
            "inputs": self.inputs,
            "required_inputs": list(self.required_inputs),
            "tools": list(self.tools),
            "procedure": list(self.procedure),
            "validation": list(self.validation),
            "output_schema": self.output_schema,
            "failure_handling": self.failure_handling,
            "heavy": self.heavy,
        }


@dataclass
class SkillContext:
    hub: Any = None  # DataHub
    memory: Any = None  # MemoryStore
    now: datetime = field(default_factory=utcnow)


@dataclass
class SkillRun:
    """Objeto entregue à função da skill para registrar avisos e fontes."""

    ctx: SkillContext
    warnings: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    degraded: bool = False

    def warn(self, msg: str, degrade: bool = True) -> None:
        self.warnings.append(msg)
        self.degraded = self.degraded or degrade

    def cite(self, obj: Series | NewsItem | SourceRef) -> None:
        if isinstance(obj, Series):
            for w in check_series(obj, self.ctx.now):
                self.warn(w)
            self.sources.append(obj.ref())
        elif isinstance(obj, NewsItem):
            check_news(obj)
            self.sources.append(
                SourceRef(obj.source, obj.retrieved_at, obj.published_at, obj.url, obj.synthetic)
            )
        elif isinstance(obj, SourceRef):
            self.sources.append(obj)


@dataclass
class SkillResult:
    skill_id: str
    version: str
    agent_id: str
    status: SkillStatus
    output: dict[str, Any]
    warnings: list[str]
    sources: list[SourceRef]
    started_at: datetime
    duration_ms: float
    input_digest: str
    error: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def ok(self) -> bool:
        return self.status in (SkillStatus.OK, SkillStatus.DEGRADED)

    @property
    def output_digest(self) -> str:
        return digest(self.output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "skill_id": self.skill_id,
            "version": self.version,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "output": to_jsonable(self.output),
            "warnings": self.warnings,
            "sources": [s.to_dict() for s in self.sources],
            "started_at": self.started_at.isoformat(),
            "duration_ms": round(self.duration_ms, 2),
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "error": self.error,
        }


def digest(obj: Any) -> str:
    def default(o: Any) -> Any:
        if isinstance(o, Series):
            return {"name": o.name, "source": o.source, "n": len(o), "last": str(o.last_date)}
        if callable(o):
            return getattr(o, "__name__", "callable")
        return str(o)

    payload = json.dumps(to_jsonable_safe(obj), sort_keys=True, default=default, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def to_jsonable_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable_safe(v) for v in obj]
    if isinstance(obj, (Series, NewsItem)) or callable(obj):
        return obj
    return to_jsonable(obj)


Validator = Callable[[dict[str, Any]], list[str]]


@dataclass
class Skill:
    spec: SkillSpec
    fn: Callable[..., dict[str, Any]]
    validators: list[Validator] = field(default_factory=list)

    def validator(self, v: Validator) -> Validator:
        self.validators.append(v)
        return v

    def __call__(self, agent_id: str, ctx: SkillContext | None = None, **inputs: Any) -> SkillResult:
        return execute(self, agent_id, ctx or SkillContext(), inputs)


REGISTRY: dict[str, Skill] = {}


def skill(**spec_kwargs: Any) -> Callable[[Callable[..., dict]], Skill]:
    def wrap(fn: Callable[..., dict]) -> Skill:
        spec_kwargs.setdefault("name", spec_kwargs["id"])
        for k in ("authorized_agents", "required_inputs", "tools", "procedure", "validation"):
            spec_kwargs[k] = tuple(spec_kwargs.get(k, ()))
        spec = SkillSpec(**spec_kwargs)
        if spec.id in REGISTRY:
            raise ValueError(f"Skill duplicada: {spec.id}")
        sk = Skill(spec, fn)
        REGISTRY[spec.id] = sk
        return sk

    return wrap


def execute(sk: Skill, agent_id: str, ctx: SkillContext, inputs: dict[str, Any]) -> SkillResult:
    started = utcnow()
    t0 = time.perf_counter()
    spec = sk.spec
    in_digest = digest(inputs)

    def result(status: SkillStatus, output: dict, run: SkillRun | None, error: str | None = None) -> SkillResult:
        COSTS.record("inferencia", spec.id, time.perf_counter() - t0, ok=status in (SkillStatus.OK, SkillStatus.DEGRADED))
        return SkillResult(
            skill_id=spec.id,
            version=spec.version,
            agent_id=agent_id,
            status=status,
            output=output,
            warnings=run.warnings if run else [],
            sources=run.sources if run else [],
            started_at=started,
            duration_ms=(time.perf_counter() - t0) * 1000,
            input_digest=in_digest,
            error=error,
        )

    if agent_id not in spec.authorized_agents and agent_id != "system:test":
        return result(SkillStatus.DENIED, {}, None, f"Agente {agent_id} não autorizado a usar {spec.id}.")
    missing = [k for k in spec.required_inputs if inputs.get(k) is None]
    if missing:
        return result(SkillStatus.FAILED, {}, None, f"Entradas obrigatórias ausentes: {', '.join(missing)}")

    run = SkillRun(ctx)
    for v in inputs.values():  # procedência automática das entradas
        try:
            if isinstance(v, (Series, NewsItem)):
                run.cite(v)
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, (Series, NewsItem)) for x in v):
                for x in v:
                    run.cite(x)
            elif isinstance(v, dict) and v and all(isinstance(x, Series) for x in v.values()):
                for x in v.values():
                    run.cite(x)
        except ProvenanceError as exc:
            return result(SkillStatus.FAILED, {}, run, str(exc))
    try:
        output = sk.fn(run, **inputs)
    except (SkillError, ProvenanceError, ValueError) as exc:
        return result(SkillStatus.FAILED, {}, run, str(exc))
    except Exception as exc:  # noqa: BLE001 - falha inesperada registrada com traceback curto
        tb = traceback.format_exc(limit=3)
        return result(SkillStatus.FAILED, {}, run, f"Erro inesperado: {exc!r}\n{tb}")

    violations: list[str] = []
    for v in sk.validators:
        violations.extend(v(output))
    if violations:
        run.warnings.extend(f"Validação: {x}" for x in violations)
        return result(SkillStatus.INVALID, output, run, "; ".join(violations))
    return result(SkillStatus.DEGRADED if run.degraded else SkillStatus.OK, output, run)


def run_skill(skill_id: str, agent_id: str, ctx: SkillContext | None = None, **inputs: Any) -> SkillResult:
    if skill_id not in REGISTRY:
        raise KeyError(f"Skill inexistente: {skill_id}")
    return REGISTRY[skill_id](agent_id, ctx, **inputs)


# --- validadores reutilizáveis -------------------------------------------------


def prob_in_unit(*keys: str) -> Validator:
    def v(out: dict) -> list[str]:
        errs = []
        for k in keys:
            x = out.get(k)
            if x is not None and not (0.0 <= float(x) <= 1.0):
                errs.append(f"{k}={x} fora de [0,1]")
        return errs

    return v


def finite(*keys: str) -> Validator:
    import math

    def v(out: dict) -> list[str]:
        errs = []
        for k in keys:
            x = out.get(k)
            if isinstance(x, (int, float)) and not math.isfinite(float(x)):
                errs.append(f"{k} não é finito")
        return errs

    return v
