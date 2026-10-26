"""Tipos fundamentais do Oráculo B3.

Toda informação que circula no sistema carrega sua procedência: fonte, horário
da observação e horário da coleta. Isso implementa as regras obrigatórias 1-3:
nada é inventado, toda fonte e horário são registrados e dados atrasados são
identificados.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any

import numpy as np


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Frequency(str, Enum):
    INTRADAY = "intraday"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    EVENT = "event"


# Idade máxima tolerada da última observação antes de o dado ser marcado como atrasado.
MAX_STALENESS: dict[Frequency, timedelta] = {
    Frequency.INTRADAY: timedelta(hours=1),
    Frequency.DAILY: timedelta(days=4),  # cobre fins de semana e feriados curtos
    Frequency.WEEKLY: timedelta(days=10),
    # séries mensais/trimestrais são datadas pelo início do período de referência e publicadas semanas depois
    Frequency.MONTHLY: timedelta(days=75),
    Frequency.QUARTERLY: timedelta(days=200),
    Frequency.ANNUAL: timedelta(days=400),
    Frequency.EVENT: timedelta(days=36500),
}


@dataclass(frozen=True)
class SourceRef:
    """Referência de procedência anexada a qualquer dado ou conclusão."""

    source: str
    retrieved_at: datetime
    as_of: datetime | None = None
    url: str | None = None
    synthetic: bool = False
    license: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["retrieved_at"] = self.retrieved_at.isoformat()
        d["as_of"] = self.as_of.isoformat() if self.as_of else None
        return d


@dataclass
class Series:
    """Série temporal com procedência obrigatória."""

    name: str
    dates: list[date]
    values: np.ndarray
    source: str
    retrieved_at: datetime
    frequency: Frequency = Frequency.DAILY
    url: str | None = None
    synthetic: bool = False
    unit: str | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError(f"Série '{self.name}' sem fonte: proibido pelo regulamento interno.")
        if self.retrieved_at is None:
            raise ValueError(f"Série '{self.name}' sem horário de coleta.")
        self.values = np.asarray(self.values, dtype=float)
        if len(self.dates) != len(self.values):
            raise ValueError(f"Série '{self.name}': datas e valores com tamanhos diferentes.")

    def __len__(self) -> int:
        return len(self.values)

    @property
    def last_date(self) -> date | None:
        return self.dates[-1] if self.dates else None

    @property
    def last_value(self) -> float | None:
        return float(self.values[-1]) if len(self.values) else None

    def staleness(self, now: datetime | None = None) -> timedelta | None:
        if not self.dates:
            return None
        now = now or utcnow()
        last = datetime.combine(self.last_date, datetime.min.time(), tzinfo=timezone.utc)
        return max(now - last, timedelta(0))

    def is_stale(self, now: datetime | None = None) -> bool:
        age = self.staleness(now)
        return age is None or age > MAX_STALENESS[self.frequency]

    def ref(self) -> SourceRef:
        as_of = (
            datetime.combine(self.last_date, datetime.min.time(), tzinfo=timezone.utc)
            if self.last_date
            else None
        )
        return SourceRef(
            source=self.source,
            retrieved_at=self.retrieved_at,
            as_of=as_of,
            url=self.url,
            synthetic=self.synthetic,
            license=self.license,
        )

    def returns(self) -> np.ndarray:
        v = self.values
        return v[1:] / v[:-1] - 1.0

    def tail(self, n: int) -> "Series":
        return Series(
            name=self.name,
            dates=self.dates[-n:],
            values=self.values[-n:],
            source=self.source,
            retrieved_at=self.retrieved_at,
            frequency=self.frequency,
            url=self.url,
            synthetic=self.synthetic,
            unit=self.unit,
            license=self.license,
        )

    def until(self, d: date) -> "Series":
        """Recorta a série até a data `d` (inclusive): base para evitar vazamento de futuro."""
        idx = [i for i, x in enumerate(self.dates) if x <= d]
        n = (idx[-1] + 1) if idx else 0
        return Series(
            name=self.name,
            dates=self.dates[:n],
            values=self.values[:n],
            source=self.source,
            retrieved_at=self.retrieved_at,
            frequency=self.frequency,
            url=self.url,
            synthetic=self.synthetic,
            unit=self.unit,
            license=self.license,
        )


@dataclass
class NewsItem:
    title: str
    source: str
    published_at: datetime
    retrieved_at: datetime
    body: str = ""
    url: str | None = None
    tickers: list[str] = field(default_factory=list)
    synthetic: bool = False


class Stance(str, Enum):
    POSITIVE = "positiva"
    NEGATIVE = "negativa"
    NEUTRAL = "neutra"
    INSUFFICIENT = "evidência insuficiente"


@dataclass
class AgentView:
    """Conclusão de um agente especialista, com evidências e procedência."""

    agent_id: str
    stance: Stance
    confidence: float  # 0..1
    summary: str
    evidence: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    skill_results: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "stance": self.stance.value,
            "peso_opiniao": round(self.confidence, 3),  # peso interno de evidência; NÃO é probabilidade (regra 6)
            "peso_qualitativo": "alta" if self.confidence >= 0.5 else ("média" if self.confidence >= 0.3 else ("baixa" if self.confidence > 0 else "nenhuma")),
            "summary": self.summary,
            "evidence": self.evidence,
            "risks": self.risks,
            "sources": [s.to_dict() for s in self.sources],
            "skill_results": self.skill_results,
            "warnings": self.warnings,
        }


def to_jsonable(obj: Any) -> Any:
    """Converte estruturas com numpy/datas em JSON serializável."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if np.isnan(f) or np.isinf(f) else f
    if isinstance(obj, float):
        return None if np.isnan(obj) or np.isinf(obj) else obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return obj
