"""Regras obrigatórias de funcionamento (seção 6 do escopo).

1. Nunca inventar notícias, cotações ou indicadores.
2. Registrar fonte e horário dos dados.
3. Identificar informações atrasadas.

As regras são aplicadas em código: qualquer saída de skill passa por
`enforce_provenance`, e relatórios finais passam por `provenance_banner`.
"""
from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .types import NewsItem, Series, SourceRef, utcnow

MANDATORY_RULES: list[str] = [
    "Nunca inventar notícias, cotações ou indicadores.",
    "Registrar fonte e horário dos dados.",
    "Identificar informações atrasadas.",
]


class ProvenanceError(ValueError):
    """Dado sem procedência verificável."""


def check_series(series: Series, now: datetime | None = None) -> list[str]:
    """Retorna avisos para uma série; levanta erro se faltar procedência."""
    if not series.source or series.retrieved_at is None:
        raise ProvenanceError(f"Série '{series.name}' sem fonte ou horário de coleta.")
    warnings: list[str] = []
    if len(series) == 0:
        warnings.append(f"Série '{series.name}' vazia ({series.source}).")
        return warnings
    if series.is_stale(now):
        age = series.staleness(now)
        warnings.append(
            f"DADO ATRASADO: '{series.name}' ({series.source}) - última observação em "
            f"{series.last_date.isoformat()} ({age.days} dias)."
        )
    if series.synthetic:
        warnings.append(
            f"DADO SINTÉTICO: '{series.name}' não representa o mercado real; uso apenas para demonstração/testes."
        )
    return warnings


def check_news(item: NewsItem) -> None:
    if not item.source or item.published_at is None or item.retrieved_at is None:
        raise ProvenanceError(f"Notícia '{item.title[:60]}' sem fonte ou horário: descartada.")


def collect_refs(series: Iterable[Series]) -> list[SourceRef]:
    return [s.ref() for s in series]


def has_synthetic(refs: Iterable[SourceRef]) -> bool:
    return any(r.synthetic for r in refs)


def provenance_banner(refs: list[SourceRef], now: datetime | None = None) -> list[str]:
    """Linhas de procedência para rodapé de relatórios."""
    now = now or utcnow()
    lines = []
    seen = set()
    for r in refs:
        key = (r.source, r.as_of, r.url)
        if key in seen:
            continue
        seen.add(key)
        as_of = r.as_of.strftime("%Y-%m-%d") if r.as_of else "n/d"
        tag = " [SINTÉTICO]" if r.synthetic else ""
        lines.append(
            f"{r.source}{tag} - dado de {as_of}, coletado em {r.retrieved_at.strftime('%Y-%m-%d %H:%M UTC')}"
            + (f" - {r.url}" if r.url else "")
        )
    return lines
