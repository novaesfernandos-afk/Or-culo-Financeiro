"""Séries fornecidas pelo usuário em CSV (ex.: dados licenciados exportados).

Formato: colunas `date,value` (data ISO). A fonte declarada é obrigatória.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from ...core.types import Frequency, Series, utcnow
from .base import DataProvider, ProviderError


class CSVProvider(DataProvider):
    name = "Arquivo CSV do usuário"
    cost_per_call = 0.0

    def __init__(self) -> None:
        self.files: dict[str, tuple[Path, str, Frequency]] = {}

    def supports(self, key: str) -> bool:
        return key in self.files

    def register(self, key: str, path: str | Path, declared_source: str, frequency: Frequency = Frequency.DAILY) -> None:
        if not declared_source:
            raise ValueError("Informe a fonte original dos dados do CSV.")
        self.files[key] = (Path(path), declared_source, frequency)

    def fetch(self, key: str, start: date | None = None, end: date | None = None) -> Series:
        path, declared, freq = self.files[key]
        if not path.exists():
            raise ProviderError(f"Arquivo não encontrado: {path}")
        dates, values = [], []
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                d = date.fromisoformat(row["date"])
                if (start and d < start) or (end and d > end):
                    continue
                dates.append(d)
                values.append(float(row["value"]))
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        return Series(
            name=key,
            dates=[dates[i] for i in order],
            values=[values[i] for i in order],
            source=f"{declared} (via CSV {path.name})",
            retrieved_at=utcnow(),
            frequency=freq,
        )
