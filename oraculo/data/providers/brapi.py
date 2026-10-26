"""brapi.dev - cotações da B3.

Alguns ativos funcionam sem token; os demais exigem `BRAPI_TOKEN`.
Verifique os termos de uso da brapi antes de uso comercial (Agente 16 - Compliance).
"""
from __future__ import annotations

import os
from datetime import date, datetime, timezone

from ...core.types import Frequency, Series, utcnow
from .base import DataProvider, ProviderError, http_get_json

BASE_URL = "https://brapi.dev/api/quote/{ticker}"


class BrapiProvider(DataProvider):
    name = "brapi.dev"
    prefixes = ("price:", "volume:", "high:", "low:")
    cost_per_call = 1.0

    def __init__(self, token: str | None = None, range_: str = "5y") -> None:
        self.token = token or os.environ.get("BRAPI_TOKEN")
        self.range = range_
        self._cache: dict[str, dict] = {}

    def _quote(self, ticker: str) -> dict:
        if ticker in self._cache:
            return self._cache[ticker]
        payload = http_get_json(
            BASE_URL.format(ticker=ticker),
            {"range": self.range, "interval": "1d", "token": self.token},
        )
        results = payload.get("results") if isinstance(payload, dict) else None
        if not results:
            raise ProviderError(f"brapi sem resultados para {ticker}: {str(payload)[:200]}")
        self._cache[ticker] = results[0]
        return results[0]

    def fetch(self, key: str, start: date | None = None, end: date | None = None) -> Series:
        field_, ticker = key.split(":", 1)
        q = self._quote(ticker.upper())
        rows = q.get("historicalDataPrice") or []
        column = {"price": "adjustedClose", "volume": "volume", "high": "high", "low": "low"}[field_]
        dates, values = [], []
        for r in sorted(rows, key=lambda r: r["date"]):
            v = r.get(column)
            if v is None and column == "adjustedClose":
                v = r.get("close")
            if v is None:
                continue
            d = datetime.fromtimestamp(r["date"], tz=timezone.utc).date()
            if (start and d < start) or (end and d > end):
                continue
            dates.append(d)
            values.append(float(v))
        if not dates:
            raise ProviderError(f"brapi sem histórico de '{column}' para {ticker}.")
        return Series(
            name=f"{field_}:{ticker.upper()}",
            dates=dates,
            values=values,
            source=f"{self.name} - {ticker.upper()} ({column})",
            retrieved_at=utcnow(),
            frequency=Frequency.DAILY,
            url=BASE_URL.format(ticker=ticker.upper()),
            unit="R$" if field_ != "volume" else "ações",
            license="Termos de uso brapi.dev",
        )
