"""Banco Central do Brasil - Sistema Gerenciador de Séries Temporais (SGS).

API pública: https://dadosabertos.bcb.gov.br/
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from ...core.types import Frequency, Series, utcnow
from .base import DataProvider, ProviderError, http_get_json

BASE_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"

# Séries SGS conhecidas. Outras podem ser registradas via `register`.
SGS_SERIES: dict[str, tuple[int, Frequency, str, str]] = {
    # chave: (código SGS, frequência, unidade, descrição)
    "selic_meta": (432, Frequency.DAILY, "% a.a.", "Meta da taxa Selic definida pelo Copom"),
    "selic_diaria": (11, Frequency.DAILY, "% a.d.", "Taxa Selic diária"),
    "ipca_mensal": (433, Frequency.MONTHLY, "% a.m.", "IPCA - variação mensal"),
    "ipca_12m": (13522, Frequency.MONTHLY, "%", "IPCA - acumulado em 12 meses"),
    "usdbrl_ptax": (1, Frequency.DAILY, "R$/US$", "Dólar americano (venda) - livre"),
}


class BCBProvider(DataProvider):
    name = "Banco Central do Brasil (SGS)"
    prefixes = ("macro:",)
    cost_per_call = 0.0  # API pública

    def __init__(self, series: dict | None = None) -> None:
        self.series = dict(SGS_SERIES)
        if series:
            self.series.update(series)

    def supports(self, key: str) -> bool:
        return key.startswith("macro:") and key.split(":", 1)[1] in self.series

    def register(self, key: str, code: int, frequency: Frequency, unit: str, description: str) -> None:
        self.series[key] = (code, frequency, unit, description)

    def fetch(self, key: str, start: date | None = None, end: date | None = None) -> Series:
        name = key.split(":", 1)[1]
        if name not in self.series:
            raise ProviderError(f"Série BCB desconhecida: {name}")
        code, freq, unit, desc = self.series[name]
        end = end or date.today()
        # A API limita séries diárias a 10 anos por consulta.
        start = start or (end - timedelta(days=365 * 5))
        url = BASE_URL.format(code=code)
        payload = http_get_json(
            url,
            {
                "formato": "json",
                "dataInicial": start.strftime("%d/%m/%Y"),
                "dataFinal": end.strftime("%d/%m/%Y"),
            },
        )
        if not isinstance(payload, list):
            raise ProviderError(f"Resposta inesperada do BCB para {name}: {str(payload)[:200]}")
        today = date.today()
        dates, values = [], []
        for row in payload:
            d = datetime.strptime(row["data"], "%d/%m/%Y").date()
            if d > today:
                # A meta Selic é publicada com vigência futura; não tratamos o futuro como observado.
                continue
            try:
                v = float(str(row["valor"]).replace(",", "."))
            except ValueError:
                continue
            dates.append(d)
            values.append(v)
        if not dates:
            raise ProviderError(f"BCB não retornou observações para {name} no período.")
        return Series(
            name=name,
            dates=dates,
            values=values,
            source=f"{self.name} - série {code} ({desc})",
            retrieved_at=utcnow(),
            frequency=freq,
            url=f"{url}?formato=json",
            unit=unit,
            license="Dados abertos BCB",
        )
