"""Interface comum para provedores de dados."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date

from ...core.costs import throttle
from ...core.types import Series

USER_AGENT = "OraculoB3/0.1 (+https://github.com/novaesfernandos-afk/or-culo-financeiro)"


class ProviderError(RuntimeError):
    """Falha ao obter dado de um provedor. Nunca substituir por valor inventado."""

    def __init__(self, msg: str, permanent: bool = False) -> None:
        super().__init__(msg)
        self.permanent = permanent  # ex.: 401/403/404 - repetir não adianta


class DataProvider(ABC):
    #: nome exibido como fonte
    name: str = "provedor"
    #: prefixos de chave atendidos (ex.: "macro:", "price:")
    prefixes: tuple[str, ...] = ()
    #: custo relativo de uma chamada (para monitoramento do Orquestrador)
    cost_per_call: float = 1.0

    def supports(self, key: str) -> bool:
        return any(key.startswith(p) for p in self.prefixes)

    @abstractmethod
    def fetch(self, key: str, start: date | None = None, end: date | None = None) -> Series:
        """Retorna a série solicitada ou levanta ProviderError."""


def http_get_json(url: str, params: dict | None = None, timeout: float = 20.0):
    if params:
        url = f"{url}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    throttle(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        hint = " (verifique o token/licença da fonte)" if exc.code in (401, 403) else ""
        raise ProviderError(f"Falha ao consultar {url.split('?')[0]}: HTTP {exc.code}{hint}", permanent=exc.code in (400, 401, 403, 404)) from exc
    except Exception as exc:  # noqa: BLE001 - convertido em erro de domínio
        raise ProviderError(f"Falha ao consultar {url.split('?')[0]}: {exc}") from exc
