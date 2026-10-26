"""Infraestrutura comum dos modos de trabalho."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..core.costs import COSTS
from ..core.types import Series
from ..data.cvm import CVMClient
from ..data.hub import DataHub, default_hub
from ..data.news import NewsCollector
from ..memory.store import MemoryStore
from ..orchestration.orchestrator import Orchestrator, Scheduler, Task
from ..skills._math import align

# Universo padrão (configurável). Sem BRAPI_TOKEN, a brapi pode limitar alguns ativos.
DEFAULT_UNIVERSE = [
    "PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "ABEV3", "WEGE3", "B3SA3",
    "SUZB3", "PRIO3", "RADL3", "EQTL3", "GGBR4", "LREN3", "MGLU3", "RENT3",
]

# Fatores para simulação de cenários: chave no DataHub e natureza do choque.
FACTORS: dict[str, dict[str, str]] = {
    "selic": {"key": "macro:selic_meta", "kind": "nivel", "unit": "p.p."},
    "dolar": {"key": "macro:usdbrl_ptax", "kind": "retorno", "unit": "%"},
    "ibovespa": {"key": "price:^BVSP", "kind": "retorno", "unit": "%"},
    "petroleo": {"key": "factor:petroleo", "kind": "retorno", "unit": "%"},
    "minerio": {"key": "factor:minerio", "kind": "retorno", "unit": "%"},
}


@dataclass
class Services:
    hub: DataHub
    memory: MemoryStore
    scheduler: Scheduler = field(default_factory=Scheduler)
    demo: bool = False
    state: dict[str, Any] = field(default_factory=dict)
    cvm: CVMClient | None = None
    news: NewsCollector | None = None

    @classmethod
    def create(cls, db_path: str | None = None, demo: bool | None = None) -> "Services":
        demo = demo if demo is not None else os.environ.get("ORACULO_DEMO") == "1"
        db = db_path or os.environ.get("ORACULO_DB", "data/oraculo.db")
        cvm = news = None
        if not demo:  # fontes reais nunca são misturadas com dados sintéticos
            cvm = CVMClient() if os.environ.get("ORACULO_CVM", "1") != "0" else None
            news = NewsCollector(cvm=cvm) if os.environ.get("ORACULO_NEWS", "1") != "0" else None
        memory = MemoryStore(db if not demo else ":memory:")
        COSTS.memory = memory  # consumo persistido (regra 11)
        return cls(hub=default_hub(demo), memory=memory, demo=demo, cvm=cvm, news=news)

    def collectors(self) -> dict:
        return {"cvm": self.cvm, "news": self.news}

    def fetch_many(self, keys: list[str], retries: int = 1) -> tuple[dict[str, Series], list[str]]:
        orch = Orchestrator(hub=self.hub)
        res = orch.run([Task(k, "A02", (lambda key: lambda _d: self.hub.get(key))(k), retries=retries, optional=True) for k in keys])
        errors = [f"{r['id']}: {r['error']}" for r in orch.log() if r["status"] != "ok"]
        return {k: v for k, v in res.items() if v is not None}, errors


@dataclass(frozen=True)
class ModeSpec:
    id: int
    slug: str
    name: str
    objective: str
    agents: tuple[str, ...]
    run: Callable[[Services, dict], dict]

    def to_dict(self) -> dict:
        return {"id": self.id, "slug": self.slug, "name": self.name, "objective": self.objective, "agents": list(self.agents)}


def adv_brl(prices: Series, volume: Series, n: int = 21) -> float:
    _, (p, v) = align(prices, volume)
    return float((p[-n:] * v[-n:]).mean()) if len(p) else 0.0


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if m.sum() >= 2 and x[m].std() > 0:
        out[m] = (x[m] - x[m].mean()) / x[m].std()
    elif m.sum() >= 1:
        out[m] = 0.0
    return out


def valued_portfolio(svc: "Services") -> tuple[list[dict], list[str], dict]:
    """Carteira valorizada pela cotação do dia; sem cotação, usa o custo e sinaliza (nunca estima preço).

    Retorna (posições, falhas de coleta, séries de preço por ticker).
    """
    stored = svc.memory.portfolio()
    data, errors = svc.fetch_many([f"price:{p['ticker']}" for p in stored]) if stored else ({}, [])
    rows = []
    for p in stored:
        s = data.get(f"price:{p['ticker']}")
        qty, pm_fees = p.get("quantity"), p.get("avg_price_fees") or p.get("avg_price")
        row = {
            "ticker": p["ticker"], "setor": p.get("sector"), "corretora": p.get("broker"), "quantidade": qty,
            "preco_medio": p.get("avg_price"), "preco_medio_taxas": p.get("avg_price_fees"), "custo": p["value"],
            "preco_atual": None, "data_preco": None, "fonte_preco": None, "atrasado": None,
            "valor_mercado": None, "resultado_rs": None, "resultado_pct": None,
        }
        if qty and s is not None:
            mv = qty * s.last_value
            row.update({"preco_atual": s.last_value, "data_preco": s.last_date.isoformat(), "fonte_preco": s.source,
                        "atrasado": s.is_stale(), "sintetico": s.synthetic, "valor_mercado": mv, "resultado_rs": mv - p["value"],
                        "resultado_pct": mv / p["value"] - 1 if p["value"] else None})
            row["valor_usado"], row["base_valor"] = mv, "mercado"
        elif qty:
            row["valor_usado"], row["base_valor"] = p["value"], "custo (cotação indisponível)"
        else:
            row["valor_usado"], row["base_valor"] = p["value"], "valor informado"
        rows.append(row)
    total = sum(r["valor_usado"] for r in rows)
    for r in rows:
        r["peso"] = r["valor_usado"] / total if total else None
    rows.sort(key=lambda r: -r["valor_usado"])
    return rows, errors, {t.split(":", 1)[1]: s for t, s in data.items()}


def parse_portfolio_csv(text: str) -> tuple[list[dict], list[str]]:
    """Lê posições em CSV: ticker; quantidade; preco_medio; preco_medio_taxas; corretora; setor.

    Aceita separador ';' ou ',' (com ';', a vírgula é decimal). Linhas inválidas são reportadas, não corrigidas.
    """
    import csv
    import io

    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return [], ["CSV vazio."]
    delim = ";" if ";" in lines[0] else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delim)
    norm = {(h or "").strip().lower(): h for h in reader.fieldnames or []}
    if "ticker" not in norm or "quantidade" not in norm or "preco_medio" not in norm:
        return [], ["Cabeçalho deve conter ao menos: ticker, quantidade, preco_medio."]

    def num(v: str | None) -> float | None:
        v = (v or "").strip()
        if not v:
            return None
        if delim == ";":
            v = v.replace(".", "").replace(",", ".") if "," in v else v
        return float(v)

    out, errs = [], []
    for i, row in enumerate(reader, start=2):
        g = lambda k: row.get(norm.get(k, ""), None)  # noqa: E731
        t = (g("ticker") or "").strip().upper()
        try:
            item = {"ticker": t, "quantity": num(g("quantidade")), "avg_price": num(g("preco_medio")),
                    "avg_price_fees": num(g("preco_medio_taxas")), "broker": (g("corretora") or "").strip() or None,
                    "sector": (g("setor") or "").strip() or None}
        except ValueError:
            errs.append(f"Linha {i} ({t or 'sem ticker'}): número inválido.")
            continue
        if not t or item["quantity"] is None or item["avg_price"] is None:
            errs.append(f"Linha {i} ({t or 'sem ticker'}): ticker, quantidade e preço médio são obrigatórios.")
            continue
        out.append(item)
    return out, errs
