"""Carteira por quantidade e preço médio: armazenamento, migração, valorização, importação CSV e API."""
from __future__ import annotations

import sqlite3

import pytest

from oraculo.memory import MemoryStore
from oraculo.modes import run_mode
from oraculo.modes.base import parse_portfolio_csv, valued_portfolio

CSV = """ticker;quantidade;preco_medio;preco_medio_taxas;corretora;setor
ITUB4;100;39,45;39,4618;Nubank;bancos
KLBN3;705;3,8162;3,8173;Clear;papel_celulose
MXRF11;64;9,54;9,5428;Clear;fii_papel
"""


def test_position_by_quantity_stores_cost_and_validates():
    m = MemoryStore()
    m.set_position("itub4", quantity=100, avg_price=39.45, avg_price_fees=39.4618, broker="Nubank", sector="bancos")
    p = m.portfolio()[0]
    assert p["ticker"] == "ITUB4" and p["quantity"] == 100 and p["value"] == pytest.approx(3946.18)
    with pytest.raises(ValueError):
        m.set_position("X", quantity=10, avg_price=None)
    with pytest.raises(ValueError):
        m.set_position("X", quantity=10, avg_price=10, avg_price_fees=9)  # taxas não reduzem o preço médio
    m.set_position("ITUB4", quantity=0)
    assert m.portfolio() == []


def test_migrates_old_portfolio_table(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE portfolio (ticker TEXT PRIMARY KEY, value REAL, sector TEXT, updated_at TEXT)")
    con.execute("INSERT INTO portfolio VALUES ('PETR4', 1000, 'petroleo_gas', '2026-01-01')")
    con.commit()
    con.close()
    m = MemoryStore(db)
    p = m.portfolio()[0]
    assert p["value"] == 1000 and p["quantity"] is None  # posição antiga preservada


def test_parse_csv_decimal_comma_and_errors():
    items, errs = parse_portfolio_csv(CSV + "BAD;x;1;;;\nSEMPM;10;;;;\n")
    assert [i["ticker"] for i in items] == ["ITUB4", "KLBN3", "MXRF11"]
    assert items[1]["avg_price"] == pytest.approx(3.8162) and items[0]["broker"] == "Nubank"
    assert len(errs) == 2
    assert parse_portfolio_csv("a,b\n1,2")[1]  # cabeçalho inválido


def test_valued_portfolio_uses_market_price_or_flags_cost(svc):
    svc.memory.set_position("PETR4", quantity=200, avg_price=41.72, avg_price_fees=41.7325)
    svc.memory.set_position("ZZZZ3", quantity=10, avg_price=5)
    svc.hub.providers[0].prefixes = ("price:PETR4", "volume:", "macro:", "factor:")  # sem cotação para ZZZZ3
    rows, errs, _ = valued_portfolio(svc)
    by = {r["ticker"]: r for r in rows}
    s = svc.hub.get("price:PETR4")
    assert by["PETR4"]["valor_mercado"] == pytest.approx(200 * s.last_value)
    assert by["PETR4"]["resultado_rs"] == pytest.approx(200 * s.last_value - 200 * 41.7325)
    assert by["ZZZZ3"]["base_valor"].startswith("custo") and by["ZZZZ3"]["preco_atual"] is None
    assert sum(r["peso"] for r in rows) == pytest.approx(1)


def test_portfolio_mode_uses_market_values(svc):
    for t, q, pm in (("PETR4", 200, 41.72), ("ITUB4", 100, 39.45), ("VALE3", 50, 60.0)):
        svc.memory.set_position(t, quantity=q, avg_price=pm)
    r = run_mode("analise_carteira", svc, {})
    assert {p["ticker"] for p in r["posicoes"]} == {"PETR4", "ITUB4", "VALE3"}
    assert r["resumo"]["posicoes_a_custo"] == []
    assert r["risco"]["status"] in ("ok", "degradado")


def test_portfolio_api_import_and_remove(svc):
    from fastapi.testclient import TestClient

    from oraculo.api.app import create_app

    c = TestClient(create_app(svc))
    r = c.post("/api/user/portfolio/import", json={"csv": CSV, "substituir": True}).json()
    assert r["importados"] == ["ITUB4", "KLBN3", "MXRF11"] and r["erros"] == []
    c.post("/api/user/portfolio", json={"ticker": "PETR4", "quantidade": "200", "preco_medio": "41,72", "corretora": "Nubank"})
    assert {p["ticker"] for p in c.get("/api/user").json()["carteira"]} == {"ITUB4", "KLBN3", "MXRF11", "PETR4"}
    assert "PETR4" not in {p["ticker"] for p in c.delete("/api/user/portfolio/PETR4").json()}
    v = c.get("/api/user/portfolio/valued").json()
    assert len(v["posicoes"]) == 3
    assert c.post("/api/user/portfolio/import", json={"csv": ""}).status_code == 400
