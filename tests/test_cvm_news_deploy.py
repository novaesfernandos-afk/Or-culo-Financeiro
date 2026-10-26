"""CVM (cadastro, fatos relevantes, DFP/ITR), coletor de notícias, integração no fluxo e proteção da implantação."""
from __future__ import annotations

import base64
import csv
import io
import zipfile
from datetime import date, datetime, timedelta, timezone

import pytest

from oraculo.core import agent_ids as A
from oraculo.core.types import NewsItem, utcnow
from oraculo.data.cvm import CVMClient
from oraculo.data.news import FEEDS, NewsCollector, mentions, parse_rss
from oraculo.skills.base import run_skill

Y = date.today().year
TODAY = date.today()
FIN_HEAD = ["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "CD_CVM", "GRUPO_DFP", "MOEDA", "ESCALA_MOEDA", "ORDEM_EXERC",
            "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA", "VL_CONTA", "ST_CONTA_FIXA"]
IND, BANK = "11.111.111/0001-11", "22.222.222/0001-22"


def _csv(header, rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("latin-1")


def _zip(path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


def _fin(cnpj, name, ref, rows, ini=None, version="1"):
    out = []
    for ordem, code, desc, val in rows:
        yr = int(ref[:4]) - (1 if ordem == "PENÚLTIMO" else 0)
        end = f"{yr}{ref[4:]}"
        out.append([cnpj, ref, version, name, "1", "DF Consolidado", "REAL", "MIL", ordem, (ini or f"{yr}-01-01").replace(ref[:4], str(yr)), end, code, desc, val, "S"])
    return out


@pytest.fixture
def cvm(tmp_path) -> CVMClient:
    c = CVMClient(cache_dir=tmp_path, max_age_hours=1e9, base_url="http://invalido.local")
    fca = Y
    _zip(tmp_path / f"fca_cia_aberta_{fca}.zip", {f"fca_cia_aberta_valor_mobiliario_{fca}.csv": _csv(
        ["CNPJ_Companhia", "Nome_Empresarial", "Codigo_Negociacao", "Data_Fim_Negociacao", "Segmento"],
        [[IND, "INDUSTRIA TESTE S.A.", "INDT3", "", "Novo Mercado"], [BANK, "BANCO TESTE S.A.", "BNKT4", "", "N1"]])})
    ipe_rows = [
        [IND, "INDUSTRIA TESTE S.A.", "1", TODAY.isoformat(), "Fato Relevante", "", "", "Aquisição de controle", TODAY.isoformat(), "AP", "p1", "1", "https://www.rad.cvm.gov.br/x"],
        [IND, "INDUSTRIA TESTE S.A.", "1", TODAY.isoformat(), "Assembleia", "", "", "Ata AGO", TODAY.isoformat(), "AP", "p2", "1", "https://www.rad.cvm.gov.br/y"],
        [IND, "INDUSTRIA TESTE S.A.", "1", "2000-01-01", "Fato Relevante", "", "", "Antigo", (TODAY - timedelta(days=90)).isoformat(), "AP", "p3", "1", "https://www.rad.cvm.gov.br/z"],
    ]
    _zip(tmp_path / f"ipe_cia_aberta_{Y}.zip", {f"ipe_cia_aberta_{Y}.csv": _csv(
        ["CNPJ_Companhia", "Nome_Companhia", "Codigo_CVM", "Data_Referencia", "Categoria", "Tipo", "Especie", "Assunto", "Data_Entrega",
         "Tipo_Apresentacao", "Protocolo_Entrega", "Versao", "Link_Download"], ipe_rows)})
    dy = Y - 1
    ref = f"{dy}-12-31"
    dre = _fin(IND, "INDUSTRIA TESTE S.A.", ref, [
        ("ÚLTIMO", "3.01", "Receita de Venda de Bens e/ou Serviços", "1000"), ("PENÚLTIMO", "3.01", "Receita de Venda de Bens e/ou Serviços", "900"),
        ("ÚLTIMO", "3.05", "Resultado Antes do Resultado Financeiro e dos Tributos", "250"),
        ("ÚLTIMO", "3.06.02", "Despesas Financeiras", "-50"),
        ("ÚLTIMO", "3.11", "Lucro/Prejuízo Consolidado do Período", "150"), ("PENÚLTIMO", "3.11", "Lucro/Prejuízo Consolidado do Período", "120"),
        ("ÚLTIMO", "3.11.01", "Atribuído a Sócios da Empresa Controladora", "150"),
        ("ÚLTIMO", "3.99.01.01", "ON", "1.5"),
    ]) + _fin(BANK, "BANCO TESTE S.A.", ref, [
        ("ÚLTIMO", "3.01", "Receitas da Intermediação Financeira", "500"),
        ("ÚLTIMO", "3.09", "Lucro/Prejuízo Consolidado do Período", "80"),
    ])
    # versão antiga reapresentada não pode prevalecer
    dre += _fin(IND, "INDUSTRIA TESTE S.A.", ref, [("ÚLTIMO", "3.01", "Receita de Venda de Bens e/ou Serviços", "1")], version="0")
    bp_head = [h for h in FIN_HEAD if h != "DT_INI_EXERC"]
    bpa = [[IND, ref, "1", "X", "1", "DF", "REAL", "MIL", "ÚLTIMO", ref, c, d, v, "S"] for c, d, v in
           (("1", "Ativo Total", "2000"), ("1.01.01", "Caixa e Equivalentes de Caixa", "150"), ("1.01.02", "Aplicações Financeiras", "50"))]
    bpp = [[IND, ref, "1", "X", "1", "DF", "REAL", "MIL", "ÚLTIMO", ref, c, d, v, "S"] for c, d, v in
           (("2.01.04", "Empréstimos e Financiamentos", "200"), ("2.02.01", "Empréstimos e Financiamentos", "300"),
            ("2.03", "Patrimônio Líquido Consolidado", "1000"))]
    dfc = _fin(IND, "X", ref, [("ÚLTIMO", "6.01", "Caixa Líquido Atividades Operacionais", "200"),
                               ("ÚLTIMO", "6.01.01.04", "Depreciação e amortização", "50"),
                               ("ÚLTIMO", "6.02.01", "Aquisições de ativos imobilizados e intangíveis", "-80")])
    cap = _csv(["CNPJ_CIA", "DT_REFER", "VERSAO", "DENOM_CIA", "QT_ACAO_ORDIN_CAP_INTEGR", "QT_ACAO_PREF_CAP_INTEGR",
                "QT_ACAO_TOTAL_CAP_INTEGR", "QT_ACAO_ORDIN_TESOURO", "QT_ACAO_PREF_TESOURO", "QT_ACAO_TOTAL_TESOURO"],
               [[IND, ref, "1", "X", "100", "0", "101", "0", "0", "1"]])  # em milhares: 100 mil ações
    _zip(tmp_path / f"dfp_cia_aberta_{dy}.zip", {
        f"dfp_cia_aberta_DRE_con_{dy}.csv": _csv(FIN_HEAD, dre), f"dfp_cia_aberta_BPA_con_{dy}.csv": _csv(bp_head, bpa),
        f"dfp_cia_aberta_BPP_con_{dy}.csv": _csv(bp_head, bpp), f"dfp_cia_aberta_DFC_MI_con_{dy}.csv": _csv(FIN_HEAD, dfc),
        f"dfp_cia_aberta_composicao_capital_{dy}.csv": cap})
    qref = f"{Y}-06-30"
    itr = _fin(IND, "INDUSTRIA TESTE S.A.", qref, [
        ("ÚLTIMO", "3.01", "Receita de Venda de Bens e/ou Serviços", "300"),
        ("ÚLTIMO", "3.11", "Lucro/Prejuízo Consolidado do Período", "60"),
        ("PENÚLTIMO", "3.01", "Receita de Venda de Bens e/ou Serviços", "250"),
        ("PENÚLTIMO", "3.11", "Lucro/Prejuízo Consolidado do Período", "40"),
    ], ini=f"{Y}-04-01") + _fin(IND, "INDUSTRIA TESTE S.A.", qref, [
        ("ÚLTIMO", "3.01", "Receita de Venda de Bens e/ou Serviços", "550"),  # acumulado do semestre: ignorado
    ])
    _zip(tmp_path / f"itr_cia_aberta_{Y}.zip", {f"itr_cia_aberta_DRE_con_{Y}.csv": _csv(FIN_HEAD, itr)})
    return c


def test_cvm_ticker_map_and_filings(cvm):
    assert cvm.company("indt3")["cnpj"] == IND
    items = cvm.filings("INDT3", since=TODAY - timedelta(days=30))
    assert [i.title for i in items] == ["Fato Relevante: Aquisição de controle"]  # assembleia e antigo fora
    assert items[0].source.startswith("CVM - Fato Relevante") and items[0].url.startswith("https://www.rad.cvm.gov.br")
    r = run_skill("global_news_monitor", A.NOTICIAS, items=items)
    assert r.output["grupos"][0]["status"] == "confirmada"


def test_cvm_annual_financials_scale_and_versions(cvm):
    fa = cvm.annual_financials("INDT3")
    f = fa["financials"]
    assert f["receita"] == 1_000_000 and f["receita_anterior"] == 900_000  # MIL -> unidades; versão 0 ignorada
    assert f["ebitda"] == 300_000 and f["divida_bruta"] == 500_000 and f["caixa"] == 200_000 and f["capex"] == 80_000
    assert f["despesa_financeira"] == 50_000
    assert fa["acoes_em_circulacao"] == 100_000 and "milhares" in fa["acoes_verificacao"]  # 150 mil / LPA 1,5
    assert "CVM" in fa["financials_source"] and fa["instituicao_financeira"] is False
    bank = cvm.annual_financials("BNKT4")
    assert bank["instituicao_financeira"] is True and "ebitda" not in bank["financials"]
    assert bank["acoes_em_circulacao"] is None  # sem LPA, sem múltiplos


def test_cvm_share_scale_rejects_inconsistency():
    assert CVMClient._resolve_shares(100, 150_000, 1.5)[0] == 100_000
    assert CVMClient._resolve_shares(100_000, 150_000, 1.5)[0] == 100_000
    assert CVMClient._resolve_shares(7, 150_000, 1.5)[0] is None


def test_cvm_latest_quarter_matches_same_quarter_last_year(cvm):
    q = cvm.latest_quarter("INDT3")
    assert q["actual"] == {"receita": 300_000, "lucro_liquido": 60_000}
    assert q["year_ago"] == {"receita": 250_000, "lucro_liquido": 40_000}
    e = run_skill("earnings_analysis", A.FUNDAMENTALISTA, actual=q["actual"], year_ago=q["year_ago"], source=q["source"])
    assert e.output["crescimento"]["lucro_liquido"] == pytest.approx(0.5)


def test_fundamental_multiples_and_bank_mode():
    fin = {"receita": 1000, "ebitda": 300, "ebit": 250, "lucro_liquido": 150, "patrimonio_liquido": 1000, "divida_bruta": 500,
           "caixa": 200, "fco": 200, "capex": 80, "ativo_total": 2000}
    r = run_skill("fundamental_analysis", A.FUNDAMENTALISTA, financials=fin, financials_source="CVM", market_cap=1500)
    m = r.output["multiplos"]
    assert m["p_l"] == pytest.approx(10) and m["p_vp"] == pytest.approx(1.5) and m["ev_ebitda"] == pytest.approx(6)
    b = run_skill("fundamental_analysis", A.FUNDAMENTALISTA, financials={"receita": 500, "lucro_liquido": 80, "patrimonio_liquido": 400},
                  financials_source="CVM", financial_institution=True)
    assert "alavancagem_controlada" not in b.output["criterios"] and b.output["indicadores"]["divida_liquida_ebitda"] is None


RSS_GOOGLE = b"""<?xml version="1.0"?><rss><channel><title>t</title>
<item><title>Petrobras sobe 3% - Valor Econ\xc3\xb4mico</title><link>https://news.google.com/a</link>
<pubDate>Mon, 26 Oct 2026 01:00:00 GMT</pubDate><source url="https://valor.globo.com">Valor Econ\xc3\xb4mico</source></item>
<item><title>Sem data</title><link>https://x</link></item>
</channel></rss>"""


def test_parse_rss_requires_pubdate_and_extracts_outlet():
    items = parse_rss(RSS_GOOGLE, FEEDS["google"], utcnow())
    assert len(items) == 1
    assert items[0].title == "Petrobras sobe 3%" and items[0].source == "Valor Econômico (via Google Notícias)"


def test_mentions_avoids_common_words():
    mk = lambda t, b="": NewsItem(t, "x", utcnow(), utcnow(), body=b)  # noqa: E731
    assert mentions(mk("Vale cai 4% com minério"), ["VALE3", "Vale"])
    assert not mentions(mk("Prorrogação vale apenas para urnas"), ["VALE3", "Vale"])
    assert not mentions(mk("Vale a pena investir?"), ["VALE3", "Vale"])
    assert mentions(mk("Mercado hoje", "Petrobras anuncia dividendos"), ["PETR4", "Petrobras"])


def test_news_collector_filters_by_company_and_date(cvm, monkeypatch):
    now = datetime.now(timezone.utc)
    general = [NewsItem("Indústria Teste anuncia fábrica", "InfoMoney", now, now, url="https://www.infomoney.com.br/a"),
               NewsItem("Outro assunto", "InfoMoney", now, now), NewsItem("INDT3 antiga", "InfoMoney", now - timedelta(days=60), now)]
    n = NewsCollector(cvm=cvm, feeds=["infomoney"])
    monkeypatch.setattr(n, "_fetch", lambda feed, q=None: general)
    items = n.collect("INDT3", days=7)
    titles = [i.title for i in items]
    assert "Fato Relevante: Aquisição de controle" in titles and "Outro assunto" not in titles and "INDT3 antiga" not in titles


def test_pipeline_uses_cvm_and_news_collectors(svc, cvm, monkeypatch):
    from oraculo.orchestration.pipeline import deep_analysis

    n = NewsCollector(cvm=cvm, feeds=[])
    r = deep_analysis("INDT3", svc.hub, svc.memory, {}, collectors={"cvm": cvm, "news": n})
    fund = r["agentes"]["A07"]
    assert fund["stance"] != "evidência insuficiente"
    assert any("Trimestre encerrado" in e for e in fund["evidence"]) and any("P/L" in e for e in fund["evidence"])
    assert any("Aquisição de controle" in c for c in r["relatorio"]["catalisadores"])
    assert "CVM" in r["relatorio_markdown"]


def test_basic_auth_protects_api(svc, monkeypatch):
    from fastapi.testclient import TestClient

    from oraculo.api.app import create_app

    monkeypatch.setenv("ORACULO_PASSWORD", "s3gredo")
    c = TestClient(create_app(svc))
    assert c.get("/healthz").status_code == 200
    assert c.get("/api/agents").status_code == 401
    bad = base64.b64encode(b"oraculo:errada").decode()
    assert c.get("/api/agents", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    good = base64.b64encode(b"oraculo:s3gredo").decode()
    assert c.get("/api/agents", headers={"Authorization": f"Basic {good}"}).status_code == 200


def test_refuses_public_bind_without_password(monkeypatch):
    from oraculo.api import app as app_mod

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.delenv("ORACULO_PASSWORD", raising=False)
    with pytest.raises(SystemExit):
        app_mod.main()


def test_demo_never_uses_real_collectors(svc):
    assert svc.demo and svc.cvm is None and svc.news is None


