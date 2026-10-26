"""Dados abertos da CVM: cadastro (FCA), fatos relevantes (IPE), balanços anuais (DFP) e trimestrais (ITR).

Portal: https://dados.cvm.gov.br/ (Licença: dados abertos do governo federal).
Os arquivos são baixados uma vez por dia para um cache local e lidos sem extração completa.
"""
from __future__ import annotations

import csv
import io
import os
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from ..core.types import NewsItem, utcnow
from ..core.costs import COSTS, throttle
from .providers.base import USER_AGENT, ProviderError

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
SOURCE = "CVM - Dados Abertos"
LICENSE = "Dados abertos CVM"

# Contas padronizadas (plano de contas CVM, demonstrações consolidadas de empresas não financeiras).
ACCOUNTS = {
    "DRE": {"3.01": "receita", "3.05": "ebit", "3.06.02": "despesa_financeira"},
    "BPA": {"1": "ativo_total", "1.01.01": "caixa", "1.01.02": "aplicacoes"},
    "BPP": {"2.01.04": "divida_cp", "2.02.01": "divida_lp", "2.03": "patrimonio_liquido"},
    "DFC_MI": {"6.01": "fco"},
}
NEWS_CATEGORIES = ("Fato Relevante", "Comunicado ao Mercado", "Aviso aos Acionistas")


@dataclass
class CVMClient:
    cache_dir: Path = field(default_factory=lambda: Path(os.environ.get("ORACULO_CACHE", "data/cache")) / "cvm")
    max_age_hours: float = 24.0
    base_url: str = BASE
    _tables: dict = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # --- download e leitura ----------------------------------------------------
    def _zip(self, doc: str, year: int) -> Path:
        name = f"{doc.lower()}_cia_aberta_{year}.zip"
        path = self.cache_dir / name
        with self._lock:
            fresh = path.exists() and time.time() - path.stat().st_mtime < self.max_age_hours * 3600
            if not fresh:
                url = f"{self.base_url}/{doc}/DADOS/{name}"
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                t0 = time.time()
                try:
                    COSTS.check("CVM")
                    throttle(url)
                    with urllib.request.urlopen(req, timeout=300) as r:
                        data = r.read()
                    COSTS.record("dados", "CVM", time.time() - t0, units=len(data) / 1e6)
                except Exception as exc:  # noqa: BLE001
                    COSTS.record("dados", "CVM", time.time() - t0, ok=False)
                    if path.exists():  # usa a cópia antiga; o horário real fica registrado na fonte
                        return path
                    code = getattr(exc, "code", None)
                    raise ProviderError(f"CVM indisponível ({url}): {exc}", permanent=code == 404) from exc
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)
        return path

    def downloaded_at(self, doc: str, year: int) -> datetime:
        p = self.cache_dir / f"{doc.lower()}_cia_aberta_{year}.zip"
        return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc) if p.exists() else utcnow()

    def _rows(self, doc: str, year: int, member: str) -> Iterator[dict]:
        with zipfile.ZipFile(self._zip(doc, year)) as z:
            with z.open(member) as fh:
                yield from csv.DictReader(io.TextIOWrapper(fh, encoding="latin-1", newline=""), delimiter=";")

    def _years(self, doc: str, ref: date | None = None) -> list[int]:
        y = (ref or date.today()).year
        return [y, y - 1] if doc != "DFP" else [y - 1, y - 2]

    # --- cadastro --------------------------------------------------------------
    def ticker_map(self) -> dict[str, dict]:
        with self._lock:
            if "tickers" in self._tables:
                return self._tables["tickers"]
            out: dict[str, dict] = {}
            for y in self._years("FCA"):
                try:
                    rows = list(self._rows("FCA", y, f"fca_cia_aberta_valor_mobiliario_{y}.csv"))
                except ProviderError:
                    continue
                for r in rows:
                    t = (r.get("Codigo_Negociacao") or "").strip().upper()
                    if t and t not in out and not r.get("Data_Fim_Negociacao"):
                        out[t] = {"cnpj": r["CNPJ_Companhia"], "nome": r["Nome_Empresarial"], "segmento": r.get("Segmento")}
                if out:
                    break
            if not out:
                raise ProviderError("Cadastro de valores mobiliários da CVM indisponível.")
            self._tables["tickers"] = out
            return out

    def company(self, ticker: str) -> dict:
        info = self.ticker_map().get(ticker.upper())
        if not info:
            raise ProviderError(f"Ticker {ticker} não encontrado no cadastro da CVM.", permanent=True)
        return info

    # --- fatos relevantes e comunicados ----------------------------------------
    def filings(self, ticker: str, since: date | None = None, categories: tuple[str, ...] = NEWS_CATEGORIES) -> list[NewsItem]:
        info = self.company(ticker)
        since = since or (date.today() - timedelta(days=30))
        items = []
        for y in sorted({since.year, date.today().year}):
            try:
                rows = self._ipe(y)
            except ProviderError:
                continue
            retrieved = self.downloaded_at("IPE", y)
            for r in rows.get(info["cnpj"], []):
                if r["Categoria"] not in categories:
                    continue
                delivered = date.fromisoformat(r["Data_Entrega"])
                if delivered < since:
                    continue
                items.append(
                    NewsItem(
                        title=f"{r['Categoria']}: {r['Assunto']}".strip(),
                        source=f"CVM - {r['Categoria']} ({r['Nome_Companhia'].strip()})",
                        published_at=datetime.combine(delivered, datetime.min.time(), tzinfo=timezone.utc),
                        retrieved_at=retrieved,
                        url=r.get("Link_Download"),
                        body=r.get("Assunto", ""),
                        tickers=[ticker.upper()],
                    )
                )
        items.sort(key=lambda n: n.published_at, reverse=True)
        return items

    def _ipe(self, year: int) -> dict[str, list[dict]]:
        key = ("IPE", year)
        with self._lock:
            if key not in self._tables:
                by_cnpj: dict[str, list[dict]] = {}
                for r in self._rows("IPE", year, f"ipe_cia_aberta_{year}.csv"):
                    if r["Categoria"] in NEWS_CATEGORIES:
                        by_cnpj.setdefault(r["CNPJ_Companhia"], []).append(r)
                self._tables[key] = by_cnpj
            return self._tables[key]

    # --- demonstrações financeiras ---------------------------------------------
    def _statement(self, doc: str, year: int, kind: str) -> dict[str, list[dict]]:
        key = (doc, year, kind)
        with self._lock:
            if key in self._tables:
                return self._tables[key]
            codes = ACCOUNTS.get(kind, {})
            by_cnpj: dict[str, list[dict]] = {}
            member = f"{doc.lower()}_cia_aberta_{kind}_con_{year}.csv"
            for r in self._rows(doc, year, member):
                code, desc = r["CD_CONTA"], r["DS_CONTA"].lower()
                keep = code in codes
                keep |= kind == "DRE" and code.count(".") == 1 and "lucro/prejuízo consolidado do período" in desc
                keep |= kind == "DRE" and code == "3.01" and "intermediação" in desc
                keep |= kind == "DRE" and code.startswith("3.99.01.")
                keep |= kind == "DRE" and "atribuído a sócios da empresa controladora" in desc
                keep |= kind == "DFC_MI" and code.startswith("6.01.01") and "deprecia" in desc
                keep |= kind == "DFC_MI" and code.startswith("6.02.") and "aquisi" in desc and ("imobilizado" in desc or "intangív" in desc)
                if keep:
                    by_cnpj.setdefault(r["CNPJ_CIA"], []).append(r)
            self._tables[key] = by_cnpj
            return by_cnpj

    def _shares(self, doc: str, year: int, cnpj: str) -> float | None:
        key = (doc, year, "capital")
        with self._lock:
            if key not in self._tables:
                t = {}
                for r in self._rows(doc, year, f"{doc.lower()}_cia_aberta_composicao_capital_{year}.csv"):
                    t[r["CNPJ_CIA"]] = r
                self._tables[key] = t
        r = self._tables[key].get(cnpj)
        if not r:
            return None
        total = float(r["QT_ACAO_TOTAL_CAP_INTEGR"] or 0) - float(r["QT_ACAO_TOTAL_TESOURO"] or 0)
        return total or None

    @staticmethod
    def _resolve_shares(raw: float | None, ni_controller: float | None, eps: float | None) -> tuple[float | None, str]:
        """A CVM recebe a quantidade de ações em unidades ou em milhares, conforme a empresa.

        A escala é confirmada pelo lucro por ação do próprio balanço (lucro da controladora / LPA).
        Sem confirmação, a quantidade é descartada (múltiplos não são calculados).
        """
        if not raw:
            return None, "quantidade de ações não informada"
        if not (ni_controller and eps):
            return None, "sem LPA para confirmar a escala da quantidade de ações"
        implied = ni_controller / eps
        for cand, label in ((raw, "unidades"), (raw * 1000, "milhares")):
            if 1 / 1.5 <= cand / implied <= 1.5:
                return cand, f"confirmada pelo LPA (informada em {label})"
        return None, f"quantidade informada ({raw:,.0f}) incompatível com lucro/LPA ({implied:,.0f})"

    @staticmethod
    def _latest_version(rows: list[dict]) -> list[dict]:
        if not rows:
            return rows
        ref = max(r["DT_REFER"] for r in rows)
        rows = [r for r in rows if r["DT_REFER"] == ref]
        v = max(int(r["VERSAO"]) for r in rows)
        return [r for r in rows if int(r["VERSAO"]) == v]

    @staticmethod
    def _value(r: dict) -> float:
        mult = 1000.0 if r.get("ESCALA_MOEDA", "").upper() == "MIL" else 1.0
        return float(r["VL_CONTA"]) * mult

    def annual_financials(self, ticker: str) -> dict:
        """Demonstrações consolidadas do último DFP disponível, no formato da skill fundamental_analysis."""
        info = self.company(ticker)
        cnpj = info["cnpj"]
        for year in self._years("DFP"):
            try:
                tables = {k: self._latest_version(self._statement("DFP", year, k).get(cnpj, [])) for k in ACCOUNTS}
            except ProviderError:
                continue
            if not tables["DRE"]:
                continue
            return self._assemble(info, ticker, tables, "DFP", year)
        raise ProviderError(f"Sem DFP consolidado na CVM para {ticker}.", permanent=True)

    def _assemble(self, info: dict, ticker: str, t: dict, doc: str, year: int) -> dict:
        cur: dict[str, float] = {}
        prev: dict[str, float] = {}
        da = capex = 0.0
        financial_institution = False
        for kind, rows in t.items():
            for r in rows:
                last = r["ORDEM_EXERC"].upper().startswith("Ú")
                if kind in ("DRE", "DFC_MI") and doc == "ITR":
                    continue
                code, desc = r["CD_CONTA"], r["DS_CONTA"].lower()
                target = cur if last else prev
                if code in ACCOUNTS[kind]:
                    target[ACCOUNTS[kind][code]] = self._value(r)
                if kind == "DRE" and "lucro/prejuízo consolidado do período" in desc:
                    target["lucro_liquido"] = self._value(r)
                if kind == "DRE" and code == "3.01" and "intermediação" in desc:
                    financial_institution = True
                if last and kind == "DRE" and code.startswith("3.99.01.") and float(r["VL_CONTA"]) > 0 and "lpa" not in cur:
                    cur["lpa"] = float(r["VL_CONTA"])  # lucro básico por ação (R$, sem escala)
                if last and kind == "DRE" and "atribuído a sócios da empresa controladora" in desc:
                    cur["lucro_controladora"] = self._value(r)
                if last and kind == "DFC_MI" and "deprecia" in desc and code.startswith("6.01.01"):
                    da += self._value(r)
                if last and kind == "DFC_MI" and code.startswith("6.02."):
                    capex += -self._value(r)
        ref = t["DRE"][0]
        fin = {
            "receita": cur.get("receita"),
            "ebit": None if financial_institution else cur.get("ebit"),
            "ebitda": (cur["ebit"] + da) if (not financial_institution and "ebit" in cur and da) else None,
            "lucro_liquido": cur.get("lucro_liquido"),
            "patrimonio_liquido": cur.get("patrimonio_liquido"),
            "ativo_total": cur.get("ativo_total"),
            "divida_bruta": None if financial_institution else (cur.get("divida_cp", 0.0) + cur.get("divida_lp", 0.0)),
            "caixa": None if financial_institution else (cur.get("caixa", 0.0) + cur.get("aplicacoes", 0.0)),
            "fco": cur.get("fco"),
            "capex": capex or None,
            "despesa_financeira": -cur["despesa_financeira"] if "despesa_financeira" in cur else None,
            "receita_anterior": prev.get("receita"),
            "lucro_anterior": prev.get("lucro_liquido"),
        }
        shares, shares_note = self._resolve_shares(self._shares(doc, year, info["cnpj"]), cur.get("lucro_controladora"), cur.get("lpa"))
        return {
            "ticker": ticker.upper(),
            "empresa": ref["DENOM_CIA"],
            "data_referencia": ref["DT_REFER"],
            "documento": f"{doc} {year}",
            "versao": ref["VERSAO"],
            "instituicao_financeira": financial_institution,
            "acoes_em_circulacao": shares,
            "acoes_verificacao": shares_note,
            "financials": {k: v for k, v in fin.items() if v is not None},
            "financials_source": (
                f"{SOURCE} - {doc} {year} consolidado, {ref['DENOM_CIA']} (ref. {ref['DT_REFER']}, versão {ref['VERSAO']}; "
                f"baixado em {self.downloaded_at(doc, year).strftime('%Y-%m-%d %H:%M UTC')})"
            ),
        }

    def latest_quarter(self, ticker: str) -> dict:
        """Último trimestre (3 meses) do ITR vs. mesmo trimestre do ano anterior, para earnings_analysis."""
        info = self.company(ticker)
        cnpj = info["cnpj"]
        for year in self._years("ITR"):
            try:
                rows = self._latest_version(self._statement("ITR", year, "DRE").get(cnpj, []))
            except ProviderError:
                continue
            if not rows:
                continue
            ref = rows[0]["DT_REFER"]
            ref_d = date.fromisoformat(ref)
            q_start = date(ref_d.year, ref_d.month - 2, 1)
            quarter_start, year_ago_start = q_start.isoformat(), q_start.replace(year=q_start.year - 1).isoformat()
            act: dict[str, float] = {}
            ago: dict[str, float] = {}
            bank = any(r["CD_CONTA"] == "3.01" and "intermediação" in r["DS_CONTA"].lower() for r in rows)
            for r in rows:
                ini = r.get("DT_INI_EXERC", "")
                is_last = r["ORDEM_EXERC"].upper().startswith("Ú")
                three_months = (date.fromisoformat(r["DT_FIM_EXERC"]) - date.fromisoformat(ini)).days < 100
                if not three_months or ini != (quarter_start if is_last else year_ago_start):
                    continue
                tgt = act if is_last else ago
                code, desc = r["CD_CONTA"], r["DS_CONTA"].lower()
                if code == "3.01":
                    tgt["receita"] = self._value(r)
                elif code == "3.05" and not bank:
                    tgt["ebit"] = self._value(r)
                elif "lucro/prejuízo consolidado do período" in desc:
                    tgt["lucro_liquido"] = self._value(r)
            if act.get("receita") and ago.get("receita"):
                return {
                    "actual": act,
                    "year_ago": ago,
                    "trimestre_encerrado_em": ref,
                    "source": f"{SOURCE} - ITR {ref} consolidado, {rows[0]['DENOM_CIA']} (versão {rows[0]['VERSAO']})",
                }
        raise ProviderError(f"Sem ITR trimestral comparável na CVM para {ticker}.", permanent=True)
