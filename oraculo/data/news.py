"""Coletor de notícias: fatos relevantes da CVM + feeds RSS de veículos profissionais.

Somente título, link, fonte e horário são usados (não o texto integral). Cada feed tem
termos de uso próprios; o Google Notícias permite apenas uso pessoal e não comercial e
por isso fica desligado por padrão (ative com ORACULO_NEWS_FEEDS=infomoney,valor,google).
"""
from __future__ import annotations

import html
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from ..core.types import NewsItem, utcnow
from .cvm import CVMClient
from ..core.costs import COSTS, throttle
from .providers.base import USER_AGENT, ProviderError


@dataclass(frozen=True)
class Feed:
    key: str
    name: str
    url: str  # pode conter {q}
    terms: str
    per_query: bool = False


FEEDS = {
    "infomoney": Feed("infomoney", "InfoMoney", "https://www.infomoney.com.br/feed/", "Termos de uso InfoMoney"),
    "infomoney_empresa": Feed(
        "infomoney_empresa", "InfoMoney", "https://www.infomoney.com.br/tudo-sobre/{q}/feed/", "Termos de uso InfoMoney", per_query=True,
    ),
    "valor": Feed("valor", "Valor Econômico", "https://valor.globo.com/rss/valor/", "Termos de uso Valor/Globo"),
    "valor_empresas": Feed("valor_empresas", "Valor Econômico", "https://valor.globo.com/rss/valor/empresas/", "Termos de uso Valor/Globo"),
    "google": Feed(
        "google", "Google Notícias", "https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419",
        "Somente uso pessoal e não comercial (termos do Google Notícias)", per_query=True,
    ),
}

DEFAULT_FEEDS = "infomoney,infomoney_empresa,valor,valor_empresas"

# Nomes populares usados para filtrar notícias gerais (o cadastro da CVM traz a razão social).
ALIASES = {
    "PETR3": ["Petrobras"], "PETR4": ["Petrobras"], "VALE3": ["Vale"], "ITUB4": ["Itaú", "Itau Unibanco"], "ITUB3": ["Itaú"],
    "BBDC4": ["Bradesco"], "BBDC3": ["Bradesco"], "BBAS3": ["Banco do Brasil"], "ABEV3": ["Ambev"], "WEGE3": ["WEG"],
    "B3SA3": ["B3"], "SUZB3": ["Suzano"], "PRIO3": ["PRIO", "PetroRio"], "RADL3": ["Raia Drogasil", "RD Saúde"],
    "EQTL3": ["Equatorial"], "GGBR4": ["Gerdau"], "LREN3": ["Renner"], "MGLU3": ["Magazine Luiza", "Magalu"], "RENT3": ["Localiza"],
}
STOPWORDS = {"S.A.", "SA", "S/A", "CIA", "COMPANHIA", "BRASILEIRO", "BRASIL", "DO", "DE", "DA", "E", "PARTICIPACOES", "PARTICIPAÇÕES", "HOLDING"}


def _text(el: ET.Element | None) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", el.text or "")).strip() if el is not None else ""


def parse_rss(xml_bytes: bytes, feed: Feed, retrieved_at: datetime) -> list[NewsItem]:
    root = ET.fromstring(xml_bytes)
    items = []
    for it in root.iter("item"):
        title = _text(it.find("title"))
        link = _text(it.find("link"))
        pub = it.find("pubDate")
        if not title or pub is None or not pub.text:
            continue  # sem horário de publicação não entra (regra 2)
        try:
            published = parsedate_to_datetime(pub.text.strip())
        except (TypeError, ValueError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        source = feed.name
        src = it.find("source")
        if feed.key == "google" and src is not None and src.text:
            outlet = src.text.strip()
            source = f"{outlet} (via Google Notícias)"
            title = re.sub(rf"\s+-\s+{re.escape(outlet)}$", "", title)
        items.append(NewsItem(title=title, source=source, published_at=published, retrieved_at=retrieved_at, url=link,
                              body=_text(it.find("description"))[:500]))
    return items


def slug(name: str) -> str:
    import unicodedata

    t = "".join(c for c in unicodedata.normalize("NFKD", name.lower()) if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", t).strip("-")


def keywords(ticker: str, company_name: str | None = None) -> list[str]:
    kws = [ticker.upper()] + ALIASES.get(ticker.upper(), [])
    if company_name and ticker.upper() not in ALIASES:
        words = [w for w in re.split(r"\s+", company_name.upper()) if w not in STOPWORDS and len(w) > 2]
        if words:
            kws.append(words[0].title())
    return kws


AMBIGUOUS = {"Vale", "B3", "PRIO", "WEG"}  # também são palavras comuns: exigem menção no título


def mentions(item: NewsItem, kws: list[str]) -> bool:
    """Nomes próprios: casamento exato de maiúsculas/minúsculas, como palavra inteira."""
    for k in kws:
        pat = rf"(?<![\w]){re.escape(k)}(?![\w])"
        text = item.title if k in AMBIGUOUS else f"{item.title} {item.body}"
        for m in re.finditer(pat, text):
            if k == "Vale" and re.match(r"\s+(a|o|lembrar|ressaltar|destacar|dizer|mencionar)\b", text[m.end():]):
                continue
            return True
    return False


@dataclass
class NewsCollector:
    cvm: CVMClient | None = None
    feeds: list[str] = field(default_factory=lambda: [f for f in os.environ.get("ORACULO_NEWS_FEEDS", DEFAULT_FEEDS).split(",") if f])
    ttl_seconds: float = 900.0
    _cache: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def _fetch(self, feed: Feed, q: str | None = None) -> list[NewsItem]:
        url = feed.url.format(q=urllib.parse.quote_plus(q or ""))
        hit = self._cache.get(url)
        if hit and time.time() - hit[0] < self.ttl_seconds:
            return hit[1]
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        t0 = time.time()
        try:
            COSTS.check(feed.name)
            throttle(url)
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            COSTS.record("dados", feed.name, time.time() - t0)
        except Exception as exc:  # noqa: BLE001
            COSTS.record("dados", feed.name, time.time() - t0, ok=False)
            raise ProviderError(f"Feed {feed.name} indisponível: {exc}", permanent=getattr(exc, "code", None) == 404) from exc
        items = parse_rss(data, feed, utcnow())
        self._cache[url] = (time.time(), items)
        return items

    def collect(self, ticker: str, days: int = 7) -> list[NewsItem]:
        """Notícias e comunicados oficiais do ativo nos últimos `days` dias. Falhas de fonte ficam em `errors`."""
        self.errors = []
        ticker = ticker.upper()
        since = datetime.now(timezone.utc) - timedelta(days=days)
        out: list[NewsItem] = []
        name = None
        if self.cvm is not None:
            try:
                name = self.cvm.company(ticker)["nome"]
                out += self.cvm.filings(ticker, since=date.today() - timedelta(days=days))
            except ProviderError as exc:
                self.errors.append(f"CVM: {exc}")
        kws = keywords(ticker, name)
        for key in self.feeds:
            feed = FEEDS.get(key.strip())
            if not feed:
                self.errors.append(f"Feed desconhecido: {key}")
                continue
            q = None
            if feed.key == "infomoney_empresa":
                if len(kws) < 2:
                    continue
                q = slug(kws[1])
            elif feed.per_query:
                q = " ".join(kws[:2])
            try:
                items = self._fetch(feed, q)
            except ProviderError as exc:
                if not (feed.key == "infomoney_empresa" and exc.permanent):  # página de assunto inexistente
                    self.errors.append(str(exc))
                continue
            for it in items:
                if it.published_at >= since and (feed.key == "google" or mentions(it, kws)):
                    it.tickers = [ticker]
                    out.append(it)
        out.sort(key=lambda n: n.published_at, reverse=True)
        return out

    def licenses(self) -> list[str]:
        return [f"{FEEDS[k].name}: {FEEDS[k].terms}" for k in self.feeds if k in FEEDS] + (["CVM: dados abertos"] if self.cvm else [])
