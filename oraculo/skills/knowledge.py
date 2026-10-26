"""Bases de conhecimento estáticas e versionadas usadas pelas skills.

São hipóteses estruturais (canais de transmissão), não fatos de mercado: toda
conclusão derivada delas deve ser confirmada com dados (betas, eventos históricos).
"""
from __future__ import annotations

from urllib.parse import urlparse

KNOWLEDGE_VERSION = "2026.10.1"

# --- Confiabilidade de fontes -------------------------------------------------
TIER1_OFFICIAL = (
    "bcb.gov.br", "b3.com.br", "cvm.gov.br", "gov.br", "ibge.gov.br", "tesouro.gov.br",
    "federalreserve.gov", "ecb.europa.eu", "sec.gov", "bls.gov", "bea.gov", "imf.org",
    "worldbank.org", "opec.org", "eia.gov", "rad.cvm.gov.br",
)
TIER2_PRESS = (
    "reuters.com", "bloomberg.com", "valor.globo.com", "valorinternational.globo.com", "ft.com",
    "wsj.com", "estadao.com.br", "folha.uol.com.br", "infomoney.com.br", "apnews.com",
    "economist.com", "cnbc.com", "g1.globo.com", "exame.com", "neofeed.com.br",
)
SOURCE_NAME_TIERS = {
    "banco central": 1, "bcb": 1, "b3": 1, "cvm": 1, "ibge": 1, "tesouro nacional": 1, "copom": 1,
    "federal reserve": 1, "fato relevante": 1, "relações com investidores": 1,
    "reuters": 2, "bloomberg": 2, "valor econômico": 2, "valor": 2, "financial times": 2,
    "wall street journal": 2, "estadão": 2, "folha": 2, "infomoney": 2, "associated press": 2,
}


def source_tier(source: str, url: str | None = None) -> int:
    """1 = oficial, 2 = imprensa profissional, 3 = não classificada."""
    host = urlparse(url).netloc.lower() if url else ""
    if host:
        if any(host == d or host.endswith("." + d) for d in TIER1_OFFICIAL) or host.startswith("ri."):
            return 1
        if any(host == d or host.endswith("." + d) for d in TIER2_PRESS):
            return 2
    s = (source or "").lower()
    tiers = [t for name, t in SOURCE_NAME_TIERS.items() if name in s]
    return min(tiers) if tiers else 3


RUMOR_MARKERS = (
    "segundo fontes", "fontes próximas", "fontes ouvidas", "rumor", "especula", "estaria ", "estariam ",
    "poderia ", "poderiam ", "teria ", "teriam ", "não confirmad", "sem confirmação",
    "reportedly", "sources said", "people familiar", "is said to", "rumour", "rumor has it", "could be considering",
)
CONFIRMATION_MARKERS = (
    "fato relevante", "comunicado ao mercado", "comunicado oficial", "anunciou", "anuncia", "aprovou", "divulgou",
    "informou em nota", "decidiu", "announced", "filed", "approved", "said in a statement", "press release",
)

# --- Léxico de sentimento financeiro (PT/EN), inspirado em Loughran-McDonald --
POSITIVE_TERMS = {
    "lucro", "alta", "crescimento", "recorde", "supera", "superou", "melhora", "aprovação", "aprovado",
    "expansão", "valorização", "ganho", "positivo", "forte", "acima do esperado", "dividendos", "recompra",
    "upgrade", "elevação de rating", "recuperação", "otimismo", "aquisição", "demanda aquecida",
    "profit", "beat", "beats", "growth", "record", "strong", "upgrade", "gain", "rally", "surge", "improve",
}
NEGATIVE_TERMS = {
    "prejuízo", "queda", "perda", "recuo", "rebaixamento", "abaixo do esperado", "crise", "default", "calote",
    "recuperação judicial", "fraude", "investigação", "multa", "processo", "greve", "guerra", "sanção", "sanções",
    "inadimplência", "endividamento", "corte", "demissões", "risco", "incerteza", "volatilidade", "pessimismo",
    "loss", "miss", "misses", "downgrade", "decline", "drop", "plunge", "fraud", "lawsuit", "war", "sanction",
    "bankruptcy", "weak", "cut", "crisis",
}
NEGATIONS = {"não", "nem", "sem", "nunca", "not", "no", "never", "without"}

# --- Mapa de transmissão de eventos para setores da B3 ------------------------
# direção: +1 favorável, -1 desfavorável; intensidade 1-3; canal explicativo.
EVENT_SECTOR_MAP: dict[str, list[tuple[str, int, int, str]]] = {
    "alta_juros_brasil": [
        ("bancos", +1, 1, "margem financeira tende a subir no curto prazo, mas inadimplência pode crescer"),
        ("seguros", +1, 2, "receita financeira das reservas"),
        ("varejo", -1, 3, "crédito mais caro e consumo discricionário menor"),
        ("construcao_civil", -1, 3, "financiamento imobiliário mais caro"),
        ("shopping", -1, 2, "custo de capital e consumo"),
        ("saneamento", -1, 1, "custo de dívida e taxa de desconto"),
        ("tecnologia", -1, 2, "ações de longa duração sensíveis à taxa de desconto"),
        ("educacao", -1, 2, "financiamento estudantil e alavancagem"),
    ],
    "queda_juros_brasil": [
        ("varejo", +1, 3, "crédito mais barato"),
        ("construcao_civil", +1, 3, "financiamento imobiliário"),
        ("shopping", +1, 2, "taxa de desconto e consumo"),
        ("tecnologia", +1, 2, "taxa de desconto"),
        ("seguros", -1, 1, "receita financeira menor"),
    ],
    "alta_petroleo": [
        ("petroleo_gas", +1, 3, "preço de realização"),
        ("aviacao", -1, 3, "querosene de aviação é o principal custo"),
        ("logistica", -1, 2, "diesel"),
        ("quimica", -1, 2, "custo de matéria-prima"),
    ],
    "queda_petroleo": [
        ("petroleo_gas", -1, 3, "preço de realização"),
        ("aviacao", +1, 3, "custo de combustível"),
        ("logistica", +1, 1, "diesel"),
    ],
    "depreciacao_real": [
        ("papel_celulose", +1, 3, "receita dolarizada, custo em reais"),
        ("mineracao", +1, 2, "receita dolarizada"),
        ("frigorificos", +1, 2, "exportações"),
        ("agronegocio", +1, 2, "exportações"),
        ("petroleo_gas", +1, 1, "receita referenciada em dólar"),
        ("aviacao", -1, 3, "custos e dívida em dólar"),
        ("varejo", -1, 1, "importados e inflação"),
    ],
    "alta_minerio": [
        ("mineracao", +1, 3, "preço de realização"),
        ("siderurgia", -1, 1, "custo de insumo, parcialmente repassado"),
    ],
    "queda_minerio": [("mineracao", -1, 3, "preço de realização")],
    "guerra_ou_conflito": [
        ("petroleo_gas", +1, 2, "prêmio de risco no petróleo"),
        ("agronegocio", +1, 1, "preços de grãos e fertilizantes"),
        ("aviacao", -1, 3, "combustível e demanda"),
        ("varejo", -1, 1, "aversão a risco e inflação"),
        ("bancos", -1, 1, "aversão a risco e fluxo estrangeiro"),
    ],
    "crise_bancaria": [
        ("bancos", -1, 3, "contágio e custo de captação"),
        ("seguros", -1, 1, "ativos financeiros"),
        ("varejo", -1, 2, "restrição de crédito"),
        ("construcao_civil", -1, 2, "restrição de crédito"),
    ],
    "sancoes_comerciais": [
        ("agronegocio", -1, 2, "acesso a mercados"),
        ("frigorificos", -1, 2, "barreiras sanitárias e tarifárias"),
        ("siderurgia", -1, 2, "tarifas de aço"),
    ],
    "alta_celulose": [("papel_celulose", +1, 3, "preço de realização")],
    "alta_soja": [("agronegocio", +1, 2, "receita"), ("frigorificos", -1, 2, "custo de ração")],
}
