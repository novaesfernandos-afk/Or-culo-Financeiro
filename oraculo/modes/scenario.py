"""Modo 5 - Simulação de cenários, inclusive a partir de perguntas em português.

Exemplos aceitos:
- "O que aconteceria com minha carteira se a Selic subisse 2 pontos percentuais?"
- "E se o petróleo caísse 20%?"
- "E se o dólar chegasse a R$ 6,50?"
"""
from __future__ import annotations

import re
import unicodedata

from ..core import agent_ids as A
from ..skills.base import SkillContext, run_skill
from .base import FACTORS, Services, valued_portfolio

ALIASES = {
    "selic": "selic", "juros": "selic", "juro": "selic",
    "dolar": "dolar", "cambio": "dolar", "usd": "dolar",
    "petroleo": "petroleo", "brent": "petroleo",
    "minerio": "minerio", "minerio de ferro": "minerio",
    "ibovespa": "ibovespa", "bolsa": "ibovespa",
}
UP = ("subisse", "subir", "suba", "aumentasse", "aumentar", "alta", "elevasse", "elevar", "valorizasse", "disparasse")
DOWN = ("caisse", "cair", "caia", "recuasse", "recuar", "queda", "baixasse", "baixar", "desvalorizasse", "cortasse", "cortar")


def _plain(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c))


def _num(s: str) -> float:
    return float(s.replace(".", "").replace(",", ".")) if "," in s else float(s)


def parse_question(q: str) -> dict:
    """Converte uma pergunta em {fator: choque}. Levanta ValueError se não entender."""
    t = _plain(q)
    factor = next((ALIASES[k] for k in sorted(ALIASES, key=len, reverse=True) if re.search(rf"\b{k}\b", t)), None)
    if not factor:
        raise ValueError("Não identifiquei o fator (Selic, dólar, petróleo, minério, Ibovespa).")
    m = re.search(r"(?:chegasse|chegar|fosse|for|atingisse|atingir)\s+(?:a|para|em)?\s*r?\$?\s*([\d.,]+)", t)
    if m:
        return {factor: {"alvo": _num(m.group(1))}}
    m = re.search(r"([\d.,]+)\s*(pontos? percentua(?:l|is)|p\.?p\.?|%|por cento|pontos?)", t)
    if not m:
        raise ValueError("Não identifiquei a magnitude do choque.")
    size = _num(m.group(1))
    sign = -1 if any(w in t for w in DOWN) else 1
    if not any(w in t for w in UP + DOWN):
        raise ValueError("Não identifiquei a direção (subir/cair).")
    unit = m.group(2)
    if FACTORS[factor]["kind"] == "nivel" and "%" in unit and "selic" == factor:
        raise ValueError("Para a Selic, informe a variação em pontos percentuais.")
    return {factor: sign * size}


def run(svc: Services, params: dict) -> dict:
    scenario = params.get("cenario")
    parsed_from = None
    if not scenario and params.get("pergunta"):
        scenario = parse_question(params["pergunta"])
        parsed_from = params["pergunta"]
    if not scenario:
        raise ValueError("Informe 'cenario' ou 'pergunta'.")
    positions = params.get("posicoes") or {p["ticker"]: p["valor_usado"] for p in valued_portfolio(svc)[0]}
    if not positions:
        raise ValueError("Carteira vazia: cadastre posições ou informe 'posicoes'.")
    unknown = [f for f in scenario if f not in FACTORS]
    if unknown:
        raise ValueError(f"Fatores desconhecidos: {unknown}. Disponíveis: {sorted(FACTORS)}")
    keys = [f"price:{t}" for t in positions] + [FACTORS[f]["key"] for f in scenario]
    data, errors = svc.fetch_many(keys)
    factor_series = {f: data[FACTORS[f]["key"]] for f in scenario if FACTORS[f]["key"] in data}
    missing = [f for f in scenario if f not in factor_series]
    if missing:
        return {"modo": "simulacao_cenarios", "erro": f"Sem série histórica para {missing}: configure a fonte (ex.: CSV licenciado) "
                "em vez de estimar sem dados.", "falhas_coleta": errors}
    ctx = SkillContext(hub=svc.hub, memory=svc.memory)
    r = run_skill(
        "scenario_simulation", A.CARTEIRA, ctx, positions=positions, scenario=scenario,
        stock_prices={t: data[f"price:{t}"] for t in positions if f"price:{t}" in data},
        factor_series=factor_series, factor_kind={f: FACTORS[f]["kind"] for f in scenario},
    )
    return {
        "modo": "simulacao_cenarios",
        "pergunta": parsed_from,
        "cenario_interpretado": {f: {"choque": v, "unidade": FACTORS[f]["unit"], "serie": FACTORS[f]["key"]} for f, v in scenario.items()},
        "resultado": r.to_dict(),
        "falhas_coleta": errors,
    }
