"""Comitê de Investimentos: três membros independentes analisam a mesma ação.

- Otimista: procura argumentos favoráveis.
- Pessimista: procura riscos e argumentos contrários.
- Auditor: verifica as evidências de cada argumento de forma independente
  (recalcula o valor a partir da evidência original, checa procedência,
  atraso, dado sintético e status do modelo) e pondera apenas o que se sustenta.

Os membros recebem cópias isoladas das evidências e não veem a tese um do
outro, o que reduz o risco de todos concordarem com a tese inicial. O comitê
apresenta as divergências e não força recomendação.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..skills.base import SkillResult


@dataclass(frozen=True)
class Evidence:
    key: str
    skill_id: str
    status: str
    output: dict
    synthetic: bool
    stale: bool
    model_status: str | None = None


def build_evidence(results: dict[str, SkillResult]) -> dict[str, Evidence]:
    ev = {}
    for key, r in results.items():
        ev[key] = Evidence(
            key=key,
            skill_id=r.skill_id,
            status=r.status.value,
            output=copy.deepcopy(r.output),
            synthetic=any(s.synthetic for s in r.sources),
            stale=any("ATRASADO" in w for w in r.warnings),
            model_status=r.output.get("status_modelo") if isinstance(r.output, dict) else None,
        )
    return ev


def _find(ev: dict[str, Evidence], skill_id: str) -> Evidence | None:
    for e in ev.values():
        if e.skill_id == skill_id and e.status in ("ok", "degradado"):
            return e
    return None


def _get(o: Any, path: str) -> Any:
    for p in path.split("."):
        if o is None:
            return None
        o = o.get(p) if isinstance(o, dict) else None
    return o


@dataclass(frozen=True)
class Rule:
    id: str
    side: str  # "favoravel" | "contraria"
    theme: str
    skill_id: str
    path: str
    predicate: Callable[[Any], bool]
    text: Callable[[Any], str]
    strength: float
    model_based: bool = False


def _r(id, side, theme, skill_id, path, pred, text, strength, model_based=False) -> Rule:  # noqa: A002
    return Rule(id, side, theme, skill_id, path, pred, text, strength, model_based)


BULL_RULES: tuple[Rule, ...] = (
    _r("tec_alta", "favoravel", "tendência", "technical_analysis", "tendencia", lambda v: v == "alta", lambda v: "Tendência técnica de alta (preço acima das médias, MMS50 ascendente)", 1.0),
    _r("macd_pos", "favoravel", "momentum", "technical_analysis", "macd.histograma", lambda v: v is not None and v > 0, lambda v: f"Momentum positivo (histograma MACD {v:+.3f})", 0.5),
    _r("qualidade", "favoravel", "fundamentos", "fundamental_analysis", "pontuacao_qualidade", lambda v: v is not None and v >= 4, lambda v: f"Qualidade fundamentalista elevada ({v}/6)", 1.5),
    _r("upside", "favoravel", "valuation", "valuation_engine", "upside_dcf", lambda v: v is not None and v > 0.15, lambda v: f"Valuation por DCF indica {v:+.0%} vs. preço atual", 1.5),
    _r("setor_top", "favoravel", "posição setorial", "sector_comparison", "pontuacao_composta", lambda v: v is not None and v >= 60, lambda v: f"Bem posicionada entre os pares ({v:.0f}/100)", 1.0),
    _r("sent_pos", "favoravel", "notícias", "news_sentiment_analysis", "escore_agregado", lambda v: v is not None and v > 0.2, lambda v: f"Noticiário com tom positivo ({v:+.2f})", 0.5),
    _r("prob_alta", "favoravel", "modelo preditivo", "price_direction_forecast", "vantagem_sobre_base", lambda v: v is not None and v > 0.05, lambda v: f"Probabilidade de alta {v*100:+.0f} p.p. acima da taxa-base", 1.5, True),
    _r("lucro_tri_alta", "favoravel", "resultado trimestral", "earnings_analysis", "crescimento.lucro_liquido", lambda v: v is not None and v > 0.10, lambda v: f"Lucro do último trimestre {v:+.0%} vs. mesmo trimestre do ano anterior", 1.0),
    _r("sazon_pos", "favoravel", "sazonalidade", "seasonality_detection", "padroes_confirmados", lambda v: bool(v), lambda v: f"Padrões sazonais confirmados fora da amostra: {', '.join(v)}", 0.3),
)

BEAR_RULES: tuple[Rule, ...] = (
    _r("tec_baixa", "contraria", "tendência", "technical_analysis", "tendencia", lambda v: v == "baixa", lambda v: "Tendência técnica de baixa (preço abaixo das médias, MMS50 descendente)", 1.0),
    _r("sobrecompra", "contraria", "momentum", "technical_analysis", "rsi14", lambda v: v is not None and v > 70, lambda v: f"RSI {v:.0f}: ativo sobrecomprado", 0.5),
    _r("vol_alta", "contraria", "risco", "technical_analysis", "volatilidade_21d_anual", lambda v: v is not None and v > 0.45, lambda v: f"Volatilidade elevada ({v:.0%} a.a.)", 1.0),
    _r("drawdown", "contraria", "risco", "portfolio_risk_analysis", "risco.max_drawdown_historico", lambda v: v is not None and v < -0.4, lambda v: f"Histórico de queda máxima de {v:.0%}", 0.75),
    _r("cauda", "contraria", "risco", "return_distribution_forecast", "quantis.p05", lambda v: v is not None and v < -0.10, lambda v: f"Cenário adverso (p05) de {v:.0%} no horizonte", 0.75, True),
    _r("alavancagem", "contraria", "fundamentos", "fundamental_analysis", "indicadores.divida_liquida_ebitda", lambda v: v is not None and v > 3.0, lambda v: f"Alavancagem elevada ({v:.1f}x dívida líquida/EBITDA)", 1.5),
    _r("accruals", "contraria", "fundamentos", "fundamental_analysis", "indicadores.accruals", lambda v: v is not None and v > 0.10, lambda v: f"Lucro pouco convertido em caixa (accruals {v:.0%})", 1.0),
    _r("lucro_tri_queda", "contraria", "resultado trimestral", "earnings_analysis", "crescimento.lucro_liquido", lambda v: v is not None and v < -0.10, lambda v: f"Lucro do último trimestre {v:+.0%} vs. mesmo trimestre do ano anterior", 1.0),
    _r("caro", "contraria", "valuation", "valuation_engine", "upside_dcf", lambda v: v is not None and v < -0.15, lambda v: f"Valuation por DCF {v:+.0%} vs. preço atual", 1.5),
    _r("setor_fraco", "contraria", "posição setorial", "sector_comparison", "pontuacao_composta", lambda v: v is not None and v <= 40, lambda v: f"Posição fraca entre os pares ({v:.0f}/100)", 1.0),
    _r("sent_neg", "contraria", "notícias", "news_sentiment_analysis", "escore_agregado", lambda v: v is not None and v < -0.2, lambda v: f"Noticiário com tom negativo ({v:+.2f})", 0.5),
    _r("rumores", "contraria", "notícias", "global_news_monitor", "rumores", lambda v: v is not None and v > 0, lambda v: f"{v} rumor(es) não confirmados em circulação", 0.3),
    _r("geo", "contraria", "geopolítica", "geopolitical_risk_scoring", "indice", lambda v: v is not None and v >= 50, lambda v: f"Risco geopolítico elevado ({v:.0f}/100)", 0.75),
    _r("prob_baixa", "contraria", "modelo preditivo", "price_direction_forecast", "vantagem_sobre_base", lambda v: v is not None and v < -0.05, lambda v: f"Probabilidade de alta {v*100:+.0f} p.p. abaixo da taxa-base", 1.5, True),
    _r("juros", "contraria", "macro", "central_bank_analysis", "fase", lambda v: v == "aperto", lambda v: "Ciclo de aperto monetário em curso", 0.5),
)


@dataclass
class Argument:
    rule_id: str
    side: str
    theme: str
    claim: str
    evidence_key: str
    claimed_value: Any
    strength: float


@dataclass
class Thesis:
    member: str
    focus: str
    arguments: list[Argument] = field(default_factory=list)


def _build_thesis(member: str, focus: str, rules: tuple[Rule, ...], ev: dict[str, Evidence]) -> Thesis:
    th = Thesis(member, focus)
    for rule in rules:
        e = _find(ev, rule.skill_id)
        if e is None:
            continue
        v = _get(e.output, rule.path)
        try:
            if rule.predicate(v):
                th.arguments.append(Argument(rule.id, rule.side, rule.theme, rule.text(v), e.key, v, rule.strength))
        except (TypeError, ValueError):
            continue
    return th


def otimista(ev: dict[str, Evidence]) -> Thesis:
    return _build_thesis("Otimista", "Valuation, crescimento e momentum", BULL_RULES, ev)


def pessimista(ev: dict[str, Evidence]) -> Thesis:
    return _build_thesis("Pessimista", "Riscos, juros, alavancagem e cauda", BEAR_RULES, ev)


RULES_BY_ID = {r.id: r for r in BULL_RULES + BEAR_RULES}

# Famílias de dados independentes. Argumentos da mesma família (ex.: vários derivados do preço)
# não contam como confirmação independente.
EVIDENCE_FAMILY = {
    "technical_analysis": "preços", "return_distribution_forecast": "preços", "price_direction_forecast": "preços",
    "portfolio_risk_analysis": "preços", "seasonality_detection": "preços", "volatility_forecast": "preços",
    "fundamental_analysis": "fundamentos", "valuation_engine": "fundamentos", "earnings_analysis": "fundamentos",
    "sector_comparison": "setor", "news_sentiment_analysis": "notícias", "global_news_monitor": "notícias",
    "central_bank_analysis": "macro", "brazil_macro_monitor": "macro", "geopolitical_risk_scoring": "geopolítica",
    "commodity_exposure": "commodities",
}


def auditor(ev: dict[str, Evidence], theses: list[Thesis], allow_synthetic: bool) -> dict:
    """Verifica cada argumento contra a evidência original e atribui peso."""
    reviews = []
    for th in theses:
        for a in th.arguments:
            rule = RULES_BY_ID[a.rule_id]
            e = ev.get(a.evidence_key)
            verdict, weight, notes = "aceito", a.strength, []
            if e is None or e.status not in ("ok", "degradado"):
                verdict, weight = "rejeitado", 0.0
                notes.append("evidência inexistente ou com falha")
            else:
                recomputed = _get(e.output, rule.path)
                if recomputed != a.claimed_value:
                    verdict, weight = "rejeitado", 0.0
                    notes.append("valor citado difere da evidência original")
                elif not rule.predicate(recomputed):
                    verdict, weight = "rejeitado", 0.0
                    notes.append("evidência não sustenta a afirmação")
                if e.skill_id == "price_direction_forecast" and (e.model_status != "producao" or not e.output.get("calibrada")):
                    verdict, weight = "rejeitado", 0.0
                    notes.append(f"probabilidade sem modelo de produção calibrado ({e.model_status})")
                if verdict == "aceito" and e.synthetic:
                    if allow_synthetic:
                        notes.append("DADO SINTÉTICO (demonstração)")
                    else:
                        verdict, weight = "rejeitado", 0.0
                        notes.append("dado sintético não sustenta conclusões")
                if verdict == "aceito" and e.stale:
                    weight *= 0.5
                    verdict = "aceito com ressalva"
                    notes.append("dado atrasado: peso reduzido à metade")
                if verdict == "aceito" and e.status == "degradado":
                    notes.append("skill executada com ressalvas")
            reviews.append({"membro": th.member, "argumento": a.claim, "tema": a.theme, "lado": a.side, "evidencia": a.evidence_key,
                            "familia": EVIDENCE_FAMILY.get(rule.skill_id, rule.skill_id), "veredito": verdict, "peso": weight,
                            "observacoes": notes})
    return {"revisoes": reviews}


def run_committee(
    results: dict[str, SkillResult],
    allow_synthetic: bool = False,
    min_accepted: int = 3,
    balance_threshold: float = 0.25,
    min_families: int = 2,
) -> dict:
    # cada membro recebe sua própria cópia das evidências, em paralelo e sem acesso às demais teses
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_bull = pool.submit(otimista, build_evidence(results))
        f_bear = pool.submit(pessimista, build_evidence(results))
        bull, bear = f_bull.result(), f_bear.result()
    audit = auditor(build_evidence(results), [bull, bear], allow_synthetic)
    acc = [r for r in audit["revisoes"] if r["peso"] > 0]
    s_bull = sum(r["peso"] for r in acc if r["lado"] == "favoravel")
    s_bear = sum(r["peso"] for r in acc if r["lado"] == "contraria")
    total = s_bull + s_bear
    balance = (s_bull - s_bear) / total if total else 0.0
    if len(acc) < min_accepted:
        conclusion = "evidência insuficiente"
        text = "Evidências validadas insuficientes para qualquer inclinação. O sistema não emite recomendação."
    elif abs(balance) < balance_threshold:
        conclusion = "inconclusivo"
        text = "Cenário inconclusivo: argumentos favoráveis e contrários com peso semelhante. O sistema apresenta as divergências e não força uma recomendação."
    elif len({r["familia"] for r in acc if r["lado"] == ("favoravel" if balance > 0 else "contraria")}) < min_families:
        side = "favorável" if balance > 0 else "desfavorável"
        fam = sorted({r["familia"] for r in acc if r["lado"] == ("favoravel" if balance > 0 else "contraria")})
        conclusion = "inconclusivo"
        text = (
            f"Cenário inconclusivo: os argumentos de lado {side} vêm de uma única família de dados ({', '.join(fam)}), "
            f"sem confirmação independente (mínimo {min_families} famílias). O sistema não força uma recomendação."
        )
    else:
        conclusion = "inclinação favorável" if balance > 0 else "inclinação desfavorável"
        text = (
            f"Evidências validadas pendem para uma {conclusion} (balanço {balance:+.2f}). "
            "Não constitui recomendação de compra ou venda; ver riscos e condições de invalidação."
        )
    themes_bull = {r["tema"] for r in acc if r["lado"] == "favoravel"}
    themes_bear = {r["tema"] for r in acc if r["lado"] == "contraria"}
    divergences = sorted(themes_bull & themes_bear)
    return {
        "membros": [
            {"membro": "Otimista", "foco": bull.focus, "argumentos": [a.claim for a in bull.arguments], "peso_validado": s_bull},
            {"membro": "Pessimista", "foco": bear.focus, "argumentos": [a.claim for a in bear.arguments], "peso_validado": s_bear},
            {"membro": "Auditor", "foco": "Validação e incertezas", "aceitos": len(acc), "rejeitados": len(audit["revisoes"]) - len(acc)},
        ],
        "revisoes_auditor": audit["revisoes"],
        "balanco": balance,
        "conclusao": conclusion,
        "texto_conclusao": text,
        "temas_em_divergencia": divergences,
        "familias_validadas": {
            "favoravel": sorted({r["familia"] for r in acc if r["lado"] == "favoravel"}),
            "contraria": sorted({r["familia"] for r in acc if r["lado"] == "contraria"}),
        },
        "independencia": "Otimista e Pessimista executados em paralelo com cópias isoladas das evidências; nenhum viu a tese do outro.",
        "dados_sinteticos": allow_synthetic,
    }
