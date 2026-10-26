"""Agente 18 - Comunicador Financeiro: relatório final em Markdown.

Regra: nunca simplificar a ponto de ocultar riscos ou incertezas. Toda seção de
riscos, divergências e avisos de dados é sempre exibida, mesmo que vazia.
"""
from __future__ import annotations

from ..core import agent_ids as A
from ..core.rules import provenance_banner
from ..core.types import AgentView, SourceRef, Stance
from ..skills.control import DISCLAIMER
from .registry import AGENTS


def _pct(x: float | None, signed: bool = True) -> str:
    if x is None:
        return "n/d"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def scenarios(quant_pkg: dict) -> list[dict]:
    d = (quant_pkg or {}).get("distribuicao")
    if not d:
        return []
    q = d["quantis"]
    h = d["horizonte_pregoes"]
    return [
        {"cenario": "Adverso", "probabilidade": "25%", "faixa": f"{_pct(q['p05'])} a {_pct(q['p25'])} (5% de chance de ser pior que {_pct(q['p05'])})"},
        {"cenario": "Base", "probabilidade": "50%", "faixa": f"{_pct(q['p25'])} a {_pct(q['p75'])} (mediana {_pct(q['p50'])})"},
        {"cenario": "Favorável", "probabilidade": "25%", "faixa": f"{_pct(q['p75'])} a {_pct(q['p95'])} (5% de chance de ser melhor que {_pct(q['p95'])})"},
        {"horizonte_pregoes": h, "metodo": "simulação histórica filtrada (EWMA + bootstrap); não inclui eventos sem precedente no histórico"},
    ]


def invalidation_conditions(dossier_results: dict, committee: dict) -> list[str]:
    conds = []
    tech = next((r.output for r in dossier_results.values() if r.skill_id == "technical_analysis" and r.ok), None)
    if tech:
        if tech["suportes"]:
            conds.append(f"Fechamento abaixo do suporte de R$ {tech['suportes'][-1]:.2f} invalidaria a leitura técnica favorável.")
        if tech["resistencias"]:
            conds.append(f"Rompimento da resistência de R$ {tech['resistencias'][0]:.2f} invalidaria a leitura técnica desfavorável.")
        if tech.get("mms200"):
            conds.append(f"Cruzamento do preço pela MMS200 (R$ {tech['mms200']:.2f}) mudaria a classificação de tendência de longo prazo.")
    fund = next((r.output for r in dossier_results.values() if r.skill_id == "fundamental_analysis" and r.ok), None)
    if fund:
        conds.append("Alavancagem acima de 3,5x dívida líquida/EBITDA ou queda da pontuação de qualidade abaixo de 3/6 no próximo balanço.")
    for rev in committee.get("revisoes_auditor", []):
        if rev["peso"] > 0 and rev["tema"] == "notícias":
            conds.append("Reversão do tom do noticiário ou desmentido oficial das notícias citadas.")
            break
    conds.append("Nova informação material (fato relevante, decisão do Copom, choque de commodities) não refletida nos dados desta análise.")
    return conds


def build_report(ticker: str, views: dict[str, AgentView], synthesis: dict, committee: dict, quant_pkg: dict,
                 dossier_results: dict, refs: list[SourceRef], warnings: list[str], synthetic: bool) -> tuple[str, dict]:
    accepted = [r for r in committee.get("revisoes_auditor", []) if r["peso"] > 0]
    rejected = [r for r in committee.get("revisoes_auditor", []) if r["peso"] == 0]
    bull = [r["argumento"] + (" (com ressalva)" if r["veredito"] != "aceito" else "") for r in accepted if r["lado"] == "favoravel"]
    bear = [r["argumento"] + (" (com ressalva)" if r["veredito"] != "aceito" else "") for r in accepted if r["lado"] == "contraria"]
    risks = sorted({x for v in views.values() for x in v.risks})
    indicators = [e for k in (A.TECNICO, A.FUNDAMENTALISTA, A.RISCO, A.ECON_BRASIL, A.QUANT) if k in views for e in views[k].evidence]
    catalysts = []
    nv = views.get(A.NOTICIAS)
    if nv:
        catalysts += [e for e in nv.evidence if e.startswith("[confirmada]")]
    eb = views.get(A.ECON_BRASIL)
    if eb:
        catalysts += [e for e in eb.evidence if e.startswith("Copom")]
    tech = next((r.output for r in dossier_results.values() if r.skill_id == "technical_analysis" and r.ok), None)
    if tech and tech["resistencias"]:
        catalysts.append(f"Teste da resistência em R$ {tech['resistencias'][0]:.2f}")
    scen = scenarios(quant_pkg)
    inval = invalidation_conditions(dossier_results, committee)

    L: list[str] = []
    L.append(f"# Análise Profunda - {ticker}")
    if synthetic:
        L.append("\n> **DADOS SINTÉTICOS - DEMONSTRAÇÃO.** Os números abaixo não representam o mercado real.")
    if any("ATRASADO" in w for w in warnings):
        L.append("\n> **Atenção: há dados atrasados nesta análise** (ver seção Avisos de dados).")
    L.append(f"\n**Conclusão:** {synthesis['conclusao']}. {synthesis['justificativa']}")
    L.append(f"\n**Comitê de Investimentos:** {committee.get('texto_conclusao', 'n/d')}")
    L.append("\n## Tese favorável")
    L += [f"- {x}" for x in bull] or ["- Nenhum argumento favorável validado pelo Auditor."]
    L.append("\n## Tese contrária")
    L += [f"- {x}" for x in bear] or ["- Nenhum argumento contrário validado pelo Auditor."]
    L.append("\n## Principais indicadores")
    L += [f"- {x}" for x in indicators] or ["- Indicadores indisponíveis."]
    L.append("\n## Catalisadores")
    L += [f"- {x}" for x in catalysts] or ["- Nenhum catalisador identificado com dados confirmados."]
    L.append("\n## Riscos")
    L += [f"- {x}" for x in risks] or ["- Nenhum risco específico detectado pelas métricas; riscos de cauda e eventos inéditos permanecem."]
    L.append("\n## Cenários probabilísticos")
    if scen:
        for s in scen[:3]:
            L.append(f"- **{s['cenario']}** ({s['probabilidade']}): {s['faixa']}")
        L.append(f"- Horizonte: {scen[3]['horizonte_pregoes']} pregões. Método: {scen[3]['metodo']}.")
    else:
        L.append("- Distribuição de retornos indisponível.")
    fc = (quant_pkg or {}).get("previsao")
    if fc:
        tag = "" if fc["status_modelo"] == "producao" else " - **modelo não validado; não usado na conclusão**"
        L.append(f"- Probabilidade de alta em {fc['horizonte_pregoes']} pregões: {fc['prob_alta']:.0%} (taxa-base {fc['taxa_base']:.0%}){tag}.")
    L.append("\n## Condições que invalidariam a tese")
    L += [f"- {x}" for x in inval]
    L.append("\n## Divergências entre agentes")
    if synthesis["contradicoes"]:
        for c in synthesis["contradicoes"]:
            n = [AGENTS[a].name for a in c["agentes"]]
            L.append(f"- {n[0]} ({c['posicoes'][0]}) x {n[1]} ({c['posicoes'][1]}): {c['resumos'][0]} | {c['resumos'][1]}")
    else:
        L.append("- Sem contradições relevantes entre especialistas com evidência.")
    if committee.get("temas_em_divergencia"):
        L.append(f"- Temas com argumentos validados dos dois lados no Comitê: {', '.join(committee['temas_em_divergencia'])}.")
    if rejected:
        L.append(f"- Argumentos rejeitados pelo Auditor: {len(rejected)} (ex.: {rejected[0]['argumento']} - {', '.join(rejected[0]['observacoes'])}).")
    L.append("\n## Posição de cada especialista")
    for k, v in sorted(views.items()):
        L.append(f"- **{AGENTS[k].name}**: {v.stance.value} (confiança {v.confidence:.0%}) - {v.summary}")
    if synthesis["frentes_sem_evidencia"]:
        L.append(f"- Sem evidência suficiente: {', '.join(AGENTS[a].name for a in synthesis['frentes_sem_evidencia'])}.")
    L.append("\n## Avisos de dados")
    uniq = sorted(set(warnings))
    L += [f"- {w}" for w in uniq] or ["- Nenhum aviso."]
    L.append("\n## Fontes e horários")
    L += [f"- {x}" for x in provenance_banner(refs)] or ["- Nenhuma fonte."]
    L.append(f"\n---\n*{DISCLAIMER}*")
    text = "\n".join(L)
    structured = {
        "tese_favoravel": bull, "tese_contraria": bear, "indicadores": indicators, "catalisadores": catalysts, "riscos": risks,
        "cenarios": scen, "condicoes_invalidacao": inval,
    }
    return text, structured


def ensure_flags(text: str, synthetic: bool, stale: bool) -> str:
    """Correções aditivas exigidas pelo Compliance (nunca remove conteúdo)."""
    if synthetic and "sintétic" not in text.lower():
        text = "> **DADOS SINTÉTICOS - DEMONSTRAÇÃO.**\n\n" + text
    if stale and "atrasad" not in text.lower():
        text = "> **Atenção: há dados atrasados nesta análise.**\n\n" + text
    if "não constitui recomendação" not in text.lower():
        text += f"\n\n---\n*{DISCLAIMER}*"
    return text


def stance_label(s: Stance) -> str:
    return s.value
