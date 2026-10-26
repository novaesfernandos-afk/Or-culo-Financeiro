"""Os oito modos de operação selecionáveis na interface."""
from __future__ import annotations

from ..core import agent_ids as A
from ..orchestration.pipeline import deep_analysis
from . import audit, event, monitoring, opportunity, portfolio, research, scenario
from .base import ModeSpec, Services  # noqa: F401


def _deep(svc: Services, params: dict) -> dict:
    return deep_analysis(params["ticker"], svc.hub, svc.memory, params, collectors=svc.collectors())


MODES: dict[str, ModeSpec] = {
    m.slug: m
    for m in [
        ModeSpec(1, "monitoramento", "Monitoramento contínuo", "Acompanhar o mercado e gerar alertas por mudança material.",
                 (A.ORQUESTRADOR, A.ECON_BRASIL, A.NOTICIAS, A.RISCO), monitoring.run),
        ModeSpec(2, "analise_profunda", "Análise profunda", "Investigação completa de uma ação com todos os especialistas e o Comitê.",
                 (A.CIIO, A.ORQUESTRADOR, A.ECON_GLOBAL, A.ECON_BRASIL, A.GEOPOLITICA, A.COMMODITIES, A.FUNDAMENTALISTA, A.TECNICO,
                  A.SETORIAL, A.FLUXO, A.QUANT, A.SAZONAL, A.NOTICIAS, A.RISCO, A.AUDITOR, A.COMPLIANCE, A.COMUNICADOR), _deep),
        ModeSpec(3, "cacador_oportunidades", "Caçador de oportunidades", "Ranking explicável de risco-retorno no universo elegível.",
                 (A.QUANT, A.FUNDAMENTALISTA, A.FLUXO, A.AUDITOR), opportunity.run),
        ModeSpec(4, "analise_evento", "Análise de evento", "Comparar antes e depois de um evento e estimar empresas afetadas.",
                 (A.NOTICIAS, A.GEOPOLITICA, A.ECON_BRASIL, A.COMMODITIES, A.SETORIAL), event.run),
        ModeSpec(5, "simulacao_cenarios", "Simulação de cenários", "Simulações condicionais com hipóteses, limitações e incertezas.",
                 (A.CARTEIRA, A.QUANT, A.ECON_BRASIL, A.RISCO), scenario.run),
        ModeSpec(6, "analise_carteira", "Análise de carteira", "Concentração, riscos, correlações e alternativas de diversificação.",
                 (A.CARTEIRA, A.RISCO), portfolio.run),
        ModeSpec(7, "pesquisa", "Pesquisa e desenvolvimento", "Criar, treinar, validar, comparar e aprovar modelos (separado da produção).",
                 (A.QUANT, A.AUDITOR), research.run),
        ModeSpec(8, "auditoria", "Auditoria", "Desempenho histórico das previsões.", (A.AUDITOR, A.CIIO), audit.run),
    ]
}


def run_mode(slug: str, svc: Services, params: dict) -> dict:
    if slug not in MODES:
        raise KeyError(f"Modo inexistente: {slug}")
    return MODES[slug].run(svc, params or {})
