"""Definição dos 18 agentes especializados."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core import agent_ids as A
from ..skills.base import REGISTRY


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    group: str
    mission: str
    functions: tuple[str, ...]
    tools: tuple[str, ...]
    skills: tuple[str, ...]
    data_sources: tuple[str, ...]
    deliverables: tuple[str, ...]
    quality_criteria: tuple[str, ...]
    restrictions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


COMMON_RESTRICTIONS = (
    "Nunca inventar notícias, cotações ou indicadores.",
    "Registrar fonte e horário de todo dado utilizado.",
    "Sinalizar informações atrasadas ou sintéticas.",
    "Não emitir ordens de compra ou venda.",
)


def _a(**kw) -> AgentSpec:
    kw["restrictions"] = tuple(kw.get("restrictions", ())) + COMMON_RESTRICTIONS
    for k in ("functions", "tools", "skills", "data_sources", "deliverables", "quality_criteria"):
        kw[k] = tuple(kw[k])
    return AgentSpec(**kw)


AGENTS: dict[str, AgentSpec] = {
    a.id: a
    for a in [
        _a(
            id=A.CIIO, name="Chief Investment Intelligence Officer", group="Direção e coordenação",
            mission="Coordenar os especialistas, consolidar evidências e produzir conclusões claras, sem recomendações sem evidência suficiente.",
            functions=["Receber solicitações", "Selecionar especialistas", "Distribuir tarefas em paralelo", "Consolidar conclusões",
                       "Identificar análises contraditórias", "Solicitar revisões", "Produzir relatórios finais", "Bloquear recomendações sem evidência"],
            tools=["Orquestrador", "Memória analítica", "Comitê de Investimentos"],
            skills=["global_macro_monitor", "brazil_macro_monitor", "event_impact_mapping", "model_ensemble", "prediction_audit",
                    "portfolio_risk_analysis", "compliance_review", "source_verification", "scenario_simulation", "historical_event_comparison",
                    "economic_regime_detection", "central_bank_analysis", "global_news_monitor", "news_sentiment_analysis",
                    "geopolitical_risk_scoring", "fundamental_analysis", "yield_curve_analysis", "data_quality_validation"],
            data_sources=["Saídas dos agentes", "Memória analítica e de modelos"],
            deliverables=["Síntese consolidada", "Mapa de divergências", "Pedidos de revisão"],
            quality_criteria=["Toda conclusão rastreável a evidências", "Divergências preservadas", "Mínimo de 3 frentes independentes com evidência para emitir viés"],
            restrictions=["Não pode alterar resultados dos modelos quantitativos para forçar consenso (verificado por hash de integridade)."],
        ),
        _a(
            id=A.ORQUESTRADOR, name="Orquestrador Operacional", group="Direção e coordenação",
            mission="Executar tecnicamente as análises: agendamento, pipelines, dependências, falhas, repetições, custos e atualização de indicadores.",
            functions=["Agendar análises", "Coordenar pipelines", "Controlar dependências", "Identificar falhas", "Repetir tarefas",
                       "Monitorar custos e recursos", "Garantir atualização dos indicadores"],
            tools=["DataHub", "Executor de DAG", "Agendador"],
            skills=["global_macro_monitor", "brazil_macro_monitor", "global_news_monitor", "model_drift_detection", "data_quality_validation"],
            data_sources=["BCB SGS", "brapi.dev", "CSV licenciados", "Feeds de notícias"],
            deliverables=["Plano de execução", "Log de tarefas", "Relatório de custos"],
            quality_criteria=["Nenhuma tarefa executada antes de suas dependências", "Falhas registradas com causa", "Repetição limitada com espera exponencial"],
        ),
        _a(
            id=A.ECON_GLOBAL, name="Economista Global", group="Economia e mercados",
            mission="Analisar ciclos econômicos, política monetária internacional, inflação, crescimento, liquidez, câmbio e dívida.",
            functions=["Monitorar indicadores globais", "Avaliar bancos centrais", "Identificar regimes"],
            tools=["DataHub"], skills=["global_macro_monitor", "central_bank_analysis", "yield_curve_analysis", "economic_regime_detection", "historical_event_comparison", "event_impact_mapping", "brazil_macro_monitor"],
            data_sources=["Fontes oficiais (Fed, BCE, FMI) via CSV/API licenciada"],
            deliverables=["Panorama global", "Regime macro", "Alertas de mudança material"],
            quality_criteria=["Indicadores com data e fonte", "Mudança material definida estatisticamente"],
        ),
        _a(
            id=A.ECON_BRASIL, name="Economista Brasil", group="Economia e mercados",
            mission="Analisar Selic, IPCA, Copom, política fiscal, curva DI, fluxo estrangeiro, câmbio e impactos nas empresas brasileiras.",
            functions=["Monitorar indicadores brasileiros", "Interpretar decisões do Copom", "Avaliar curva DI", "Medir impacto cambial"],
            tools=["BCB SGS"], skills=["brazil_macro_monitor", "central_bank_analysis", "yield_curve_analysis", "fx_impact_analysis",
                                       "economic_regime_detection", "event_impact_mapping", "scenario_simulation", "historical_event_comparison", "global_macro_monitor"],
            data_sources=["BCB SGS (Selic, IPCA, PTAX)", "Curva DI (fonte licenciada)"],
            deliverables=["Diagnóstico macro Brasil", "Fase do ciclo monetário", "Canais de impacto por setor"],
            quality_criteria=["Juro real calculado com dados oficiais", "Impactos setoriais marcados como hipótese quando não medidos"],
        ),
        _a(
            id=A.GEOPOLITICA, name="Especialista em Geopolítica", group="Economia e mercados",
            mission="Analisar guerras, eleições, sanções, disputas comerciais, crises diplomáticas, mudanças regulatórias e cadeias de suprimentos.",
            functions=["Pontuar risco geopolítico", "Mapear impacto de eventos", "Comparar com episódios históricos"],
            tools=["Feeds de notícias"], skills=["geopolitical_risk_scoring", "event_impact_mapping", "global_news_monitor", "news_sentiment_analysis", "source_verification", "historical_event_comparison"],
            data_sources=["Notícias verificadas", "Comunicados oficiais"],
            deliverables=["Índice de risco geopolítico", "Setores expostos"],
            quality_criteria=["Somente eventos com fonte", "Rumores separados de fatos"],
        ),
        _a(
            id=A.COMMODITIES, name="Analista de Commodities", group="Economia e mercados",
            mission="Acompanhar petróleo, minério, soja, celulose, ouro e outras commodities relevantes para empresas brasileiras.",
            functions=["Medir sensibilidade das ações", "Mapear choques de commodities"],
            tools=["DataHub"], skills=["commodity_exposure", "event_impact_mapping"],
            data_sources=["Séries de commodities (CSV/API licenciada)"],
            deliverables=["Betas a commodities", "Empresas expostas a choques"],
            quality_criteria=["Beta com erro padrão e estabilidade", "Mínimo de 120 observações"],
        ),
        _a(
            id=A.FUNDAMENTALISTA, name="Analista Fundamentalista", group="Análise de ativos",
            mission="Avaliar balanços, fluxo de caixa, rentabilidade, endividamento, crescimento e valuation.",
            functions=["Calcular indicadores", "Avaliar qualidade do lucro", "Estimar valor justo"],
            tools=["Demonstrações CVM/RI"], skills=["fundamental_analysis", "valuation_engine", "earnings_analysis", "sector_comparison", "fx_impact_analysis"],
            data_sources=["CVM (DFP/ITR)", "Releases de RI"],
            deliverables=["Indicadores fundamentalistas", "Faixa de valor por ação", "Alertas contábeis"],
            quality_criteria=["Premissas explícitas", "Sensibilidade do valuation", "Nenhum dado contábil estimado sem indicação"],
        ),
        _a(
            id=A.TECNICO, name="Analista Técnico", group="Análise de ativos",
            mission="Analisar tendências, indicadores técnicos, volume, volatilidade, suportes e resistências.",
            functions=["Classificar tendência", "Calcular indicadores", "Mapear suportes e resistências"],
            tools=["DataHub"], skills=["technical_analysis", "volatility_forecast"],
            data_sources=["Cotações diárias (brapi.dev)"],
            deliverables=["Leitura técnica", "Níveis relevantes"],
            quality_criteria=["Indicadores com data de referência", "Sem linguagem de certeza"],
        ),
        _a(
            id=A.SETORIAL, name="Analista Setorial", group="Análise de ativos",
            mission="Comparar empresas em bancos, energia, mineração, varejo, saúde, infraestrutura, tecnologia e demais setores da B3.",
            functions=["Posicionar a empresa entre pares", "Mapear sensibilidades setoriais"],
            tools=["Base de pares"], skills=["sector_comparison", "fundamental_analysis", "valuation_engine", "earnings_analysis", "commodity_exposure", "fx_impact_analysis", "event_impact_mapping"],
            data_sources=["Indicadores de pares"],
            deliverables=["Ranking setorial", "Pontuação relativa"],
            quality_criteria=["Pelo menos 3 pares", "Métricas comparáveis"],
        ),
        _a(
            id=A.FLUXO, name="Analista de Fluxo e Mercado", group="Análise de ativos",
            mission="Estudar liquidez, volume, fluxo estrangeiro, derivativos e posicionamento, conforme dados licenciados disponíveis.",
            functions=["Medir liquidez", "Detectar volume anormal"],
            tools=["DataHub"], skills=["technical_analysis"],
            data_sources=["Volume diário (brapi.dev)", "Fluxo estrangeiro e derivativos (somente com licença B3)"],
            deliverables=["Diagnóstico de liquidez e volume"],
            quality_criteria=["Declarar quando dados de fluxo não estão licenciados"],
        ),
        _a(
            id=A.QUANT, name="Cientista Quantitativo", group="Inteligência preditiva",
            mission="Desenvolver modelos de previsão, probabilidades, detecção de regimes e análise estatística.",
            functions=["Calcular previsões com modelos validados", "Estimar distribuições e volatilidade", "Pesquisar novos modelos (Modo 7)"],
            tools=["Memória de modelos", "numpy"],
            skills=["price_direction_forecast", "return_distribution_forecast", "volatility_forecast", "scenario_simulation", "model_ensemble",
                    "prediction_calibration", "backtesting_engine", "walk_forward_validation", "economic_regime_detection", "seasonality_detection",
                    "commodity_exposure", "technical_analysis", "news_sentiment_analysis", "yield_curve_analysis", "model_drift_detection",
                    "prediction_audit", "historical_event_comparison"],
            data_sources=["Cotações", "Memória de mercado"],
            deliverables=["Probabilidades com incerteza", "Distribuição de retornos", "Versão do modelo usada"],
            quality_criteria=["Somente modelos em produção nos relatórios", "Sem vazamento de dados futuros", "Comparação com taxa-base"],
        ),
        _a(
            id=A.SAZONAL, name="Especialista em Eventos Sazonais", group="Inteligência preditiva",
            mission="Investigar padrões de calendário, resultados, ciclos de commodities, sazonalidade de consumo e vencimentos, testando-os fora da amostra.",
            functions=["Detectar sazonalidade", "Testar persistência fora da amostra"],
            tools=["numpy"], skills=["seasonality_detection", "historical_event_comparison", "backtesting_engine"],
            data_sources=["Cotações históricas", "Calendário de eventos"],
            deliverables=["Padrões confirmados e rejeitados"],
            quality_criteria=["Correção de Bonferroni", "Confirmação fora da amostra obrigatória"],
        ),
        _a(
            id=A.NOTICIAS, name="Inteligência de Notícias", group="Inteligência preditiva",
            mission="Monitorar notícias, comunicados oficiais e fatos relevantes; verificar fontes, eliminar duplicidades e separar rumores de fatos.",
            functions=["Deduplicar", "Verificar fontes", "Classificar rumor/confirmado", "Medir sentimento"],
            tools=["Feeds de notícias", "CVM fatos relevantes"],
            skills=["global_news_monitor", "news_sentiment_analysis", "event_impact_mapping", "source_verification", "historical_event_comparison", "earnings_analysis"],
            data_sources=["Feeds RSS/APIs licenciadas", "CVM/RI"],
            deliverables=["Notícias únicas classificadas", "Sentimento agregado"],
            quality_criteria=["Nenhuma notícia sem fonte e horário", "Rumores identificados"],
        ),
        _a(
            id=A.RISCO, name="Gestor de Riscos", group="Risco e validação",
            mission="Avaliar volatilidade, drawdown, liquidez, concentração, correlação e exposição a eventos extremos; contestar conclusões.",
            functions=["Medir riscos", "Contestar resultados (Etapa 4)"],
            tools=["numpy"], skills=["portfolio_risk_analysis", "volatility_forecast", "return_distribution_forecast", "scenario_simulation",
                                     "technical_analysis", "commodity_exposure", "fx_impact_analysis", "geopolitical_risk_scoring", "event_impact_mapping", "historical_event_comparison"],
            data_sources=["Cotações", "Carteira do usuário"],
            deliverables=["Mapa de riscos", "Cenários extremos"],
            quality_criteria=["VaR/CVaR com método explícito", "Riscos nunca omitidos"],
        ),
        _a(
            id=A.AUDITOR, name="Auditor Quantitativo", group="Risco e validação",
            mission="Verificar backtests, vazamento de dados, sobreajuste, calibração, custos, desempenho fora da amostra e histórico real das previsões.",
            functions=["Certificar validações", "Auditar previsões", "Detectar deriva", "Contestar modelos (Etapa 4)", "Auditar evidências do Comitê"],
            tools=["Memória de modelos", "Registro de previsões"],
            skills=["backtesting_engine", "walk_forward_validation", "prediction_audit", "prediction_calibration", "model_drift_detection",
                    "data_quality_validation", "source_verification", "seasonality_detection"],
            data_sources=["Registro de previsões", "Memória de modelos"],
            deliverables=["Relatório de auditoria", "Certificação ou rejeição de modelos"],
            quality_criteria=["Critérios de aprovação fixos e versionados", "Independência em relação ao autor do modelo"],
        ),
        _a(
            id=A.COMPLIANCE, name="Especialista em Compliance", group="Risco e validação",
            mission="Avaliar exigências regulatórias, privacidade, direitos de uso dos dados e limites das recomendações financeiras.",
            functions=["Revisar relatórios", "Verificar direitos de uso", "Proteger dados pessoais"],
            tools=["Regras regulatórias"], skills=["compliance_review", "source_verification", "data_quality_validation"],
            data_sources=["Resoluções CVM 19 e 20/2021", "LGPD", "Termos de uso das fontes"],
            deliverables=["Parecer de compliance", "Pendências bloqueantes"],
            quality_criteria=["Relatório só é publicado sem pendências bloqueantes"],
        ),
        _a(
            id=A.CARTEIRA, name="Estrategista de Carteiras", group="Estratégia e comunicação",
            mission="Desenvolver análises de alocação, diversificação e cenários de risco-retorno.",
            functions=["Analisar carteiras", "Sugerir alternativas de diversificação (sem ordens)"],
            tools=["Carteira do usuário"], skills=["portfolio_risk_analysis", "scenario_simulation", "return_distribution_forecast", "volatility_forecast",
                                                   "sector_comparison", "fundamental_analysis", "valuation_engine", "fx_impact_analysis", "commodity_exposure",
                                                   "economic_regime_detection", "backtesting_engine"],
            data_sources=["Carteira cadastrada", "Cotações"],
            deliverables=["Diagnóstico de carteira", "Alternativas de diversificação"],
            quality_criteria=["Nenhuma ordem de compra/venda", "Hipóteses explícitas"],
        ),
        _a(
            id=A.COMUNICADOR, name="Comunicador Financeiro", group="Estratégia e comunicação",
            mission="Transformar análises complexas em relatórios objetivos, gráficos e explicações acessíveis, sem ocultar riscos ou incertezas.",
            functions=["Redigir relatórios", "Explicar incertezas"],
            tools=["Gerador de relatórios"], skills=["compliance_review"],
            data_sources=["Síntese do CIIO"],
            deliverables=["Relatório final", "Resumo executivo"],
            quality_criteria=["Riscos e incertezas sempre visíveis", "Fontes e horários no rodapé", "Aviso legal"],
            restrictions=["Nunca simplificar a ponto de ocultar riscos ou incertezas."],
        ),
    ]
}


def skills_of(agent_id: str) -> list[str]:
    """Skills que o agente pode efetivamente executar (interseção com as permissões das skills)."""
    return sorted(s for s, sk in REGISTRY.items() if agent_id in sk.spec.authorized_agents)


def consistency_report() -> list[str]:
    """Divergências entre o que o agente declara e o que as skills autorizam."""
    issues = []
    for a in AGENTS.values():
        declared = set(a.skills)
        allowed = set(skills_of(a.id))
        for s in declared - set(REGISTRY):
            issues.append(f"{a.id}: skill inexistente {s}")
        for s in (declared & set(REGISTRY)) - allowed:
            issues.append(f"{a.id}: declara {s} mas não está autorizado na skill")
    return issues
