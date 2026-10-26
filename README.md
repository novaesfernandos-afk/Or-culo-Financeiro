# Oráculo B3

Plataforma de inteligência de investimentos para a B3 com arquitetura multiagente: 18 agentes especializados,
32 skills executáveis e testáveis, 8 modos de trabalho, memória operacional em 4 níveis, orquestração
de tarefas, Comitê de Investimentos e mecanismos de auditoria.

> O Oráculo B3 **não emite ordens de compra ou venda** e não constitui recomendação de investimento.
> Toda saída é probabilística, traz fonte e horário dos dados e passa por revisão de compliance.

## Como rodar

```bash
pip install -e ".[dev]"

# Demonstração com dados SINTÉTICOS (sem rede; tudo marcado como sintético)
ORACULO_DEMO=1 oraculo-b3          # http://127.0.0.1:8000

# Dados reais (Banco Central SGS + brapi.dev)
export BRAPI_TOKEN=...             # sem token, a brapi libera só alguns ativos (ex.: PETR4, VALE3, ITUB4, MGLU3)
oraculo-b3

pytest                             # 134 testes, sem acesso à rede
```

| Variável | Uso |
|----------|-----|
| `ORACULO_DEMO=1` | Modo demonstração (somente dados sintéticos; CVM e notícias desligadas) |
| `ORACULO_DB` | Banco SQLite (padrão `data/oraculo.db`) |
| `ORACULO_CACHE` | Cache dos arquivos da CVM (padrão `data/cache`) |
| `BRAPI_TOKEN` | Token da brapi.dev |
| `ORACULO_CVM=0` / `ORACULO_NEWS=0` | Desliga a CVM ou o coletor de notícias |
| `ORACULO_NEWS_FEEDS` | Feeds RSS (padrão `infomoney,infomoney_empresa,valor,valor_empresas`; `google` é opcional) |
| `ORACULO_USER` / `ORACULO_PASSWORD` | Usuário e senha de acesso; **obrigatório** para expor fora da máquina local |
| `HOST` / `PORT` | Endereço do servidor |

## Regras obrigatórias (aplicadas em código)

| # | Regra | Como é aplicada |
|---|-------|-----------------|
| 1 | Nunca inventar notícias, cotações ou indicadores | `Series` exige fonte; dado indisponível gera erro, nunca valor substituto |
| 2 | Registrar fonte e horário | Toda série, notícia e resultado de skill carrega fonte, data de referência e horário de coleta |
| 3 | Identificar informações atrasadas | Limites por frequência; aviso no relatório e no Painel; peso reduzido no Comitê |
| 4 | Não apresentar probabilidades sem calibração | Previsão direcional só de modelo de produção com mapa de calibração; faixas de cenário só viram probabilidade se a cobertura histórica medida bater com a nominal |
| 5 | Não usar informações futuras em testes | Atributos só com passado, execução defasada no backtest, expurgo no walk-forward, detecção de vazamento |
| 6 | Opiniões não são probabilidades | Pesos dos agentes exibidos como "baixa/média/alta"; índice agregado marcado como não-probabilidade |
| 7 | Não prometer lucros | Compliance bloqueia promessas, certezas e linguagem de ordem; aviso legal obrigatório |
| 8 | Separar experimental de produção | Análises gravadas por ambiente; Modo 7 isolado; previsões só registradas com modelo de produção |
| 9 | Respeitar limites das fontes | Intervalo mínimo por servidor; Google Notícias desligado por padrão; erros 401/403 não são repetidos |
| 10 | Não negociar | Não há integração com corretoras nem rotas de ordens (testado) |
| 11 | Monitorar custos | Chamadas e processamento por dia, com orçamentos por provedor; exibidos no Painel |
| 12 | Exibir dados insuficientes | "Evidência insuficiente" por agente e lista no Painel |

Cada regra tem teste em `tests/test_regras.py`. O planejamento técnico (capacidades da plataforma, dependências,
custos e mapeamento de pastas) está em [`docs/PLANEJAMENTO.md`](docs/PLANEJAMENTO.md).

Dados sintéticos (modo demonstração) são marcados em toda a cadeia, sinalizados no relatório, rejeitados pelo
Auditor fora da demonstração e nunca podem originar um modelo de produção.

## Arquitetura

```
oraculo/
  core/           tipos com procedência, regras obrigatórias, IDs dos agentes
  data/           DataHub (roteamento, cache, custos) + provedores: BCB SGS, brapi, CSV licenciado, sintético
  skills/         as 32 skills (macro, mercado, inteligência, preditivas, controle) + framework versionado
  agents/         18 agentes, lógica dos especialistas, Comitê de Investimentos, síntese (CIIO), comunicador
  orchestration/  orquestrador em DAG (paralelo, dependências, repetições, custos), agendador, fluxo de 7 etapas
  modes/          os 8 modos de trabalho
  memory/         SQLite: mercado, analítica, modelos (com governança), usuário, registro de previsões
  api/ + web/     API FastAPI e interface web
```

### Agentes

| ID | Agente | Grupo |
|----|--------|-------|
| A01 | Chief Investment Intelligence Officer | Direção e coordenação |
| A02 | Orquestrador Operacional | Direção e coordenação |
| A03 | Economista Global | Economia e mercados |
| A04 | Economista Brasil | Economia e mercados |
| A05 | Especialista em Geopolítica | Economia e mercados |
| A06 | Analista de Commodities | Economia e mercados |
| A07 | Analista Fundamentalista | Análise de ativos |
| A08 | Analista Técnico | Análise de ativos |
| A09 | Analista Setorial | Análise de ativos |
| A10 | Analista de Fluxo e Mercado | Análise de ativos |
| A11 | Cientista Quantitativo | Inteligência preditiva |
| A12 | Especialista em Eventos Sazonais | Inteligência preditiva |
| A13 | Inteligência de Notícias | Inteligência preditiva |
| A14 | Gestor de Riscos | Risco e validação |
| A15 | Auditor Quantitativo | Risco e validação |
| A16 | Especialista em Compliance | Risco e validação |
| A17 | Estrategista de Carteiras | Estratégia e comunicação |
| A18 | Comunicador Financeiro | Estratégia e comunicação |

Cada agente tem missão, funções, ferramentas, skills, fontes de dados, entregáveis, critérios de qualidade e
restrições (`agents/registry.py`, também em `GET /api/agents`). As permissões são verificadas na execução:
um agente que chama uma skill não autorizada recebe `negado`.

### Skills

Cada skill (`skills/*.py`) é uma função Python registrada com `SkillSpec` versionada: identificador, objetivo,
agentes autorizados, entradas, ferramentas, procedimento, critérios de validação, formato de saída e tratamento
de falhas. A execução registra status (`ok`, `degradado`, `invalido`, `falhou`, `negado`), avisos, fontes,
duração e hashes de entrada/saída.

- **Macro:** `global_macro_monitor`, `brazil_macro_monitor`, `central_bank_analysis`, `yield_curve_analysis`, `fx_impact_analysis`, `economic_regime_detection`
- **Mercado:** `fundamental_analysis`, `valuation_engine`, `technical_analysis`, `sector_comparison`, `earnings_analysis`, `commodity_exposure`
- **Inteligência:** `global_news_monitor`, `news_sentiment_analysis`, `event_impact_mapping`, `geopolitical_risk_scoring`, `seasonality_detection`, `historical_event_comparison`
- **Preditivas:** `price_direction_forecast`, `return_distribution_forecast`, `volatility_forecast`, `scenario_simulation`, `model_ensemble`, `prediction_calibration`
- **Controle:** `backtesting_engine`, `walk_forward_validation`, `portfolio_risk_analysis`, `prediction_audit`, `data_quality_validation`, `source_verification`, `model_drift_detection`, `compliance_review`

### Modos de trabalho

| # | Modo | Destaques |
|---|------|-----------|
| 1 | Monitoramento contínuo | Ciclos leves; modelos pesados só sob mudança material e com resfriamento de 6 h por ativo; alertas do usuário; agendador |
| 2 | Análise profunda | Fluxo de 7 etapas + Comitê; tese favorável/contrária, indicadores, catalisadores, riscos, cenários e condições de invalidação |
| 3 | Caçador de oportunidades | Ranking explicável por fatores invariantes ao preço nominal (preço baixo não é oportunidade) |
| 4 | Análise de evento | Antes x depois, retorno anormal vs. Ibovespa, mapa de impacto setorial e comparação histórica |
| 5 | Simulação de cenários | Aceita perguntas em português ("E se o dólar chegasse a R$ 6,50?"); hipóteses, limitações e alerta de choque fora do histórico |
| 6 | Análise de carteira | Concentração, VaR/CVaR, correlações, contribuição de risco, liquidez e alternativas de menor correlação; sem ordens |
| 7 | Pesquisa e desenvolvimento | Treino, validação walk-forward com expurgo, comparação de estratégias, ciclo de aprovação |
| — | Painel | Página inicial: Selic, IPCA e dólar com fonte e horário, ativos e carteira, alertas, análises, modelos em produção, custos do dia, saúde das fontes e dados insuficientes |
| 8 | Auditoria | Acerto direcional, erro médio, Brier/ECE, retorno líquido de custos, por horizonte e por modelo, piores falhas |

### Fluxo de colaboração (Modo 2)

1. **Coleta:** o Orquestrador busca cotações, volume, Ibovespa e Selic/IPCA/câmbio em paralelo, com repetições.
2. **Análise paralela:** 10 especialistas executam suas skills; sem dados, a posição é "evidência insuficiente".
3. **Modelagem:** o Cientista Quantitativo usa o modelo em **produção**; o resultado recebe um hash de integridade.
4. **Contestação:** o Gestor de Riscos, o Auditor e o **Comitê de Investimentos** verificam os resultados.
5. **Síntese:** o CIIO consolida e preserva divergências; exige ao menos 3 frentes com evidência; se o hash quantitativo mudar, bloqueia.
6. **Comunicação:** relatório em Markdown; o Compliance bloqueia linguagem de ordem, promessas, falta de fontes, dados pessoais e dados sintéticos ou atrasados não sinalizados.
7. **Registro:** análise, divergências, hipóteses de invalidação e previsão (com versão do modelo e hash dos dados) vão para a memória.

### Comitê de Investimentos

Três membros independentes analisam a mesma ação:

- **Otimista:** procura argumentos favoráveis (valuation, crescimento, momentum).
- **Pessimista:** procura riscos (juros, alavancagem, volatilidade, cauda).
- **Auditor:** recalcula cada argumento a partir da evidência original e rejeita valores divergentes, modelos não
  validados e dados sintéticos; dados atrasados recebem metade do peso.

Otimista e Pessimista rodam em paralelo, cada um com uma cópia isolada das evidências. A conclusão só sai do que
o Auditor validou e exige argumentos de **ao menos duas famílias de dados independentes** (preços, fundamentos,
macro, notícias, setor...). Fora disso, o resultado é "inconclusivo" ou "evidência insuficiente", com as
divergências exibidas. Também está disponível em `POST /api/committee`.

### Memória e governança de modelos

- **Mercado:** séries e eventos (eventos exigem fonte).
- **Analítica:** análises, hipóteses com condições de invalidação, divergências.
- **Modelos:** versões imutáveis, parâmetros, métricas, validação e trilha de eventos.
- **Usuário:** preferências, ativos acompanhados, alertas, objetivos e carteira.

Ciclo de vida: `experimental` (A11 cria) → `validado`/`rejeitado` (só A15 certifica, por critérios fixos:
habilidade de Brier ≥ 1%, superar a base em ≥ 60% dos blocos, ECE ≤ 0,05, acerto acima de "sempre alta",
sem vazamento) → `aprovado` (responsável humano `user:<nome>`, diferente do autor) → `producao` (humano; a
versão anterior é arquivada). Parâmetros de versões não experimentais não podem ser alterados.

## Fontes de dados

| Fonte | Uso | Observação |
|-------|-----|------------|
| Banco Central (SGS) | Selic meta, Selic diária, IPCA mensal e 12 meses, dólar PTAX | API pública |
| brapi.dev | Cotações e volume diários (até 5 anos) | Token necessário para a maioria dos ativos e para `^BVSP` |
| CVM - Dados Abertos | Cadastro (ticker → CNPJ), fatos relevantes e comunicados (IPE), balanço anual (DFP) e trimestral (ITR) | Pública; baixada 1x/dia para o cache |
| InfoMoney e Valor (RSS) | Notícias por empresa (título, link, fonte e horário) | Uso sujeito aos termos de cada veículo |
| Google Notícias (RSS) | Busca por empresa | **Desligado por padrão**: o feed só permite uso pessoal e não comercial |
| CSV do usuário | Qualquer série licenciada (commodities, curva DI, indicadores globais) | Fonte declarada obrigatória |
| Sintético | Demonstração e testes | Sempre marcado como sintético |

Na Análise Profunda, o Orquestrador busca automaticamente na CVM o último balanço anual consolidado e o último
trimestre (comparado ao mesmo trimestre do ano anterior), além dos fatos relevantes e notícias dos últimos 14 dias.
Com o preço atual, o Analista Fundamentalista calcula P/L, P/VP, EV/EBITDA e rendimento do fluxo de caixa livre.
Detalhes que evitam números errados:

- A quantidade de ações vem da CVM em unidades ou em milhares, conforme a empresa. A escala é confirmada pelo lucro
  por ação do próprio balanço; sem confirmação, os múltiplos não são calculados.
- Bancos e seguradoras são identificados pelo plano de contas; critérios de dívida, EBITDA e caixa não se aplicam a eles.
- Notícias de veículos gerais só entram se citarem a empresa (nomes ambíguos como "Vale" e "B3" exigem menção no título).
- Fatos relevantes da CVM são fonte oficial (confirmada). Matérias com "dizem fontes" ou "estaria" são classificadas como rumor.

Dados próprios podem substituir as fontes automáticas na requisição do Modo 2 (`noticias`, `financials` +
`financials_source`, `valuation`, `pares`, `eventos_geopoliticos`).

## Publicar online (Render)

O repositório inclui `Dockerfile` e `render.yaml`:

1. Crie uma conta em https://render.com e conecte o GitHub.
2. Em **New → Blueprint**, escolha este repositório. O Render lê o `render.yaml`.
3. Ao final, em **Environment**, copie a senha gerada em `ORACULO_PASSWORD` (usuário: `oraculo`) e, se tiver,
   informe o `BRAPI_TOKEN`.
4. Acesse a URL `https://oraculo-b3-....onrender.com`; o navegador pede usuário e senha.

No plano gratuito o serviço "dorme" após inatividade e o disco é temporário: análises, carteira e modelos se perdem
ao reiniciar. Para persistir, use um plano pago com o bloco `disk` do `render.yaml`. O container roda sem privilégios
de administrador e se recusa a iniciar exposto sem senha.

## Próximos passos sugeridos

- Fontes licenciadas para commodities, curva DI, fluxo estrangeiro e indicadores globais.
- Narrativa opcional via LLM no Comunicador, sempre sujeita à revisão de compliance.
