# Oráculo B3 — Planejamento técnico

Este documento responde à seção 8 do escopo: capacidades da plataforma, arquitetura multiagente, dependências,
custos estimados, mapeamento da estrutura de pastas sugerida e estado de cada prioridade.

## 1. O que a plataforma de desenvolvimento oferece e o que foi implementado no backend

O Oráculo B3 é desenvolvido com um assistente de programação (Claude Code) em um ambiente de nuvem. Esse
ambiente escreve, testa e publica código; ele **não** fornece agentes autônomos, memória ou tarefas agendadas
para o aplicativo em produção. Por isso, conforme o item 8 do escopo, tudo foi implementado no próprio backend:

| Capacidade | Suporte nativo da plataforma | Implementação no Oráculo B3 |
|------------|------------------------------|-----------------------------|
| Agentes | Não para o app em produção | 18 agentes como módulos Python com regras determinísticas (`oraculo/agents/`) |
| Skills | Não | 32 funções versionadas, com permissões e testes (`oraculo/skills/`) |
| Orquestração e paralelismo | Não | Executor de DAG com dependências, repetição e custos (`oraculo/orchestration/`) |
| Tarefas agendadas | Existem para sessões de desenvolvimento, não para o app | Agendador interno do Modo 1 (`Scheduler`) |
| Memória | Não | SQLite com 4 níveis + registro de previsões (`oraculo/memory/`) |
| Integrações | Acesso de rede no ambiente de desenvolvimento | Conectores próprios: BCB, CVM, brapi, RSS, CSV (`oraculo/data/`) |

**Transparência sobre os agentes:** os agentes do Oráculo B3 não são modelos de linguagem autônomos nem prompts
estáticos. Cada um é código que executa skills determinísticas sobre dados reais e aplica regras explícitas para
formar sua posição. Isso torna os resultados reproduzíveis, testáveis e auditáveis. Um modelo de linguagem pode
ser acoplado no futuro (ex.: redação do Comunicador), sempre sujeito ao Compliance e ao controle de custos.

## 2. Arquitetura

```
Usuário ─► Interface web (Painel + 8 modos) ─► API FastAPI (autenticação, cabeçalhos de segurança)
                                                   │
                                   ┌───────────────┼──────────────────────┐
                                   ▼               ▼                      ▼
                             Modos/Workflows   Comitê de Investimentos   Painel
                                   │
                    Orquestrador (A02): DAG paralelo, repetições, custos
                                   │
   ┌──────────────┬────────────────┼────────────────┬──────────────────┐
   ▼              ▼                ▼                ▼                  ▼
 Especialistas  Quant (A11)    Risco (A14)     Auditor (A15)     Compliance (A16)
 (A03–A10,A12,A13)   │              │                │                  │
   └──────────── Skills (32, versionadas, com permissão por agente) ────┘
                                   │
          DataHub ─► BCB SGS · brapi · CVM (IPE/DFP/ITR/FCA) · RSS · CSV
          CostMonitor (orçamentos, consumo) · throttle por servidor
                                   │
          Memória: mercado · analítica (produção/experimental) · modelos (governança) · usuário · previsões
```

Fluxo da Análise Profunda: Coleta → Análise paralela → Modelagem (só modelo de produção calibrado) →
Contestação (Risco, Auditor, Comitê) → Síntese (CIIO, com hash de integridade) → Comunicação + Compliance → Registro.

## 3. Estrutura de pastas: sugerida × implementada

A estrutura foi adaptada a um pacote Python único (mais simples de testar e implantar). Correspondência:

| Sugerida | Implementada |
|----------|--------------|
| `apps/web/`, `apps/api/` | `oraculo/web/index.html`, `oraculo/api/` (`app.py`, `dashboard.py`) |
| `agents/director/` | `oraculo/agents/chief.py` |
| `agents/orchestrator/` | `oraculo/orchestration/orchestrator.py` |
| `agents/{macro,fundamentals,technical,geopolitics,commodities,news}/` | `oraculo/agents/specialists.py` |
| `agents/quantitative/`, `agents/risk/`, `agents/audit/` | `specialists.py` (`quant`, `risco`, `auditor`) + `agents/committee.py` |
| `agents/compliance/` | skill `compliance_review` + etapa 6 do fluxo |
| (registro dos 18 agentes) | `oraculo/agents/registry.py` |
| `skills/{macro,market,intelligence,prediction,validation}/` | `oraculo/skills/{macro,market,intelligence,predictive,control}.py` |
| `workflows/*` | `oraculo/modes/*` e `oraculo/orchestration/pipeline.py` |
| `data/connectors/` | `oraculo/data/providers/`, `cvm.py`, `news.py` |
| `data/pipelines/`, `data/schemas/` | `oraculo/data/hub.py`, `oraculo/core/types.py` |
| `models/{training,inference,evaluation,registry}/` | `modes/research.py`, `skills/predictive.py`, `skills/control.py`, `memory/store.py` |
| `tests/` | `tests/` (134 testes) |
| `docs/` | `docs/` |
| `infrastructure/` | `Dockerfile`, `render.yaml` (na raiz, exigência do Render) |

## 4. Dependências

**Bibliotecas:** Python ≥ 3.11, numpy, FastAPI, Uvicorn. Testes: pytest, httpx. Sem pandas/scipy (menos
superfície de dependências).

**Serviços externos:**

| Serviço | Credencial | Observação |
|---------|-----------|------------|
| Banco Central (SGS) | Não | API pública |
| CVM – Dados Abertos | Não | Arquivos anuais; baixados 1x/dia (~35 MB) |
| brapi.dev | `BRAPI_TOKEN` (opcional) | Sem token, só alguns ativos e sem Ibovespa |
| InfoMoney, Valor (RSS) | Não | Títulos e links; termos de uso de cada veículo |
| Google Notícias (RSS) | Não | Desligado por padrão: só uso pessoal e não comercial |
| Commodities, curva DI, fluxo estrangeiro, indicadores globais | Licença do fornecedor | Entram via CSV com fonte declarada |
| Render (hospedagem) | Conta do usuário | Blueprint em `render.yaml` |

## 5. Custos estimados

Os preços de fornecedores mudam; os valores abaixo indicam **o que gera custo**, não cotações. Confirme sempre
no site de cada fornecedor.

| Item | Situação atual | Custo |
|------|----------------|-------|
| Dados BCB e CVM | Usados | Gratuitos |
| brapi.dev | Plano gratuito funciona com poucos ativos | Plano pago necessário para cobertura ampla — consultar preço |
| Notícias RSS | Usadas | Gratuitas, sujeitas aos termos |
| Dados licenciados (DI, fluxo, commodities) | Não contratados | Variável por fornecedor (B3, provedores de mercado) |
| Hospedagem | Render plano gratuito (disco temporário) | Plano pago para disco persistente — consultar preço |
| Modelos de linguagem | Não utilizados | Zero hoje; se adotados, entram no monitor de custos |

**Consumo medido (dados reais, uma Análise Profunda da PETR4):** cerca de 4 s de execução, 13 chamadas a fontes
(6 brapi, 3 BCB, 4 RSS) e menos de 1 s de processamento de skills, além do download diário da CVM. Esses números
aparecem no Painel ("Custos de hoje") e ficam gravados por dia (`/api/costs`). Orçamentos diários por provedor
são configuráveis em `ORACULO_ORCAMENTO` (padrão: 1.000 chamadas/dia na brapi).

## 6. Prioridades do escopo — estado

| Prioridade | Itens | Estado |
|------------|-------|--------|
| 1 | Infraestrutura de dados confiável | Concluído: BCB, brapi, CVM, RSS, CSV; procedência, atraso, qualidade, cache, throttle |
| 1 | Segurança | Concluído: senha obrigatória para exposição pública, cabeçalhos HTTP, container sem privilégios, governança de modelos |
| 1 | Dashboard | Concluído: Painel com macro, ativos, carteira, alertas, análises, modelos, custos, saúde das fontes, dados insuficientes |
| 1 | Análises fundamentais | Concluído: DFP/ITR da CVM, múltiplos, tratamento de bancos |
| 2 | Agentes, skills, workflows, auditoria | Concluído: 18 agentes, 32 skills, 8 modos, Comitê, Modo 8 |
| 3 | Modelos preditivos validados e alertas | Infraestrutura concluída (treino, walk-forward, calibração, aprovação humana, alertas). **Nenhum modelo foi aprovado ainda**: a aprovação é decisão humana no Modo 7 |
| 4 | Otimização, simulações avançadas, cobertura | Parcial: simulação de cenários com perguntas; cobertura depende de token da brapi e dados licenciados |

## 7. Regras obrigatórias

As 12 regras estão em `oraculo/core/rules.py` junto com onde cada uma é aplicada (`RULE_ENFORCEMENT`, também em
`/api/info`) e são verificadas por `tests/test_regras.py`.

## 8. Próximos passos sugeridos

1. Token da brapi para cobrir o universo completo e o Ibovespa (benchmark dos retornos anormais).
2. Treinar e validar modelos no Modo 7 com dados reais; aprovar somente os que passarem nos critérios.
3. Fontes licenciadas de curva DI, commodities e indicadores globais (via CSV ou conector dedicado).
4. Banco de dados gerenciado (PostgreSQL) quando houver múltiplos usuários.
