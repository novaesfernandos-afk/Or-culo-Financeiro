"""Fluxo de colaboração entre agentes para a análise de uma ação (7 etapas).

1. Coleta (Orquestrador)  2. Análise paralela (especialistas)  3. Modelagem (Quant)
4. Contestação (Risco, Auditor e Comitê)  5. Síntese (CIIO)  6. Comunicação (Comunicador + Compliance)
7. Registro (memória analítica e registro de previsões)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from ..agents import committee as committee_mod
from ..agents import specialists as sp
from ..agents.chief import synthesize
from ..agents.communicator import build_report, ensure_flags
from ..core import agent_ids as A
from ..core.rules import has_synthetic
from ..core.types import NewsItem, to_jsonable, utcnow
from ..data.hub import DataHub
from ..memory.store import MemoryStore
from ..skills.base import SkillContext, digest, run_skill
from .orchestrator import Orchestrator, Task

MACRO_KEYS = ("selic_meta", "ipca_12m", "usdbrl_ptax")
DIRECTION_MODEL_ID = "direction_logit_h{h}"


def news_from_dicts(items: list[dict]) -> list[NewsItem]:
    out = []
    for it in items or []:
        out.append(
            NewsItem(
                title=it["titulo"],
                source=it["fonte"],
                published_at=datetime.fromisoformat(it["publicado_em"]),
                retrieved_at=datetime.fromisoformat(it["coletado_em"]) if it.get("coletado_em") else utcnow(),
                body=it.get("texto", ""),
                url=it.get("url"),
                tickers=it.get("tickers", []),
            )
        )
    return out


def collect(d: sp.Dossier, hub: DataHub, orch: Orchestrator, commodity_keys: list[str]) -> None:
    """Etapa 1 - o Orquestrador coleta os dados em paralelo."""
    t = d.ticker

    def fetch(key: str):
        return lambda _deps: hub.get(key)

    tasks = [
        Task("preco", A.ORQUESTRADOR, fetch(f"price:{t}"), retries=2, description=f"Cotações de {t}"),
        Task("volume", A.ORQUESTRADOR, fetch(f"volume:{t}"), retries=2, optional=True, description=f"Volume de {t}"),
        Task("benchmark", A.ORQUESTRADOR, fetch("price:^BVSP"), retries=1, optional=True, description="Ibovespa"),
    ]
    tasks += [Task(f"macro:{k}", A.ORQUESTRADOR, fetch(f"macro:{k}"), retries=2, optional=True, description=f"Macro {k}") for k in MACRO_KEYS]
    tasks += [Task(f"cmd:{k}", A.ORQUESTRADOR, fetch(k), retries=1, optional=True, description=f"Commodity {k}") for k in commodity_keys]
    res = orch.run(tasks)
    d.prices, d.volume, d.benchmark = res.get("preco"), res.get("volume"), res.get("benchmark")
    d.macro = {k: res[f"macro:{k}"] for k in MACRO_KEYS if res.get(f"macro:{k}") is not None}
    d.commodities = {k.split(":", 1)[1]: res[f"cmd:{k}"] for k in commodity_keys if res.get(f"cmd:{k}") is not None}
    d.collection_errors = [f"{r['id']}: {r['error']}" for r in orch.log() if r["status"] != "ok"]


def deep_analysis(ticker: str, hub: DataHub, memory: MemoryStore, options: dict | None = None) -> dict[str, Any]:
    o = options or {}
    ticker = ticker.upper().strip()
    ctx = SkillContext(hub=hub, memory=memory)
    d = sp.Dossier(
        ticker=ticker, ctx=ctx, horizon=int(o.get("horizonte", 5)), sector=o.get("setor"),
        financials=o.get("financials"), financials_source=o.get("financials_source"), valuation_inputs=o.get("valuation"),
        peers=o.get("pares"), news=news_from_dicts(o.get("noticias", [])), geo_events=o.get("eventos_geopoliticos", []),
    )
    stages: dict[str, Any] = {}

    # Etapa 1 - Coleta
    orch = Orchestrator(hub=hub)
    collect(d, hub, orch, o.get("commodities", []))
    stages["1_coleta"] = {"tarefas": orch.log(), "resumo": orch.summary}
    if d.prices is None:
        return {"ticker": ticker, "conclusao": "evidência insuficiente", "erro": "Cotações indisponíveis; nenhuma análise realizada sem dados reais.",
                "etapas": stages}

    # Etapa 2 - Análise paralela dos especialistas
    specialists = {
        A.ECON_BRASIL: sp.economista_brasil, A.ECON_GLOBAL: sp.economista_global, A.GEOPOLITICA: sp.geopolitica,
        A.COMMODITIES: sp.commodities, A.FUNDAMENTALISTA: sp.fundamentalista, A.TECNICO: sp.tecnico, A.SETORIAL: sp.setorial,
        A.FLUXO: sp.fluxo, A.SAZONAL: sp.sazonal, A.NOTICIAS: sp.noticias,
    }
    orch2 = Orchestrator(hub=hub)
    views_res = orch2.run([Task(aid, aid, (lambda f: lambda _deps: f(d))(fn), retries=0, optional=True) for aid, fn in specialists.items()])
    views = {aid: v for aid, v in views_res.items() if v is not None}
    for aid in specialists:
        if aid not in views:
            views[aid] = sp._insufficient(aid, f"Falha na execução: {orch2.records[aid].error}")
    stages["2_analise_paralela"] = {"tarefas": orch2.log(), "resumo": orch2.summary}

    # Etapa 3 - Modelagem com modelo validado (produção), quando existir
    prod = memory.production_model(DIRECTION_MODEL_ID.format(h=d.horizon))
    qview, quant_pkg = sp.quant(d, prod)
    views[A.QUANT] = qview
    quant_digest = digest(quant_pkg)
    stages["3_modelagem"] = {"modelo_producao": f"{prod['model_id']}@{prod['version']}" if prod else None, "hash_integridade": quant_digest}

    # Etapa 4 - Contestação: Risco, Auditor e Comitê de Investimentos
    views[A.RISCO] = sp.risco(d, quant_pkg)
    views[A.AUDITOR] = sp.auditor(d, quant_pkg, quant_digest, digest(quant_pkg))
    synthetic = has_synthetic(s for r in d.results.values() for s in r.sources)
    committee = committee_mod.run_committee(d.results, allow_synthetic=synthetic and bool(o.get("permitir_sinteticos", synthetic)))
    stages["4_contestacao"] = {"risco": views[A.RISCO].to_dict(), "auditor": views[A.AUDITOR].to_dict(), "comite": committee}

    # Etapa 5 - Síntese (o CIIO lê, mas não altera, o pacote quantitativo)
    synthesis = synthesize(views, committee, quant_pkg, integrity_ok=True)
    if digest(quant_pkg) != quant_digest:  # verificação posterior: a síntese não pode alterar o resultado quantitativo
        synthesis = synthesize(views, committee, quant_pkg, integrity_ok=False)
    stages["5_sintese"] = synthesis

    # Etapa 6 - Comunicação + Compliance
    refs = [s for r in d.results.values() for s in r.sources]
    warnings = [w for r in d.results.values() for w in r.warnings] + d.collection_errors
    stale = any("ATRASADO" in w for w in warnings)
    report, structured = build_report(ticker, views, synthesis, committee, quant_pkg, d.results, refs, warnings, synthetic)
    sources_names = sorted({r.source for r in refs})
    comp = run_skill("compliance_review", A.COMPLIANCE, ctx, text=report, sources_count=len(sources_names), uses_synthetic=synthetic,
                     stale_inputs=stale, sources=sources_names, licensed_sources=o.get("fontes_licenciadas"))
    if comp.ok and not comp.output["aprovado"]:
        report = ensure_flags(report, synthetic, stale)
        comp = run_skill("compliance_review", A.COMPLIANCE, ctx, text=report, sources_count=len(sources_names), uses_synthetic=synthetic,
                         stale_inputs=stale, sources=sources_names, licensed_sources=o.get("fontes_licenciadas"))
    published = comp.ok and comp.output["aprovado"]
    stages["6_comunicacao"] = {"compliance": comp.to_dict(), "publicado": published}

    # Etapa 7 - Registro
    result = {
        "ticker": ticker,
        "conclusao": synthesis["conclusao"],
        "relatorio_markdown": report if published else None,
        "relatorio_bloqueado_motivo": None if published else comp.output.get("pendencias") if comp.ok else comp.error,
        "relatorio": structured,
        "comite": committee,
        "sintese": synthesis,
        "agentes": {k: v.to_dict() for k, v in views.items()},
        "quant": quant_pkg,
        "dados_sinteticos": synthetic,
        "dados_atrasados": stale,
        "etapas": stages,
        "skills_executadas": [r.to_dict() for r in d.results.values()],
    }
    dissent = synthesis["contradicoes"] + [{"comite_temas": committee["temas_em_divergencia"]}]
    aid = memory.save_analysis("analise_profunda", ticker, {"ticker": ticker, **{k: v for k, v in o.items() if k != "noticias"}},
                               to_jsonable({k: v for k, v in result.items() if k != "skills_executadas"}),
                               sorted(views), dissent, synthesis["conclusao"])
    fc = quant_pkg.get("previsao")
    if fc:
        pid = memory.log_prediction(
            ticker=ticker, ref_date=fc["data_referencia"], horizon=fc["horizonte_pregoes"], prob_up=fc["prob_alta"],
            expected_return=(quant_pkg.get("distribuicao") or {}).get("media"), model_version=fc["modelo"], model_status=fc["status_modelo"],
            data_digest=quant_digest, rationale=synthesis["justificativa"], analysis_id=aid, synthetic=synthetic,
        )
        stages["7_registro"] = {"analise_id": aid, "previsao_id": pid}
    else:
        stages["7_registro"] = {"analise_id": aid, "previsao_id": None}
    for inv in structured["condicoes_invalidacao"][:3]:
        memory.add_hypothesis(aid, f"{ticker}: {synthesis['conclusao']}", inv)
    result["analise_id"] = aid
    return to_jsonable(result)

