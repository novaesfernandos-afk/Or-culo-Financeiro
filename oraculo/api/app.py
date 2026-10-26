"""API HTTP (FastAPI) e interface web do Oráculo B3."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import base64
import os
import secrets

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .. import __version__
from ..agents import AGENTS, skills_of
from ..core.costs import COSTS
from ..core.rules import MANDATORY_RULES, RULE_ENFORCEMENT
from .dashboard import build as build_dashboard
from ..core.types import to_jsonable
from ..memory.store import GovernanceError
from ..modes import MODES, Services, run_mode
from ..skills import REGISTRY
from ..skills.knowledge import EVENT_SECTOR_MAP
from ..modes.base import DEFAULT_UNIVERSE, FACTORS

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(services: Services | None = None) -> FastAPI:
    svc = services or Services.create()
    app = FastAPI(title="Oráculo B3", version=__version__)
    app.state.svc = svc
    password = os.environ.get("ORACULO_PASSWORD")
    username = os.environ.get("ORACULO_USER", "oraculo")

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        """Com ORACULO_PASSWORD definido (obrigatório em implantação pública), exige usuário e senha."""
        if password and request.url.path != "/healthz":
            ok = False
            header = request.headers.get("authorization", "")
            if header.lower().startswith("basic "):
                try:
                    user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
                    ok = secrets.compare_digest(user, username) and secrets.compare_digest(pwd, password)
                except (ValueError, UnicodeDecodeError):
                    ok = False
            if not ok:
                return Response("Autenticação necessária", status_code=401, headers={"WWW-Authenticate": 'Basic realm="Oraculo B3"'})
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'"
        )
        if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict:
        return {"ok": True}

    def guard(fn, *a, **kw) -> Any:
        try:
            return to_jsonable(fn(*a, **kw))
        except (ValueError, KeyError, GovernanceError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/info")
    def info() -> dict:
        return {"versao": __version__, "modo_demonstracao": svc.demo, "regras_obrigatorias": MANDATORY_RULES,
                "aplicacao_das_regras": RULE_ENFORCEMENT,
                "fontes_noticias": svc.news.licenses() if svc.news else [], "cvm_ativa": svc.cvm is not None,
                "universo_padrao": DEFAULT_UNIVERSE, "fatores_cenario": FACTORS, "tipos_evento": sorted(EVENT_SECTOR_MAP),
                "uso_dados": svc.hub.usage()}

    @app.get("/api/dashboard")
    def dashboard() -> dict:
        return guard(build_dashboard, svc)

    @app.get("/api/costs")
    def costs() -> dict:
        return {"hoje": COSTS.today(), "historico": svc.memory.usage()}

    @app.get("/api/agents")
    def agents() -> list[dict]:
        return [{**a.to_dict(), "skills_autorizadas": skills_of(a.id)} for a in AGENTS.values()]

    @app.get("/api/skills")
    def skills() -> list[dict]:
        return [s.spec.to_dict() for s in REGISTRY.values()]

    @app.get("/api/skills/{skill_id}")
    def skill(skill_id: str) -> dict:
        if skill_id not in REGISTRY:
            raise HTTPException(404, "Skill inexistente")
        return REGISTRY[skill_id].spec.to_dict()

    @app.get("/api/modes")
    def modes() -> list[dict]:
        return [m.to_dict() for m in sorted(MODES.values(), key=lambda m: m.id)]

    @app.post("/api/modes/{slug}")
    def run(slug: str, params: dict = Body(default_factory=dict)) -> JSONResponse:
        return JSONResponse(guard(run_mode, slug, svc, params))

    @app.post("/api/committee")
    def committee(params: dict = Body(...)) -> dict:
        r = guard(run_mode, "analise_profunda", svc, params)
        return {"ticker": r.get("ticker"), "comite": r.get("comite"), "conclusao_final": r.get("conclusao"), "analise_id": r.get("analise_id")}

    # --- Memória ---------------------------------------------------------------
    m = svc.memory

    @app.get("/api/memory/analyses")
    def analyses(subject: str | None = None) -> list[dict]:
        return guard(m.analyses, subject)

    @app.get("/api/memory/analyses/{aid}")
    def analysis(aid: str) -> dict:
        r = m.get_analysis(aid)
        if r is None:
            raise HTTPException(404, "Análise inexistente")
        return to_jsonable(r)

    @app.get("/api/memory/predictions")
    def predictions() -> list[dict]:
        return m.predictions(include_synthetic=svc.demo)

    @app.get("/api/memory/models")
    def models() -> list[dict]:
        return m.models()

    @app.get("/api/memory/models/{model_id}/history")
    def model_history(model_id: str) -> list[dict]:
        return m.model_history(model_id)

    @app.get("/api/memory/alerts")
    def alert_log() -> list[dict]:
        return m.alert_log()

    @app.get("/api/user")
    def user() -> dict:
        return {"preferencias": m.prefs(), "acompanhados": m.watchlist(), "alertas": m.alerts(), "objetivos": m.goals(), "carteira": m.portfolio()}

    @app.post("/api/user/watchlist")
    def watch(body: dict = Body(...)) -> list[str]:
        guard(m.watch, body["ticker"], body.get("nota", ""))
        return m.watchlist()

    @app.delete("/api/user/watchlist/{ticker}")
    def unwatch(ticker: str) -> list[str]:
        m.unwatch(ticker)
        return m.watchlist()

    @app.post("/api/user/portfolio")
    def position(body: dict = Body(...)) -> list[dict]:
        guard(m.set_position, body["ticker"], float(body["valor"]), body.get("setor"))
        return m.portfolio()

    @app.post("/api/user/alerts")
    def add_alert(body: dict = Body(...)) -> list[dict]:
        if body.get("tipo") not in ("preco_acima", "preco_abaixo", "variacao_diaria_pct"):
            raise HTTPException(400, "tipo deve ser preco_acima, preco_abaixo ou variacao_diaria_pct")
        m.add_alert(body["ticker"], {"tipo": body["tipo"], "valor": float(body["valor"])})
        return m.alerts()

    @app.post("/api/user/prefs")
    def prefs(body: dict = Body(...)) -> dict:
        for k, v in body.items():
            m.set_pref(k, v)
        return m.prefs()

    @app.post("/api/user/goals")
    def goal(body: dict = Body(...)) -> list[dict]:
        m.add_goal(body["texto"])
        return m.goals()

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "localhost") and not os.environ.get("ORACULO_PASSWORD"):
        raise SystemExit("Defina ORACULO_PASSWORD antes de expor o Oráculo B3 fora da máquina local.")
    uvicorn.run(create_app(), host=host, port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
