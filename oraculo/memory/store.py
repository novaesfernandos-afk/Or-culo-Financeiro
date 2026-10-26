"""Memória operacional em quatro níveis (SQLite).

- Memória de mercado: séries históricas e eventos.
- Memória analítica: análises, hipóteses, decisões, evidências e divergências.
- Memória de modelos: versões, parâmetros, métricas, validação e trilha de aprovação.
- Memória do usuário: preferências, ativos acompanhados, alertas, objetivos e carteira.

Também mantém o registro de previsões (Etapa 7) para auditoria futura.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..core.types import Series, to_jsonable, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_series (
    key TEXT, date TEXT, value REAL, source TEXT, retrieved_at TEXT, synthetic INTEGER,
    PRIMARY KEY (key, date)
);
CREATE TABLE IF NOT EXISTS market_events (
    id TEXT PRIMARY KEY, date TEXT, type TEXT, description TEXT, source TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY, created_at TEXT, mode TEXT, subject TEXT, request TEXT,
    result TEXT, agents TEXT, dissent TEXT, conclusion TEXT
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY, analysis_id TEXT, created_at TEXT, text TEXT, invalidation TEXT, status TEXT
);
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT, version TEXT, created_at TEXT, created_by TEXT, status TEXT,
    params TEXT, metrics TEXT, validation TEXT, description TEXT,
    PRIMARY KEY (model_id, version)
);
CREATE TABLE IF NOT EXISTS model_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, version TEXT, action TEXT, actor TEXT, at TEXT, note TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY, made_at TEXT, ref_date TEXT, ticker TEXT, horizon INTEGER, prob_up REAL,
    expected_return REAL, model_version TEXT, model_status TEXT, data_digest TEXT, rationale TEXT,
    analysis_id TEXT, realized_return REAL, benchmark_return REAL, resolved_at TEXT, synthetic INTEGER
);
CREATE TABLE IF NOT EXISTS user_prefs (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS watchlist (ticker TEXT PRIMARY KEY, added_at TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY, ticker TEXT, condition TEXT, created_at TEXT, active INTEGER, last_triggered TEXT
);
CREATE TABLE IF NOT EXISTS goals (id TEXT PRIMARY KEY, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS portfolio (ticker TEXT PRIMARY KEY, value REAL, sector TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS usage_log (
    day TEXT, kind TEXT, name TEXT, calls INTEGER, failures INTEGER, units REAL, seconds REAL, PRIMARY KEY (day, kind, name)
);
CREATE TABLE IF NOT EXISTS alert_log (id TEXT PRIMARY KEY, at TEXT, kind TEXT, message TEXT, payload TEXT);
"""

MODEL_STATUSES = ("experimental", "validado", "aprovado", "producao", "rejeitado", "arquivado")
# Limiares mínimos para um modelo ser considerado validado fora da amostra.
VALIDATION_GATES = {"habilidade_brier_min": 0.01, "blocos_melhores_que_base_min_frac": 0.6, "n_previsoes_min": 250, "ece_max": 0.05}


class GovernanceError(PermissionError):
    """Violação das regras de versionamento e aprovação de modelos."""


def _j(x: Any) -> str:
    return json.dumps(to_jsonable(x), ensure_ascii=False)


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(analyses)")}
            if "environment" not in cols:  # migração: produção x experimental (regra 8)
                self._conn.execute("ALTER TABLE analyses ADD COLUMN environment TEXT DEFAULT 'producao'")
                self._conn.commit()
            pcols = {r[1] for r in self._conn.execute("PRAGMA table_info(portfolio)")}
            for col, typ in (("quantity", "REAL"), ("avg_price", "REAL"), ("avg_price_fees", "REAL"), ("broker", "TEXT")):
                if col not in pcols:  # migração: carteira por quantidade e preço médio
                    self._conn.execute(f"ALTER TABLE portfolio ADD COLUMN {col} {typ}")
            self._conn.commit()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, args)
            self._conn.commit()
            return cur

    def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    # --- Memória de mercado ----------------------------------------------------
    def save_series(self, key: str, s: Series) -> int:
        rows = [(key, d.isoformat(), float(v), s.source, s.retrieved_at.isoformat(), int(s.synthetic)) for d, v in zip(s.dates, s.values)]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO market_series VALUES (?,?,?,?,?,?)", rows)
            self._conn.commit()
        return len(rows)

    def load_series(self, key: str) -> Series | None:
        rows = self._all("SELECT * FROM market_series WHERE key=? ORDER BY date", (key,))
        if not rows:
            return None
        return Series(
            name=key,
            dates=[date.fromisoformat(r["date"]) for r in rows],
            values=[r["value"] for r in rows],
            source=rows[-1]["source"] + " (memória local)",
            retrieved_at=datetime.fromisoformat(rows[-1]["retrieved_at"]),
            synthetic=bool(rows[-1]["synthetic"]),
        )

    def add_event(self, d: str, type_: str, description: str, source: str) -> str:
        if not source:
            raise ValueError("Evento sem fonte não pode ser registrado.")
        eid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO market_events VALUES (?,?,?,?,?,?)", (eid, d, type_, description, source, utcnow().isoformat()))
        return eid

    def events(self, type_: str | None = None) -> list[dict]:
        if type_:
            return self._all("SELECT * FROM market_events WHERE type=? ORDER BY date", (type_,))
        return self._all("SELECT * FROM market_events ORDER BY date")

    # --- Memória analítica -----------------------------------------------------
    def save_analysis(self, mode: str, subject: str, request: dict, result: dict, agents: list[str], dissent: list, conclusion: str,
                      environment: str = "producao") -> str:
        if environment not in ("producao", "experimental"):
            raise ValueError("Ambiente deve ser 'producao' ou 'experimental'.")
        aid = uuid.uuid4().hex[:12]
        self._exec(
            "INSERT INTO analyses (id, created_at, mode, subject, request, result, agents, dissent, conclusion, environment) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (aid, utcnow().isoformat(), mode, subject, _j(request), _j(result), _j(agents), _j(dissent), conclusion, environment),
        )
        return aid

    def analyses(self, subject: str | None = None, limit: int = 50, environment: str | None = "producao") -> list[dict]:
        sql = "SELECT id, created_at, mode, subject, conclusion, dissent, environment FROM analyses WHERE 1=1"
        args: tuple = ()
        if subject:
            sql += " AND subject=?"
            args += (subject,)
        if environment:
            sql += " AND environment=?"
            args += (environment,)
        rows = self._all(sql + " ORDER BY created_at DESC LIMIT ?", args + (limit,))
        for r in rows:
            r["dissent"] = json.loads(r["dissent"])
        return rows

    def get_analysis(self, aid: str) -> dict | None:
        rows = self._all("SELECT * FROM analyses WHERE id=?", (aid,))
        if not rows:
            return None
        r = rows[0]
        for k in ("request", "result", "agents", "dissent"):
            r[k] = json.loads(r[k])
        return r

    def add_hypothesis(self, analysis_id: str, text: str, invalidation: str) -> str:
        hid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO hypotheses VALUES (?,?,?,?,?,?)", (hid, analysis_id, utcnow().isoformat(), text, invalidation, "aberta"))
        return hid

    # --- Memória de modelos (com governança) -------------------------------------
    def register_model(self, model_id: str, version: str, created_by: str, params: dict, description: str = "") -> None:
        if self._all("SELECT 1 FROM models WHERE model_id=? AND version=?", (model_id, version)):
            raise GovernanceError(f"Versão {model_id}@{version} já existe; versões são imutáveis. Crie uma nova versão.")
        self._exec(
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)",
            (model_id, version, utcnow().isoformat(), created_by, "experimental", _j(params), _j({}), _j({}), description),
        )
        self._model_event(model_id, version, "registrado", created_by, description)

    def update_model_params(self, model_id: str, version: str, actor: str, params: dict) -> None:
        m = self.get_model(model_id, version)
        if m is None:
            raise KeyError(f"{model_id}@{version} inexistente")
        if m["status"] != "experimental":
            raise GovernanceError(
                f"{model_id}@{version} está '{m['status']}': parâmetros só podem mudar em nova versão experimental."
            )
        self._exec("UPDATE models SET params=? WHERE model_id=? AND version=?", (_j(params), model_id, version))
        self._model_event(model_id, version, "parametros_alterados", actor, "")

    def attach_validation(self, model_id: str, version: str, actor: str, validation: dict) -> bool:
        """Anexa o relatório de validação fora da amostra. Só o Auditor Quantitativo (A15) certifica."""
        if actor != "A15":
            raise GovernanceError("Somente o Auditor Quantitativo (A15) certifica validações.")
        m = self.get_model(model_id, version)
        if m is None:
            raise KeyError(f"{model_id}@{version} inexistente")
        agg = validation.get("agregado", {})
        gates = {
            "habilidade_brier": agg.get("habilidade_brier", -1) > VALIDATION_GATES["habilidade_brier_min"],
            "consistencia_blocos": agg.get("n_blocos", 0) > 0
            and agg.get("blocos_melhores_que_base", 0) / agg["n_blocos"] >= VALIDATION_GATES["blocos_melhores_que_base_min_frac"],
            "tamanho_amostra": agg.get("n_previsoes", 0) >= VALIDATION_GATES["n_previsoes_min"],
            "calibracao": agg.get("ece", 1) <= VALIDATION_GATES["ece_max"],
            "sem_vazamento": not validation.get("vazamento_detectado", True),
            "acerto_acima_de_sempre_alta": agg.get("acerto", 0) > agg.get("acerto_sempre_alta", 1),
        }
        passed = all(gates.values())
        status = "validado" if passed else "rejeitado"
        self._exec(
            "UPDATE models SET status=?, metrics=?, validation=? WHERE model_id=? AND version=?",
            (status, _j(agg), _j({"gates": gates, "limiares": VALIDATION_GATES}), model_id, version),
        )
        self._model_event(model_id, version, status, actor, _j(gates))
        return passed

    def approve_model(self, model_id: str, version: str, approver: str, note: str = "") -> None:
        """Aprovação humana. Agentes não podem aprovar, nem o próprio autor."""
        if not approver.startswith("user:"):
            raise GovernanceError("Aprovação exige um responsável humano (formato 'user:<nome>'). Agentes não aprovam modelos.")
        m = self.get_model(model_id, version)
        if m is None:
            raise KeyError(f"{model_id}@{version} inexistente")
        if m["created_by"] == approver:
            raise GovernanceError("O autor do modelo não pode aprová-lo.")
        if m["status"] != "validado":
            raise GovernanceError(f"Somente modelos 'validado' podem ser aprovados (atual: {m['status']}).")
        self._exec("UPDATE models SET status='aprovado' WHERE model_id=? AND version=?", (model_id, version))
        self._model_event(model_id, version, "aprovado", approver, note)

    def promote_model(self, model_id: str, version: str, actor: str) -> None:
        if not actor.startswith("user:"):
            raise GovernanceError("Promoção à produção exige um responsável humano.")
        m = self.get_model(model_id, version)
        if m is None or m["status"] != "aprovado":
            raise GovernanceError("Somente modelos aprovados podem ir para produção.")
        for old in self._all("SELECT version FROM models WHERE model_id=? AND status='producao'", (model_id,)):
            self._exec("UPDATE models SET status='arquivado' WHERE model_id=? AND version=?", (model_id, old["version"]))
            self._model_event(model_id, old["version"], "arquivado", actor, f"substituído por {version}")
        self._exec("UPDATE models SET status='producao' WHERE model_id=? AND version=?", (model_id, version))
        self._model_event(model_id, version, "producao", actor, "")

    def reject_model(self, model_id: str, version: str, actor: str, note: str) -> None:
        self._exec("UPDATE models SET status='rejeitado' WHERE model_id=? AND version=?", (model_id, version))
        self._model_event(model_id, version, "rejeitado", actor, note)

    def get_model(self, model_id: str, version: str) -> dict | None:
        rows = self._all("SELECT * FROM models WHERE model_id=? AND version=?", (model_id, version))
        return self._decode_model(rows[0]) if rows else None

    def production_model(self, model_id: str) -> dict | None:
        rows = self._all("SELECT * FROM models WHERE model_id=? AND status='producao'", (model_id,))
        return self._decode_model(rows[0]) if rows else None

    def models(self) -> list[dict]:
        return [self._decode_model(r, with_params=False) for r in self._all("SELECT * FROM models ORDER BY model_id, created_at")]

    def model_history(self, model_id: str) -> list[dict]:
        return self._all("SELECT * FROM model_events WHERE model_id=? ORDER BY id", (model_id,))

    def _decode_model(self, r: dict, with_params: bool = True) -> dict:
        r = dict(r)
        for k in ("params", "metrics", "validation"):
            r[k] = json.loads(r[k]) if (with_params or k != "params") else None
        return r

    def _model_event(self, model_id: str, version: str, action: str, actor: str, note: str) -> None:
        self._exec(
            "INSERT INTO model_events (model_id, version, action, actor, at, note) VALUES (?,?,?,?,?,?)",
            (model_id, version, action, actor, utcnow().isoformat(), note),
        )

    # --- Registro de previsões (Etapa 7) ------------------------------------------
    def log_prediction(
        self, ticker: str, ref_date: str, horizon: int, prob_up: float, model_version: str, model_status: str,
        data_digest: str, rationale: str, expected_return: float | None = None, analysis_id: str | None = None,
        synthetic: bool = False,
    ) -> str:
        pid = uuid.uuid4().hex[:12]
        self._exec(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, utcnow().isoformat(), ref_date, ticker, horizon, prob_up, expected_return, model_version, model_status,
             data_digest, rationale, analysis_id, None, None, None, int(synthetic)),
        )
        return pid

    def predictions(self, include_synthetic: bool = False) -> list[dict]:
        sql = "SELECT * FROM predictions" + ("" if include_synthetic else " WHERE synthetic=0") + " ORDER BY made_at"
        return self._all(sql)

    def resolve_prediction(self, pid: str, realized_return: float, benchmark_return: float | None = None) -> None:
        self._exec(
            "UPDATE predictions SET realized_return=?, benchmark_return=?, resolved_at=? WHERE id=? AND realized_return IS NULL",
            (realized_return, benchmark_return, utcnow().isoformat(), pid),
        )

    def resolve_due_predictions(self, price_lookup) -> int:
        """Resolve previsões vencidas. `price_lookup(ticker) -> Series` (dados reais)."""
        n = 0
        for p in self._all("SELECT * FROM predictions WHERE realized_return IS NULL"):
            s = price_lookup(p["ticker"])
            if s is None:
                continue
            ref = date.fromisoformat(p["ref_date"])
            idx = [i for i, d in enumerate(s.dates) if d <= ref]
            if not idx:
                continue
            i0 = idx[-1]
            if i0 + p["horizon"] < len(s.values):
                self.resolve_prediction(p["id"], float(s.values[i0 + p["horizon"]] / s.values[i0] - 1))
                n += 1
        return n

    # --- Memória do usuário ------------------------------------------------------
    def set_pref(self, key: str, value: Any) -> None:
        self._exec("INSERT OR REPLACE INTO user_prefs VALUES (?,?)", (key, _j(value)))

    def prefs(self) -> dict:
        return {r["key"]: json.loads(r["value"]) for r in self._all("SELECT * FROM user_prefs")}

    def watch(self, ticker: str, note: str = "") -> None:
        self._exec("INSERT OR REPLACE INTO watchlist VALUES (?,?,?)", (ticker.upper(), utcnow().isoformat(), note))

    def unwatch(self, ticker: str) -> None:
        self._exec("DELETE FROM watchlist WHERE ticker=?", (ticker.upper(),))

    def watchlist(self) -> list[str]:
        return [r["ticker"] for r in self._all("SELECT ticker FROM watchlist ORDER BY ticker")]

    def add_alert(self, ticker: str, condition: dict) -> str:
        aid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO alerts VALUES (?,?,?,?,?,?)", (aid, ticker.upper(), _j(condition), utcnow().isoformat(), 1, None))
        return aid

    def alerts(self, active_only: bool = True) -> list[dict]:
        rows = self._all("SELECT * FROM alerts" + (" WHERE active=1" if active_only else ""))
        for r in rows:
            r["condition"] = json.loads(r["condition"])
        return rows

    def add_goal(self, text: str) -> str:
        gid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO goals VALUES (?,?,?)", (gid, text, utcnow().isoformat()))
        return gid

    def goals(self) -> list[dict]:
        return self._all("SELECT * FROM goals ORDER BY created_at")

    def set_position(
        self,
        ticker: str,
        value: float | None = None,
        sector: str | None = None,
        quantity: float | None = None,
        avg_price: float | None = None,
        avg_price_fees: float | None = None,
        broker: str | None = None,
    ) -> None:
        """Grava uma posição por quantidade e preço médio (preferível) ou por valor em R$.

        Com quantidade, `value` guarda o custo (quantidade x preço médio com taxas); o valor de
        mercado é calculado com a cotação do dia. Quantidade ou valor zero remove a posição.
        """
        ticker = ticker.upper().strip()
        if quantity is not None:
            if quantity <= 0:
                self._exec("DELETE FROM portfolio WHERE ticker=?", (ticker,))
                return
            if not avg_price or avg_price <= 0:
                raise ValueError(f"{ticker}: informe o preço médio (maior que zero).")
            if avg_price_fees is not None and avg_price_fees < avg_price:
                raise ValueError(f"{ticker}: preço médio com taxas não pode ser menor que o preço médio sem taxas.")
            value = quantity * (avg_price_fees or avg_price)
        elif value is None or value <= 0:
            self._exec("DELETE FROM portfolio WHERE ticker=?", (ticker,))
            return
        self._exec(
            "INSERT OR REPLACE INTO portfolio (ticker, value, sector, updated_at, quantity, avg_price, avg_price_fees, broker) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (ticker, value, sector, utcnow().isoformat(), quantity, avg_price, avg_price_fees, broker),
        )

    def remove_position(self, ticker: str) -> None:
        self._exec("DELETE FROM portfolio WHERE ticker=?", (ticker.upper().strip(),))

    def portfolio(self) -> list[dict]:
        return self._all("SELECT * FROM portfolio ORDER BY value DESC")

    def add_usage(self, day: str, kind: str, name: str, seconds: float, units: float, ok: bool) -> None:
        self._exec(
            "INSERT INTO usage_log VALUES (?,?,?,1,?,?,?) ON CONFLICT(day, kind, name) DO UPDATE SET "
            "calls=calls+1, failures=failures+excluded.failures, units=units+excluded.units, seconds=seconds+excluded.seconds",
            (day, kind, name, 0 if ok else 1, units, seconds),
        )

    def usage(self, days: int = 30) -> list[dict]:
        return self._all("SELECT * FROM usage_log ORDER BY day DESC, kind, calls DESC LIMIT ?", (days * 50,))

    def log_alert(self, kind: str, message: str, payload: dict | None = None) -> str:
        aid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO alert_log VALUES (?,?,?,?,?)", (aid, utcnow().isoformat(), kind, message, _j(payload or {})))
        return aid

    def alert_log(self, limit: int = 100) -> list[dict]:
        return self._all("SELECT * FROM alert_log ORDER BY at DESC LIMIT ?", (limit,))
