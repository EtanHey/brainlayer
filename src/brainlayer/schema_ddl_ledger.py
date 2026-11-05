"""Record runtime schema (DDL) changes in the ``migration_events`` ledger (#618).

The runtime schema-ensure path (``VectorStore._init_db``) can create, alter, drop
and recreate objects -- including the ``chunks_fts*`` FTS5 tables -- every time a
writable store opens. Before this module, none of that left a ledger row, so a
repaired schema had no recorded time, actor or before/after state.

What gets recorded
------------------
An audit snapshots ``sqlite_schema`` (object name, type, owning table and a hash
of the normalized SQL -- never row contents) before the ensure path runs and
again after it. Only when the two differ is one ``migration_events`` row written:

* ``id`` ``runtime-ddl:<uuid>``; ``from_pattern`` ``runtime-ddl`` marks the kind
* ``detected_at`` epoch seconds, ``actor`` process name + pid, ``path`` the code
  entrypoint that ran the DDL
* ``prior_fingerprint`` / ``result_fingerprint`` and ``affected_objects``
* ``status`` ``success`` or ``failure``, ``error``, ``duration_seconds``
* ``details`` JSON: created/dropped/altered object names and atomic-step outcomes

A validation-only open, or an ensure pass that leaves the schema identical,
writes nothing.

Transaction boundaries (explicit, because they are not all one transaction)
--------------------------------------------------------------------------
* ``_init_db`` historically runs DDL in implicit per-statement autocommit, so a
  failure part-way can leave earlier statements committed. A failure event says
  exactly that: ``details.ddl_committed`` and ``affected_objects`` list only the
  objects whose committed definition actually differs from the prior snapshot,
  and ``result_fingerprint`` stays NULL -- a failed pass never claims a result.
* Destructive repairs run inside :func:`atomic_ddl_step` (a SAVEPOINT). If one
  fails, its DDL is rolled back and the step is listed under
  ``details.rolled_back_steps``; nothing in it is reported as committed.
* A success event is written in its own transaction immediately after the DDL;
  the ledger column upgrade, the result snapshot and the insert are atomic with
  each other (taken under the write lock). A failure event is necessarily written
  after the failed work rolled back -- it cannot live inside the transaction it
  reports on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

EVENT_KIND = "runtime-ddl"
EVENT_ID_PREFIX = f"{EVENT_KIND}:"
_MAX_ERROR_CHARS = 500

# Same extended column set that source_class_migration._ensure_ledgers adds, so
# the explicit migration and the runtime audit share one ledger shape.
LEDGER_EXTENSION_COLUMNS: tuple[tuple[str, str], ...] = (
    ("actor", "TEXT"),
    ("path", "TEXT"),
    ("prior_fingerprint", "TEXT"),
    ("result_fingerprint", "TEXT"),
    ("affected_objects", "TEXT"),
    ("status", "TEXT"),
    ("error", "TEXT"),
    ("duration_seconds", "REAL"),
    ("details", "TEXT"),
)

SchemaObjects = dict[str, tuple[str, str, str]]


def _normalized_sql_hash(sql: str | None) -> str:
    normalized = " ".join(str(sql or "").split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def schema_objects(conn: Any) -> SchemaObjects:
    """Return ``{name: (type, tbl_name, sql_hash)}`` for every user schema object.

    Reads ``sqlite_schema`` only. ``sqlite_*`` internal objects are excluded; table
    contents are never read.
    """
    return {
        str(name): (str(object_type), str(tbl_name), _normalized_sql_hash(sql))
        for name, object_type, tbl_name, sql in conn.execute(
            "SELECT name, type, tbl_name, sql FROM sqlite_schema WHERE name NOT LIKE 'sqlite\\_%' ESCAPE '\\'"
        )
    }


def schema_fingerprint(objects: SchemaObjects) -> str:
    payload = sorted((name, *meta) for name, meta in objects.items())
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def diff_schema(prior: SchemaObjects, result: SchemaObjects) -> dict[str, list[str]]:
    return {
        "created": sorted(result.keys() - prior.keys()),
        "dropped": sorted(prior.keys() - result.keys()),
        "altered": sorted(name for name in prior.keys() & result.keys() if prior[name] != result[name]),
    }


def _affected(changes: dict[str, list[str]]) -> list[str]:
    return sorted({name for names in changes.values() for name in names})


def _columns(conn: Any, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}


def ensure_ddl_ledger(conn: Any) -> None:
    """Create/upgrade ``migration_events`` to the shared extended shape. Idempotent."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS migration_events (
            id TEXT PRIMARY KEY,
            from_pattern TEXT NOT NULL,
            to_pattern TEXT NOT NULL,
            commit_hash TEXT NOT NULL,
            repo TEXT NOT NULL,
            detected_at REAL NOT NULL,
            confidence REAL,
            memories_weakened INTEGER DEFAULT 0
        )
        """
    )
    existing = _columns(conn, "migration_events")
    for name, sql_type in LEDGER_EXTENSION_COLUMNS:
        if name not in existing:
            conn.execute(f'ALTER TABLE migration_events ADD COLUMN "{name}" {sql_type}')


def default_actor() -> str:
    override = os.environ.get("BRAINLAYER_DDL_ACTOR", "").strip()
    if override:
        return override
    program = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "python"
    return f"{program or 'python'}:pid={os.getpid()}"


def _package_version() -> str:
    try:
        from . import __version__

        return f"brainlayer-{__version__}"
    except Exception:  # pragma: no cover - defensive
        return "brainlayer-unknown"


def _in_transaction(conn: Any) -> bool:
    getautocommit = getattr(conn, "getautocommit", None)
    if getautocommit is not None:
        return not getautocommit()
    return bool(getattr(conn, "in_transaction", False))


@dataclass
class RuntimeDDLAudit:
    """Before/after schema audit for one runtime schema-ensure pass."""

    conn: Any
    path: str
    actor: str = field(default_factory=default_actor)
    prior: SchemaObjects = field(default_factory=dict)
    started_monotonic: float = field(default_factory=time.monotonic)
    completed_steps: list[dict[str, Any]] = field(default_factory=list)
    rolled_back_steps: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False

    @classmethod
    def begin(cls, conn: Any, *, path: str, actor: str | None = None) -> "RuntimeDDLAudit":
        audit = cls(conn=conn, path=path, actor=actor or default_actor())
        audit.prior = schema_objects(conn)
        return audit

    def _insert(
        self,
        *,
        status: str,
        result_fingerprint: str | None,
        affected: list[str],
        error: str | None,
        details: dict[str, Any],
    ) -> str:
        event_id = f"{EVENT_ID_PREFIX}{uuid.uuid4().hex}"
        self.conn.execute(
            """
            INSERT INTO migration_events(
                id, from_pattern, to_pattern, commit_hash, repo, detected_at,
                confidence, memories_weakened, actor, path, prior_fingerprint,
                result_fingerprint, affected_objects, status, error,
                duration_seconds, details
            ) VALUES (?, ?, 'schema-ensure', ?, 'brainlayer', ?, 1.0, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                EVENT_KIND,
                _package_version(),
                time.time(),
                self.actor,
                self.path,
                schema_fingerprint(self.prior),
                result_fingerprint,
                json.dumps(affected),
                status,
                error,
                time.monotonic() - self.started_monotonic,
                json.dumps(details, sort_keys=True),
            ),
        )
        return event_id

    def finish_success(self) -> str | None:
        """Write a success event iff the schema changed. Returns the event id or None."""
        if self.finished:
            return None
        self.finished = True
        if schema_objects(self.conn) == self.prior:
            return None
        nested = _in_transaction(self.conn)
        self.conn.execute("SAVEPOINT runtime_ddl_ledger" if nested else "BEGIN IMMEDIATE")
        try:
            ensure_ddl_ledger(self.conn)
            result = schema_objects(self.conn)
            changes = diff_schema(self.prior, result)
            event_id = self._insert(
                status="success",
                result_fingerprint=schema_fingerprint(result),
                affected=_affected(changes),
                error=None,
                details={
                    "changes": changes,
                    "completed_steps": self.completed_steps,
                    "rolled_back_steps": self.rolled_back_steps,
                    "transaction": "implicit_per_statement; event in its own transaction after DDL",
                },
            )
        except BaseException:
            if nested:
                self.conn.execute("ROLLBACK TO runtime_ddl_ledger")
                self.conn.execute("RELEASE runtime_ddl_ledger")
            else:
                self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("RELEASE runtime_ddl_ledger" if nested else "COMMIT")
        return event_id

    def finish_failure(self, exc: BaseException, *, retryable: bool = False) -> str | None:
        """Best-effort failure event. Never claims a result fingerprint; never raises.

        Any transaction the failure left open is rolled back first, so the observed
        schema is what is actually committed. ``affected_objects`` names only objects
        whose committed definition differs from the prior snapshot. A failure that
        committed nothing and rolled back no atomic repair step left the schema
        untouched -- like a no-op pass, it writes no event (the error still reaches
        the caller; a retryable one is retried).
        """
        if self.finished:
            return None
        self.finished = True
        try:
            if _in_transaction(self.conn):
                self.conn.execute("ROLLBACK")
            changes = diff_schema(self.prior, schema_objects(self.conn))
            committed = _affected(changes)
            if not committed and not self.rolled_back_steps:
                return None
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                ensure_ddl_ledger(self.conn)
                # The ledger upgrade above is our own DDL, not the failed pass's.
                event_id = self._insert(
                    status="failure",
                    result_fingerprint=None,
                    affected=committed,
                    error=f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS],
                    details={
                        "ddl_committed": bool(committed),
                        "committed_changes": changes,
                        "completed_steps": self.completed_steps,
                        "rolled_back_steps": self.rolled_back_steps,
                        "retryable": retryable,
                        "transaction": "implicit_per_statement; failure event written after rollback",
                    },
                )
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            self.conn.execute("COMMIT")
            return event_id
        except Exception as ledger_exc:  # never mask the original failure
            logger.warning(
                "runtime DDL failure could not be recorded in migration_events: %s: %s",
                type(ledger_exc).__name__,
                ledger_exc,
            )
            return None


@contextmanager
def atomic_ddl_step(conn: Any, audit: RuntimeDDLAudit | None, name: str, *, targets: list[str]) -> Iterator[None]:
    """Run a destructive DDL step all-or-nothing inside a SAVEPOINT.

    On failure the step's DDL is rolled back, the step is noted on ``audit`` as
    rolled back, and the original exception propagates.
    """
    savepoint = f"ddl_step_{name}"
    conn.execute(f"SAVEPOINT {savepoint}")
    step = {"step": name, "targets": sorted(targets)}
    try:
        yield
    except BaseException as exc:
        try:
            conn.execute(f"ROLLBACK TO {savepoint}")
            conn.execute(f"RELEASE {savepoint}")
        except Exception as rollback_exc:  # pragma: no cover - connection already broken
            logger.error("atomic DDL step %s could not roll back: %s", name, rollback_exc)
            step["rollback_error"] = f"{type(rollback_exc).__name__}: {rollback_exc}"[:_MAX_ERROR_CHARS]
        if audit is not None:
            audit.rolled_back_steps.append({**step, "error": f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]})
        raise
    conn.execute(f"RELEASE {savepoint}")
    if audit is not None:
        audit.completed_steps.append(step)
