"""T3 thread side tables: which T3 thread a session memory belongs to.

Two small tables join a session's chunks (by the existing ``chunks.conversation_id``)
to the T3 thread that drove it, so ``brain_expand`` can show the thread's title,
project and link. No chunk is rewritten.

* ``t3_threads``: one row per T3 thread.
* ``t3_thread_sessions``: one row per provider thread of a T3 thread, with the
  BrainLayer ``conversation_id`` of its session once a session chunk exists.

t3-ingest is the only writer (``write_t3_thread_rows``). BrainBar only reads.
``text_owner``, ``owner_reason`` and ``session_high_water`` are computed and
stored for measurement; nothing reads them to decide a write yet (T3 PR 3.3).

The tables are OPTIONAL in the runtime schema contract (``runtime_store``): a
database without them keeps every writer working, and t3-ingest reports
``t3_side_tables.state = "absent"`` in its health instead of writing. They reach
an existing database only through ``python -m brainlayer.t3_thread_tables``, a
reviewed migration the lead runs; a fresh database gets them from ``VectorStore``.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MIGRATION_NAME = "2026_11_19_t3_thread_tables_v1"

T3_THREADS_DDL = """CREATE TABLE IF NOT EXISTS t3_threads (
    thread_id TEXT PRIMARY KEY,
    state_db TEXT NOT NULL,
    title TEXT NOT NULL,
    project_id TEXT,
    project_name TEXT,
    forked_from TEXT,
    created_at TEXT NOT NULL,
    deleted_at TEXT,
    text_owner TEXT NOT NULL CHECK (text_owner IN ('session', 't3')),
    owner_reason TEXT NOT NULL,
    last_message_at TEXT,
    session_high_water TEXT,
    updated_at TEXT NOT NULL
)"""

T3_THREAD_SESSIONS_DDL = """CREATE TABLE IF NOT EXISTS t3_thread_sessions (
    thread_id TEXT NOT NULL,
    provider_thread_id TEXT NOT NULL,
    driver TEXT NOT NULL,
    provider TEXT NOT NULL,
    native_session_id TEXT,
    conversation_id TEXT,
    is_active INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (thread_id, provider_thread_id)
)"""

T3_THREAD_SESSIONS_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_t3_thread_sessions_conversation ON t3_thread_sessions(conversation_id)"
)

T3_THREAD_SCHEMA_OBJECTS = {
    "t3_threads": "table",
    "t3_thread_sessions": "table",
    "idx_t3_thread_sessions_conversation": "index",
}

# The session layer watches exactly these drivers. Nothing is auto-detected.
ELIGIBLE_DRIVERS = frozenset({"claudeAgent", "codex", "cursor"})
# Older T3 schemas have no ``driver`` column; the provider name then stands in.
_PROVIDER_DRIVER_ALIASES = {"codex-worker": "codex"}

OWNER_SESSION = "session"
OWNER_T3 = "t3"
REASON_NO_SESSION_ID = "no_session_id"
REASON_MULTIPLE_SESSIONS = "multiple_sessions"
REASON_DRIVER_NOT_WATCHED = "driver_not_watched"
REASON_NO_SESSION_CHUNKS = "no_session_chunks"
REASON_SESSION_CHUNKS = "session_chunks"


def driver_for(provider: str, driver: str | None) -> str:
    return driver or _PROVIDER_DRIVER_ALIASES.get(provider, provider)


@dataclass(frozen=True)
class T3ProviderThread:
    provider_thread_id: str
    provider: str
    driver: str
    native_session_id: str | None
    is_active: bool


@dataclass(frozen=True)
class T3ThreadCatalogEntry:
    thread_id: str
    state_db: str
    title: str
    project_id: str | None
    project_name: str | None
    forked_from: str | None
    created_at: str
    deleted_at: str | None
    last_message_at: str | None
    provider_threads: tuple[T3ProviderThread, ...] = ()


def ensure_t3_thread_tables(cursor: Any) -> None:
    """Create the side tables and their index if absent. Idempotent."""
    cursor.execute(T3_THREADS_DDL)
    cursor.execute(T3_THREAD_SESSIONS_DDL)
    cursor.execute(T3_THREAD_SESSIONS_INDEX_DDL)


def t3_thread_tables_present(cursor: Any) -> bool:
    names = {
        row[0]: row[1]
        for row in cursor.execute(
            "SELECT name, type FROM sqlite_schema WHERE name IN (?, ?, ?)",
            tuple(T3_THREAD_SCHEMA_OBJECTS),
        )
    }
    return names == T3_THREAD_SCHEMA_OBJECTS


def side_table_state(db_path: Path) -> str:
    """``absent`` | ``empty`` | ``populated``, read through a read-only connection."""
    conn = sqlite3.connect(f"{Path(db_path).absolute().as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        conn.execute("PRAGMA query_only=ON")
        if not t3_thread_tables_present(conn):
            return "absent"
        return "populated" if conn.execute("SELECT EXISTS(SELECT 1 FROM t3_threads)").fetchone()[0] else "empty"
    finally:
        conn.close()


def _active_clause(cursor: Any) -> str:
    from .archive_lifecycle import lifecycle_active_clauses_present

    cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
    clauses = lifecycle_active_clauses_present(cols)
    return " AND ".join(clauses) if clauses else "1=1"


def _codex_conversations(cursor: Any) -> dict[str, str]:
    """Map a Codex native id to its ``rollout-<time>-<uuid>`` conversation.

    One index-only range read. If two conversations ever end in the same id,
    the lexicographically first wins, so the answer is stable across runs.
    """
    found: dict[str, str] = {}
    for (conversation_id,) in cursor.execute(
        "SELECT DISTINCT conversation_id FROM chunks"
        " WHERE conversation_id >= 'rollout-' AND conversation_id < 'rollout.'"
        " ORDER BY conversation_id"
    ):
        if isinstance(conversation_id, str) and len(conversation_id) > 36:
            found.setdefault(conversation_id[-36:], conversation_id)
    return found


def _session_high_water(cursor: Any, conversation_id: str, active: str) -> tuple[int, str | None]:
    """Count and high-water mark of the session layer's own rows for one conversation.

    ``last_seen_at`` counts: a repeat inside one session bumps it instead of adding a row.
    """
    row = cursor.execute(
        "SELECT count(*), strftime('%Y-%m-%dT%H:%M:%fZ', max(max(julianday(created_at)),"
        " coalesce(max(julianday(last_seen_at)), 0)))"
        f" FROM chunks WHERE conversation_id = ? AND source != 't3' AND {active}",  # noqa: S608
        (conversation_id,),
    ).fetchone()
    return int(row[0] or 0), (row[1] if row[0] else None)


def _ownership(
    entry: T3ThreadCatalogEntry, conversations: dict[str, str | None], high_water: dict[str, str | None]
) -> tuple[str, str, str | None]:
    native = [provider for provider in entry.provider_threads if provider.native_session_id]
    if not native:
        return OWNER_T3, REASON_NO_SESSION_ID, None
    if len(native) > 1:
        return OWNER_T3, REASON_MULTIPLE_SESSIONS, None
    session = native[0]
    if session.driver not in ELIGIBLE_DRIVERS:
        return OWNER_T3, REASON_DRIVER_NOT_WATCHED, None
    conversation_id = conversations.get(session.provider_thread_id)
    if conversation_id is None:
        return OWNER_T3, REASON_NO_SESSION_CHUNKS, None
    return OWNER_SESSION, REASON_SESSION_CHUNKS, high_water.get(conversation_id)


def plan_t3_thread_rows(cursor: Any, entries: Iterable[T3ThreadCatalogEntry]) -> dict[str, Any]:
    """Resolve conversations and ownership with reads only; no write lock is held."""
    entries = list(entries)
    active = _active_clause(cursor)
    codex: dict[str, str] | None = None
    seen: dict[str, tuple[int, str | None]] = {}
    conversations: dict[str, str | None] = {}
    for entry in entries:
        for provider in entry.provider_threads:
            native = provider.native_session_id
            candidate = None
            if native and provider.driver in ELIGIBLE_DRIVERS:
                if provider.driver == "codex":
                    if codex is None:
                        codex = _codex_conversations(cursor)
                    candidate = codex.get(native)
                else:
                    candidate = native
            if candidate is not None and candidate not in seen:
                seen[candidate] = _session_high_water(cursor, candidate, active)
            conversations[provider.provider_thread_id] = (
                candidate if candidate is not None and seen[candidate][0] > 0 else None
            )
    high_water = {conversation_id: value[1] for conversation_id, value in seen.items()}
    threads = []
    sessions = []
    for entry in entries:
        owner, reason, water = _ownership(
            entry,
            {p.provider_thread_id: conversations.get(p.provider_thread_id) for p in entry.provider_threads},
            high_water,
        )
        threads.append(
            (
                entry.thread_id,
                entry.state_db,
                entry.title,
                entry.project_id,
                entry.project_name,
                entry.forked_from,
                entry.created_at,
                entry.deleted_at,
                owner,
                reason,
                entry.last_message_at,
                water,
            )
        )
        for provider in entry.provider_threads:
            sessions.append(
                (
                    entry.thread_id,
                    provider.provider_thread_id,
                    provider.driver,
                    provider.provider,
                    provider.native_session_id,
                    conversations.get(provider.provider_thread_id),
                    int(provider.is_active),
                )
            )
    return {"threads": threads, "sessions": sessions}


def _existing(cursor: Any, thread_ids: list[str]) -> tuple[dict[str, tuple], dict[str, set[tuple]]]:
    threads: dict[str, tuple] = {}
    sessions: dict[str, set[tuple]] = {}
    for start in range(0, len(thread_ids), 500):
        ids = json.dumps(thread_ids[start : start + 500])
        for row in cursor.execute(
            "SELECT thread_id, state_db, title, project_id, project_name, forked_from, created_at, deleted_at,"
            " text_owner, owner_reason, last_message_at, session_high_water FROM t3_threads"
            " WHERE thread_id IN (SELECT value FROM json_each(?))",
            (ids,),
        ):
            threads[row[0]] = tuple(row)
        for row in cursor.execute(
            "SELECT thread_id, provider_thread_id, driver, provider, native_session_id, conversation_id, is_active"
            " FROM t3_thread_sessions WHERE thread_id IN (SELECT value FROM json_each(?))",
            (ids,),
        ):
            sessions.setdefault(row[0], set()).add(tuple(row))
    return threads, sessions


def _diff(cursor: Any, plan: dict[str, Any]) -> list[tuple[tuple, set[tuple], bool, bool]]:
    """Planned threads whose row or session set differs from what is stored."""
    by_thread: dict[str, set[tuple]] = {}
    for session in plan["sessions"]:
        by_thread.setdefault(session[0], set()).add(tuple(session))
    existing_threads, existing_sessions = _existing(cursor, [row[0] for row in plan["threads"]])
    changes = []
    for row in plan["threads"]:
        planned_sessions = by_thread.get(row[0], set())
        thread_changed = existing_threads.get(row[0]) != tuple(row)
        sessions_changed = existing_sessions.get(row[0], set()) != planned_sessions
        if thread_changed or sessions_changed:
            changes.append((tuple(row), planned_sessions, thread_changed, sessions_changed))
    return changes


def count_changes(cursor: Any, plan: dict[str, Any]) -> int:
    """How many planned threads would be written; read-only, so an idle run needs no writer."""
    return len(_diff(cursor, plan))


def write_t3_thread_rows(cursor: Any, plan: dict[str, Any], *, now: str | None = None) -> dict[str, int]:
    """Upsert planned rows inside the caller's write transaction.

    A row is rewritten only when its content changed, so ``updated_at`` says when
    the thread's facts last changed and a full scan does not churn the WAL. A
    thread's session rows are replaced as a set, so a provider thread T3 dropped
    does not linger.
    """
    now = now or datetime.now(timezone.utc).isoformat()
    changes = _diff(cursor, plan)
    counts = {"threads_written": 0, "sessions_written": 0, "threads_unchanged": len(plan["threads"]) - len(changes)}
    for row, planned_sessions, thread_changed, sessions_changed in changes:
        thread_id = row[0]
        if thread_changed:
            cursor.execute(
                "INSERT INTO t3_threads(thread_id, state_db, title, project_id, project_name, forked_from,"
                " created_at, deleted_at, text_owner, owner_reason, last_message_at, session_high_water, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(thread_id) DO UPDATE SET state_db=excluded.state_db, title=excluded.title,"
                " project_id=excluded.project_id, project_name=excluded.project_name,"
                " forked_from=excluded.forked_from, created_at=excluded.created_at, deleted_at=excluded.deleted_at,"
                " text_owner=excluded.text_owner, owner_reason=excluded.owner_reason,"
                " last_message_at=excluded.last_message_at, session_high_water=excluded.session_high_water,"
                " updated_at=excluded.updated_at",
                (*row, now),
            )
            counts["threads_written"] += 1
        if sessions_changed:
            cursor.execute("DELETE FROM t3_thread_sessions WHERE thread_id = ?", (thread_id,))
            for session in sorted(planned_sessions):
                cursor.execute(
                    "INSERT INTO t3_thread_sessions(thread_id, provider_thread_id, driver, provider,"
                    " native_session_id, conversation_id, is_active, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (*session, now),
                )
                counts["sessions_written"] += 1
    return counts


def summarize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    owners: dict[str, int] = {}
    for row in plan["threads"]:
        owners[row[9]] = owners.get(row[9], 0) + 1
    resolved_threads = {row[0] for row in plan["sessions"] if row[5] is not None}
    return {
        "threads": len(plan["threads"]),
        "sessions": len(plan["sessions"]),
        "threads_with_conversation": len(resolved_threads),
        "owner_reasons": dict(sorted(owners.items())),
    }


def apply_migration(db_path: Path) -> dict[str, Any]:
    """Create the side tables on an existing database, through the runtime writer.

    The lead runs this after review. It adds two tables and one index and touches
    no existing row; the ``schema_migrations`` row records when it ran.
    """
    from .runtime_store import open_writer_store

    with open_writer_store(Path(db_path)) as store:
        cursor = store.conn.cursor()
        before = t3_thread_tables_present(cursor)
        cursor.execute("BEGIN IMMEDIATE")
        try:
            ensure_t3_thread_tables(cursor)
            cursor.execute(
                "INSERT OR IGNORE INTO schema_migrations(name, applied_at, details) VALUES (?, ?, ?)",
                (
                    MIGRATION_NAME,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps({"objects": sorted(T3_THREAD_SCHEMA_OBJECTS)}),
                ),
            )
            cursor.execute("COMMIT")
        except BaseException:
            cursor.execute("ROLLBACK")
            raise
        return {"db": str(db_path), "already_present": before, "present": t3_thread_tables_present(cursor)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create the T3 thread side tables (T3 PR 3.1).")
    parser.add_argument("--db", required=True, type=Path, help="BrainLayer database to migrate")
    parser.add_argument("--apply", action="store_true", help="create the tables; without it, only report")
    args = parser.parse_args(argv)
    if not args.apply:
        print(json.dumps({"db": str(args.db), "state": side_table_state(args.db), "applied": False}))
        return 0
    print(json.dumps({**apply_migration(args.db), "applied": True}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
