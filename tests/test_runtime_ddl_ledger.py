"""#618: runtime schema repair must record DDL changes in the migration ledger.

Every scenario drives the real runtime schema-ensure path (``VectorStore`` open ->
``_init_db``) against a private scratch SQLite file. No production DB is touched.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import apsw
import pytest
import sqlite_vec

import brainlayer.source_class_migration as source_class_migration
from brainlayer.runtime_store import ReadonlyStore, WriterRuntimeStore, check_fts_counts
from brainlayer.schema_ddl_ledger import (
    EVENT_ID_PREFIX,
    ensure_ddl_ledger,
    schema_fingerprint,
    schema_objects,
)
from brainlayer.vector_store import VectorStore

SENTINEL = "PRIVATE-CONTENT-SENTINEL-618"
EVENT_COLUMNS = (
    "id, detected_at, actor, path, prior_fingerprint, result_fingerprint, "
    "affected_objects, status, error, details, from_pattern, to_pattern, commit_hash, repo"
)


def _connect(path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def _events(path: Path) -> list[dict]:
    conn = _connect(path)
    try:
        ledger_cols = {r[1] for r in conn.execute("PRAGMA table_info(migration_events)")}
        if "actor" not in ledger_cols:  # no ledger, or never extended: no runtime-ddl event exists
            return []
        cols = [c.strip() for c in EVENT_COLUMNS.split(",")]
        rows = conn.execute(
            f"SELECT {EVENT_COLUMNS} FROM migration_events WHERE id LIKE ? ORDER BY detected_at",
            (f"{EVENT_ID_PREFIX}%",),
        ).fetchall()
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


def _schema(path: Path) -> dict:
    conn = _connect(path)
    try:
        return schema_objects(conn)
    finally:
        conn.close()


def _current_db(path: Path) -> None:
    """A fully current runtime DB holding one chunk with sentinel content."""
    with VectorStore(path) as store:
        store.conn.execute(
            "INSERT INTO chunks(id, content, metadata, source_file) VALUES ('c1', ?, '{}', 'synthetic.jsonl')",
            (SENTINEL,),
        )
    # Creating a fresh DB is itself runtime DDL; start each scenario from a clean ledger.
    conn = _connect(path)
    conn.execute("DELETE FROM migration_events")
    conn.close()


def _degrade_to_legacy_fts(path: Path, *, plant_operational_conflict: bool = False) -> None:
    """Reproduce the pre-repair shape: single-column chunks_fts without its sync triggers."""
    conn = _connect(path)
    for trigger in ("chunks_fts_insert", "chunks_fts_delete", "chunks_fts_update"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP TABLE chunks_fts")
    conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, chunk_id UNINDEXED)")
    conn.execute("INSERT INTO chunks_fts(content, chunk_id) VALUES (?, 'c1')", (SENTINEL,))
    if plant_operational_conflict:
        # Forces CREATE VIRTUAL TABLE chunks_fts_operational to fail part-way through
        # the repair: fts5 cannot create its shadow table over an existing one.
        conn.execute("DROP TRIGGER IF EXISTS chunks_fts_operational_insert")
        conn.execute("DROP TABLE chunks_fts_operational")
        conn.execute("CREATE TABLE chunks_fts_operational_data (id INTEGER PRIMARY KEY, block BLOB)")
    conn.close()


def _assert_no_content(event: dict) -> None:
    assert SENTINEL not in json.dumps(event, default=str)


def test_runtime_fts_repair_writes_one_ledger_event_and_repeat_open_writes_none(tmp_path):
    db = tmp_path / "scratch.db"
    _current_db(db)
    _degrade_to_legacy_fts(db)
    prior = _schema(db)
    before = time.time()

    with VectorStore(db) as store:
        fts_cols = {row[1] for row in store.conn.execute("PRAGMA table_info(chunks_fts)")}
    assert {"summary", "key_facts"} <= fts_cols  # the repair really ran

    events = _events(db)
    assert len(events) == 1
    event = events[0]
    result = _schema(db)
    assert event["status"] == "success"
    assert event["error"] is None
    assert event["prior_fingerprint"] == schema_fingerprint(prior)
    assert event["result_fingerprint"] == schema_fingerprint(result)
    assert event["prior_fingerprint"] != event["result_fingerprint"]
    assert before <= event["detected_at"] <= time.time()
    assert event["actor"]
    assert event["path"].endswith("VectorStore._init_db")
    affected = json.loads(event["affected_objects"])
    assert {"chunks_fts", "chunks_fts_insert", "chunks_fts_delete", "chunks_fts_update"} <= set(affected)
    details = json.loads(event["details"])
    assert "chunks_fts" in details["changes"]["altered"]
    assert [step["step"] for step in details["completed_steps"]] == ["fts_schema_rebuild"]
    assert details["rolled_back_steps"] == []
    assert (event["from_pattern"], event["to_pattern"], event["repo"]) == ("runtime-ddl", "schema-ensure", "brainlayer")
    _assert_no_content(event)

    # Repeat open is a no-op ensure pass: same schema, no new event.
    with VectorStore(db):
        pass
    # The repaired schema satisfies the runtime contract (definition hashes included).
    with WriterRuntimeStore(db):
        pass
    assert _schema(db) == result
    assert len(_events(db)) == 1


def test_read_only_validation_and_noop_ensure_write_no_event(tmp_path):
    db = tmp_path / "scratch.db"
    _current_db(db)
    schema = _schema(db)

    with ReadonlyStore(db):
        pass
    check_fts_counts(db)
    with WriterRuntimeStore(db):
        pass
    with VectorStore(db, readonly=True):
        pass
    with VectorStore(db):  # writable ensure over an already-current schema
        pass

    assert _events(db) == []
    assert _schema(db) == schema


def test_failed_fts_repair_rolls_back_and_records_failure_without_result(tmp_path):
    db = tmp_path / "scratch.db"
    _current_db(db)
    _degrade_to_legacy_fts(db, plant_operational_conflict=True)
    prior = _schema(db)

    with pytest.raises(apsw.SQLError, match="chunks_fts_operational_data"):
        VectorStore(db)

    # The destructive drop/recreate rolled back: legacy chunks_fts and its row survive.
    assert _schema(db) == prior
    conn = _connect(db)
    try:
        assert conn.execute("SELECT content FROM chunks_fts WHERE chunk_id = 'c1'").fetchall() == [(SENTINEL,)]
    finally:
        conn.close()

    events = _events(db)
    assert len(events) == 1
    event = events[0]
    assert event["status"] == "failure"
    assert event["result_fingerprint"] is None
    assert event["prior_fingerprint"] == schema_fingerprint(prior)
    assert json.loads(event["affected_objects"]) == []
    assert "chunks_fts_operational_data" in event["error"]
    details = json.loads(event["details"])
    assert details["ddl_committed"] is False
    assert details["committed_changes"] == {"altered": [], "created": [], "dropped": []}
    assert [step["step"] for step in details["rolled_back_steps"]] == ["fts_schema_rebuild"]
    assert details["completed_steps"] == []
    _assert_no_content(event)


def test_fresh_database_creation_is_recorded_once(tmp_path):
    db = tmp_path / "fresh.db"
    with VectorStore(db):
        pass
    events = _events(db)
    assert len(events) == 1
    assert events[0]["status"] == "success"
    assert events[0]["prior_fingerprint"] == schema_fingerprint({})
    assert "chunks" in json.loads(events[0]["affected_objects"])
    with VectorStore(db):
        pass
    assert len(_events(db)) == 1


def test_runtime_ledger_shape_matches_explicit_migration_ledger(tmp_path):
    explicit = tmp_path / "explicit.db"
    runtime = tmp_path / "runtime.db"
    with sqlite3.connect(explicit) as conn:
        source_class_migration._ensure_ledgers(conn)
    with sqlite3.connect(runtime) as conn:
        ensure_ddl_ledger(conn)
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY, applied_at TEXT NOT NULL, details TEXT)""")

    def cols(path):
        with sqlite3.connect(path) as conn:
            return [tuple(r[1:]) for r in conn.execute("PRAGMA table_info(migration_events)")]

    assert cols(explicit) == cols(runtime)

    # Each ensure is a no-op over the other's ledger, in both directions.
    db = tmp_path / "scratch.db"
    with VectorStore(db):
        pass
    schema = _schema(db)
    with sqlite3.connect(db) as conn:
        source_class_migration._ensure_ledgers(conn)
    assert _schema(db) == schema
    with sqlite3.connect(explicit) as conn:
        before = [tuple(r) for r in conn.execute("PRAGMA table_info(migration_events)")]
        ensure_ddl_ledger(conn)
        assert [tuple(r) for r in conn.execute("PRAGMA table_info(migration_events)")] == before


def test_ledger_write_failure_after_committed_ddl_is_loud_not_silent(tmp_path, caplog):
    db = tmp_path / "scratch.db"
    _current_db(db)
    _degrade_to_legacy_fts(db)
    conn = _connect(db)
    conn.execute(
        "CREATE TRIGGER reject_runtime_ddl_event BEFORE INSERT ON migration_events "
        "BEGIN SELECT RAISE(ABORT, 'ledger unavailable'); END"
    )
    conn.close()

    with caplog.at_level("ERROR", logger="brainlayer.vector_store"):
        with VectorStore(db) as store:
            fts_cols = {row[1] for row in store.conn.execute("PRAGMA table_info(chunks_fts)")}

    assert "summary" in fts_cols  # the committed repair is not undone by a ledger failure
    assert "migration_events record failed" in caplog.text
    assert "ledger unavailable" in caplog.text
    assert _events(db) == []


def test_failure_after_autocommitted_ddl_reports_exactly_what_committed(tmp_path):
    db = tmp_path / "scratch.db"
    _current_db(db)
    conn = _connect(db)
    # chunks_fts_insert is recreated (autocommit) before the chunk_tags section, where a
    # view squatting on the chunk_tags name makes the ensure pass fail.
    conn.execute("DROP TRIGGER chunks_fts_insert")
    for trigger in ("chunk_tags_insert", "chunk_tags_update", "chunk_tags_update_clear", "chunk_tags_delete"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
    conn.execute("DROP TABLE chunk_tags")
    conn.execute("CREATE VIEW chunk_tags AS SELECT id AS chunk_id, tags AS tag FROM chunks")
    conn.close()
    prior = _schema(db)

    with pytest.raises(apsw.SQLError):
        VectorStore(db)

    observed = _schema(db)
    assert "chunks_fts_insert" in observed and "chunks_fts_insert" not in prior
    events = _events(db)
    assert len(events) == 1
    event = events[0]
    assert event["status"] == "failure"
    assert event["result_fingerprint"] is None
    assert event["prior_fingerprint"] == schema_fingerprint(prior)
    details = json.loads(event["details"])
    assert details["ddl_committed"] is True
    committed = {name for names in details["committed_changes"].values() for name in names}
    assert committed == set(json.loads(event["affected_objects"]))
    # Exactly the objects whose committed definition changed -- minus our own ledger upgrade.
    assert committed == {n for n in prior.keys() | observed.keys() if prior.get(n) != observed.get(n)} - {
        "migration_events"
    }
    assert "chunks_fts_insert" in committed
    _assert_no_content(event)
