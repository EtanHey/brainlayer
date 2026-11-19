"""T3 PR 3.1: a session memory shows its T3 thread through two side tables.

Every scenario uses a private T3 fixture shaped like production's V2 projection
and a scratch BrainLayer database. No real database is opened.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from brainlayer.embeddings import EmbeddedChunk

CODEX_NATIVE = "01a1f2c6-5375-7eb1-8582-ca7f3f7344ff"
CODEX_CONVERSATION = f"rollout-2026-11-19T03-44-13-{CODEX_NATIVE}"
CURSOR_NATIVE = "agent-b65ce155-6e59-405c-902f-4e8790ebab90"

# thread_id, project_id, title, active provider thread, deleted_at
THREADS = [
    ("th-claude", "p-bl", "Claude thread", "pt-claude", None),
    ("th-codex", "p-bl", "Codex worker thread", "pt-codex", None),
    ("th-cursor", "p-golems", "Cursor thread", "pt-cursor", None),
    ("th-multi", "p-bl", "Two sessions", "pt-multi-2", None),
    ("th-agy", "p-golems", "Antigravity thread", "pt-agy", None),
    ("th-none", "p-bl", "No provider yet", None, None),
    ("th-deleted", "p-bl", "Deleted thread", "pt-deleted", "2026-11-18T10:00:00Z"),
]
# provider_thread_id, thread_id, provider, driver, native id
PROVIDERS = [
    ("pt-claude", "th-claude", "claudeAgent", "claudeAgent", "sess-claude"),
    ("pt-codex", "th-codex", "codex-worker", "codex", CODEX_NATIVE),
    ("pt-cursor", "th-cursor", "cursor", "cursor", CURSOR_NATIVE),
    ("pt-multi-1", "th-multi", "claudeAgent", "claudeAgent", "sess-multi-claude"),
    ("pt-multi-2", "th-multi", "codex", "codex", "01a1aaaa-0000-7000-8000-000000000002"),
    ("pt-agy", "th-agy", "antigravity", "antigravity", "agy-native-1"),
    ("pt-deleted", "th-deleted", "claudeAgent", "claudeAgent", "sess-deleted"),
]


def _t3_fixture(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE projection_projects (project_id TEXT PRIMARY KEY, title TEXT NOT NULL);
            CREATE TABLE orchestration_v2_projection_threads (
                thread_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, title TEXT NOT NULL,
                default_provider TEXT NOT NULL, active_provider_thread_id TEXT, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL, deleted_at TEXT, payload_json TEXT NOT NULL);
            CREATE TABLE orchestration_v2_projection_messages (
                message_id TEXT PRIMARY KEY, thread_id TEXT, role TEXT, created_at TEXT,
                payload_json TEXT, streaming INTEGER, updated_at TEXT DEFAULT '2026-11-19T00:00:00Z');
            CREATE TABLE orchestration_v2_projection_provider_threads (
                provider_thread_id TEXT PRIMARY KEY, thread_id TEXT, provider TEXT NOT NULL, driver TEXT,
                payload_json TEXT NOT NULL);
            CREATE TABLE orchestration_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE,
                aggregate_kind TEXT, stream_id TEXT, stream_version INTEGER,
                event_type TEXT, occurred_at TEXT, command_id TEXT, causation_event_id TEXT,
                correlation_id TEXT, actor_kind TEXT, payload_json TEXT, metadata_json TEXT,
                application_event_version INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE orchestration_v2_projection_metadata (
                projection_name TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                last_sequence INTEGER NOT NULL, updated_at TEXT NOT NULL);
            INSERT INTO orchestration_v2_projection_metadata VALUES ('event-log-pruned-through', 2, 0, '');
            INSERT INTO orchestration_events(event_type, stream_id, occurred_at, payload_json)
                VALUES ('bootstrap', 'project', '2026-11-19T00:00:00Z', '{}');
        """)
        conn.executemany(
            "INSERT INTO projection_projects VALUES (?, ?)", [("p-bl", "brainlayer"), ("p-golems", "golems")]
        )
        for index, (thread_id, project_id, title, active, deleted_at) in enumerate(THREADS):
            created = f"2026-11-19T0{index}:00:00Z"
            conn.execute(
                "INSERT INTO orchestration_v2_projection_threads VALUES (?, ?, ?, 'claudeAgent', ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    project_id,
                    title,
                    active,
                    created,
                    created,
                    deleted_at,
                    json.dumps({"forkedFrom": {"type": "node", "nodeId": f"node:{thread_id}"}}),
                ),
            )
            conn.execute(
                "INSERT INTO orchestration_v2_projection_messages VALUES (?, ?, 'user', ?, ?, 0, ?)",
                (f"m-{thread_id}", thread_id, created, json.dumps({"text": f"hello from {thread_id}"}), created),
            )
        conn.executemany(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES (?, ?, ?, ?, ?)",
            [
                (pt, thread_id, provider, driver, json.dumps({"nativeThreadRef": {"nativeId": native}}))
                for pt, thread_id, provider, driver, native in PROVIDERS
            ],
        )
    return path


def _session_chunk(conn, chunk_id, conversation_id, created_at, *, last_seen_at=None, source="realtime_watcher"):
    conn.execute(
        "INSERT INTO chunks(id, content, metadata, source_file, conversation_id, source, created_at,"
        " last_seen_at, project, content_type) VALUES (?, ?, '{}', 'session.jsonl', ?, ?, ?, ?, 'brainlayer',"
        " 'assistant_text')",
        (chunk_id, f"session text {chunk_id}", conversation_id, source, created_at, last_seen_at),
    )


def _brainlayer_db(path: Path, *, side_tables: bool = True) -> Path:
    from brainlayer.vector_store import VectorStore

    with VectorStore(path) as store:
        conn = store.conn
        _session_chunk(conn, "s-claude-1", "sess-claude", "2026-11-19T00:00:05Z")
        _session_chunk(conn, "s-claude-2", "sess-claude", "2026-11-19T00:01:00Z", last_seen_at="2026-11-19T00:09:00Z")
        _session_chunk(conn, "s-codex-1", CODEX_CONVERSATION, "2026-11-19T01:00:05Z")
        _session_chunk(conn, "s-multi-1", "sess-multi-claude", "2026-11-19T03:00:05Z")
        if not side_tables:
            conn.execute("DROP INDEX idx_t3_thread_sessions_conversation")
            conn.execute("DROP TABLE t3_thread_sessions")
            conn.execute("DROP TABLE t3_threads")
    return path


@pytest.fixture
def env(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.ingest import t3

    monkeypatch.setattr(
        index_new, "embed_chunks", lambda chunks, **kwargs: [EmbeddedChunk(c, [0.1] * 1024) for c in chunks]
    )
    source = _t3_fixture(tmp_path / "statev2.sqlite")
    destination = _brainlayer_db(tmp_path / "brainlayer.db")

    def run(db=destination):
        return t3.ingest_t3(
            source,
            db_path=db,
            health_path=tmp_path / "t3-health.json",
            bookmark_path=tmp_path / f"bookmark-{Path(db).stem}.json",
            projection_version=2,
        )

    run.source, run.db, run.tmp = source, destination, tmp_path
    return run


def _rows(db: Path, sql: str, params=()):
    conn = sqlite3.connect(f"{db.absolute().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _threads(db: Path) -> dict[str, tuple]:
    return {
        row[0]: row[1:]
        for row in _rows(
            db,
            "SELECT thread_id, title, project_name, text_owner, owner_reason, session_high_water, deleted_at"
            " FROM t3_threads",
        )
    }


def _event(source: Path, thread_id: str, kind: str = "thread.metadata-updated") -> None:
    with sqlite3.connect(source) as conn:
        conn.execute(
            "INSERT INTO orchestration_events(event_type, stream_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
            (kind, thread_id, "2026-11-19T12:00:00Z", json.dumps({"id": thread_id})),
        )


def _rename(source: Path, thread_id: str, title: str) -> None:
    with sqlite3.connect(source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_threads SET title = ? WHERE thread_id = ?", (title, thread_id))


# --- schema contract -------------------------------------------------------------------------


def test_fresh_database_has_the_side_tables_pinned_by_the_runtime_contract(tmp_path):
    from brainlayer.runtime_store import _REQUIRED_SQL_HASHES, ReadonlyStore, _schema_sql_hash
    from brainlayer.vector_store import VectorStore

    db = tmp_path / "fresh.db"
    VectorStore(db).close()
    rows = dict(_rows(db, "SELECT name, sql FROM sqlite_schema WHERE name LIKE '%t3_thread%' AND sql IS NOT NULL"))
    assert set(rows) == {"t3_threads", "t3_thread_sessions", "idx_t3_thread_sessions_conversation"}
    for name, sql in rows.items():
        assert _schema_sql_hash(sql) in _REQUIRED_SQL_HASHES[name]
    with ReadonlyStore(db):
        pass


def test_runtime_writer_still_opens_a_database_the_lead_has_not_migrated(tmp_path):
    from brainlayer.runtime_store import SchemaFingerprintMismatch, WriterRuntimeStore

    db = _brainlayer_db(tmp_path / "unmigrated.db", side_tables=False)
    with WriterRuntimeStore(db):
        pass
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t3_threads (thread_id TEXT PRIMARY KEY)")
    conn.close()
    with pytest.raises(SchemaFingerprintMismatch, match="t3_threads"):
        WriterRuntimeStore(db)


def test_migration_adds_the_side_tables_and_touches_no_row(tmp_path, capsys):
    from brainlayer import t3_thread_tables

    db = _brainlayer_db(tmp_path / "unmigrated.db", side_tables=False)
    counts = "SELECT (SELECT count(*) FROM chunks), (SELECT count(*) FROM chunk_id_alias), (SELECT count(*) FROM dedupe_audit)"
    before = _rows(db, counts)

    assert t3_thread_tables.main(["--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "absent"
    first = t3_thread_tables.apply_migration(db)
    assert first["already_present"] is False and first["present"] is True
    second = t3_thread_tables.apply_migration(db)
    assert second["already_present"] is True

    assert _rows(db, counts) == before
    assert _rows(db, "SELECT count(*) FROM schema_migrations WHERE name = ?", (t3_thread_tables.MIGRATION_NAME,)) == [
        (1,)
    ]
    assert t3_thread_tables.side_table_state(db) == "empty"


# --- the writer --------------------------------------------------------------------------------


def test_every_thread_gets_one_row_and_every_provider_thread_one_session_row(env):
    result = env()
    assert result.side_tables["state"] == "empty" and result.side_tables["scope"] == "all"

    threads = _threads(env.db)
    assert set(threads) == {row[0] for row in THREADS}
    assert threads["th-claude"] == (
        "Claude thread",
        "brainlayer",
        "session",
        "session_chunks",
        "2026-11-19T00:09:00.000Z",  # last_seen_at beats created_at
        None,
    )
    assert threads["th-codex"][2:4] == ("session", "session_chunks")
    assert threads["th-cursor"][2:4] == ("t3", "no_session_chunks")
    assert threads["th-multi"][2:4] == ("t3", "multiple_sessions")
    assert threads["th-agy"][2:4] == ("t3", "driver_not_watched")
    assert threads["th-none"][2:4] == ("t3", "no_session_id")
    assert threads["th-deleted"][5] == "2026-11-18T10:00:00Z"
    assert all(row[4] is None for key, row in threads.items() if row[2] == "t3")

    sessions = {
        row[0]: row[1:]
        for row in _rows(
            env.db,
            "SELECT provider_thread_id, thread_id, driver, provider, native_session_id, conversation_id, is_active"
            " FROM t3_thread_sessions",
        )
    }
    assert set(sessions) == {row[0] for row in PROVIDERS}
    assert sessions["pt-claude"][4:] == ("sess-claude", 1)
    assert sessions["pt-codex"][1:5] == ("codex", "codex-worker", CODEX_NATIVE, CODEX_CONVERSATION)
    assert sessions["pt-cursor"][4] is None  # no session chunk yet
    assert sessions["pt-multi-1"][4:] == ("sess-multi-claude", 0)
    assert sessions["pt-multi-2"][4:] == (None, 1)
    assert sessions["pt-agy"][4] is None
    forked = _rows(env.db, "SELECT forked_from FROM t3_threads WHERE thread_id = 'th-claude'")[0][0]
    assert json.loads(forked) == {"type": "node", "nodeId": "node:th-claude"}


def test_side_tables_change_no_chunk_alias_or_audit_row(env):
    """The same T3 input writes the same chunks whether or not the side tables exist."""
    bare = _brainlayer_db(env.tmp / "bare.db", side_tables=False)
    assert env().side_tables["threads_written"] == len(THREADS)
    assert env(bare).side_tables == {"state": "absent"}
    snapshot = (
        "SELECT id, content, conversation_id, source, metadata FROM chunks ORDER BY id",
        "SELECT * FROM chunk_id_alias ORDER BY 1",
        "SELECT count(*) FROM dedupe_audit",
    )
    for sql in snapshot:
        assert _rows(env.db, sql) == _rows(bare, sql)
    assert _rows(bare, "SELECT count(*) FROM sqlite_schema WHERE name LIKE 't3_thread%'") == [(0,)]
    health = json.loads((env.tmp / "t3-health.json").read_text())
    assert health["t3_side_tables"] == {"state": "absent"}


def test_a_rename_reaches_the_table_on_the_next_incremental_run(env):
    env()
    _rename(env.source, "th-claude", "Renamed thread")
    _event(env.source, "th-claude")
    result = env()
    assert result.mode == "incremental"
    assert result.side_tables["scope"] == "touched"
    assert result.side_tables["threads_written"] == 1
    assert _threads(env.db)["th-claude"][0] == "Renamed thread"
    assert _threads(env.db)["th-codex"][0] == "Codex worker thread"


def test_a_rename_reaches_the_table_on_a_bounded_run_after_a_prune(env):
    """A bounded window selects messages only, so the catalog rebuilds every thread."""
    env()
    _rename(env.source, "th-agy", "Renamed during the gap")
    _event(env.source, "th-agy")
    with sqlite3.connect(env.source) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_metadata SET last_sequence = 2"
            " WHERE projection_name = 'event-log-pruned-through'"
        )
    result = env()
    assert result.mode == "bounded"
    assert result.side_tables["scope"] == "all"
    assert _threads(env.db)["th-agy"][0] == "Renamed during the gap"


def test_an_idle_run_takes_no_writer(env, monkeypatch):
    env()

    def forbidden(*args, **kwargs):
        pytest.fail("an idle run opened the writer")

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", forbidden)
    result = env()
    assert result.mode == "incremental" and result.events_read == 0
    assert result.side_tables["state"] == "populated"


def test_a_side_table_failure_holds_the_bookmark_and_the_next_run_repairs_it(env, monkeypatch):
    from brainlayer import t3_thread_tables

    env()
    bookmark = env.tmp / "bookmark-brainlayer.json"
    held = bookmark.read_text()
    _rename(env.source, "th-codex", "Renamed while writes fail")
    _event(env.source, "th-codex")

    def fail(*args, **kwargs):
        raise RuntimeError("side table write refused")

    monkeypatch.setattr(t3_thread_tables, "write_t3_thread_rows", fail)
    with pytest.raises(RuntimeError, match="side table write refused"):
        env()
    assert bookmark.read_text() == held
    health = json.loads((env.tmp / "t3-health.json").read_text())
    assert health["alerting"] is True and "side_table_failure" in health["alert_reasons"]
    assert health["bookmark_held"] == "side_table_failure"
    assert _threads(env.db)["th-codex"][0] == "Codex worker thread"

    monkeypatch.undo()
    from brainlayer import index_new

    monkeypatch.setattr(
        index_new, "embed_chunks", lambda chunks, **kwargs: [EmbeddedChunk(c, [0.1] * 1024) for c in chunks]
    )
    env()
    assert _threads(env.db)["th-codex"][0] == "Renamed while writes fail"


def test_an_unreadable_inactive_provider_payload_is_counted_and_never_blocks_ingest(env):
    with sqlite3.connect(env.source) as conn:
        conn.execute(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES"
            " ('pt-claude-old', 'th-claude', 'claudeAgent', 'claudeAgent', '{not json')"
        )
    result = env()
    assert result.chunks_indexed == len(THREADS)
    assert result.side_tables["invalid_provider_payloads"] == 1
    # The unreadable provider thread has no native id, so th-claude keeps one session.
    assert _threads(env.db)["th-claude"][2] == "session"
    assert _rows(
        env.db, "SELECT native_session_id FROM t3_thread_sessions WHERE provider_thread_id='pt-claude-old'"
    ) == [(None,)]


def test_session_rows_follow_t3_when_a_provider_thread_disappears(env):
    env()
    with sqlite3.connect(env.source) as conn:
        conn.execute("DELETE FROM orchestration_v2_projection_provider_threads WHERE provider_thread_id='pt-multi-2'")
        conn.execute(
            "UPDATE orchestration_v2_projection_threads SET active_provider_thread_id='pt-multi-1'"
            " WHERE thread_id='th-multi'"
        )
    _event(env.source, "th-multi")
    env()
    assert _rows(env.db, "SELECT provider_thread_id, is_active FROM t3_thread_sessions WHERE thread_id='th-multi'") == [
        ("pt-multi-1", 1)
    ]
    assert _threads(env.db)["th-multi"][2:4] == ("session", "session_chunks")
