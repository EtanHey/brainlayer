"""Behavioral tests for the read-only T3 SQLite ingestion adapter."""

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.embeddings import EmbeddedChunk
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue


def _create_t3_fixture(path: Path, *, drift: bool = False) -> Path:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE projection_threads (
                thread_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE projection_thread_messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE projection_thread_sessions (
                thread_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                provider_name TEXT,
                provider_session_id TEXT,
                provider_thread_id TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE provider_session_runtime (
                thread_id TEXT PRIMARY KEY,
                provider_name TEXT NOT NULL,
                adapter_key TEXT NOT NULL,
                status TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                resume_cursor_json TEXT,
                runtime_payload_json TEXT
            );
            """
        )
        if drift:
            conn.execute("ALTER TABLE projection_thread_messages RENAME COLUMN text TO body")

        conn.executemany(
            "INSERT INTO projection_threads VALUES (?, ?, ?, ?, ?)",
            [
                ("thread-1", "brainlayer", "Mirrored thread", "2026-07-01T00:00:00Z", "2026-07-01T00:02:00Z"),
                ("thread-2", "golems", "Unmirrored thread", "2026-07-02T00:00:00Z", "2026-07-02T00:01:00Z"),
            ],
        )
        if not drift:
            conn.executemany(
                "INSERT INTO projection_thread_messages VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("message-1", "thread-1", "user", "u", "2026-07-01T00:00:01Z", "2026-07-01T00:00:01Z"),
                    (
                        "message-2",
                        "thread-1",
                        "assistant",
                        "assistant reply",
                        "2026-07-01T00:00:02Z",
                        "2026-07-01T00:00:02Z",
                    ),
                    (
                        "message-3",
                        "thread-2",
                        "user",
                        "a useful unmirrored prompt",
                        "2026-07-02T00:00:01Z",
                        "2026-07-02T00:00:01Z",
                    ),
                ],
            )
        conn.execute(
            "INSERT INTO projection_thread_sessions VALUES (?, ?, ?, ?, ?, ?)",
            ("thread-1", "stopped", "codex", None, None, "2026-07-01T00:02:00Z"),
        )
        conn.execute(
            "INSERT INTO provider_session_runtime VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "thread-1",
                "codex",
                "codex",
                "stopped",
                "2026-07-01T00:02:00Z",
                json.dumps({"threadId": "provider-session-1"}),
                json.dumps({"cwd": "/Users/test/Gits/brainlayer"}),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return path


def test_t3_reader_maps_messages_and_thread_provider_linkage(tmp_path):
    from brainlayer.ingest.t3 import T3Reader

    state_db = _create_t3_fixture(tmp_path / "state.sqlite")

    threads = T3Reader(state_db, health_path=tmp_path / "t3-health.json").read_threads()

    assert [thread.thread_id for thread in threads] == ["thread-1", "thread-2"]
    assert [message.message_id for message in threads[0].messages] == ["message-1", "message-2"]
    assert threads[0].provider_session_id == "provider-session-1"
    assert threads[0].mirrored is True
    assert threads[1].provider_session_id is None
    assert threads[1].mirrored is False


def test_t3_reader_does_not_require_unused_session_projection(tmp_path):
    from brainlayer.ingest.t3 import T3Reader

    state_db = _create_t3_fixture(tmp_path / "state.sqlite")
    with sqlite3.connect(state_db) as conn:
        conn.execute("DROP TABLE projection_thread_sessions")

    threads = T3Reader(state_db, health_path=tmp_path / "t3-health.json").read_threads()

    assert len(threads) == 2
    assert threads[0].provider_session_id == "provider-session-1"


def test_t3_reader_opens_source_with_readonly_wal_safe_uri(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import T3Reader

    state_db = _create_t3_fixture(tmp_path / "state.sqlite")
    connect_calls = []
    real_connect = sqlite3.connect

    def capture_connect(database, *args, **kwargs):
        connect_calls.append((database, kwargs))
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr("brainlayer.ingest.t3.sqlite3.connect", capture_connect)

    T3Reader(state_db, health_path=tmp_path / "t3-health.json").read_threads()

    assert connect_calls[0][0] == f"file:{state_db}?mode=ro&immutable=0"
    assert connect_calls[0][1]["uri"] is True
    assert connect_calls[0][1]["isolation_level"] is None


def test_t3_schema_drift_raises_alarm_and_writes_health(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import T3Reader

    state_db = _create_t3_fixture(tmp_path / "state.sqlite", drift=True)
    health_path = tmp_path / "t3-health.json"
    alarms = []

    def capture_alarm(code, message, context):
        alarms.append((code, message, context))
        raise BrainLayerAlarm(code, message, context)

    monkeypatch.setattr("brainlayer.ingest.t3.raise_alarm", capture_alarm)

    with pytest.raises(BrainLayerAlarm) as raised:
        T3Reader(state_db, health_path=health_path).read_threads()

    assert raised.value.code == "t3_schema_drift"
    assert alarms[0][2]["missing_columns"]["projection_thread_messages"] == ["text"]
    health = json.loads(health_path.read_text())
    assert health["alerting"] is True
    assert "schema_drift" in health["alert_reasons"]
    assert health["failures"][0]["code"] == "t3_schema_drift"


def test_t3_ingestion_keeps_short_messages_and_sets_first_class_provenance(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import ingest_t3

    state_db = _create_t3_fixture(tmp_path / "state.sqlite")
    indexed: list[Chunk] = []

    def capture_index(chunks, *, source_file, project, db_path):
        indexed.extend(chunks)
        assert source_file == str(state_db)
        assert project is None
        assert db_path == tmp_path / "brainlayer.db"
        return len(chunks)

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", capture_index)

    result = ingest_t3(
        state_db,
        db_path=tmp_path / "brainlayer.db",
        health_path=tmp_path / "t3-health.json",
    )

    assert result.threads_seen == 2
    assert result.threads_ingested == 2
    assert result.messages_seen == 3
    assert result.messages_ingested == 3
    assert result.messages_skipped == {}
    assert result.duplicates_accepted == 1
    assert len(indexed) == 3
    assert {chunk.metadata["provenance_class"] for chunk in indexed} == {"t3-thread"}
    assert {chunk.metadata["source"] for chunk in indexed} == {"t3"}
    assert {chunk.metadata["project"] for chunk in indexed} == {"brainlayer", "golems"}
    assert indexed[0].metadata["t3_provider_name"] == "codex"
    assert indexed[0].metadata["t3_provider_session_id"] == "provider-session-1"
    assert indexed[0].metadata["t3_mirrored"] is True
    assert {chunk.metadata["conversation_id"] for chunk in indexed} == {"thread-1", "thread-2"}
    assert {chunk.metadata["chunk_id"] for chunk in indexed} == {
        "t3:thread-1:message-1:0",
        "t3:thread-1:message-2:0",
        "t3:thread-2:message-3:0",
    }


def test_indexer_preserves_stable_identity_timestamp_and_provenance(monkeypatch):
    from brainlayer import index_new

    chunk = Chunk(
        content="T3 message",
        content_type=ContentType.USER_MESSAGE,
        value=ContentValue.HIGH,
        metadata={
            "chunk_id": "t3:thread-1:message-1:0",
            "created_at": "2026-07-01T00:00:01Z",
            "provenance_class": "t3-thread",
            "source": "t3",
            "session_id": "thread-1",
            "sender": "user",
        },
        char_count=11,
    )
    captured = {}

    class FakeStore:
        def upsert_chunks(self, chunks, embeddings, *, deadline_monotonic=None):
            captured["chunks"] = chunks
            captured["embeddings"] = embeddings
            return len(chunks)

    monkeypatch.setattr(index_new, "embed_chunks", lambda chunks, on_progress=None: [EmbeddedChunk(chunk, [0.1])])

    assert index_new.index_chunks_to_sqlite([chunk], source_file="/missing/state.sqlite", store=FakeStore()) == 1
    assert captured["chunks"][0]["id"] == "t3:thread-1:message-1:0"
    assert captured["chunks"][0]["created_at"] == "2026-07-01T00:00:01Z"
    assert captured["chunks"][0]["provenance_class"] == "t3-thread"


def test_ingest_t3_cli_is_a_real_production_entrypoint(tmp_path, monkeypatch):
    from brainlayer.cli import app
    from brainlayer.ingest.t3 import T3IngestionResult

    captured = {}

    def fake_ingest(state_db_path, *, db_path, health_path, dry_run):
        captured.update(
            state_db_path=state_db_path,
            db_path=db_path,
            health_path=health_path,
            dry_run=dry_run,
        )
        return T3IngestionResult(
            threads_seen=45,
            threads_ingested=45,
            messages_seen=2349,
            messages_ingested=2349,
            chunks_planned=2506,
            chunks_indexed=2506,
            duplicates_accepted=34,
        )

    monkeypatch.setattr("brainlayer.ingest.t3.ingest_t3", fake_ingest)
    state_db = tmp_path / "state.sqlite"
    db_path = tmp_path / "brainlayer.db"
    health_path = tmp_path / "t3-health.json"

    result = CliRunner().invoke(
        app,
        [
            "ingest-t3",
            "--state-db",
            str(state_db),
            "--db",
            str(db_path),
            "--health-path",
            str(health_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "state_db_path": state_db,
        "db_path": db_path,
        "health_path": health_path,
        "dry_run": False,
    }
    assert "chunks_indexed=2506" in result.output


def test_read_t3_threads_export_is_removed():
    import brainlayer.ingest as ingest

    assert not hasattr(ingest, "read_t3_threads")
