"""Behavioral tests for the read-only T3 SQLite ingestion adapter."""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest
from typer.testing import CliRunner

from brainlayer.alarm import BrainLayerAlarm
from brainlayer.embeddings import EmbeddedChunk
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue


def _is_uuid(value: str | None) -> bool:
    if value is None:
        return False
    try:
        UUID(value)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


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
            CREATE TABLE projection_projects (
                project_id TEXT PRIMARY KEY,
                title TEXT NOT NULL
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
        conn.executemany(
            "INSERT INTO projection_projects VALUES (?, ?)",
            [("brainlayer", "BrainLayer"), ("golems", "Golems")],
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
    assert threads[0].project_name == "BrainLayer"
    assert threads[0].mirrored is True
    assert threads[1].provider_session_id is None
    assert threads[1].project_name == "Golems"
    assert threads[1].mirrored is False


def test_t3_project_mapping_missing_row_falls_back_to_none(tmp_path):
    from brainlayer.ingest.t3 import T3Reader

    state_db = _create_t3_fixture(tmp_path / "state.sqlite")
    with sqlite3.connect(state_db) as conn:
        conn.execute("DELETE FROM projection_projects WHERE project_id = ?", ("golems",))

    threads = T3Reader(state_db, health_path=tmp_path / "t3-health.json").read_threads()

    assert threads[0].project_name == "BrainLayer"
    assert threads[1].project_name is None


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
    assert {chunk.metadata["project"] for chunk in indexed} == {"BrainLayer", "Golems"}
    assert all(not _is_uuid(chunk.metadata["project"]) for chunk in indexed)
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


@pytest.mark.parametrize("projection_version", [1, 2])
def test_ingest_t3_cli_is_a_real_production_entrypoint(tmp_path, monkeypatch, projection_version):
    from brainlayer.cli import app
    from brainlayer.ingest.t3 import T3IngestionResult

    captured = {}

    def fake_ingest(state_db_path, *, db_path, health_path, dry_run, projection_version):
        captured.update(
            state_db_path=state_db_path,
            db_path=db_path,
            health_path=health_path,
            dry_run=dry_run,
            projection_version=projection_version,
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
            "--projection-version",
            str(projection_version),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "state_db_path": state_db,
        "db_path": db_path,
        "health_path": health_path,
        "dry_run": False,
        "projection_version": projection_version,
    }
    assert "chunks_indexed=2506" in result.output


def test_read_t3_threads_export_is_removed():
    import brainlayer.ingest as ingest

    assert not hasattr(ingest, "read_t3_threads")


def _create_v2_fixture(path):
    # V2 stores retain legacy projections: choosing the wrong tables must be observable.
    _create_t3_fixture(path)
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE orchestration_v2_projection_threads (
                thread_id TEXT PRIMARY KEY, project_id TEXT, title TEXT, created_at TEXT,
                default_provider TEXT, active_provider_thread_id TEXT);
            CREATE TABLE orchestration_v2_projection_messages (
                message_id TEXT PRIMARY KEY, thread_id TEXT, role TEXT, created_at TEXT,
                payload_json TEXT, streaming INTEGER);
            CREATE TABLE orchestration_v2_projection_provider_threads (
                provider_thread_id TEXT PRIMARY KEY, provider TEXT, payload_json TEXT);
        """)
        conn.executemany(
            "INSERT INTO orchestration_v2_projection_threads VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("v2-a", "brainlayer", "V2 A", "2026-10-09T01:00:00Z", "codex", "provider-a"),
                ("v2-b", "golems", "V2 B", "2026-10-09T02:00:00Z", "claude", None),
            ],
        )
        conn.execute(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES (?, ?, ?)",
            ("provider-a", "codex", json.dumps({"nativeThreadRef": {"nativeId": "native-a"}})),
        )
        conn.executemany(
            "INSERT INTO orchestration_v2_projection_messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("v2-user", "v2-a", "user", "2026-10-09T01:00:01Z", json.dumps({"text": "u"}), 0),
                (
                    "v2-assistant",
                    "v2-b",
                    "assistant",
                    "2026-10-09T02:00:01Z",
                    json.dumps({"text": "settled V2 response"}),
                    0,
                ),
                ("v2-streaming", "v2-a", "assistant", "2026-10-09T03:00:00Z", json.dumps({"text": "partial"}), 1),
            ],
        )
    return path


def test_v2_reader_uses_selected_projection_preserves_all_projects_and_waits_for_streaming(tmp_path):
    from brainlayer.ingest.t3 import T3Reader

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    before = state.read_bytes()
    reader = T3Reader(state, health_path=None, projection_version=2)
    threads = reader.read_threads()
    assert [t.thread_id for t in threads] == ["v2-a", "v2-b"]
    assert [t.project_name for t in threads] == ["BrainLayer", "Golems"]
    assert threads[0].provider_session_id == "native-a"
    assert threads[0].mirrored is True
    assert threads[1].provider_name == "claude"
    assert threads[1].mirrored is False
    assert [m.message_id for t in threads for m in t.messages] == ["v2-user", "v2-assistant"]
    assert state.read_bytes() == before
    with sqlite3.connect(state) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET streaming=0 WHERE message_id='v2-streaming'")
    assert len(reader.read_threads()[0].messages) == 2
    assert [t.thread_id for t in T3Reader(state, health_path=None).read_threads()] == ["thread-1", "thread-2"]


@pytest.mark.parametrize("bad_payload", ["{broken", "[]", '{"text": null}'])
def test_v2_invalid_payload_alarms_before_indexing(tmp_path, monkeypatch, bad_payload):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(state) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET payload_json=? WHERE message_id='v2-user'", (bad_payload,)
        )

    def no_index(*args, **kwargs):
        pytest.fail("invalid snapshot must not index partial content")

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", no_index)
    health = tmp_path / "health.json"
    with pytest.raises(BrainLayerAlarm, match="t3_payload_invalid"):
        ingest_t3(state, db_path=tmp_path / "dest.db", health_path=health, projection_version=2)
    assert json.loads(health.read_text())["alerting"] is True


@pytest.mark.parametrize(
    "native_ref",
    [
        "invalid-ref",
        17,
        True,
        [],
        {"nativeId": ""},
        {"nativeId": " \t\n"},
        {"nativeId": " native-a"},
        {"nativeId": "native-a "},
        {"nativeId": 17},
        {"nativeId": []},
    ],
)
def test_v2_invalid_native_reference_alarms_before_indexing(tmp_path, monkeypatch, native_ref):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(state) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_provider_threads SET payload_json=? WHERE provider_thread_id='provider-a'",
            (json.dumps({"nativeThreadRef": native_ref}),),
        )

    def no_index(*args, **kwargs):
        pytest.fail("invalid native reference must not index partial content")

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", no_index)
    health_path = tmp_path / "health.json"
    destination = tmp_path / "dest.db"
    with pytest.raises(BrainLayerAlarm, match="t3_payload_invalid"):
        ingest_t3(state, db_path=destination, health_path=health_path, projection_version=2)
    health = json.loads(health_path.read_text())
    assert health["alerting"] is True
    assert health["failures"][0]["code"] == "t3_payload_invalid"
    assert health["failures"][0]["error_type"] == "T3PayloadError"
    assert not destination.exists()


@pytest.mark.parametrize(
    "native_ref,native_id", [(None, None), ({"nativeId": None}, None), ({"nativeId": "native-a"}, "native-a")]
)
def test_v2_valid_native_reference_preserves_nullable_linkage(tmp_path, monkeypatch, native_ref, native_id):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(state) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_provider_threads SET payload_json=? WHERE provider_thread_id='provider-a'",
            (json.dumps({"nativeThreadRef": native_ref}),),
        )
    captured = []

    def index(chunks, **kwargs):
        captured.extend(chunks)
        return len(chunks)

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", index)
    health_path = tmp_path / "health.json"
    result = ingest_t3(state, db_path=tmp_path / "dest.db", health_path=health_path, projection_version=2)
    assert result.chunks_indexed == 2
    assert captured[0].metadata["t3_mirrored"] is True
    assert captured[0].metadata["t3_provider_session_id"] == native_id
    assert json.loads(health_path.read_text())["alerting"] is False


def test_explicit_v2_schema_drift_never_falls_back_to_healthy_legacy(tmp_path):
    from brainlayer.ingest.t3 import T3Reader

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(state) as conn:
        conn.execute("ALTER TABLE orchestration_v2_projection_messages RENAME COLUMN payload_json TO body")
    with pytest.raises(BrainLayerAlarm, match="t3_schema_drift"):
        T3Reader(state, health_path=None, projection_version=2).read_threads()


def test_v2_ingest_preserves_stable_identity_and_dry_run_has_no_destination_write(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "statev2.sqlite")
    captured = []

    def index(chunks, **kwargs):
        captured.extend(chunks)
        assert kwargs["source_file"] == str(state)
        return len(chunks)

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", index)
    destination = tmp_path / "dest.db"
    planned = ingest_t3(state, db_path=destination, health_path=None, projection_version=2, dry_run=True)
    assert planned.chunks_planned == 2 and planned.chunks_indexed == 0
    assert captured == [] and not destination.exists()
    health = tmp_path / "health.json"
    result = ingest_t3(state, db_path=destination, health_path=health, projection_version=2)
    assert result.chunks_indexed == 2
    assert [c.metadata["chunk_id"] for c in captured] == ["t3:v2-a:v2-user:0", "t3:v2-b:v2-assistant:0"]
    assert captured[0].metadata["created_at"] == "2026-10-09T01:00:01Z"
    assert all(c.metadata["provenance_class"] == "t3-thread" for c in captured)
    assert json.loads(health.read_text())["projection_version"] == 2


@pytest.mark.parametrize("failure", [RuntimeError("local embedding failed"), KeyboardInterrupt(), SystemExit(2)])
def test_v2_index_failure_records_alert_and_preserves_original_exception(tmp_path, monkeypatch, failure):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "health.json"

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", fail)
    with pytest.raises(type(failure)) as raised:
        ingest_t3(state, db_path=tmp_path / "destination.db", health_path=health, projection_version=2)
    assert raised.value is failure
    result = json.loads(health.read_text())
    assert result["alerting"] is True and result["alert_reasons"] == ["indexing_failure"]
    assert result["chunks_planned"] == 2
    assert result["failures"][-1]["error_type"] == type(failure).__name__
    assert result["failures"][-1]["code"] == "t3_indexing_failure"
    # Read-only retry reports its actual dry-run result without ever claiming a persisted write.
    planned = ingest_t3(
        state, db_path=tmp_path / "destination.db", health_path=health, projection_version=2, dry_run=True
    )
    assert planned.chunks_indexed == 0 and not (tmp_path / "destination.db").exists()
    assert json.loads(health.read_text())["alerting"] is False


def test_v2_real_indexer_model_guard_failure_health(tmp_path, monkeypatch):
    state = _create_v2_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "health.json"
    destination = tmp_path / "destination.db"
    monkeypatch.setenv("BRAINLAYER_FORBID_EMBEDDING_MODEL", "1")
    # Earlier declared model tests may warm the process-global cache. The guard forbids loading,
    # so use a fresh real CLI process rather than assuming an in-process model is still cold.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "brainlayer",
            "ingest-t3",
            "--state-db",
            str(state),
            "--projection-version",
            "2",
            "--db",
            str(destination),
            "--health-path",
            str(health),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "refusing to load" in result.stderr
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is True and snapshot["alert_reasons"] == ["indexing_failure"]
    assert snapshot["failures"][-1]["error_type"] == "RuntimeError"
    assert not destination.exists()

def test_v2_partial_embeddings_fail_health_preserve_rows_and_replay(tmp_path, monkeypatch):
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    source = _create_v2_fixture(tmp_path / "state.sqlite")
    destination = tmp_path / "destination.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()

    def partial(chunks, **kwargs):
        return [EmbeddedChunk(chunk=chunks[0], embedding=[0.1] * 1024)]

    monkeypatch.setattr("brainlayer.index_new.embed_chunks", partial)
    with pytest.raises(RuntimeError, match="indexed 1 of 2 eligible chunks"):
        ingest_t3(source, db_path=destination, health_path=health, projection_version=2)
    assert json.loads(health.read_text())["alerting"] is True
    store = VectorStore(destination)
    assert list(store.conn.execute("SELECT id FROM chunks")) == [("t3:v2-a:v2-user:0",)]
    assert list(store.conn.execute("SELECT chunk_id FROM chunk_vectors")) == [("t3:v2-a:v2-user:0",)]
    store.conn.execute("UPDATE chunks SET ingested_at=1000")
    first_created = store.conn.execute("SELECT created_at FROM chunks").fetchone()[0]
    store.close()

    def complete(chunks, **kwargs):
        return [EmbeddedChunk(chunk=chunk, embedding=[0.1] * 1024) for chunk in chunks]

    monkeypatch.setattr("brainlayer.index_new.embed_chunks", complete)
    result = ingest_t3(source, db_path=destination, health_path=health, projection_version=2)
    assert result.chunks_indexed == 2 and json.loads(health.read_text())["alerting"] is False
    store = VectorStore(destination, readonly=True)
    expected = {"t3:v2-a:v2-user:0", "t3:v2-b:v2-assistant:0"}
    assert {row[0] for row in store.conn.execute("SELECT id FROM chunks")} == expected
    assert {row[0] for row in store.conn.execute("SELECT chunk_id FROM chunk_vectors")} == expected
    assert store.conn.execute(
        "SELECT created_at,ingested_at FROM chunks WHERE id=?", ("t3:v2-a:v2-user:0",)
    ).fetchone() == (first_created, 1000)
    store.close()
