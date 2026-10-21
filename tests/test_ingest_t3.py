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

    assert connect_calls[0][0] == f"{state_db.absolute().as_uri()}?mode=ro&immutable=0"
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

    def capture_index(chunks, *, source_file, project, db_path, budget=None):
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
    assert [t.thread_id for t in T3Reader(state, health_path=None, projection_version=1).read_threads()] == [
        "thread-1",
        "thread-2",
    ]


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
    from brainlayer.vector_store import VectorStore

    VectorStore(destination).close()
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
    with VectorStore(destination, readonly=True) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0


def test_v2_partial_embeddings_fail_health_before_write_and_replay(tmp_path, monkeypatch):
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    source = _create_v2_fixture(tmp_path / "state.sqlite")
    destination = tmp_path / "destination.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()

    def partial(chunks, **kwargs):
        target = next(chunk for chunk in chunks if chunk.metadata["chunk_id"] == "t3:v2-a:v2-user:0")
        return [EmbeddedChunk(chunk=target, embedding=[0.1] * 1024)]

    monkeypatch.setattr("brainlayer.index_new.embed_chunks", partial)
    with pytest.raises(ValueError, match="do not match admitted"):
        ingest_t3(source, db_path=destination, health_path=health, projection_version=2)
    assert json.loads(health.read_text())["alerting"] is True
    store = VectorStore(destination)
    assert list(store.conn.execute("SELECT id FROM chunks")) == []
    assert list(store.conn.execute("SELECT chunk_id FROM chunk_vectors")) == []
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
    store.close()


def test_default_sources_prefer_v2_and_keep_legacy(tmp_path, monkeypatch):
    from brainlayer.t3_sources import resolve_t3_sources

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    legacy = _create_t3_fixture(root / "state.sqlite")
    assert resolve_t3_sources() == [(legacy, 1)]
    modern = _create_v2_fixture(root / "statev2.sqlite")
    assert resolve_t3_sources() == [(modern, 2), (legacy, 1)]
    assert resolve_t3_sources(legacy, 1) == [(legacy, 1)]


def test_default_ingest_combines_sources_and_deduplicates_existing_ids(tmp_path, monkeypatch):
    from brainlayer.ingest import t3

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    _create_v2_fixture(root / "statev2.sqlite")
    _create_v2_fixture(root / "state.sqlite")
    captured = []
    monkeypatch.setattr(t3, "_index_chunks", lambda chunks, **kwargs: captured.extend(chunks) or len(chunks))
    result = t3.ingest_t3(db_path=tmp_path / "copy.db", health_path=tmp_path / "health.json")
    assert result.messages_ingested == 2
    assert len(captured) == 2
    assert len({c.metadata["chunk_id"] for c in captured}) == 2
    health = json.loads((tmp_path / "health.json").read_text())
    assert health["alerting"] is False
    assert len(health["state_db_paths"]) == 2


def test_writer_busy_cli_exits_cleanly_with_alerting_retry_health(tmp_path, monkeypatch):
    from brainlayer.cli import app
    from brainlayer.ingest import t3
    from brainlayer.vector_store import WriterInUseError

    source = _create_t3_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "health.json"

    def busy(*args, **kwargs):
        raise WriterInUseError("another writer owns destination")

    monkeypatch.setattr(t3, "_index_chunks", busy)
    result = CliRunner().invoke(app, ["ingest-t3", "--state-db", str(source), "--health-path", str(health)])
    assert result.exit_code == 0, result.output
    assert "retry" in result.output.lower()
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is True
    assert snapshot["alert_reasons"] == ["writer_busy"]
    assert snapshot["failures"][-1]["error_type"] == "WriterInUseError"


def test_default_ingest_retains_distinct_legacy_messages_with_source_paths(tmp_path, monkeypatch):
    from brainlayer.ingest import t3

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    modern = _create_v2_fixture(root / "statev2.sqlite")
    legacy = _create_t3_fixture(root / "state.sqlite")
    indexed = {}

    def capture(chunks, *, source_file, **kwargs):
        indexed[source_file] = [c.metadata["chunk_id"] for c in chunks]
        return len(chunks)

    monkeypatch.setattr(t3, "_index_chunks", capture)
    result = t3.ingest_t3(db_path=tmp_path / "copy.db", health_path=tmp_path / "health.json")
    assert result.messages_ingested == 5
    assert len(indexed[str(modern)]) == 2
    assert len(indexed[str(legacy)]) == 3


def test_broken_default_v2_never_falls_back_to_legacy(tmp_path, monkeypatch):
    from brainlayer.ingest import t3

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    _create_t3_fixture(root / "statev2.sqlite")
    _create_t3_fixture(root / "state.sqlite")
    health = tmp_path / "health.json"
    with pytest.raises(BrainLayerAlarm, match="t3_schema_drift"):
        t3.ingest_t3(dry_run=True, health_path=health)
    assert json.loads(health.read_text())["alerting"] is True


def test_unreadable_default_legacy_source_records_alerting_health(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import ingest_t3

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    (root / "state.sqlite").write_bytes(b"not a SQLite database")
    health = tmp_path / "health.json"
    with pytest.raises(BrainLayerAlarm, match="t3_source_unavailable"):
        ingest_t3(dry_run=True, health_path=health)
    assert json.loads(health.read_text())["alert_reasons"] == ["source_selection_failure"]


def test_unchanged_second_ingest_embeds_and_upserts_zero(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    source = _create_v2_fixture(tmp_path / "statev2.sqlite")
    destination = tmp_path / "copy.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()
    embedded = []
    upserted = []
    real_upsert = VectorStore.upsert_chunks

    def embed(chunks, **kwargs):
        embedded.extend(chunks)
        return [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]

    def upsert(self, chunks, vectors, **kwargs):
        upserted.extend(chunks)
        return real_upsert(self, chunks, vectors, **kwargs)

    monkeypatch.setattr(index_new, "embed_chunks", embed)
    monkeypatch.setattr(VectorStore, "upsert_chunks", upsert)
    assert ingest_t3(source, db_path=destination, health_path=health, projection_version=2).chunks_indexed == 2
    with VectorStore(destination, readonly=True) as store:
        before = list(store.conn.execute("SELECT id,seen_count FROM chunks ORDER BY id"))
        audit_before = store.conn.execute("SELECT COUNT(*) FROM dedupe_audit").fetchone()[0]
    embedded.clear()
    upserted.clear()
    second = ingest_t3(source, db_path=destination, health_path=health, projection_version=2)
    assert second.chunks_indexed == 0
    assert embedded == [] and upserted == []
    with VectorStore(destination, readonly=True) as store:
        assert list(store.conn.execute("SELECT id,seen_count FROM chunks ORDER BY id")) == before
        assert store.conn.execute("SELECT COUNT(*) FROM dedupe_audit").fetchone()[0] == audit_before

    # One edited message is eligible again; the other unchanged message stays untouched.
    with sqlite3.connect(source) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET payload_json=? WHERE message_id='v2-user'",
            (json.dumps({"text": "updated prompt"}),),
        )
    changed = ingest_t3(source, db_path=destination, health_path=health, projection_version=2)
    assert changed.chunks_indexed == 1
    assert len(embedded) == len(upserted) == 1
    assert embedded[0].content == "updated prompt"
    embedded.clear()
    upserted.clear()
    assert ingest_t3(source, db_path=destination, health_path=health, projection_version=2).chunks_indexed == 0
    assert embedded == [] and upserted == []


def test_busy_writer_is_checked_before_embedding(tmp_path, monkeypatch):
    from brainlayer import index_new, runtime_store
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import WriterInUseError

    source = _create_t3_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "health.json"
    embedded = []

    def busy(*args, **kwargs):
        raise WriterInUseError("busy")

    monkeypatch.setattr(runtime_store, "open_writer_store", busy)
    monkeypatch.setattr(index_new, "open_writer_store", busy)
    monkeypatch.setattr(index_new, "embed_chunks", lambda chunks, **kwargs: embedded.extend(chunks) or [])
    with pytest.raises(WriterInUseError):
        ingest_t3(source, db_path=tmp_path / "copy.db", health_path=health)
    assert embedded == []
    assert json.loads(health.read_text())["alert_reasons"] == ["writer_busy"]


def test_explicit_v2_path_auto_detects_without_version(tmp_path):
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.t3_sources import resolve_t3_sources

    source = _create_v2_fixture(tmp_path / "named.sqlite")
    assert resolve_t3_sources(source) == [(source, 2)]
    assert ingest_t3(source, health_path=None, dry_run=True).messages_seen == 2

    from brainlayer.cli import app

    result = CliRunner().invoke(
        app, ["ingest-t3", "--state-db", str(source), "--health-path", str(tmp_path / "health.json"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "messages_seen=2" in result.output


@pytest.mark.parametrize("legacy_failure", ["corrupt", "schema_drift"])
def test_failed_legacy_does_not_block_valid_v2(tmp_path, monkeypatch, legacy_failure):
    from brainlayer.ingest.t3 import ingest_t3

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    _create_v2_fixture(root / "statev2.sqlite")
    if legacy_failure == "corrupt":
        (root / "state.sqlite").write_bytes(b"corrupt legacy")
    else:
        _create_t3_fixture(root / "state.sqlite", drift=True)
    health = tmp_path / "health.json"
    indexed = []
    monkeypatch.setattr(
        "brainlayer.ingest.t3._index_chunks", lambda chunks, **kwargs: indexed.extend(chunks) or len(chunks)
    )
    result = ingest_t3(db_path=tmp_path / "copy.db", health_path=health)
    assert result.messages_seen == 2 and result.chunks_indexed == len(indexed) == 2
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is True
    assert snapshot["alert_reasons"] == ["legacy_source_failure"]


def test_real_v1_v2_overlap_prefers_modern_payload(tmp_path, monkeypatch):
    from brainlayer.ingest import t3

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    modern = _create_v2_fixture(root / "statev2.sqlite")
    legacy = _create_t3_fixture(root / "state.sqlite")
    with sqlite3.connect(legacy) as conn:
        conn.execute("UPDATE projection_threads SET thread_id='v2-a' WHERE thread_id='thread-1'")
        conn.execute("UPDATE projection_thread_messages SET thread_id='v2-a' WHERE thread_id='thread-1'")
        conn.execute(
            "UPDATE projection_thread_messages SET message_id='v2-user',text='stale legacy' WHERE message_id='message-1'"
        )
    captured = []
    monkeypatch.setattr(t3, "_index_chunks", lambda chunks, **kwargs: captured.extend(chunks) or len(chunks))
    result = t3.ingest_t3(db_path=tmp_path / "copy.db", health_path=None)
    assert result.messages_seen == 4
    overlap = [c for c in captured if c.metadata["chunk_id"] == "t3:v2-a:v2-user:0"]
    assert len(overlap) == 1 and overlap[0].content == "u"


def test_s2_run_caps_embedding_and_releases_writer_while_embedding(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from brainlayer import index_new, runtime_store
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    source = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(source) as conn:
        conn.execute("DELETE FROM orchestration_v2_projection_messages")
        conn.executemany(
            "INSERT INTO orchestration_v2_projection_messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    f"cap-{i}",
                    "v2-a",
                    "user",
                    f"2026-10-11T01:00:{i % 60:02d}Z",
                    json.dumps({"text": f"distinct cap prompt {i}"}),
                    0,
                )
                for i in range(225)
            ],
        )
    destination = tmp_path / "copy.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()
    active = [False]
    real_open = runtime_store.open_writer_store

    @contextmanager
    def tracked_open(*args, **kwargs):
        with real_open(*args, **kwargs) as store:
            active[0] = True
            try:
                yield store
            finally:
                active[0] = False

    def embed(chunks, **kwargs):
        assert not active[0], "writer hold spans embedding"
        embedded.extend(chunks)
        return [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]

    embedded = []
    monkeypatch.setattr(runtime_store, "open_writer_store", tracked_open)
    monkeypatch.setattr(index_new, "embed_chunks", embed)
    first = ingest_t3(source, db_path=destination, health_path=health)
    assert len(embedded) == first.chunks_indexed == 200
    snapshot = json.loads(health.read_text())
    assert snapshot["chunks_deferred"] == 25
    assert snapshot["alerting"] is False
    embedded.clear()
    second = ingest_t3(source, db_path=destination, health_path=health)
    assert len(embedded) == second.chunks_indexed == 25
    embedded.clear()
    third = ingest_t3(source, db_path=destination, health_path=health)
    assert third.chunks_indexed == 0 and embedded == []
    with VectorStore(destination, readonly=True) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 225


def test_s2_recheck_skips_chunk_indexed_by_another_writer_during_embedding(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.vector_store import VectorStore

    source = _create_v2_fixture(tmp_path / "statev2.sqlite")
    destination = tmp_path / "copy.db"
    VectorStore(destination).close()

    def embed(chunks, **kwargs):
        computed = [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]
        # A separate writer can persist a selected chunk while T3 has no writer hold.
        with VectorStore(destination) as other:
            index_new.index_chunks_to_sqlite([chunks[0]], str(source), store=other, precomputed_embeddings=computed[:1])
        return computed

    monkeypatch.setattr(index_new, "embed_chunks", embed)
    result = ingest_t3(source, db_path=destination, health_path=tmp_path / "health.json")
    assert result.chunks_indexed == 1
    with VectorStore(destination, readonly=True) as store:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        assert store.conn.execute("SELECT seen_count FROM chunks WHERE id='t3:v2-a:v2-user:0'").fetchone()[0] == 1
        assert store.conn.execute("SELECT COUNT(*) FROM dedupe_audit").fetchone()[0] == 0


def test_s2_cap_is_shared_across_modern_and_legacy_sources(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.ingest import t3
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / ".t3/userdata"
    root.mkdir(parents=True)
    _create_v2_fixture(root / "statev2.sqlite")
    _create_t3_fixture(root / "state.sqlite")
    destination = tmp_path / "copy.db"
    VectorStore(destination).close()
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 3)
    embedded = []

    def embed(chunks, **kwargs):
        embedded.extend(chunks)
        return [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]

    monkeypatch.setattr(index_new, "embed_chunks", embed)
    first = t3.ingest_t3(db_path=destination, health_path=tmp_path / "health.json")
    assert len(embedded) == first.chunks_indexed == 3
    assert first.chunks_deferred == 2
    embedded.clear()
    second = t3.ingest_t3(db_path=destination, health_path=tmp_path / "health.json")
    assert len(embedded) == second.chunks_indexed == 2
    assert second.chunks_deferred == 0


def test_s2_precomputed_vectors_reject_mismatched_scrubbed_content():
    from dataclasses import replace

    from brainlayer import index_new

    chunk = Chunk(
        content="admitted text",
        content_type=ContentType.USER_MESSAGE,
        value=ContentValue.HIGH,
        metadata={"chunk_id": "t3:thread:message:0"},
        char_count=13,
    )

    class Store:
        def upsert_chunks(self, *args, **kwargs):
            pytest.fail("mismatched precomputed embedding must not reach persistence")

    computed = [EmbeddedChunk(replace(chunk, content="stale text"), [0.1] * 1024)]
    with pytest.raises(ValueError, match="do not match admitted"):
        index_new.index_chunks_to_sqlite([chunk], "source.sqlite", store=Store(), precomputed_embeddings=computed)


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("scrub_to_empty", [False, True])
def test_empty_t3_inputs_are_handled_before_embedding_and_budget(tmp_path, monkeypatch, mixed, scrub_to_empty):
    from brainlayer import index_new
    from brainlayer.ingest import t3
    from brainlayer.vector_store import VectorStore

    blank = "synthetic body removed by scrub" if scrub_to_empty else " \n"
    bodies = [blank]
    if mixed:
        bodies = ["First surviving synthetic T3 input.", blank, "Last surviving synthetic T3 input."]
    chunks = [
        Chunk(
            body,
            ContentType.USER_MESSAGE,
            ContentValue.HIGH,
            {"chunk_id": f"t3:empty-fixture:{i}", "created_at": "2026-10-18T00:00:00Z", "allow_duplicate": True},
            len(body),
        )
        for i, body in enumerate(bodies)
    ]
    if scrub_to_empty:
        real_scrub = index_new.scrub_for_storage
        monkeypatch.setattr(
            index_new,
            "scrub_for_storage",
            lambda text, metadata: ("", metadata) if text == blank else real_scrub(text, metadata),
        )
    embedded = []

    def embed(chunks, **kwargs):
        embedded.extend(chunks)
        return [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]

    monkeypatch.setattr(index_new, "embed_chunks", embed)
    destination = tmp_path / "copy.db"
    VectorStore(destination).close()
    expected = [c for c in chunks if c.content != blank]
    budget = t3._IndexBudget(remaining=len(chunks))
    assert t3._index_chunks(
        chunks, source_file="fixture.sqlite", project=None, db_path=destination, budget=budget
    ) == len(expected)
    assert {c.metadata["chunk_id"] for c in embedded} == {c.metadata["chunk_id"] for c in expected}
    assert budget.remaining == 1 and budget.deferred == 0
    with VectorStore(destination, readonly=True) as store:
        assert dict(store.conn.execute("SELECT id, content FROM chunks")) == {
            c.metadata["chunk_id"]: c.content for c in expected
        }
        assert store.conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    embedded.clear()
    replay_budget = t3._IndexBudget(remaining=2)
    assert (
        t3._index_chunks(chunks, source_file="fixture.sqlite", project=None, db_path=destination, budget=replay_budget)
        == 0
    )
    assert not embedded
    assert replay_budget.remaining == 2 and replay_budget.deferred == 0


def test_s2_cap_indexes_newest_messages_first(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.ingest import t3
    from brainlayer.vector_store import VectorStore

    source = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(source) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET created_at='2026-03-06T00:00:00Z' WHERE message_id='v2-user'"
        )
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET created_at='2026-10-11T15:00:00Z' WHERE message_id='v2-assistant'"
        )
    destination = tmp_path / "copy.db"
    VectorStore(destination).close()
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 1)
    embedded = []

    def embed(chunks, **kwargs):
        embedded.extend(chunks)
        return [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]

    monkeypatch.setattr(index_new, "embed_chunks", embed)
    t3.ingest_t3(source, db_path=destination, health_path=None)
    assert [c.metadata["chunk_id"] for c in embedded] == ["t3:v2-b:v2-assistant:0"]


# Production shape from the MBP statev2 projection (2026-10-18, names replaced): the delegate-task name ends in
# "...mockapi-r1", so the derived chunk id reads "<label containing api>:<long value>" to the
# at-rest scrubber, which rewrites it. chunk_write refuses a rewritten identity, and the whole
# T3 run failed on every pass that selected one of these messages.
_SCRUB_TRIPPING_THREAD = (
    "thread:delegated-task:command%3Amcp%3A00000000-1111-4222-8333-444444444444%3Adelegate-task%3Ademo-voice-mockapi-r1"
)
_SCRUB_TRIPPING_MESSAGE = "message:provider:codex:native-item:msg_0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c4b5a697887"


def test_t3_chunk_ids_that_trip_the_secret_scrubber_still_index_and_stay_stable(tmp_path, monkeypatch):
    from brainlayer import index_new
    from brainlayer.ingest.t3 import ingest_t3
    from brainlayer.pipeline.secret_scrub import scrub_for_storage
    from brainlayer.vector_store import VectorStore

    raw_id = f"t3:{_SCRUB_TRIPPING_THREAD}:{_SCRUB_TRIPPING_MESSAGE}:0"
    assert scrub_for_storage(raw_id)[0] != raw_id, "fixture must reproduce the production false positive"

    source = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(source) as conn:
        conn.execute(
            "INSERT INTO orchestration_v2_projection_threads VALUES (?, ?, ?, ?, ?, ?)",
            (_SCRUB_TRIPPING_THREAD, "brainlayer", "delegated", "2026-10-18T01:13:00Z", "codex", None),
        )
        conn.execute(
            "INSERT INTO orchestration_v2_projection_messages VALUES (?, ?, ?, ?, ?, ?)",
            (
                _SCRUB_TRIPPING_MESSAGE,
                _SCRUB_TRIPPING_THREAD,
                "assistant",
                "2026-10-18T01:13:38Z",
                json.dumps({"text": "delegated worker reply"}),
                0,
            ),
        )
    destination = tmp_path / "copy.db"
    health = tmp_path / "health.json"
    VectorStore(destination).close()
    embedded = []

    def embed(chunks, **kwargs):
        embedded.extend(chunks)
        return [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks]

    monkeypatch.setattr(index_new, "embed_chunks", embed)
    result = ingest_t3(source, db_path=destination, health_path=health, projection_version=2)
    assert result.chunks_indexed == 3
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is False and snapshot["failures"] == []

    with VectorStore(destination, readonly=True) as store:
        rows = dict(
            store.conn.execute("SELECT id, metadata FROM chunks WHERE content='delegated worker reply'").fetchall()
        )
        ids = sorted(store.conn.execute("SELECT id FROM chunks").fetchall())
    assert len(rows) == 1
    (stored_id,) = rows
    # The stored identity survives the scrubber unchanged, so replay finds the same row.
    assert scrub_for_storage(stored_id)[0] == stored_id
    assert stored_id.startswith("t3:") and stored_id.endswith(":0")
    # Ordinary ids keep their readable scheme; only the tripping one is digested.
    assert ("t3:v2-a:v2-user:0",) in ids and ("t3:v2-b:v2-assistant:0",) in ids
    # The origin still traces: the T3 message and thread ride in metadata.
    metadata = json.loads(rows[stored_id])
    assert metadata["message_id"] == _SCRUB_TRIPPING_MESSAGE
    assert metadata["source_uri"] == f"t3://thread/{_SCRUB_TRIPPING_THREAD}/message/{_SCRUB_TRIPPING_MESSAGE}"

    embedded.clear()
    assert ingest_t3(source, db_path=destination, health_path=health, projection_version=2).chunks_indexed == 0
    assert embedded == []


def test_t3_indexing_failure_health_names_the_underlying_error(tmp_path, monkeypatch):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "health.json"

    def fail(*args, **kwargs):
        raise ValueError("chunk identifier requires redaction; refusing to change identity")

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", fail)
    with pytest.raises(ValueError):
        ingest_t3(state, db_path=tmp_path / "destination.db", health_path=health, projection_version=2)
    failure = json.loads(health.read_text())["failures"][-1]
    assert failure["code"] == "t3_indexing_failure"
    assert failure["error"] == "chunk identifier requires redaction; refusing to change identity"


def test_t3_conversation_id_that_trips_the_secret_scrubber_is_digested():
    from brainlayer.ingest.t3 import T3Message, T3Thread, _message_chunks
    from brainlayer.pipeline.secret_scrub import scrub_for_storage

    thread_id = "deploy-api-run:Zq8vN3kLp0Wx7RtY2mBc5HjD9sFa"
    assert scrub_for_storage(thread_id)[0] != thread_id
    thread = T3Thread(thread_id=thread_id, project_id="p", title="t", created_at="2026-10-18T00:00:00Z")
    message = T3Message("m1", thread_id, "user", "hello there", "2026-10-18T00:00:01Z")
    (chunk,) = _message_chunks(thread, message)
    for key in ("chunk_id", "conversation_id"):
        assert scrub_for_storage(chunk.metadata[key])[0] == chunk.metadata[key]
    assert chunk.metadata["conversation_id"].startswith("t3:sha256:")
    assert chunk.metadata["t3_thread_id"] == thread_id
    assert _message_chunks(thread, message)[0].metadata == chunk.metadata


_SYNTHETIC_PROVIDER_KEY = "sk-proj-" + "Q7vXk2Lm9PzR4tWb" * 3
_SYNTHETIC_LABELED_VALUE = "Zq8vN3kLp0Wx7RtY2mBc5HjD9sFa"


@pytest.mark.parametrize(
    "message",
    [
        f"embedding backend failed: Authorization {_SYNTHETIC_PROVIDER_KEY} api_key={_SYNTHETIC_LABELED_VALUE}",
        # A token straddling the 500-char cap must be scrubbed before truncation hides its shape.
        # Truncating first would keep "sk-proj-Q7vXk2Lm9Pz": too short to match, still a leak.
        "x" * 480 + f" {_SYNTHETIC_PROVIDER_KEY} api_key={_SYNTHETIC_LABELED_VALUE}",
    ],
    ids=["inline", "straddles_cap"],
)
def test_t3_indexing_failure_health_scrubs_secrets_from_the_error(tmp_path, monkeypatch, message):
    from brainlayer.ingest.t3 import ingest_t3

    state = _create_v2_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "health.json"
    failure = RuntimeError(message)

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr("brainlayer.ingest.t3._index_chunks", fail)
    with pytest.raises(RuntimeError) as raised:
        ingest_t3(state, db_path=tmp_path / "destination.db", health_path=health, projection_version=2)
    assert raised.value is failure
    text = health.read_text()
    assert _SYNTHETIC_LABELED_VALUE[:12] not in text
    assert "sk-proj-Q7vX" not in text and "Q7vXk2Lm9PzR4tWb" not in text
    error = json.loads(text)["failures"][-1]["error"]
    assert len(error) <= 500
    if not message.startswith("x"):
        assert "[REDACTED:openai]" in error and "api_key=[REDACTED:assignment]" in error


@pytest.fixture(
    params=[
        (
            f"reader failed: {_SYNTHETIC_PROVIDER_KEY} api_key={_SYNTHETIC_LABELED_VALUE}",
            "reader failed: [REDACTED:openai] api_key=[REDACTED:assignment]",
        ),
        ("x" * 480 + f" {_SYNTHETIC_PROVIDER_KEY}", "x" * 480 + " [REDACTED:openai]"),
        ("x" * 480 + f" api_key={_SYNTHETIC_LABELED_VALUE}", "x" * 480 + " api_key=[REDACTED:assignment]"),
    ],
    ids=["inline", "provider_straddles_cap", "assignment_straddles_cap"],
)
def health_exception_text(request):
    return request.param


@pytest.mark.parametrize("error_kind", ["os", "sqlite", "operational", "payload"])
def test_t3_reader_failure_health_scrubs_exception_text(tmp_path, monkeypatch, health_exception_text, error_kind):
    from brainlayer.ingest.t3 import T3PayloadError, T3Reader

    state = _create_t3_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "t3-health.json"
    message, scrubbed = health_exception_text
    error_types = {
        "os": OSError,
        "sqlite": sqlite3.DatabaseError,
        "operational": sqlite3.OperationalError,
        "payload": T3PayloadError,
    }
    error = error_types[error_kind](message)
    reader = T3Reader(state, health_path=health)

    def fail(*args):
        raise error

    monkeypatch.setattr(reader, "_read_snapshot", fail)
    expected_code = "t3_payload_invalid" if error_kind == "payload" else "t3_reader_failed"
    with pytest.raises(BrainLayerAlarm) as raised:
        reader.read_threads()
    assert raised.value.code == expected_code
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is True
    assert snapshot["alert_reasons"] == ["reader_failure"]
    failure = snapshot["failures"][0]
    assert failure["code"] == expected_code
    assert failure["error_type"] == type(error).__name__
    assert failure["error"] == scrubbed[:500]


@pytest.mark.parametrize("schema_error", ["no such table", "no such column"])
def test_t3_schema_race_health_scrubs_sqlite_error(tmp_path, monkeypatch, health_exception_text, schema_error):
    from brainlayer.ingest.t3 import T3Reader

    state = _create_t3_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "t3-health.json"
    message, scrubbed = health_exception_text
    # Put the token at the same boundary regardless of the diagnostic prefix length.
    prefix = f"{schema_error}: "
    if message.startswith("x"):
        message, scrubbed = message[len(prefix) :], scrubbed[len(prefix) :]
    error = sqlite3.OperationalError(prefix + message)
    reader = T3Reader(state, health_path=health)

    def fail(*args):
        raise error

    monkeypatch.setattr(reader, "_read_snapshot", fail)
    with pytest.raises(BrainLayerAlarm) as raised:
        reader.read_threads()
    assert raised.value.code == "t3_schema_drift"
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is True
    assert snapshot["alert_reasons"] == ["schema_drift"]
    failure = snapshot["failures"][0]
    assert failure["error_type"] == "OperationalError"
    assert failure["sqlite_error"] == (prefix + scrubbed)[:500]
    assert failure["error"] == (prefix + scrubbed)[:500]


def test_t3_source_selection_failure_health_scrubs_alarm_message(tmp_path, monkeypatch, health_exception_text):
    from brainlayer.ingest import t3

    state = _create_t3_fixture(tmp_path / "state.sqlite")
    health = tmp_path / "t3-health.json"
    message, scrubbed = health_exception_text
    code = "t3_source_unavailable"
    prefix = f"{code}: "
    if message.startswith("x"):
        message, scrubbed = message[len(prefix) :], scrubbed[len(prefix) :]
    alarm = BrainLayerAlarm(code, message, {"state_db_path": str(state)})
    resolve_sources = t3.resolve_t3_sources

    def fail_initial_probe(state_db_path, projection_version, **kwargs):
        if kwargs.get("tolerate_legacy_failure"):
            raise alarm
        return resolve_sources(state_db_path, projection_version, **kwargs)

    monkeypatch.setattr(t3, "resolve_t3_sources", fail_initial_probe)
    with pytest.raises(BrainLayerAlarm) as raised:
        t3.ingest_t3(state, health_path=health, dry_run=True, projection_version=1)
    assert raised.value is alarm
    snapshot = json.loads(health.read_text())
    assert snapshot["alerting"] is True
    assert snapshot["alert_reasons"] == ["source_selection_failure"]
    failure = snapshot["failures"][0]
    assert failure["code"] == code
    assert failure["message"] == (prefix + scrubbed)[:500]
