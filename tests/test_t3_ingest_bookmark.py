"""Replay ordering with T3's production event and prune-marker shapes."""

import json
import sqlite3

import pytest

from brainlayer.ingest import t3
from tests.test_ingest_t3 import _create_t3_fixture, _create_v2_fixture


@pytest.fixture
def run(tmp_path, monkeypatch):
    source = _create_v2_fixture(tmp_path / "statev2.sqlite")
    with sqlite3.connect(source) as conn:
        conn.executescript("""
            CREATE TABLE orchestration_events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE,
                aggregate_kind TEXT, stream_id TEXT, stream_version INTEGER,
                event_type TEXT, occurred_at TEXT, command_id TEXT, causation_event_id TEXT,
                correlation_id TEXT, actor_kind TEXT, payload_json TEXT, metadata_json TEXT,
                application_event_version INTEGER NOT NULL DEFAULT 1);
            CREATE TABLE orchestration_v2_projection_metadata (
                projection_name TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                last_sequence INTEGER NOT NULL, updated_at TEXT NOT NULL);
            INSERT INTO orchestration_v2_projection_metadata VALUES ('event-log-pruned-through',2,0,'');
        """)
    indexed = {}
    calls = []

    def index(chunks, *, budget, **kwargs):
        calls.append(len(chunks))
        changed = [c for c in chunks if indexed.get(c.metadata["chunk_id"]) != c.content]
        selected = changed[: budget.remaining]
        budget.remaining -= len(selected)
        budget.deferred += len(changed) - len(selected)
        indexed.update((c.metadata["chunk_id"], c.content) for c in selected)
        return len(selected)

    monkeypatch.setattr(t3, "_index_chunks", index)
    bookmark = tmp_path / "bookmark.json"

    def invoke():
        return t3.ingest_t3(
            source, db_path=tmp_path / "dest.db", health_path=tmp_path / "health.json", bookmark_path=bookmark
        )

    invoke.source, invoke.bookmark, invoke.indexed, invoke.calls = source, bookmark, indexed, calls
    return invoke


def event(run, message="v2-streaming", *, kind="message.updated", text=None):
    with sqlite3.connect(run.source) as conn:
        if text is not None:
            conn.execute(
                "UPDATE orchestration_v2_projection_messages SET streaming=0,payload_json=? WHERE message_id=?",
                (json.dumps({"text": text}), message),
            )
        conn.execute(
            "INSERT INTO orchestration_events(event_type,stream_id,payload_json) VALUES (?,?,?)",
            (kind, "v2-a", json.dumps({"id": message, "threadId": "v2-a"})),
        )


def test_idle_window_reads_zero_payloads_and_never_indexes(run):
    run()
    run.calls.clear()
    result = run()
    assert result.messages_seen == 0
    assert result.mode == "incremental" and result.events_read == 0
    assert run.calls == []


def test_stream_finalization_after_bookmark_is_ingested_once(run):
    event(run)
    run()
    assert not any("v2-streaming" in key for key in run.indexed)
    event(run, text="final answer")
    result = run()
    assert result.messages_seen == result.chunks_indexed == 1
    assert run().chunks_indexed == 0
    assert sum("v2-streaming" in key for key in run.indexed) == 1


def test_pruned_window_falls_back_and_sets_bookmark(run):
    run()
    event(run, text="completed")
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_metadata SET last_sequence=1")
    result = run()
    assert result.mode == "full" and result.messages_seen == 3
    assert result.bookmark_to == 1 and run().mode == "incremental"


def test_replaced_database_identity_forces_full_scan(run):
    run()
    replacement = run.source.with_suffix(".replacement")
    with sqlite3.connect(run.source) as source, sqlite3.connect(replacement) as target:
        source.backup(target)
    replacement.replace(run.source)
    assert run().mode == "full"


def test_indexing_error_holds_and_replays_same_window(run, monkeypatch):
    run()
    before = run.bookmark.read_bytes()
    event(run, text="completed")
    original = t3._index_chunks
    with monkeypatch.context() as patch:
        patch.setattr(t3, "_index_chunks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("failure")))
        with pytest.raises(RuntimeError, match="failure"):
            run()
    assert run.bookmark.read_bytes() == before
    monkeypatch.setattr(t3, "_index_chunks", original)
    result = run()
    assert result.events_read == 1 and result.chunks_indexed == 1


def test_budget_deferral_holds_until_remaining_chunks_finish(run, monkeypatch):
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 1)
    first = run()
    assert first.chunks_deferred == 1 and first.bookmark_held == "chunks_deferred"
    assert not run.bookmark.exists()
    second = run()
    assert second.chunks_indexed == 1 and second.bookmark_held is None
    assert run().mode == "incremental" and run().messages_seen == 0


def test_event_after_ceiling_is_picked_up_next_run(run, monkeypatch):
    run()
    original = t3.T3Reader._read_v2_snapshot

    def mid_scan(conn, selection=None):
        # WAL allows the app writer to commit while our read snapshot is open.
        event(run, text="late completion")
        return original(conn, selection)

    with sqlite3.connect(run.source) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    with monkeypatch.context() as patch:
        patch.setattr(t3.T3Reader, "_read_v2_snapshot", staticmethod(mid_scan))
        assert run().messages_seen == 0
    result = run()
    assert result.events_read == 1 and result.chunks_indexed == 1


def test_thread_rename_reloads_messages_with_same_ids(run):
    run()
    event(run, kind="thread.metadata-updated")
    result = run()
    assert result.mode == "incremental" and result.messages_seen == 1
    assert result.chunks_indexed == 0


def test_unchanged_legacy_source_is_skipped(run, tmp_path, monkeypatch):
    legacy = _create_t3_fixture(tmp_path / "state.sqlite")
    monkeypatch.setattr(t3, "resolve_t3_sources", lambda *a, **k: [(run.source, 2), (legacy, 1)])
    run()
    assert run().messages_seen == 0
    with sqlite3.connect(legacy) as conn:
        conn.execute("UPDATE projection_thread_messages SET text='changed' WHERE message_id='message-1'")
    assert run().messages_seen == 2
