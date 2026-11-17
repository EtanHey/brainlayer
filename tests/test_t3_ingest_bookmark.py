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
            ALTER TABLE orchestration_v2_projection_messages ADD COLUMN updated_at TEXT DEFAULT '2026-10-09T00:00:00Z';
            INSERT INTO orchestration_events(event_type,stream_id,occurred_at,payload_json)
                VALUES ('bootstrap','project','2026-11-17T00:00:00Z','{}');
        """)
    indexed = {}
    calls = []

    def index(chunks, *, budget, **kwargs):
        calls.append(len(chunks))
        changed = [c for c in chunks if indexed.get(c.metadata["chunk_id"]) != c.content]
        if getattr(budget, "pending_ids", None) is not None:
            changed = [c for c in changed if c.metadata["chunk_id"] in budget.pending_ids]
        changed.sort(key=t3._newest_chunk_key, reverse=not getattr(budget, "oldest_first", False))
        selected = changed[: budget.remaining]
        budget.remaining -= len(selected)
        budget.deferred += len(changed) - len(selected)
        if hasattr(budget, "deferred_ids"):
            budget.deferred_ids.update(c.metadata["chunk_id"] for c in changed[len(selected) :])
        indexed.update((c.metadata["chunk_id"], c.content) for c in selected)
        return len(selected)

    real_index = t3._index_chunks
    monkeypatch.setattr(t3, "_index_chunks", index)
    bookmark = tmp_path / "bookmark.json"

    def invoke(*, discover=False):
        return t3.ingest_t3(
            None if discover else source,
            db_path=tmp_path / "dest.db",
            health_path=tmp_path / "health.json",
            bookmark_path=bookmark,
        )

    invoke.source, invoke.bookmark, invoke.indexed, invoke.calls = source, bookmark, indexed, calls
    invoke.real_index = real_index
    return invoke


def event(run, message="v2-streaming", *, kind="message.updated", text=None):
    with sqlite3.connect(run.source) as conn:
        if text is not None:
            conn.execute(
                "UPDATE orchestration_v2_projection_messages SET streaming=0,payload_json=? WHERE message_id=?",
                (json.dumps({"text": text}), message),
            )
        conn.execute(
            "INSERT INTO orchestration_events(event_type,stream_id,payload_json,occurred_at) VALUES (?,?,?,?)",
            (kind, "v2-a", json.dumps({"id": message, "threadId": "v2-a"}), "2026-11-17T01:00:00Z"),
        )


def test_idle_window_reads_zero_payloads_and_never_indexes(run, monkeypatch):
    run()
    run.calls.clear()
    result = run()
    assert result.messages_seen == 0
    assert result.mode == "incremental" and result.events_read == 0
    assert run.calls == []

    def forbidden(*args, **kwargs):
        pytest.fail("no-op attempted writer pidfile or embedding")

    monkeypatch.setattr("brainlayer.runtime_store.open_writer_store", forbidden)
    monkeypatch.setattr("brainlayer.index_new.embed_chunks", forbidden)
    monkeypatch.setattr(t3, "_index_chunks", run.real_index)
    assert run().messages_seen == 0


def test_pruned_bookmark_bounds_payloads_and_keeps_margin_and_finalization(run):
    run()
    event(run, text="finalized during gap")
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_metadata SET last_sequence=2")
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET updated_at=? WHERE message_id='v2-streaming'",
            ("2026-11-17T00:30:00Z",),
        )
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET updated_at=? WHERE message_id='v2-user'",
            ("2026-11-16T23:55:00Z",),
        )
    result = run()
    assert result.mode == "bounded" and result.messages_seen == 2
    assert result.chunks_indexed == 1
    assert any("v2-streaming" in key for key in run.indexed)
    assert next(iter(json.loads(run.bookmark.read_text())["sources"].values()))["last_sequence"] == 2
    assert run().messages_seen == 0


@pytest.mark.parametrize("stamp", [None, "invalid"])
def test_pruned_bookmark_without_usable_timestamp_keeps_full_read(run, stamp):
    run()
    state = json.loads(run.bookmark.read_text())
    next(iter(state["sources"].values()))["occurred_at"] = stamp
    run.bookmark.write_text(json.dumps(state))
    event(run, text="completed")
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_metadata SET last_sequence=2")
    result = run()
    assert result.mode == "full" and result.messages_seen == 3


@pytest.mark.parametrize("sequence", [1, -1])
def test_missing_prune_marker_keeps_full_read_even_for_malformed_bookmark(run, sequence):
    run()
    state = json.loads(run.bookmark.read_text())
    next(iter(state["sources"].values()))["last_sequence"] = sequence
    run.bookmark.write_text(json.dumps(state))
    with sqlite3.connect(run.source) as conn:
        conn.execute("DELETE FROM orchestration_v2_projection_metadata")
    result = run()
    assert result.mode == "full" and result.messages_seen == 2


def test_bounded_fallback_deferral_replays_every_pending_chunk(run, monkeypatch):
    run()
    event(run, text="finalized")
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_metadata SET last_sequence=2")
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET updated_at='2026-11-17T00:30:00Z', "
            "payload_json=? WHERE streaming=0",
            (json.dumps({"text": "changed during prune gap"}),),
        )
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 1)
    first = run()
    assert first.mode == "bounded" and first.chunks_deferred == 2
    assert run().chunks_deferred == 1
    assert run().chunks_deferred == 0
    assert all(content == "changed during prune gap" for content in run.indexed.values())
    assert run().messages_seen == 0


def test_phase_timings_present_in_result_and_health_for_full_and_idle(run):
    for _ in range(2):
        result = run()
        health = json.loads(run.bookmark.with_name("health.json").read_text())
        for phase in ("read_ms", "chunk_ms", "change_check_ms", "embed_ms", "write_ms"):
            assert getattr(result, phase) >= 0
            assert health[phase] == getattr(result, phase)


def test_phase_timings_measure_real_indexing_and_failure_health(run, tmp_path, monkeypatch):
    from brainlayer.embeddings import EmbeddedChunk
    from brainlayer.vector_store import VectorStore

    destination = tmp_path / "dest.db"
    VectorStore(destination).close()
    monkeypatch.setattr(t3, "_index_chunks", run.real_index)
    monkeypatch.setattr(
        "brainlayer.index_new.embed_chunks",
        lambda chunks: [EmbeddedChunk(chunk, [0.1] * 1024) for chunk in chunks],
    )
    result = run()
    assert result.chunks_indexed == 2
    for phase in ("read_ms", "chunk_ms", "change_check_ms", "embed_ms", "write_ms"):
        assert getattr(result, phase) > 0
    event(run, text="updated")

    def failed_embed(chunks):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr("brainlayer.index_new.embed_chunks", failed_embed)
    with pytest.raises(RuntimeError, match="embedding failed"):
        run()
    health = json.loads(run.bookmark.with_name("health.json").read_text())
    assert health["bookmark_held"] == "indexing_failure"
    assert health["read_ms"] > 0 and health["chunk_ms"] > 0
    assert health["change_check_ms"] > 0 and health["embed_ms"] > 0
    assert health["write_ms"] == 0


def test_unchanged_legacy_does_not_read_v2_identity_set(run, tmp_path, monkeypatch):
    legacy = _create_t3_fixture(tmp_path / "state.sqlite")
    _select_sources(run, legacy, monkeypatch)
    run(discover=True)
    queries = []
    original = t3.T3Reader._connect

    def connect(reader):
        conn = original(reader)
        conn.set_trace_callback(queries.append)
        return conn

    monkeypatch.setattr(t3.T3Reader, "_connect", connect)
    assert run(discover=True).messages_seen == 0
    assert not any(
        query.strip() == "SELECT thread_id, message_id FROM orchestration_v2_projection_messages" for query in queries
    )


def test_stream_finalization_after_bookmark_is_ingested_once(run):
    event(run)
    run()
    assert not any("v2-streaming" in key for key in run.indexed)
    event(run, text="final answer")
    result = run()
    assert result.messages_seen == 2 and result.chunks_indexed == 1
    assert run().chunks_indexed == 0
    assert sum("v2-streaming" in key for key in run.indexed) == 1


def test_pruned_window_falls_back_and_sets_bookmark(run):
    run()
    event(run, text="completed")
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_metadata SET last_sequence=2")
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET updated_at='2026-11-17T01:00:00Z' "
            "WHERE message_id='v2-streaming'"
        )
    result = run()
    assert result.mode == "bounded" and result.messages_seen == 1
    assert result.bookmark_to == 2 and run().mode == "incremental"


def test_replaced_database_identity_forces_full_scan(run):
    run()
    replacement = run.source.with_suffix(".replacement")
    with sqlite3.connect(run.source) as source, sqlite3.connect(replacement) as target:
        source.backup(target)
    replacement.replace(run.source)
    result = run()
    assert result.mode == "full" and result.messages_seen == 2


@pytest.mark.parametrize("busy", [False, True])
def test_indexing_error_holds_and_replays_same_window(run, monkeypatch, busy):
    from brainlayer.vector_store import WriterInUseError

    run()
    before = run.bookmark.read_bytes()
    event(run, text="completed")
    original = t3._index_chunks
    with monkeypatch.context() as patch:
        error = WriterInUseError("failure") if busy else RuntimeError("failure")
        patch.setattr(t3, "_index_chunks", lambda *a, **k: (_ for _ in ()).throw(error))
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
    assert next(iter(json.loads(run.bookmark.read_text())["sources"].values()))["last_sequence"] is None
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
    _select_sources(run, legacy, monkeypatch)
    run(discover=True)
    assert run(discover=True).messages_seen == 0
    with sqlite3.connect(legacy) as conn:
        conn.execute("UPDATE projection_thread_messages SET text='changed' WHERE message_id='message-1'")
    assert run(discover=True).messages_seen == 3
    assert run.indexed["t3:thread-1:message-1:0"] == "changed"


def _select_sources(run, legacy, monkeypatch):
    original = t3.resolve_t3_sources

    def resolve(path=None, version=None, **kwargs):
        if path is not None:
            return original(path, version, **kwargs)
        return [(run.source, 2), (legacy, 1)]

    monkeypatch.setattr(t3, "resolve_t3_sources", resolve)


@pytest.mark.parametrize("touched", [False, True])
def test_changed_legacy_cannot_overwrite_untouched_v2_message(run, tmp_path, monkeypatch, touched):
    legacy = _create_t3_fixture(tmp_path / "state.sqlite")
    with sqlite3.connect(legacy) as conn:
        conn.execute("UPDATE projection_threads SET thread_id='v2-b' WHERE thread_id='thread-1'")
        conn.execute("UPDATE projection_thread_messages SET thread_id='v2-b' WHERE thread_id='thread-1'")
        conn.execute("UPDATE projection_thread_messages SET message_id='v2-assistant' WHERE message_id='message-1'")
    _select_sources(run, legacy, monkeypatch)
    run(discover=True)
    authoritative = run.indexed["t3:v2-b:v2-assistant:0"]
    with sqlite3.connect(legacy) as conn:
        conn.execute("UPDATE projection_thread_messages SET text='stale legacy copy' WHERE message_id='v2-assistant'")
        conn.execute("UPDATE projection_thread_messages SET text='legacy-only update' WHERE message_id='message-2'")
    if touched:
        event(run, text="new V2 content in another thread")
    result = run(discover=True)
    assert result.mode == "incremental"
    assert run.indexed["t3:v2-b:v2-assistant:0"] == authoritative
    assert run.indexed["t3:v2-b:message-2:0"] == "legacy-only update"


def test_legacy_fingerprint_oserror_recovers_with_v2(run, tmp_path, monkeypatch):
    legacy = _create_t3_fixture(tmp_path / "state.sqlite")
    _select_sources(run, legacy, monkeypatch)
    original = t3.Path.stat

    def vanished(path, *args, **kwargs):
        if path == legacy:
            raise FileNotFoundError("optional legacy disappeared after discovery")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(t3.Path, "stat", vanished)
    result = run(discover=True)
    assert result.chunks_indexed == 2 and result.bookmark_held == "source_failure"
    health = json.loads((tmp_path / "health.json").read_text())
    assert health["alert_reasons"] == ["legacy_source_failure"]
    assert health["failures"][-1]["code"] == "t3_reader_failed"


@pytest.mark.parametrize("fallback", ["replacement", "prune", "restored"])
def test_full_scan_deferral_never_reuses_old_sequence(run, monkeypatch, fallback):
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_events SET sequence=100")
    run()
    if fallback == "prune":
        state = json.loads(run.bookmark.read_text())
        next(iter(state["sources"].values())).pop("occurred_at")
        run.bookmark.write_text(json.dumps(state))
    # Both threads change, but one thread's last event precedes the old bookmark.
    with sqlite3.connect(run.source) as conn:
        conn.execute(
            "UPDATE orchestration_v2_projection_messages SET payload_json=?", (json.dumps({"text": "changed"}),)
        )
        conn.execute("DELETE FROM orchestration_events")
        conn.executemany(
            "INSERT INTO orchestration_events(sequence,stream_id,event_type) VALUES (?,?,'message.updated')",
            [(50, "v2-b"), (80 if fallback == "restored" else 200, "v2-a")],
        )
        if fallback == "prune":
            conn.execute("UPDATE orchestration_v2_projection_metadata SET last_sequence=150")
    if fallback == "replacement":
        replacement = run.source.with_suffix(".replacement")
        with sqlite3.connect(run.source) as source, sqlite3.connect(replacement) as target:
            source.backup(target)
        replacement.replace(run.source)
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 1)
    first = run()
    assert first.mode == "full" and first.chunks_deferred == 1
    held = json.loads(run.bookmark.read_text())
    pending = set(held["pending_chunks"])
    assert len(pending) == 1
    second = run()
    assert second.chunks_indexed == 1 and second.chunks_deferred == 0
    assert all(run.indexed[key] == "changed" for key in pending)
    assert next(iter(held["sources"].values()))["last_sequence"] is None
    assert "pending_chunks" not in json.loads(run.bookmark.read_text())
    assert run().mode == "incremental" and run().messages_seen == 0


def test_lagging_projection_does_not_skip_committed_flip(run):
    run()
    event(run)
    with sqlite3.connect(run.source) as conn:
        conn.execute("INSERT INTO orchestration_v2_projection_metadata VALUES ('thread-projections',2,1,'')")
    assert run().bookmark_to == 1
    with sqlite3.connect(run.source) as conn:
        conn.execute("UPDATE orchestration_v2_projection_messages SET streaming=0 WHERE message_id='v2-streaming'")
        conn.execute(
            "UPDATE orchestration_v2_projection_metadata SET last_sequence=2 WHERE projection_name='thread-projections'"
        )
    assert run().chunks_indexed == 1


def test_restored_head_below_bookmark_forces_full_scan(run):
    event(run)
    run()
    with sqlite3.connect(run.source) as conn:
        conn.execute("DELETE FROM orchestration_events WHERE sequence=2")
    result = run()
    assert result.mode == "full" and result.messages_seen == 2


def test_kill_before_bookmark_write_replays_committed_chunks(run, monkeypatch):
    run()
    before = run.bookmark.read_bytes()
    event(run, text="committed before interruption")

    def killed(*args):
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(t3.t3_bookmark, "save", killed)
        with pytest.raises(KeyboardInterrupt):
            run()
    assert run.bookmark.read_bytes() == before
    result = run()
    assert result.events_read == 1 and result.chunks_indexed == 0


def test_over_200_changed_freezes_window_despite_new_arrivals(run):
    def add_messages(count, prefix, updated):
        with sqlite3.connect(run.source) as conn:
            conn.executemany(
                "INSERT INTO orchestration_v2_projection_messages VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        f"{prefix}-{i}",
                        "v2-a",
                        "user",
                        updated,
                        json.dumps({"text": f"{prefix} content {i}"}),
                        0,
                        updated,
                    )
                    for i in range(count)
                ],
            )

    add_messages(205, "backlog", "2026-10-09T01:00:00Z")
    first = run()
    assert first.chunks_deferred == 7
    add_messages(201, "incoming", "2026-11-17T01:00:00Z")
    event(run, kind="unfamiliar.import-or-revert")
    second = run()
    assert second.chunks_indexed == 7 and second.chunks_deferred == 0 and second.bookmark_to == 1
    assert run().chunks_indexed == 200
    assert run().chunks_indexed == 1
    assert run().messages_seen == 0


@pytest.mark.parametrize("replace_identity", [False, True])
def test_pending_ids_are_discarded_when_source_is_restored_or_replaced(run, monkeypatch, replace_identity):
    monkeypatch.setattr(t3, "DEFAULT_T3_MAX_CHANGED_CHUNKS", 1)
    assert run().chunks_deferred == 1
    assert json.loads(run.bookmark.read_text())["pending_chunks"]
    with sqlite3.connect(run.source) as conn:
        conn.execute(
            "INSERT INTO orchestration_v2_projection_messages VALUES "
            "('restored-message','v2-a','user','2026-10-10', '{\"text\":\"restored content\"}',0,'2026-10-10')"
        )
        if not replace_identity:
            conn.execute("DELETE FROM orchestration_events")
    if replace_identity:
        replacement = run.source.with_suffix(".restored")
        with sqlite3.connect(run.source) as source, sqlite3.connect(replacement) as target:
            source.backup(target)
        replacement.replace(run.source)
    assert run().chunks_deferred == 1
    assert run().chunks_indexed == 1
    assert len(run.indexed) == 3
