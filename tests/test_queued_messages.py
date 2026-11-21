"""Queued steering reaches the same storage and accounting path as delivered input."""

import json
import sqlite3

import pytest

from brainlayer.pipeline.classify import classify_content
from brainlayer.vector_store import VectorStore
from brainlayer.watcher import JSONLTailer, JSONLWatcher, growth_entry_class, normalize_provider_entry
from brainlayer.watcher_bridge import create_flush_callback

TEXT = "Stop changing the storage contract; please keep the user's original words."
STAMP = "2026-11-20T12:00:00Z"


def enqueue(content=TEXT, operation="enqueue"):
    return {"type": "queue-operation", "operation": operation, "content": content, "timestamp": STAMP}


def replay(tmp_path, entries):
    source = tmp_path / "projects" / "-Users-example-Gits-example" / "session.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text("".join(json.dumps(e) + "\n" for e in entries))
    db = tmp_path / "test.db"
    VectorStore(db).close()
    flush = create_flush_callback(db, arbitrated=False)
    tailer = JSONLTailer(str(source))
    while lines := tailer.read_new_lines(max_lines=1):
        normalized = [normalize_provider_entry(e, "claude") for e in lines]
        flush([{**e, "_source_file": str(source)} for e in normalized if e is not None])
    return sqlite3.connect(db), tailer, source


@pytest.mark.parametrize("content", [TEXT, [{"type": "text", "text": TEXT}]])
def test_enqueue_only_is_stored_verbatim_with_origin_and_time(tmp_path, content):
    conn, tailer, source = replay(tmp_path, [enqueue(content)])
    rows = conn.execute("SELECT content, content_type, created_at, metadata, tags FROM chunks").fetchall()
    conn.close()
    assert len(rows) == 1
    content, kind, stamp, metadata, tags = rows[0]
    assert (content, kind, stamp) == (TEXT, "user_message", STAMP)
    assert json.loads(metadata)["entry_type"] == "queue-operation"
    assert json.loads(metadata)["queued"] is True
    assert json.loads(metadata)["sender"] == "user"
    assert "correction" in json.loads(tags)
    assert tailer.offset == source.stat().st_size


@pytest.mark.parametrize("queued_first", [True, False])
def test_delivered_and_queued_input_fold_in_both_orders(tmp_path, queued_first):
    queued = enqueue()
    user = {"type": "user", "message": {"role": "user", "content": TEXT}, "timestamp": "2026-11-20T12:01:00Z"}
    conn, _, _ = replay(tmp_path, [queued, user] if queued_first else [user, queued])
    assert conn.execute("SELECT count(*), max(seen_count) FROM chunks WHERE archived_at IS NULL").fetchone() == (1, 2)
    assert conn.execute("SELECT count(*) FROM chunk_id_alias").fetchone() == (1,)
    conn.close()


@pytest.mark.parametrize(
    "entry",
    [
        enqueue(operation="remove"),
        enqueue(operation="dequeue"),
        enqueue(""),
        enqueue("   "),
        enqueue({"text": TEXT}),
        enqueue(123),
        enqueue([{"type": "text", "text": 5}]),
        enqueue([{"type": "image"}]),
    ],
)
def test_other_queue_shapes_are_skipped(tmp_path, entry):
    conn, tailer, source = replay(tmp_path, [entry])
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,)
    conn.close()
    assert classify_content(entry) is None
    assert growth_entry_class(entry, "claude") == "neutral"
    assert tailer.offset == source.stat().st_size


def test_enqueue_counts_as_content_and_respects_line_budget(tmp_path):
    entries = [enqueue(), enqueue(operation="remove"), enqueue(TEXT + " Again.")]
    source = tmp_path / "session.jsonl"
    source.write_text("".join(json.dumps(e) + "\n" for e in entries))
    tailer = JSONLTailer(str(source))
    first = tailer.read_new_lines(max_lines=1)
    assert len(first) == 1
    assert tailer.offset < source.stat().st_size
    assert growth_entry_class(first[0], "claude") == "recognised"
    rest = tailer.read_new_lines(max_lines=10)
    assert [growth_entry_class(e, "claude") for e in rest] == ["neutral", "recognised"]
    assert tailer.offset == source.stat().st_size


@pytest.mark.parametrize(
    "payload",
    [
        "<task-notification><task-id>background-7</task-id><status>completed</status></task-notification>",
        "<task-notification><result> </result><status>completed</status></task-notification>",
    ],
)
def test_queued_notification_without_result_stores_nothing(tmp_path, payload):
    conn, _, _ = replay(tmp_path, [enqueue(payload)])
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,)
    conn.close()


def test_queued_notification_with_result_keeps_assistant_path(tmp_path):
    result = "The background worker completed the storage audit and verified the session attribution contract."
    payload = "<task-notification><task-id>worker-7</task-id><result>" + result + "</result></task-notification>"
    conn, _, _ = replay(tmp_path, [enqueue(payload)])
    rows = conn.execute("SELECT content, content_type, metadata FROM chunks").fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0][:2] == (result, "assistant_text")
    assert json.loads(rows[0][2])["sender"] == "assistant"


def test_withdrawing_a_draft_keeps_the_words_at_enqueue_time(tmp_path):
    conn, _, _ = replay(tmp_path, [enqueue(), {"type": "queue-operation", "operation": "popAll"}])
    assert conn.execute("SELECT content, created_at FROM chunks").fetchall() == [(TEXT, STAMP)]
    conn.close()


@pytest.mark.parametrize("budget", [0, 5])
def test_watcher_catches_up_and_confirms_enqueue_bytes(tmp_path, budget):
    source = tmp_path / "sessions" / "session.jsonl"
    source.parent.mkdir()
    entries = [enqueue(), enqueue(operation="remove"), enqueue(TEXT + " Again.")]
    source.write_text("".join(json.dumps(e) + "\n" for e in entries))
    db = tmp_path / "test.db"
    VectorStore(db).close()
    watcher = JSONLWatcher(
        watch_dir=tmp_path,
        registry_path=tmp_path / "offsets.json",
        health_path=tmp_path / "health.json",
        on_flush=create_flush_callback(db, arbitrated=False),
        batch_size=1,
        max_lines_per_file=1,
        catch_up_budget_s=budget,
    )
    watcher.poll_once()
    if budget == 0:
        assert watcher._tailers[str(source)].offset < source.stat().st_size
        watcher.poll_once()
        watcher.poll_once()
    assert watcher._tailers[str(source)].offset == source.stat().st_size
    assert watcher.registry.get(str(source))[0] == source.stat().st_size
    assert watcher._max_offset_lag_bytes([str(source)]) == 0
