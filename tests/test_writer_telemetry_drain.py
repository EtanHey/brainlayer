from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import apsw
import pytest

from brainlayer.drain import burn_drain_once, drain_once
from brainlayer.queue_io import enqueue_store
from brainlayer.vector_store import VectorStore


def test_lock_wait_and_lock_held_are_separate(tmp_path, monkeypatch):
    from brainlayer.writer_telemetry import writer_span

    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "timing.db"
    owner = apsw.Connection(str(db_path))
    owner.execute("CREATE TABLE sample(value)")
    contender = apsw.Connection(str(db_path))
    owner.execute("BEGIN IMMEDIATE")
    waiting, acquired, release, finish = (threading.Event() for _ in range(4))
    errors = []
    boundaries = {}

    def writer():
        try:
            conn = contender

            def busy(_attempt):
                waiting.set()
                return release.wait(3)

            conn.set_busy_handler(busy)
            with writer_span(
                conn, db_path=db_path, producer="drain", lane="realtime", operation="barrier", sample_fts=False
            ):
                conn.execute("BEGIN IMMEDIATE")
                acquired.set()
                assert finish.wait(3)
                conn.execute("INSERT INTO sample VALUES (1)")
                conn.execute("COMMIT")
                boundaries["committed"] = time.monotonic()
            conn.close()
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=writer)
    worker.start()
    try:
        assert waiting.wait(3)
        assert not acquired.is_set()
        boundaries["wait_observed"] = time.monotonic()
        # A causal gate, not an assumption that two racing connections overlapped.
        time.sleep(0.05)
        owner.execute("ROLLBACK")
        release.set()
        assert acquired.wait(3)
        time.sleep(0.05)
        finish.set()
    finally:
        release.set()
        finish.set()
        worker.join(5)
        owner.close()
    assert not worker.is_alive()
    assert not errors
    event = _finished(log_path, "barrier")[0]
    assert event["lock_wait_ms"] >= 50
    assert event["lock_held_ms"] >= 50
    assert event["lock_held_ms"] < (boundaries["committed"] - boundaries["wait_observed"]) * 1000 - 40
    assert event["lock_acquired_at"] is not None


def _configure(monkeypatch, tmp_path: Path) -> Path:
    log_path = tmp_path / "writer-telemetry.jsonl"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "heartbeats"))
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    monkeypatch.setenv("BRAINLAYER_DRAIN_POST_COMMIT_YIELD_MS", "0")
    return log_path


def _finished(log_path: Path, operation: str) -> list[dict]:
    if not log_path.exists():
        return []
    return [
        event
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line)).get("event") == "txn_finished" and event.get("operation") == operation
    ]


def _seed_store(db_path: Path) -> None:
    store = VectorStore(db_path)
    store.close()


def test_live_drain_records_queue_wait_lane_and_source_from_original_enqueue(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _seed_store(db_path)
    log_path.unlink(missing_ok=True)
    queued = enqueue_store(content="Unique queued telemetry memory", source="mcp", queue_dir=queue_dir)
    event = {**json.loads(queued.read_text()), "queued_at": time.time() - 2.0}
    queued.write_text(json.dumps(event) + "\n")

    assert drain_once(db_path=db_path, queue_dir=queue_dir, batch_size=1, log_path=tmp_path / "drain.log") == 1

    events = _finished(log_path, "apply_file")
    assert len(events) == 1
    event = events[0]
    assert event["producer"] == "drain"
    assert event["lane"] == "interactive"
    assert event["queue_source"] == "mcp"
    assert event["queue_wait_ms"] >= 1_500
    assert event["rows_planned"] == 1
    assert event["outcome"] == "commit"


def test_burn_drain_records_one_span_for_selected_batch(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _seed_store(db_path)
    log_path.unlink(missing_ok=True)
    enqueue_store(content="First unique burn telemetry memory", source="mcp", queue_dir=queue_dir)
    enqueue_store(content="Second unique burn telemetry memory", source="mcp", queue_dir=queue_dir)

    result = burn_drain_once(
        db_path=db_path,
        queue_dir=queue_dir,
        batch_size=10,
        max_events_per_transaction=10,
        log_path=tmp_path / "drain.log",
    )

    assert result.applied_events == 2
    events = _finished(log_path, "burn_apply")
    assert len(events) == 1
    assert events[0]["rows_planned"] == 2
    assert events[0]["queue_source"] == "mcp"
    assert events[0]["outcome"] == "commit"


def test_hotlane_vector_write_emits_bounded_transaction_span(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    _seed_store(db_path)
    store = VectorStore(db_path)
    store.conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, source) VALUES (?, ?, '{}', 'test', 'mcp')",
        ("hotlane-chunk", "stable hotlane content"),
    )
    store.close()
    log_path.unlink(missing_ok=True)
    importlib.invalidate_caches()
    sys.modules.pop("scripts.hotlane_brainbar_daemon", None)
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")

    assert (
        hotlane._write_embedded_vectors(
            db_path,
            [hotlane.EmbeddedVector("hotlane-chunk", "stable hotlane content", [0.25] * 1024)],
        )
        == 1
    )

    events = _finished(log_path, "write_vectors")
    assert len(events) == 1
    assert events[0]["producer"] == "hotlane"
    assert events[0]["lane"] == "hotlane"
    assert events[0]["rows_planned"] == 1
    assert events[0]["outcome"] == "commit"


def test_failed_begin_never_claims_lock_held(tmp_path, monkeypatch):
    import pytest

    from brainlayer.writer_telemetry import writer_span

    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "failed-begin.db"
    owner = apsw.Connection(str(db_path))
    try:
        owner.execute("CREATE TABLE sample(value)")
        conn = apsw.Connection(str(db_path))
        owner.execute("BEGIN IMMEDIATE")
        with (
            pytest.raises(apsw.BusyError),
            writer_span(
                conn, db_path=db_path, producer="drain", lane="realtime", operation="failed_begin", sample_fts=False
            ),
        ):
            conn.execute("BEGIN IMMEDIATE")
        conn.close()
        owner.execute("ROLLBACK")
    finally:
        owner.close()
    event = _finished(log_path, "failed_begin")[0]
    assert event["lock_held_ms"] is None
    assert event["lock_acquired_at"] is None
    assert event["lock_wait_ms"] >= 0


def test_ingest_stage_timings_separate_queue_encode_begin_apply_commit_and_checkpoint(tmp_path, monkeypatch):
    import brainlayer.drain as drain
    import brainlayer.writer_telemetry as telemetry

    log_path = _configure(monkeypatch, tmp_path)
    db_path, queue_dir = tmp_path / "stages.db", tmp_path / "queue"
    _seed_store(db_path)
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    queued = enqueue_store(content="stage measurement fixture", queue_dir=queue_dir)
    event = json.loads(queued.read_text())
    event["queued_at"] = 1000.0
    drain._preserve_remaining_events(queued, [event])
    os.utime(queued, (1100, 1100))
    assert json.loads(queued.read_text())["queued_at"] == 1000
    clock = [0.0]
    monkeypatch.setattr(telemetry, "_MONOTONIC", lambda: clock[0])
    monkeypatch.setattr(telemetry, "_WALL_TIME", lambda: 1010.0 + clock[0], raising=False)
    original_open, original_apply = drain._open_connection, drain._apply_event

    class TimedConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            clock[0] += {"BEGIN IMMEDIATE": 0.02, "COMMIT": 0.04}.get(sql, 0)
            return self.connection.execute(sql, *args)

    def apply(*args):
        clock[0] += 0.03
        return original_apply(*args)

    checkpoint_entered, release_checkpoint = threading.Event(), threading.Event()

    def checkpoint(*args):
        assert any(json.loads(line)["event"] == "ingest_committed" for line in log_path.read_text().splitlines())
        checkpoint_entered.set()
        assert release_checkpoint.wait(3)
        clock[0] += 0.05
        return 0

    monkeypatch.setattr(drain, "_open_connection", lambda path: TimedConnection(original_open(path)))
    monkeypatch.setattr(drain, "_apply_event", apply)
    monkeypatch.setattr(drain, "_post_commit_checkpoint", checkpoint)
    results, errors = [], []

    def run():
        try:
            results.append(drain.drain_once(db_path=db_path, queue_dir=queue_dir, log_path=tmp_path / "drain.log"))
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert checkpoint_entered.wait(3), errors
        assert not results
    finally:
        release_checkpoint.set()
        worker.join(3)
    assert not errors
    assert results == [1]
    result = [
        json.loads(line) for line in log_path.read_text().splitlines() if json.loads(line)["event"] == "ingest_finished"
    ][-1]
    assert result["encode_ms"] == 0
    assert result["begin_ms"] == pytest.approx(20)
    assert result["apply_ms"] == pytest.approx(30)
    assert result["commit_ms"] == pytest.approx(40)
    assert result["checkpoint_ms"] == pytest.approx(50)
    assert result["queue_wait_ms"] == pytest.approx(10000)
    assert result["queue_to_commit_ms"] == pytest.approx(10090)


def test_ingest_watermarks_follow_line_queue_fts_and_vector_commit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import brainlayer.queue_io as queue_io
    import brainlayer.watcher as watcher
    import brainlayer.writer_telemetry as telemetry
    from brainlayer.drain import _preserve_remaining_events
    from brainlayer.watcher_bridge import create_flush_callback

    log_path = _configure(monkeypatch, tmp_path)
    db_path, queue = tmp_path / "watermarks.db", tmp_path / "queue"
    _seed_store(db_path)
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    monkeypatch.setattr(watcher, "time", SimpleNamespace(time=lambda: 1000.0, monotonic=time.monotonic))
    monkeypatch.setattr(queue_io, "time", SimpleNamespace(time=lambda: 1002.0))
    monkeypatch.setattr(telemetry, "_WALL_TIME", lambda: 1005.0)
    source = tmp_path / "session.jsonl"
    content = (
        "watermarkfixture documents the measured stages from a new conversation line to committed searchable memory."
    )
    source.write_text(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": content}}) + "\n")
    entries = watcher.JSONLTailer(str(source)).read_new_lines()
    entries[0]["_source_file"] = str(source)
    assert create_flush_callback(db_path, arbitrated=True)(entries) == 1
    spool = next(queue.glob("*.jsonl"))
    event = json.loads(spool.read_text())
    _preserve_remaining_events(spool, [event])
    os.utime(spool, (1100, 1100))
    assert json.loads(spool.read_text())["queued_at"] == 1002
    assert drain_once(db_path=db_path, queue_dir=queue, log_path=tmp_path / "drain.log") == 1
    conn = apsw.Connection(str(db_path))
    (chunk_id,) = conn.execute("SELECT id FROM chunks WHERE content = ?", (content,)).fetchone()
    assert conn.execute("SELECT content FROM chunks_fts WHERE chunks_fts MATCH 'watermarkfixture'").fetchone() == (
        content,
    )
    conn.close()
    hotlane = importlib.import_module("scripts.hotlane_brainbar_daemon")
    monkeypatch.setattr(telemetry, "_WALL_TIME", lambda: 1008.0)
    assert hotlane._write_embedded_vectors(db_path, [hotlane.EmbeddedVector(chunk_id, content, [0.25] * 1024)]) == 1
    events = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if json.loads(line)["event"] == "ingest_committed"
    ]
    assert [(event["producer"], event["observed_to_commit_ms"]) for event in events] == [
        ("drain", 5000),
        ("hotlane", 8000),
    ]
    assert events[-1]["queue_to_commit_ms"] == 6000
    assert not any(value in json.dumps(events) for value in (content, str(source)))
