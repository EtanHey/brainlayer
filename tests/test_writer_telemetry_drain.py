from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

from brainlayer.drain import burn_drain_once, drain_once
from brainlayer.queue_io import enqueue_store
from brainlayer.vector_store import VectorStore


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


def test_live_drain_records_queue_wait_lane_and_source_from_spool_mtime(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    queue_dir = tmp_path / "queue"
    _seed_store(db_path)
    log_path.unlink(missing_ok=True)
    queued = enqueue_store(content="Unique queued telemetry memory", source="mcp", queue_dir=queue_dir)
    old = time.time() - 2.0
    os.utime(queued, (old, old))

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
