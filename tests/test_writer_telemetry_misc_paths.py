from __future__ import annotations

import json
from pathlib import Path

import pytest

from brainlayer.cli import _RewindArchiveBatcher
from brainlayer.enrichment_controller import _apply_enrichment
from brainlayer.vector_store import VectorStore


def _configure(monkeypatch, tmp_path: Path) -> Path:
    log_path = tmp_path / "writer-telemetry.jsonl"
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY", "1")
    monkeypatch.setenv("BRAINLAYER_WRITER_TELEMETRY_PATH", str(log_path))
    monkeypatch.setenv("BRAINLAYER_WRITER_HEARTBEAT_DIR", str(tmp_path / "heartbeats"))
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    return log_path


def _finished(log_path: Path, operation: str) -> list[dict]:
    if not log_path.exists():
        return []
    return [
        event
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line)).get("event") == "txn_finished" and event.get("operation") == operation
    ]


def _insert_chunk(store: VectorStore, chunk_id: str, *, conversation_id: str = "session-one") -> None:
    store.conn.execute(
        """
        INSERT INTO chunks (id, content, metadata, source_file, source, conversation_id, content_type)
        VALUES (?, ?, '{}', 'watcher.jsonl', 'realtime_watcher', ?, 'assistant_text')
        """,
        (chunk_id, f"Content for {chunk_id}", conversation_id),
    )


def test_rewind_archive_flush_emits_watcher_span(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    _insert_chunk(store, "rewind-one", conversation_id="session-one")
    _insert_chunk(store, "rewind-two", conversation_id="session-two")
    store.close()
    log_path.unlink(missing_ok=True)
    batcher = _RewindArchiveBatcher(
        db_path,
        batch_size=10,
        flush_interval_ms=1_000,
    )
    batcher.add("session-one")
    batcher.add("session-two")

    try:
        assert batcher.flush("test") == 2
    finally:
        batcher.close()

    events = _finished(log_path, "rewind_archive")
    assert len(events) == 1
    assert events[0]["producer"] == "watcher"
    assert events[0]["lane"] == "realtime"
    assert events[0]["rows_planned"] is None
    assert events[0]["sessions_planned"] == 2
    assert events[0]["rows_touched"] == 2
    assert events[0]["outcome"] == "commit"


def test_direct_enrichment_apply_emits_savepoint_span(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    _insert_chunk(store, "enrich-one")
    log_path.unlink(missing_ok=True)

    try:
        _apply_enrichment(
            store,
            {"id": "enrich-one", "content": "Content for enrich-one"},
            {"summary": "Observed enrichment", "tags": ["telemetry"], "entities": []},
        )
    finally:
        store.close()

    events = _finished(log_path, "apply")
    assert len(events) == 1
    assert events[0]["producer"] == "enrichment"
    assert events[0]["lane"] == "enrichment"
    assert events[0]["transaction_mode"] == "savepoint"
    assert events[0]["rows_planned"] == 1
    assert events[0]["outcome"] == "commit"


def test_direct_enrichment_apply_preserves_exception_and_records_rollback(tmp_path, monkeypatch):
    log_path = _configure(monkeypatch, tmp_path)
    db_path = tmp_path / "brainlayer.db"
    store = VectorStore(db_path)
    _insert_chunk(store, "enrich-error")
    log_path.unlink(missing_ok=True)

    def fail_update(**_kwargs):
        raise RuntimeError("synthetic enrichment failure")

    monkeypatch.setattr(store, "update_enrichment", fail_update)
    try:
        with pytest.raises(RuntimeError, match="synthetic enrichment failure"):
            _apply_enrichment(
                store,
                {"id": "enrich-error", "content": "Content for enrich-error"},
                {"summary": "will fail"},
            )
    finally:
        store.close()

    events = _finished(log_path, "apply")
    assert len(events) == 1
    assert events[0]["outcome"] == "rollback"
    assert "synthetic enrichment failure" in events[0]["error"]
