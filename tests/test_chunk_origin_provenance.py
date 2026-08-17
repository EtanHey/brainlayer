"""Unique chunk_origin provenance coverage (shared drain/pipeline cases live in test_enrichment_preserves_chunk_origin)."""

from __future__ import annotations

import json
import sqlite3


def test_meta_research_enrichment_queue_does_not_stamp_backend(monkeypatch, tmp_path):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path))
    monkeypatch.setattr(controller, "GEMINI_REALTIME_MODEL", "gemini-test-model")

    chunk = {"id": "meta-1", "content": "# research note"}
    controller._enqueue_meta_research_write(chunk)

    [path] = tmp_path.glob("enrichment-*.jsonl")
    event = json.loads(path.read_text(encoding="utf-8").strip())

    assert event["chunk_origin"] is None


def test_meta_research_batcher_does_not_stamp_backend(monkeypatch, tmp_path):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path))
    monkeypatch.setattr(controller, "GEMINI_REALTIME_MODEL", "gemini-test-model")

    chunk = {"id": "meta-1", "content": "# research note"}
    batcher = controller._EnrichmentWriteBatcher(max_batch=10)
    batcher.enqueue(chunk, controller._meta_research_enrichment(chunk), counted_as="skipped")
    batcher.flush()

    [path] = tmp_path.glob("enrichment-*.jsonl")
    event = json.loads(path.read_text(encoding="utf-8").strip())

    assert event["chunk_origin"] is None


def test_direct_apply_enrichment_stamps_backend_origin(monkeypatch):
    from unittest.mock import MagicMock

    from brainlayer import enrichment_controller as controller

    monkeypatch.setattr(controller, "GEMINI_REALTIME_MODEL", "gemini-test-model")

    store = MagicMock()
    controller._apply_enrichment(
        store,
        {"id": "chunk-1", "content": "content"},
        {"summary": "summary", "entities": []},
    )

    assert "chunk_origin" not in store.update_enrichment.call_args.kwargs
    assert store.update_enrichment.call_args.kwargs["enrichment_model"] == "gemini-test-model"


def test_queue_io_preserves_enrichment_chunk_origin(tmp_path):
    from brainlayer.queue_io import enqueue_enrichment_updates

    path = enqueue_enrichment_updates(
        [
            {
                "chunk_id": "chunk-1",
                "enrichment": {"summary": "summary"},
                "chunk_origin": "gemini-2.5-flash-lite",
            }
        ],
        queue_dir=tmp_path,
    )

    event = json.loads(path.read_text(encoding="utf-8").strip())

    assert event["chunk_origin"] == "gemini-2.5-flash-lite"


def test_backfill_chunk_origin_provenance_dry_run_and_apply(tmp_path):
    from brainlayer.chunk_origin_backfill import backfill_chunk_origin_provenance

    db_path = tmp_path / "brainlayer.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            source TEXT,
            enrichment_version TEXT,
            enrich_status TEXT,
            enriched_at TEXT,
            summary_v2 TEXT,
            chunk_origin TEXT DEFAULT 'unknown'
        )
        """
    )
    rows = [
        ("manual-abc", "manual memory", "{}", "manual.md", "unknown", "1.0", "success", "2026-06-01", None, "unknown"),
        (
            "r81-abc",
            "enriched memory",
            json.dumps({"prompt_version": "r81"}),
            "watcher.jsonl",
            "claude_code",
            "1.0",
            "success",
            "2026-06-01",
            None,
            "unknown",
        ),
        (
            "unknown-abc",
            "ordinary memory",
            "{}",
            "watcher.jsonl",
            "claude_code",
            "1.0",
            "success",
            "2026-06-01",
            None,
            "unknown",
        ),
        (
            "existing-abc",
            "existing memory",
            "{}",
            "watcher.jsonl",
            "claude_code",
            "1.0",
            "success",
            "2026-06-01",
            None,
            "precompact_checkpoint",
        ),
    ]
    conn.executemany(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, source, enrichment_version,
            enrich_status, enriched_at, summary_v2, chunk_origin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()

    dry = backfill_chunk_origin_provenance(db_path, apply=False, batch_size=2, checkpoint_every=1)

    assert dry.scanned == 3
    assert dry.inferred == {"manual": 1}
    assert dry.updated == 0

    applied = backfill_chunk_origin_provenance(db_path, apply=True, batch_size=2, checkpoint_every=1)

    assert applied.updated == 1
    with sqlite3.connect(db_path) as verify:
        assert dict(verify.execute("SELECT id, chunk_origin FROM chunks")) == {
            "manual-abc": "manual",
            "r81-abc": "unknown",
            "unknown-abc": "unknown",
            "existing-abc": "precompact_checkpoint",
        }
