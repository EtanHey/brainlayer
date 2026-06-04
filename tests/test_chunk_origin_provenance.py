from __future__ import annotations

import json
import sqlite3


def test_enrichment_update_payload_stamps_gemini_model(monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setattr(controller, "GEMINI_REALTIME_MODEL", "gemini-test-model")

    payload = controller._enrichment_update_payload(
        {"id": "chunk-1", "content": "content"},
        {"summary": "summary", "entities": []},
    )

    assert payload["chunk_origin"] == "gemini-test-model"


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


def test_drain_enrichment_update_stamps_chunk_origin():
    from brainlayer.drain import _apply_enrichment

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            summary TEXT,
            enriched_at TEXT,
            enrich_status TEXT,
            chunk_origin TEXT DEFAULT 'unknown'
        )
        """
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, chunk_origin) VALUES (?, ?, '{}', 'test.jsonl', ?)",
        ("chunk-1", "content", "unknown"),
    )

    _apply_enrichment(
        conn,
        {
            "chunk_id": "chunk-1",
            "enrichment": {"summary": "summary"},
            "chunk_origin": "gemini-2.5-flash-lite",
        },
    )

    assert conn.execute("SELECT summary, chunk_origin FROM chunks WHERE id = 'chunk-1'").fetchone() == (
        "summary",
        "gemini-2.5-flash-lite",
    )


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
    assert dry.inferred == {"manual": 1, "gemini-2.5-flash-lite": 1}
    assert dry.updated == 0

    applied = backfill_chunk_origin_provenance(db_path, apply=True, batch_size=2, checkpoint_every=1)

    assert applied.updated == 2
    with sqlite3.connect(db_path) as verify:
        assert dict(verify.execute("SELECT id, chunk_origin FROM chunks")) == {
            "manual-abc": "manual",
            "r81-abc": "gemini-2.5-flash-lite",
            "unknown-abc": "unknown",
            "existing-abc": "precompact_checkpoint",
        }
