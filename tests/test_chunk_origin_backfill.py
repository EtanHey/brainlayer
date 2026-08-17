"""Backfill must not re-stamp enrichment model names into chunk_origin."""

from __future__ import annotations

import json
import sqlite3


def test_backfill_does_not_stamp_enrichment_model_as_chunk_origin(tmp_path):
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
    conn.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, source, enrichment_version,
            enrich_status, enriched_at, summary_v2, chunk_origin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "enriched-abc",
            "ordinary watcher memory",
            json.dumps(
                {
                    "enrichment_model": "gemini-2.5-flash-lite",
                    "enrichment_backend": "gemini-flex",
                    "model": "groq",
                    "backend": "mlx",
                }
            ),
            "watcher.jsonl",
            "claude_code",
            "r82-hybrid-taxonomy",
            "success",
            "2026-06-01",
            None,
            "unknown",
        ),
    )
    conn.commit()
    conn.close()

    applied = backfill_chunk_origin_provenance(db_path, apply=True, batch_size=10, checkpoint_every=1)

    assert applied.updated == 0
    assert applied.inferred == {}
    with sqlite3.connect(db_path) as verify:
        assert verify.execute("SELECT chunk_origin FROM chunks WHERE id = 'enriched-abc'").fetchone()[0] == "unknown"


def test_backfill_does_not_infer_gemini_from_r81_prompt_version(tmp_path):
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
    conn.execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, source, enrichment_version,
            enrich_status, enriched_at, summary_v2, chunk_origin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "r81-abc",
            "enriched memory",
            json.dumps({"prompt_version": "r81"}),
            "watcher.jsonl",
            "claude_code",
            "r81",
            "success",
            "2026-06-01",
            "prompt=r81",
            "unknown",
        ),
    )
    conn.commit()
    conn.close()

    applied = backfill_chunk_origin_provenance(db_path, apply=True, batch_size=10, checkpoint_every=1)

    assert applied.updated == 0
    assert applied.inferred == {}
    with sqlite3.connect(db_path) as verify:
        assert verify.execute("SELECT chunk_origin FROM chunks WHERE id = 'r81-abc'").fetchone()[0] == "unknown"
