"""Drain enrichment writes must not clobber chunk_origin."""

from __future__ import annotations

import json
import sqlite3


def test_drain_enrichment_does_not_change_chunk_origin_and_records_enriched_by():
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
            chunk_origin TEXT DEFAULT 'unknown',
            enrichment_model TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO chunks (id, content, metadata, source_file, chunk_origin) VALUES (?, ?, '{}', 'test.jsonl', ?)",
        ("chunk-1", "ordinary watcher memory", "unknown"),
    )

    _apply_enrichment(
        conn,
        {
            "chunk_id": "chunk-1",
            "enrichment": {"summary": "summary"},
            "chunk_origin": "gemini-2.5-flash-lite",
            "enrichment_model": "gemini-2.5-flash-lite",
        },
    )

    summary, origin, metadata_raw, model = conn.execute(
        "SELECT summary, chunk_origin, metadata, enrichment_model FROM chunks WHERE id = 'chunk-1'"
    ).fetchone()
    metadata = json.loads(metadata_raw)

    assert summary == "summary"
    assert origin == "unknown"
    assert metadata["enriched_by"] == "gemini-2.5-flash-lite"
    assert model == "gemini-2.5-flash-lite"


def test_drain_enrichment_preserves_non_unknown_chunk_origin():
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
        ("chunk-1", "[PreCompact checkpoint]\ntimestamp: 2026-05-16", "precompact_checkpoint"),
    )

    _apply_enrichment(
        conn,
        {
            "chunk_id": "chunk-1",
            "enrichment": {"summary": "summary"},
            "chunk_origin": "gemini-2.5-flash-lite",
            "enrichment_model": "gemini-2.5-flash-lite",
        },
    )

    summary, origin, metadata_raw = conn.execute(
        "SELECT summary, chunk_origin, metadata FROM chunks WHERE id = 'chunk-1'"
    ).fetchone()
    metadata = json.loads(metadata_raw)

    assert summary == "summary"
    assert origin == "precompact_checkpoint"
    assert metadata["enriched_by"] == "gemini-2.5-flash-lite"
