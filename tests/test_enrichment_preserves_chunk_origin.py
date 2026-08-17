"""Enrichment must never overwrite chunk_origin; model lives in metadata.enriched_by."""

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


def test_update_enrichment_does_not_fill_unknown_chunk_origin():
    from brainlayer.session_repo import SessionMixin

    class Store(SessionMixin):
        _has_chunk_origin = True

    store = Store()
    store.db_path = None
    store.conn = sqlite3.connect(":memory:")
    store.conn.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            enriched_at TEXT,
            enrich_status TEXT,
            summary TEXT,
            chunk_origin TEXT DEFAULT 'unknown',
            enrichment_model TEXT,
            metadata TEXT DEFAULT '{}'
        )
        """
    )
    store.conn.execute("INSERT INTO chunks (id, chunk_origin) VALUES (?, ?)", ("unknown-1", "unknown"))

    store.update_enrichment(
        "unknown-1",
        summary="summary",
        chunk_origin="gemini-2.5-flash-lite",
        enrichment_model="gemini-2.5-flash-lite",
    )

    origin, metadata_raw = store.conn.execute(
        "SELECT chunk_origin, metadata FROM chunks WHERE id = 'unknown-1'"
    ).fetchone()
    metadata = json.loads(metadata_raw or "{}")

    assert origin == "unknown"
    assert metadata["enriched_by"] == "gemini-2.5-flash-lite"


def test_enrichment_payload_keeps_model_in_metadata_not_chunk_origin(monkeypatch):
    from brainlayer import enrichment_controller as controller

    monkeypatch.setattr(controller, "GEMINI_REALTIME_MODEL", "gemini-test-model")

    payload = controller._enrichment_update_payload(
        {"id": "chunk-1", "content": "content"},
        {"summary": "summary", "entities": []},
    )

    assert payload.get("chunk_origin") in (None, "")
    assert payload["enrichment_model"] == "gemini-test-model"
    assert (payload["enrichment"].get("enrichment_metadata") or {})["enriched_by"] == "gemini-test-model"


def test_local_enrichment_pipeline_does_not_stamp_backend_as_chunk_origin():
    from unittest.mock import MagicMock, patch

    from brainlayer.pipeline import enrichment

    store = MagicMock()
    store.get_context.return_value = {"context": []}
    chunk = {
        "id": "chunk-mlx",
        "content": "content that should be enriched",
        "content_type": "user_message",
        "project": "brainlayer",
        "conversation_id": None,
        "position": None,
    }

    with (
        patch.object(enrichment, "build_prompt", return_value="prompt"),
        patch.object(enrichment, "call_llm", return_value='{"summary":"ok summary","tags":["test"]}'),
        patch.object(enrichment, "parse_enrichment", return_value={"summary": "ok summary", "tags": ["test"]}),
    ):
        result = enrichment._enrich_one(store, chunk, with_context=False, backend="mlx")

    assert result is True
    kwargs = store.update_enrichment.call_args.kwargs
    assert kwargs.get("chunk_origin") in (None, "")
    assert kwargs["enrichment_model"] == enrichment.MLX_MODEL
