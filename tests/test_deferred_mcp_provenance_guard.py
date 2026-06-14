"""Regression guard for deferred MCP queue provenance.

The deferred-MCP provenance rule is intentionally duplicated in
enrichment_controller.py and provenance_integration.py. These tests fail if
either copy stops treating source='mcp' + source_file='brainlayer-queue' as
Etan-authored direct input.
"""

from __future__ import annotations

from typing import Any

import pytest

DIRECT_QUEUE_CASES = [
    pytest.param("manual", "brainlayer-store", "RAW-ETAN-DIRECT", id="manual-brainlayer-store"),
    pytest.param("manual", "brainbar-store", "RAW-ETAN-DIRECT", id="manual-brainbar-store"),
    pytest.param("manual", "brainlayer-queue", "RAW-ETAN-DIRECT", id="manual-brainlayer-queue"),
    pytest.param("mcp", "brainlayer-queue", "RAW-ETAN-DIRECT", id="deferred-mcp-brainlayer-queue"),
    pytest.param("mcp", "other-queue", "AGENT-INFERENCE", id="mcp-other-source-file"),
    pytest.param("claude_code", "brainlayer-queue", "AGENT-INFERENCE", id="claude-code-brainlayer-queue"),
]


def _row(source: str, source_file: str) -> dict[str, Any]:
    return {
        "id": f"chunk-{source}-{source_file}",
        "content": "PRIMARY_BACKEND: Groq",
        "content_type": "note",
        "sender": None,
        "created_at": "2026-06-14T00:00:00Z",
        "provenance_class": None,
        "source": source,
        "source_file": source_file,
    }


def _enrichment_controller_class(source: str, source_file: str) -> str:
    from brainlayer.enrichment_controller import _derive_chunk_provenance_class

    return _derive_chunk_provenance_class(_row(source, source_file))


def _provenance_integration_claim(source: str, source_file: str):
    from brainlayer.provenance_integration import _row_to_claim

    return _row_to_claim(_row(source, source_file), "enrichment")


@pytest.mark.parametrize(("source", "source_file", "expected"), DIRECT_QUEUE_CASES)
def test_deferred_mcp_queue_rows_classify_as_user_anchored_in_both_paths(source, source_file, expected):
    claim = _provenance_integration_claim(source, source_file)

    assert _enrichment_controller_class(source, source_file) == expected
    assert claim.provenance_class == expected
    assert claim.user_anchored is (expected != "AGENT-INFERENCE")


@pytest.mark.parametrize(("source", "source_file", "_expected"), DIRECT_QUEUE_CASES)
def test_duplicate_deferred_queue_predicates_stay_logically_in_sync(source, source_file, _expected):
    controller_class = _enrichment_controller_class(source, source_file)
    integration_claim = _provenance_integration_claim(source, source_file)

    assert integration_claim.provenance_class == controller_class
    assert integration_claim.user_anchored is (controller_class != "AGENT-INFERENCE")


def test_on_disk_deferred_mcp_row_resolves_direct_and_remains_queryable(tmp_path):
    from brainlayer import enrichment_controller as controller
    from brainlayer.provenance_integration import resolve_entity_conflicts
    from brainlayer.vector_store import VectorStore

    db_path = tmp_path / "brainlayer.db"
    chunk = {
        "id": "chunk-deferred-mcp-queue",
        "content": "PRIMARY_BACKEND: Groq",
        "metadata": {},
        "source_file": "brainlayer-queue",
        "project": "brainlayer",
        "content_type": "note",
        "value_type": "HIGH",
        "char_count": len("PRIMARY_BACKEND: Groq"),
        "source": "mcp",
        "sender": None,
        "created_at": "2026-06-14T00:00:00Z",
    }

    store = VectorStore(db_path)
    try:
        store.upsert_chunks([chunk], [[0.01] * 1024])
        controller._apply_enrichment(store, chunk, {"summary": "queued MCP store", "entities": []})
        store.upsert_entity("e-enrichment", "concept", "enrichment", canonical_name="enrichment")
        store.link_entity_chunk("e-enrichment", chunk["id"], context="PRIMARY_BACKEND: Groq", mention_type="explicit")

        row = (
            store.conn.cursor()
            .execute("SELECT source, source_file, provenance_class FROM chunks WHERE id = ?", (chunk["id"],))
            .fetchone()
        )
        assert row == ("mcp", "brainlayer-queue", "RAW-ETAN-DIRECT")

        report = resolve_entity_conflicts(store, "enrichment", dry_run=True)
        primary_backend = report.resolutions["PRIMARY_BACKEND"].authoritative
        assert primary_backend is not None
        assert primary_backend.id == chunk["id"]
        assert primary_backend.provenance_class == "RAW-ETAN-DIRECT"
        assert primary_backend.user_anchored is True

        results = store.search(query_text="PRIMARY_BACKEND", n_results=5, source_filter="mcp")
        assert results["ids"] == [[chunk["id"]]]
        assert results["metadatas"][0][0]["provenance_class"] == "RAW-ETAN-DIRECT"
    finally:
        store.close()
