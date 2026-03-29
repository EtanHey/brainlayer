"""Tests for brain_digest MCP mode routing."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.asyncio
async def test_brain_digest_mode_digest_preserves_current_behavior(monkeypatch):
    from brainlayer.mcp.store_handler import _brain_digest

    monkeypatch.setattr("brainlayer.mcp.store_handler._get_vector_store", lambda: MagicMock())
    monkeypatch.setattr(
        "brainlayer.mcp.store_handler._get_embedding_model",
        lambda: SimpleNamespace(embed_query=lambda text: [0.1] * 1024),
    )
    monkeypatch.setattr("brainlayer.mcp.store_handler._normalize_project_name", lambda p: p)
    monkeypatch.setattr(
        "brainlayer.pipeline.digest.digest_content",
        lambda **kwargs: {"digest_id": "digest-123", "summary": "ok", "kwargs_content": kwargs["content"]},
    )

    result = await _brain_digest(content="raw research", mode="digest", title="T")

    text = result.content[0].text
    assert "brain_digest (digest)" in text
    assert "\u250c" in text  # box-drawing top-left


@pytest.mark.asyncio
async def test_brain_digest_mode_enrich_without_content_runs_enrich_realtime(monkeypatch):
    from brainlayer.mcp.store_handler import _brain_digest

    store = MagicMock()
    monkeypatch.setattr("brainlayer.mcp.store_handler._get_vector_store", lambda: store)
    monkeypatch.setattr(
        "brainlayer.enrichment_controller.enrich_realtime",
        lambda store, limit=25: SimpleNamespace(
            mode="realtime", attempted=limit, enriched=2, skipped=0, failed=0, errors=[]
        ),
    )

    result = await _brain_digest(mode="enrich", limit=7)

    text = result.content[0].text
    assert "brain_digest" in text
    assert "7" in text  # attempted count


@pytest.mark.asyncio
async def test_brain_digest_mode_enrich_returns_structured_enrichment_result(monkeypatch):
    from brainlayer.mcp.store_handler import _brain_digest

    monkeypatch.setattr("brainlayer.mcp.store_handler._get_vector_store", lambda: MagicMock())
    monkeypatch.setattr(
        "brainlayer.enrichment_controller.enrich_realtime",
        lambda store, limit=25: SimpleNamespace(
            mode="realtime", attempted=3, enriched=2, skipped=1, failed=0, errors=["note"]
        ),
    )

    result = await _brain_digest(mode="enrich", limit=3)

    text = result.content[0].text
    assert "brain_digest" in text
    assert "3" in text  # attempted
    assert "2" in text  # enriched


@pytest.mark.asyncio
async def test_brain_digest_missing_content_with_mode_digest_errors():
    from brainlayer.mcp.store_handler import _brain_digest

    result = await _brain_digest(content=None, mode="digest")

    assert result.isError is True
    assert "content is required" in result.content[0].text.lower()


def test_brain_digest_input_schema_includes_mode_and_limit():
    from brainlayer.mcp import list_tools

    tools = asyncio.run(list_tools())
    digest = next(t for t in tools if t.name == "brain_digest")
    props = digest.inputSchema["properties"]

    assert "mode" in props
    assert props["mode"]["enum"] == ["digest", "enrich", "connect"]
    assert "limit" in props
