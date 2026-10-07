"""Tests for brain_digest MCP mode routing."""

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
@pytest.mark.parametrize("options", [{}, {"limit": 7}, {"content": "synthetic content", "limit": 3}])
async def test_brain_digest_retired_enrich_rejects_before_database_or_producer(monkeypatch, options):
    from brainlayer.mcp.store_handler import _brain_digest

    database = MagicMock(side_effect=AssertionError("retired digest mode opened DB"))
    producer = MagicMock(side_effect=AssertionError("retired digest mode invoked producer"))
    monkeypatch.setattr("brainlayer.mcp.store_handler._get_vector_store", database)
    monkeypatch.setattr("brainlayer.enrichment_controller.enrich_realtime", producer)
    result = await _brain_digest(mode="enrich", **options)
    assert result.is_error is True
    assert "Enrichment has been retired" in result.content[0].text
    database.assert_not_called()
    producer.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["full", "operator"])
async def test_brain_digest_retired_enrich_dispatch_never_opens_database(monkeypatch, profile):
    from brainlayer import mcp
    from brainlayer.mcp.palette import ToolPalette

    database = MagicMock(side_effect=AssertionError("retired digest dispatch opened DB"))
    monkeypatch.setattr(mcp, "_tool_palette", ToolPalette(profile))
    monkeypatch.setattr("brainlayer.mcp.store_handler._get_vector_store", database)
    result = await mcp.call_tool("brain_digest", {"mode": "enrich", "limit": 7})
    assert result.is_error is True
    assert "Enrichment has been retired" in result.content[0].text
    database.assert_not_called()


@pytest.mark.asyncio
async def test_brain_digest_connect_keeps_local_proposal_routing(monkeypatch):
    from brainlayer.mcp.store_handler import _brain_digest

    monkeypatch.setattr("brainlayer.mcp.store_handler._get_vector_store", lambda: MagicMock())
    monkeypatch.setattr(
        "brainlayer.mcp.store_handler._get_embedding_model",
        lambda: SimpleNamespace(embed_query=lambda text: [0.1] * 1024),
    )
    monkeypatch.setattr("brainlayer.mcp.store_handler._normalize_project_name", lambda project: project)
    connect = MagicMock(return_value={"mode": "connect", "proposal": {"content": "synthetic content"}, "stats": {}})
    monkeypatch.setattr("brainlayer.pipeline.digest.digest_connect", connect)
    result = await _brain_digest(content="synthetic content", mode="connect", project="fixture", limit=7)
    assert result.is_error is not True
    assert connect.call_args.kwargs["content"] == "synthetic content"
    assert connect.call_args.kwargs["project"] == "fixture"


@pytest.mark.asyncio
async def test_brain_digest_missing_content_with_mode_digest_errors():
    from brainlayer.mcp.store_handler import _brain_digest

    result = await _brain_digest(content=None, mode="digest")

    assert result.is_error is True
    assert "content is required" in result.content[0].text.lower()


def test_brain_digest_input_schema_advertises_only_digest_and_connect():
    from brainlayer.mcp import _full_tool_definitions

    tools = _full_tool_definitions()
    digest = next(t for t in tools if t.name == "brain_digest")
    props = digest.input_schema["properties"]

    assert "mode" in props
    assert props["mode"]["enum"] == ["digest", "connect"]
    assert "limit" not in props
    assert "enrich" not in digest.description.lower()
