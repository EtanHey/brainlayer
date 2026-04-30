from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brainlayer.mcp.search_handler import _brain_search, _exact_chunk_lookup_result


@pytest.mark.asyncio
async def test_brain_search_exact_chunk_id_query_bypasses_hybrid_search():
    """Free-text chunk IDs should short-circuit to an exact chunk lookup."""
    chunk_id = "brainbar-ddf12232"
    mock_store = MagicMock()
    mock_store.get_chunk.return_value = {
        "id": chunk_id,
        "content": "VoiceBar follow-up note about search recall regression",
        "source_file": "docs/repro.md",
        "project": "brainlayer",
        "content_type": "note",
        "importance": 9,
        "created_at": "2026-04-30T09:15:00Z",
        "summary": "Search recall regression repro",
        "tags": '["fts", "regression"]',
    }

    with (
        patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
        patch(
            "brainlayer.mcp.search_handler._search",
            new=AsyncMock(side_effect=AssertionError("exact chunk-id query should bypass hybrid search")),
        ),
    ):
        result = await _brain_search(query=chunk_id, detail="compact")

    _, structured = result
    assert structured["total"] == 1
    assert structured["results"][0]["chunk_id"] == chunk_id
    assert structured["results"][0]["project"] == "brainlayer"
    assert structured["results"][0]["summary"] == "Search recall regression repro"


@pytest.mark.asyncio
async def test_brain_search_exact_chunk_id_defaults_missing_project_to_unknown():
    """Exact chunk lookup should keep compact results stable when project is null."""
    chunk_id = "brainbar-nullproj01"
    mock_store = MagicMock()
    mock_store.get_chunk.return_value = {
        "id": chunk_id,
        "content": "Chunk without project metadata",
        "source_file": "docs/repro.md",
        "project": None,
        "content_type": "note",
        "importance": 3,
        "created_at": "2026-04-30T09:15:00Z",
        "summary": "Null project repro",
        "tags": '["fts"]',
    }

    with (
        patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
        patch(
            "brainlayer.mcp.search_handler._search",
            new=AsyncMock(side_effect=AssertionError("exact chunk-id query should bypass hybrid search")),
        ),
    ):
        _, structured = await _brain_search(query=chunk_id, detail="compact")

    assert structured["results"][0]["project"] == "unknown"


def test_exact_chunk_lookup_skips_lifecycle_managed_chunks():
    """Exact chunk lookup must respect default lifecycle filtering."""
    mock_store = MagicMock()
    mock_store.get_chunk.return_value = {
        "id": "brainbar-archived01",
        "content": "Archived chunk",
        "project": "brainlayer",
        "archived_at": "2026-04-30T09:15:00Z",
    }

    assert _exact_chunk_lookup_result("brainbar-archived01", mock_store, "compact") is None


@pytest.mark.asyncio
async def test_brain_search_chunk_id_context_routing_wins_over_exact_lookup():
    """Explicit chunk_id context expansion should run before exact-id short-circuiting."""
    chunk_id = "brainbar-ddf12232"
    mock_store = MagicMock()
    mock_store.get_chunk.return_value = {
        "id": chunk_id,
        "content": "Exact chunk content",
        "source_file": "docs/repro.md",
        "project": "brainlayer",
        "created_at": "2026-04-30T09:15:00Z",
    }

    with (
        patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
        patch("brainlayer.mcp.search_handler._context", new=AsyncMock(return_value=["context window"])) as context_mock,
        patch(
            "brainlayer.mcp.search_handler._search",
            new=AsyncMock(side_effect=AssertionError("chunk_id routing should bypass hybrid search")),
        ),
    ):
        result = await _brain_search(query=chunk_id, chunk_id=chunk_id, detail="compact")

    assert result == ["context window"]
    context_mock.assert_awaited_once_with(chunk_id=chunk_id, before=3, after=3)


@pytest.mark.asyncio
async def test_brain_search_exact_chunk_id_respects_project_scope():
    """Exact chunk-id bypass must not leak chunks outside the active project scope."""
    chunk_id = "brainbar-ddf12232"
    mock_store = MagicMock()
    mock_store.get_chunk.return_value = {
        "id": chunk_id,
        "content": "VoiceBar follow-up note about search recall regression",
        "source_file": "docs/repro.md",
        "project": "voicelayer",
        "content_type": "note",
        "importance": 9,
        "created_at": "2026-04-30T09:15:00Z",
        "summary": "Search recall regression repro",
        "tags": '["fts", "regression"]',
    }

    with (
        patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
        patch(
            "brainlayer.mcp.search_handler._search",
            new=AsyncMock(return_value=(["fallback"], {"total": 0, "results": []})),
        ) as search_mock,
    ):
        result = await _brain_search(query=chunk_id, project="brainlayer", detail="compact")

    assert result == (["fallback"], {"total": 0, "results": []})
    search_mock.assert_awaited_once()
