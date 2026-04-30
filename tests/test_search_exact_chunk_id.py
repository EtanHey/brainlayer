from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brainlayer.mcp.search_handler import _brain_search


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
