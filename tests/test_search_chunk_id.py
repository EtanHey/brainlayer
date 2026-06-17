"""Tests for brain_search returning chunk_id in results.

Verifies that both compact and full detail modes include chunk_id
from results["ids"] rather than metadata (which doesn't carry it).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from brainlayer.mcp.search_handler import _exact_chunk_lookup_result, _search


def _make_search_results(chunk_ids, documents, metadatas=None, distances=None):
    """Build a mock hybrid_search return value."""
    n = len(chunk_ids)
    if metadatas is None:
        metadatas = [
            {
                "source_file": f"session_{i}.jsonl",
                "project": "test-project",
                "content_type": "user_message",
                "value_type": "TEXT",
                "char_count": len(documents[i]),
                "summary": f"Summary {i}",
                "tags": json.dumps(["test"]),
                "importance": 5,
                "intent": "question",
                "created_at": "2026-03-01T12:00:00",
                "source": "manual",
            }
            for i in range(n)
        ]
    if distances is None:
        distances = [0.1 * (i + 1) for i in range(n)]
    return {
        "ids": [chunk_ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.count.return_value = 100
    store.enrich_results_with_session_context.side_effect = lambda r: r
    return store


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.embed_query.return_value = [0.0] * 1024
    return model


class TestSearchReturnsChunkId:
    """brain_search results must include chunk_id so brain_update can target them."""

    @pytest.mark.asyncio
    async def test_compact_results_include_chunk_id(self, mock_store, mock_model):
        """Compact detail mode must include chunk_id in every result."""
        chunk_ids = ["abc-123", "def-456", "ghi-789"]
        documents = ["First document", "Second document", "Third document"]
        mock_store.hybrid_search.return_value = _make_search_results(chunk_ids, documents)

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            result = await _search(query="test query", detail="compact")

        # Result is (text_contents, structured_dict)
        _, structured = result
        assert structured["total"] == 3
        for i, r in enumerate(structured["results"]):
            assert "chunk_id" in r, f"Result {i} missing chunk_id"
            assert r["chunk_id"] == chunk_ids[i], f"Result {i} chunk_id mismatch"

    @pytest.mark.asyncio
    async def test_compact_results_include_lightweight_provenance_fields(self, mock_store, mock_model):
        """Compact search should expose provenance_class and superseded_by when present."""
        chunk_ids = ["authoritative-1"]
        documents = ["Primary backend is Gemini"]
        metadatas = [
            {
                "source_file": "session.jsonl",
                "project": "brainlayer",
                "content_type": "user_message",
                "created_at": "2026-06-09T12:00:00Z",
                "provenance_class": "RAW-ETAN-DIRECT",
                "superseded_by": None,
            }
        ]
        mock_store.hybrid_search.return_value = _make_search_results(chunk_ids, documents, metadatas=metadatas)

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            result = await _search(query="primary backend", detail="compact")

        _, structured = result
        assert structured["results"][0]["provenance_class"] == "RAW-ETAN-DIRECT"
        assert "superseded_by" not in structured["results"][0]

    @pytest.mark.asyncio
    async def test_full_results_include_chunk_id(self, mock_store, mock_model):
        """Full detail mode must include chunk_id in every result."""
        chunk_ids = ["abc-123", "def-456"]
        documents = ["First document content", "Second document content"]
        mock_store.hybrid_search.return_value = _make_search_results(chunk_ids, documents)

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            result = await _search(query="test query", detail="full")

        _, structured = result
        assert structured["total"] == 2
        for i, r in enumerate(structured["results"]):
            assert "chunk_id" in r, f"Result {i} missing chunk_id in full mode"
            assert r["chunk_id"] == chunk_ids[i]

    @pytest.mark.asyncio
    async def test_chunk_id_not_dependent_on_metadata(self, mock_store, mock_model):
        """chunk_id must come from results['ids'], not metadata (which doesn't have it)."""
        chunk_ids = ["real-chunk-id-001"]
        documents = ["Some content"]
        # Metadata explicitly has NO chunk_id key
        metadatas = [
            {
                "source_file": "session.jsonl",
                "project": "test",
                "content_type": "user_message",
                "value_type": "TEXT",
                "char_count": 12,
                "created_at": "2026-03-01T12:00:00",
            }
        ]
        mock_store.hybrid_search.return_value = _make_search_results(chunk_ids, documents, metadatas=metadatas)

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            # Test compact
            _, compact = await _search(query="test", detail="compact")
            assert compact["results"][0]["chunk_id"] == "real-chunk-id-001"

            # Test full
            _, full = await _search(query="test", detail="full")
            assert full["results"][0]["chunk_id"] == "real-chunk-id-001"

    @pytest.mark.asyncio
    async def test_full_text_output_exposes_chunk_id(self, mock_store, mock_model):
        """Full detail rendered TEXT must include the chunk_id so it can be chained
        into brain_update/brain_expand/brain_supersede/brain_archive from a search hit."""
        chunk_ids = ["text-id-001", "text-id-002"]
        documents = ["First document content", "Second document content"]
        mock_store.hybrid_search.return_value = _make_search_results(chunk_ids, documents)

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            content, _ = await _search(query="test query", detail="full")

        text = "\n".join(getattr(c, "text", "") for c in content)
        assert "- ID: text-id-001" in text
        assert "- ID: text-id-002" in text

    @pytest.mark.asyncio
    async def test_compact_text_output_hides_chunk_id(self, mock_store, mock_model):
        """Compact rendered TEXT must NOT expose the chunk_id."""
        chunk_ids = ["hidden-id-001"]
        documents = ["Some content"]
        mock_store.hybrid_search.return_value = _make_search_results(chunk_ids, documents)

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            content, _ = await _search(query="test query", detail="compact")

        text = "\n".join(getattr(c, "text", "") for c in content)
        assert "hidden-id-001" not in text
        assert "- ID:" not in text

    @pytest.mark.asyncio
    async def test_exact_lookup_full_text_output_exposes_chunk_id(self, mock_store, mock_model):
        """Exact chunk-id lookup must honor detail=full in rendered text."""
        mock_store.get_chunk.return_value = {
            "id": "exact-id-001",
            "content": "Exact document content",
            "source_file": "session.jsonl",
            "project": "test-project",
            "created_at": "2026-03-01T12:00:00",
            "importance": 5,
            "summary": "Exact summary",
        }

        content, structured = _exact_chunk_lookup_result("exact-id-001", mock_store, detail="full")

        text = "\n".join(getattr(c, "text", "") for c in content)
        assert structured["results"][0]["chunk_id"] == "exact-id-001"
        assert "- ID: exact-id-001" in text

    @pytest.mark.asyncio
    async def test_empty_results_no_crash(self, mock_store, mock_model):
        """Empty result set should not crash."""
        mock_store.hybrid_search.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            result = await _search(query="nothing", detail="compact")
            _, structured = result
            assert structured["total"] == 0
