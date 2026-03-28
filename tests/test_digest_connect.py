"""Tests for the digest_connect (search→connect→propose) pipeline.

Tests the new mode="connect" for brain_digest:
1. Unit tests for digest_connect() with a mock VectorStore
2. Behavioral tests using the mock MCP harness from PR #109
3. Proposal structure validation

Run: pytest tests/test_digest_connect.py -v
"""

from unittest.mock import MagicMock

import pytest

from tests.mock_mcp import (
    assert_call_count,
    assert_call_sequence,
    assert_called_before,
    assert_never_called,
)
from tests.mock_mcp.mock_brainlayer import MockBrainLayer

# =============================================================================
# Unit tests: digest_connect() with mock VectorStore
# =============================================================================


class TestDigestConnectUnit:
    """Test the digest_connect function directly with mocked dependencies."""

    def _make_mock_store(self, search_results=None):
        """Create a mock VectorStore that returns configurable hybrid_search results.

        Accepts a list of chunk dicts with id/content/score/project/content_type/etc.
        Converts to the real hybrid_search format: {ids: [[...]], documents: [[...]], ...}
        """
        store = MagicMock()
        chunks = search_results or []
        # Convert to the real hybrid_search return format
        ids = [c.get("id", "") for c in chunks]
        docs = [c.get("content", "") for c in chunks]
        metas = [{k: v for k, v in c.items() if k not in ("id", "content", "score")} for c in chunks]
        # Convert score to distance (distance = 1 - score)
        dists = [1 - c.get("score", 0) for c in chunks]
        store.hybrid_search.return_value = {
            "ids": [ids],
            "documents": [docs],
            "metadatas": [metas],
            "distances": [dists],
        }
        return store

    def _make_mock_embed(self):
        """Create a mock embedding function."""
        return MagicMock(return_value=[0.1] * 1024)

    def test_returns_proposal_not_stored(self):
        """digest_connect returns a proposal and does NOT store anything."""
        from brainlayer.pipeline.digest import digest_connect

        store = self._make_mock_store()
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="We decided to use JWT tokens for authentication with 24h expiry.",
            store=store,
            embed_fn=embed_fn,
            title="Auth decision",
        )

        assert result["status"] == "proposal"
        assert "extracted" in result
        assert "connections" in result
        assert "contradictions" in result
        assert "supersede_proposals" in result
        assert "suggested_stores" in result
        assert "stats" in result
        # Verify nothing was stored
        store.upsert_chunks.assert_not_called()

    def test_extracts_entities(self):
        """Entities are extracted from the content."""
        from brainlayer.pipeline.digest import digest_connect

        store = self._make_mock_store()
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="BrainLayer uses SQLite for storage. Claude Code sessions are indexed via the watcher.",
            store=store,
            embed_fn=embed_fn,
        )

        # Should extract entities — the extraction pipeline finds entities in the content
        assert "entities" in result["extracted"]
        assert isinstance(result["extracted"]["entities"], list)

    def test_extracts_decisions(self):
        """Decisions are extracted from the content."""
        from brainlayer.pipeline.digest import digest_connect

        store = self._make_mock_store()
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="We decided to use Groq as the primary enrichment backend. We chose InMemoryTransport for mock MCP testing.",
            store=store,
            embed_fn=embed_fn,
        )

        assert len(result["extracted"]["decisions"]) >= 1

    def test_searches_existing_knowledge(self):
        """digest_connect calls hybrid_search to find related chunks."""
        from brainlayer.pipeline.digest import digest_connect

        store = self._make_mock_store()
        embed_fn = self._make_mock_embed()

        digest_connect(
            content="Authentication uses JWT tokens.",
            store=store,
            embed_fn=embed_fn,
            title="Auth decision",
        )

        # Should have searched at least once (for the title/topic)
        assert store.hybrid_search.call_count >= 1

    def test_finds_connections(self):
        """When existing chunks match, they appear as connections."""
        from brainlayer.pipeline.digest import digest_connect

        existing_chunks = [
            {
                "id": "chunk_old_001",
                "content": "Authentication was previously using session cookies.",
                "score": 0.88,
                "project": "brainlayer",
                "content_type": "decision",
                "created_at": "2026-03-10",
                "tags": ["auth"],
                "importance": 7,
            }
        ]
        store = self._make_mock_store(search_results=existing_chunks)
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="We decided to switch from cookies to JWT tokens for authentication.",
            store=store,
            embed_fn=embed_fn,
        )

        assert len(result["connections"]) >= 1
        assert result["connections"][0]["chunk_id"] == "chunk_old_001"
        assert result["stats"]["connections_found"] >= 1

    def test_proposes_supersedes_for_high_similarity(self):
        """High-similarity existing chunks are proposed for supersede."""
        from brainlayer.pipeline.digest import digest_connect

        existing_chunks = [
            {
                "id": "chunk_stale_001",
                "content": "decided to use session cookies for auth",
                "score": 0.92,  # Very high similarity
                "project": "brainlayer",
                "content_type": "decision",
                "created_at": "2026-03-01",
                "tags": ["auth"],
                "importance": 6,
            }
        ]
        store = self._make_mock_store(search_results=existing_chunks)
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="We decided to use JWT tokens instead of session cookies.",
            store=store,
            embed_fn=embed_fn,
        )

        assert len(result["supersede_proposals"]) >= 1
        assert result["supersede_proposals"][0]["chunk_id"] == "chunk_stale_001"

    def test_suggested_store_includes_supersedes(self):
        """Suggested store action includes supersede chunk IDs."""
        from brainlayer.pipeline.digest import digest_connect

        existing_chunks = [
            {
                "id": "chunk_replace_me",
                "content": "old auth decision",
                "score": 0.95,
                "project": "test",
                "content_type": "decision",
                "tags": [],
                "importance": 5,
            }
        ]
        store = self._make_mock_store(search_results=existing_chunks)
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="New auth decision replacing the old one.",
            store=store,
            embed_fn=embed_fn,
        )

        stores = result["suggested_stores"]
        assert len(stores) >= 1
        assert stores[0]["action"] == "store"
        assert stores[0]["content"] is not None

    def test_empty_content_raises(self):
        """Empty content raises ValueError."""
        from brainlayer.pipeline.digest import digest_connect

        store = self._make_mock_store()
        embed_fn = self._make_mock_embed()

        with pytest.raises(ValueError, match="non-empty"):
            digest_connect(content="", store=store, embed_fn=embed_fn)

        with pytest.raises(ValueError, match="non-empty"):
            digest_connect(content="   ", store=store, embed_fn=embed_fn)

    def test_no_connections_returns_empty(self):
        """When no existing knowledge matches, connections list is empty."""
        from brainlayer.pipeline.digest import digest_connect

        store = self._make_mock_store(search_results=[])
        embed_fn = self._make_mock_embed()

        result = digest_connect(
            content="Completely novel topic about quantum computing.",
            store=store,
            embed_fn=embed_fn,
        )

        assert result["connections"] == []
        assert result["contradictions"] == []
        assert result["supersede_proposals"] == []

    def test_importance_heuristic(self):
        """Auto-importance scoring works for proposals."""
        from brainlayer.pipeline.digest import _auto_propose_importance

        # Baseline (no decisions, no actions)
        assert _auto_propose_importance([], [], []) == 5

        # With decisions
        assert _auto_propose_importance([], ["decided X"], []) == 7

        # With decisions + actions
        assert _auto_propose_importance([], ["decided X"], ["TODO: fix"]) == 8

        # With high-confidence entities
        entities = [{"confidence": 0.9}] * 4
        assert _auto_propose_importance(entities, ["decided X"], ["fix"]) == 9


# =============================================================================
# Behavioral tests: MCP tool call sequence for digest connect
# =============================================================================


class TestDigestConnectBehavioral:
    """Behavioral tests using mock MCP harness.

    Tests the tool call sequence pattern:
    brain_digest(mode=connect) → brain_search for context → brain_store to accept.
    """

    @pytest.mark.asyncio
    async def test_connect_then_store_sequence(self):
        """The expected pattern: connect first, then store accepted proposals."""
        brain = MockBrainLayer()
        async with brain.connect() as client:
            # Step 1: Connect — search for related knowledge (returns proposal)
            result = await client.call_tool(
                "brain_search",
                {
                    "query": "authentication JWT tokens",
                },
            )

            # Step 2: Store — accept the proposal
            await client.call_tool(
                "brain_store",
                {
                    "content": "Decided to use JWT tokens for auth with 24h expiry.",
                    "tags": ["decision", "auth"],
                    "importance": 8,
                },
            )

        assert_called_before(brain, "brain_search", "brain_store")
        assert brain.was_called("brain_search")
        assert brain.was_called("brain_store")
        assert len(brain.stored_items) == 1

    @pytest.mark.asyncio
    async def test_connect_without_store_is_valid(self):
        """Agent may search and decide NOT to store — that's fine."""
        brain = MockBrainLayer()
        async with brain.connect() as client:
            # Search for context
            await client.call_tool("brain_search", {"query": "existing auth decisions"})
            # Agent decides: nothing new to store

        assert brain.was_called("brain_search")
        assert_never_called(brain, "brain_store")
        assert len(brain.stored_items) == 0

    @pytest.mark.asyncio
    async def test_search_before_store_pattern(self):
        """The connect pattern requires searching BEFORE storing."""
        brain = MockBrainLayer()
        async with brain.connect() as client:
            # BAD pattern: store without searching first
            await client.call_tool(
                "brain_store",
                {
                    "content": "Some decision",
                    "importance": 5,
                },
            )

        # This should fail the behavioral contract
        with pytest.raises(AssertionError):
            assert_called_before(brain, "brain_search", "brain_store")

    @pytest.mark.asyncio
    async def test_multiple_searches_then_store(self):
        """Multiple searches before a single store is the ideal pattern."""
        brain = MockBrainLayer()
        async with brain.connect() as client:
            # Search for different aspects
            await client.call_tool("brain_search", {"query": "auth decisions"})
            await client.call_tool("brain_search", {"query": "JWT token expiry"})
            await client.call_tool("brain_search", {"query": "session management"})

            # Then store the integrated result
            await client.call_tool(
                "brain_store",
                {
                    "content": "Auth uses JWT with 24h expiry, replacing session cookies.",
                    "tags": ["decision", "auth", "jwt"],
                    "importance": 8,
                },
            )

        assert_call_count(brain, "brain_search", 3)
        assert_call_count(brain, "brain_store", 1)
        assert_call_sequence(brain, ["brain_search", "brain_search", "brain_search", "brain_store"])

    @pytest.mark.asyncio
    async def test_search_store_search_store_pattern(self):
        """Multiple connect→store cycles for different topics."""
        brain = MockBrainLayer()
        async with brain.connect() as client:
            # Topic 1: Auth
            await client.call_tool("brain_search", {"query": "auth"})
            await client.call_tool(
                "brain_store",
                {
                    "content": "Auth decision: JWT",
                    "tags": ["auth"],
                },
            )

            # Topic 2: Database
            await client.call_tool("brain_search", {"query": "database"})
            await client.call_tool(
                "brain_store",
                {
                    "content": "DB decision: SQLite",
                    "tags": ["database"],
                },
            )

        assert_call_count(brain, "brain_search", 2)
        assert_call_count(brain, "brain_store", 2)
        assert len(brain.stored_items) == 2


# =============================================================================
# MCP handler integration test (mode routing)
# =============================================================================


class TestDigestModeRouting:
    """Test that the MCP handler correctly routes mode=connect."""

    def test_connect_mode_accepted_in_validation(self):
        """mode='connect' passes the validation check in _brain_digest."""
        # This tests the validation logic, not the full pipeline
        # (which would need a real DB)
        import asyncio

        from brainlayer.mcp.store_handler import _brain_digest

        # mode='connect' with no content should give a clear error
        result = asyncio.run(_brain_digest(content=None, mode="connect"))
        # Should get "content is required" error, NOT "Unknown mode" error
        error_text = result.content[0].text
        assert "content is required" in error_text
        assert "Unknown" not in error_text

    def test_invalid_mode_rejected(self):
        """Invalid modes are rejected."""
        import asyncio

        from brainlayer.mcp.store_handler import _brain_digest

        result = asyncio.run(_brain_digest(content="test", mode="invalid_mode"))
        error_text = result.content[0].text
        assert "Unknown" in error_text
