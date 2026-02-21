"""Tests for brainlayer_store — write-side MCP tool.

TDD: Tests written BEFORE implementation.

brainlayer_store allows agents to persistently store:
- ideas, mistakes, decisions, learnings, todos, bookmarks

Stored items go into the chunks table with source='manual',
are embedded at write time, and return related existing memories.
"""

import json
from datetime import datetime, timezone

import pytest

from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Mock embedding function that returns a fixed 1024-dim vector."""

    def _embed(text: str) -> list[float]:
        # Return a deterministic but text-dependent vector
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


# ── Unit Tests: store_memory() ────────────────────────────────


class TestStoreMemory:
    """Test the core store_memory function."""

    def test_store_returns_id(self, store, mock_embed):
        """Storing a memory returns a chunk ID."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Always retry API calls with exponential backoff",
            memory_type="learning",
            project="golems",
        )
        assert result["id"] is not None
        assert isinstance(result["id"], str)
        assert len(result["id"]) > 0

    def test_store_persists_to_db(self, store, mock_embed):
        """Stored memory is retrievable from the database."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Use sqlite-vec for local vector search",
            memory_type="decision",
            project="brainlayer",
        )

        # Verify it's in the chunks table
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT * FROM chunks WHERE id = ?", (result["id"],)))
        assert len(rows) == 1

    def test_store_sets_source_manual(self, store, mock_embed):
        """Stored memories have source='manual'."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="This is a manual note",
            memory_type="note",
            project="test",
        )

        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT source FROM chunks WHERE id = ?", (result["id"],)))
        assert rows[0][0] == "manual"

    def test_store_sets_content_type(self, store, mock_embed):
        """content_type is set based on memory_type."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="JWT is better than session cookies for our API",
            memory_type="decision",
            project="golems",
        )

        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT content_type FROM chunks WHERE id = ?", (result["id"],)))
        assert rows[0][0] == "decision"

    def test_store_with_tags(self, store, mock_embed):
        """Tags are stored as JSON array."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Never use rm -rf without confirmation",
            memory_type="mistake",
            project="golems",
            tags=["safety", "bash", "destructive"],
        )

        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT tags FROM chunks WHERE id = ?", (result["id"],)))
        tags = json.loads(rows[0][0])
        assert "safety" in tags
        assert "bash" in tags

    def test_store_with_importance(self, store, mock_embed):
        """Importance score is stored."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Critical: never expose API keys in logs",
            memory_type="learning",
            project="golems",
            importance=9,
        )

        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT importance FROM chunks WHERE id = ?", (result["id"],)))
        assert rows[0][0] == 9.0

    def test_store_creates_embedding(self, store, mock_embed):
        """Stored memory gets embedded at write time."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Use bge-large-en-v1.5 for embeddings",
            memory_type="decision",
            project="brainlayer",
        )

        # Verify embedding exists in chunk_vectors
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (result["id"],)))
        assert rows[0][0] == 1

    def test_store_sets_timestamps(self, store, mock_embed):
        """Stored memory has created_at and enriched_at timestamps."""
        from brainlayer.store import store_memory

        before = datetime.now(timezone.utc).isoformat()
        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Timestamps matter",
            memory_type="note",
            project="test",
        )
        after = datetime.now(timezone.utc).isoformat()

        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT created_at, enriched_at FROM chunks WHERE id = ?", (result["id"],)))
        created_at = rows[0][0]
        assert created_at >= before
        assert created_at <= after

    def test_store_returns_related_memories(self, store, mock_embed):
        """Storing a memory returns related existing memories."""
        from brainlayer.store import store_memory

        # Store a first memory
        store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Authentication uses JWT tokens with RS256",
            memory_type="decision",
            project="golems",
        )

        # Store a second related memory — should find the first
        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="JWT tokens expire after 24 hours",
            memory_type="learning",
            project="golems",
        )

        assert "related" in result
        # May or may not find related depending on mock embedding similarity
        assert isinstance(result["related"], list)


class TestStoreValidation:
    """Test input validation for store_memory."""

    def test_empty_content_rejected(self, store, mock_embed):
        """Empty content raises ValueError."""
        from brainlayer.store import store_memory

        with pytest.raises(ValueError, match="content"):
            store_memory(
                store=store,
                embed_fn=mock_embed,
                content="",
                memory_type="note",
                project="test",
            )

    def test_invalid_type_rejected(self, store, mock_embed):
        """Invalid memory_type raises ValueError."""
        from brainlayer.store import store_memory

        with pytest.raises(ValueError, match="type"):
            store_memory(
                store=store,
                embed_fn=mock_embed,
                content="test content",
                memory_type="invalid_type",
                project="test",
            )

    def test_valid_types_accepted(self, store, mock_embed):
        """All valid memory types are accepted."""
        from brainlayer.store import VALID_MEMORY_TYPES, store_memory

        for mtype in VALID_MEMORY_TYPES:
            result = store_memory(
                store=store,
                embed_fn=mock_embed,
                content=f"Test content for {mtype}",
                memory_type=mtype,
                project="test",
            )
            assert result["id"] is not None

    def test_importance_clamped(self, store, mock_embed):
        """Importance is clamped to 1-10 range."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Over-important note",
            memory_type="note",
            project="test",
            importance=15,
        )
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT importance FROM chunks WHERE id = ?", (result["id"],)))
        assert rows[0][0] == 10.0

    def test_project_optional(self, store, mock_embed):
        """Project is optional — defaults to None/unknown."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="General note without project",
            memory_type="note",
        )
        assert result["id"] is not None


class TestStoreMCPIntegration:
    """Test that brainlayer_store is properly wired into the MCP server."""

    def test_store_tool_listed(self):
        """brainlayer_store appears in the tool list."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]
        assert "brainlayer_store" in tool_names

    def test_store_tool_has_write_annotations(self):
        """brainlayer_store has destructive/non-read-only annotations."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brainlayer_store")
        # Write tool should NOT be read-only
        assert store_tool.annotations.readOnlyHint is False

    def test_store_tool_input_schema(self):
        """brainlayer_store has correct required fields."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brainlayer_store")
        schema = store_tool.inputSchema
        assert "content" in schema["properties"]
        assert "type" in schema["properties"]
        assert "content" in schema["required"]
        assert "type" in schema["required"]
