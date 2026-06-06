"""Tests for brainlayer_store — write-side MCP tool.

TDD: Tests written BEFORE implementation.

brainlayer_store allows agents to persistently store:
- ideas, mistakes, decisions, learnings, todos, bookmarks

Stored items go into the chunks table with source='manual',
are embedded at write time, and return related existing memories.
"""

import json
import time
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

    def test_store_created_at_override_does_not_backdate_write_timestamps(self, store, mock_embed):
        """Queued replay created_at stays separate from write-time timestamps."""
        from brainlayer.store import store_memory

        reservation_created_at = "2026-01-01T00:00:00+00:00"
        before = datetime.now(timezone.utc).isoformat()
        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Replay timestamp separation matters",
            memory_type="note",
            project="test",
            created_at=reservation_created_at,
        )
        after = datetime.now(timezone.utc).isoformat()

        cursor = store.conn.cursor()
        row = cursor.execute(
            "SELECT created_at, enriched_at, last_seen_at FROM chunks WHERE id = ?",
            (result["id"],),
        ).fetchone()

        assert row is not None
        assert row[0] == reservation_created_at
        assert before <= row[1] <= after
        assert before <= row[2] <= after

    def test_store_sets_ingested_at(self, store, mock_embed):
        """Stored memory has a fresh Unix ingested_at timestamp."""
        from brainlayer.store import store_memory

        before = int(time.time())
        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="brain_store ingested_at regression coverage",
            memory_type="note",
            project="test",
        )
        after = int(time.time())

        cursor = store.conn.cursor()
        row = cursor.execute("SELECT ingested_at FROM chunks WHERE id = ?", (result["id"],)).fetchone()

        assert row is not None
        assert row[0] is not None
        assert before - 5 <= row[0] <= after + 5

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

    def test_store_merges_duplicate_memory(self, store, mock_embed):
        """Duplicate brain_store writes collapse into one canonical chunk."""
        from brainlayer.store import store_memory

        content = "Duplicate manual memory should merge through direct brain_store writes"
        first = store_memory(
            store=store,
            embed_fn=mock_embed,
            content=content,
            memory_type="learning",
            project="brainlayer",
            tags=["first"],
            importance=4,
        )
        second = store_memory(
            store=store,
            embed_fn=mock_embed,
            content=content,
            memory_type="learning",
            project="brainlayer",
            tags=["second"],
            importance=9,
        )

        row = (
            store.conn.cursor()
            .execute("SELECT id, seen_count, importance, tags FROM chunks WHERE source = 'manual'")
            .fetchone()
        )
        alias = store.conn.cursor().execute("SELECT canonical_chunk_id FROM chunk_id_alias").fetchone()

        assert second["id"] == first["id"]
        assert row == (first["id"], 2, 9.0, '["first", "second"]')
        assert alias == (first["id"],)

    def test_duplicate_store_uses_available_embedding_when_canonical_has_none(self, store, mock_embed):
        """A later synchronous duplicate should backfill a missing canonical vector."""
        from brainlayer.store import store_memory

        content = "Duplicate manual memory should reuse available exact embedding"
        first = store_memory(
            store=store,
            embed_fn=None,
            content=content,
            memory_type="learning",
            project="brainlayer",
        )
        assert store.conn.cursor().execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 0

        second = store_memory(
            store=store,
            embed_fn=mock_embed,
            content=content,
            memory_type="learning",
            project="brainlayer",
        )

        vector_count = (
            store.conn.cursor()
            .execute("SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = ?", (first["id"],))
            .fetchone()[0]
        )
        assert second["id"] == first["id"]
        assert vector_count == 1

    def test_duplicate_store_busy_retry_keeps_incoming_chunk_id_stable(self, store, mock_embed, monkeypatch):
        """A BusyError after merge rollback should retry with the original incoming ID."""
        import apsw

        from brainlayer.store import store_memory

        cursor = store.conn.cursor()
        cursor.execute(
            "INSERT INTO kg_entities(id, name, entity_type) VALUES (?, ?, ?)",
            ("entity-retry", "Retry Entity", "project"),
        )
        first = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Duplicate retry memory should survive one busy rollback",
            memory_type="learning",
            project="brainlayer",
        )

        original_link = store.link_entity_chunk
        calls = {"count": 0}

        def flaky_link_entity_chunk(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise apsw.BusyError("database is locked")
            return original_link(*args, **kwargs)

        monkeypatch.setattr(store, "link_entity_chunk", flaky_link_entity_chunk)

        second = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Duplicate retry memory should survive one busy rollback",
            memory_type="learning",
            project="brainlayer",
            entity_id="entity-retry",
        )

        row = cursor.execute("SELECT seen_count FROM chunks WHERE id = ?", (first["id"],)).fetchone()
        link = cursor.execute(
            "SELECT chunk_id FROM kg_entity_chunks WHERE entity_id = ?",
            ("entity-retry",),
        ).fetchone()
        assert second["id"] == first["id"]
        assert row == (2,)
        assert link == (first["id"],)

    def test_supplied_chunk_id_retry_is_idempotent(self, store, mock_embed):
        """Retrying a queued write with the promised chunk ID should not violate the primary key."""
        from brainlayer.store import store_memory

        chunk_id = "manual-queuedretry01"
        content = "Queued brain_store retry must reuse the promised chunk id"

        first = store_memory(
            store=store,
            embed_fn=mock_embed,
            content=content,
            memory_type="learning",
            project="brainlayer",
            tags=["first"],
            importance=4,
            chunk_id=chunk_id,
        )
        second = store_memory(
            store=store,
            embed_fn=mock_embed,
            content=content,
            memory_type="learning",
            project="brainlayer",
            tags=["second"],
            importance=9,
            chunk_id=chunk_id,
        )

        row = (
            store.conn.cursor()
            .execute("SELECT seen_count, importance, tags FROM chunks WHERE id = ?", (chunk_id,))
            .fetchone()
        )
        audit = (
            store.conn.cursor()
            .execute(
                "SELECT chunk_id_dropped, chunk_id_kept, mechanism FROM dedupe_audit WHERE chunk_id_kept = ?",
                (chunk_id,),
            )
            .fetchone()
        )

        assert first["id"] == chunk_id
        assert second["id"] == chunk_id
        assert row == (2, 9.0, '["first", "second"]')
        assert audit == (chunk_id, chunk_id, "sha256_same_id")

    def test_changed_duplicate_embedding_runs_after_write_transaction(self, store):
        """Merged-content embedding should not hold the write transaction open."""
        from brainlayer.store import store_memory

        calls = []

        def embed(content: str):
            calls.append((content, store.conn.getautocommit()))
            return [0.1] * 1024

        first_words = [f"token{i}" for i in range(100)]
        second_words = first_words.copy()
        second_words[0] = "changed0"

        first = store_memory(
            store=store,
            embed_fn=embed,
            content=" ".join(first_words),
            memory_type="learning",
            project="brainlayer",
        )
        second = store_memory(
            store=store,
            embed_fn=embed,
            content=" ".join(second_words),
            memory_type="learning",
            project="brainlayer",
        )

        merged_embedding_calls = [call for call in calls if "\n\n2. " in call[0]]
        assert second["id"] == first["id"]
        assert merged_embedding_calls
        assert all(autocommit for _, autocommit in merged_embedding_calls)


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

    def test_system_prompt_content_rejected(self, store, mock_embed):
        """Prompt scaffolding should be rejected instead of polluting memory."""
        from brainlayer.store import store_memory

        with pytest.raises(ValueError, match="system prompt"):
            store_memory(
                store=store,
                embed_fn=mock_embed,
                content="# Base Context\n\nYou are a coding agent.\n\n## IRON RULES",
                memory_type="note",
                project="test",
            )


class TestStoreMCPIntegration:
    """Test that brain_store is properly wired into the MCP server."""

    def test_store_tool_listed(self):
        """brain_store appears in the tool list."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]
        assert "brain_store" in tool_names

    def test_store_tool_has_write_annotations(self):
        """brain_store has destructive/non-read-only annotations."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        # Write tool should NOT be read-only
        assert store_tool.annotations.readOnlyHint is False

    def test_store_tool_input_schema(self):
        """brain_store has correct required fields — only content required, type is optional."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        schema = store_tool.inputSchema
        assert "content" in schema["properties"]
        assert "type" in schema["properties"]
        assert "content" in schema["required"]
        # type is now optional (auto-detected from content)
        assert "type" not in schema["required"]
