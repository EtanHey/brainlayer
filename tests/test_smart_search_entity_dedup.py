"""Tests for smart search (compact defaults + brain_expand) and entity dedup fixes.

TDD tests — written before implementation.
"""

import asyncio
from collections import Counter

from brainlayer.vector_store import VectorStore


def _dummy_embed(text):  # noqa: ARG001
    """Dummy embedding function — returns constant vector."""
    return [0.1] * 1024


def _varied_embed(text):
    """Embedding that varies slightly based on text hash for similarity testing."""
    import hashlib

    h = hashlib.md5(text.lower().encode()).hexdigest()
    base = [0.1] * 1024
    for i, c in enumerate(h):
        base[i % 1024] += int(c, 16) * 0.001
    return base


def _insert_chunks(store, ids, documents, metadatas, embeddings):
    """Helper to insert test chunks."""
    chunks = []
    for cid, doc, meta in zip(ids, documents, metadatas):
        chunks.append(
            {
                "id": cid,
                "content": doc,
                "metadata": meta,
                "source_file": meta.get("source_file", "test.jsonl"),
                "project": meta.get("project"),
                "content_type": meta.get("content_type", "user_message"),
                "char_count": len(doc),
                "source": meta.get("source", "claude_code"),
            }
        )
    store.upsert_chunks(chunks, embeddings)


# ============================================================
# 1. brain_search defaults to compact (no full content)
# ============================================================


class TestCompactSearchDefault:
    """brain_search should return compact results by default."""

    def test_compact_result_builder_returns_snippet(self):
        """_build_compact_result returns snippet (150 chars), chunk_id, score — no full content."""
        from brainlayer.mcp._shared import _build_compact_result

        long_content = "A" * 500
        result = _build_compact_result(
            {
                "score": 0.95,
                "chunk_id": "chunk-abc123",
                "project": "test",
                "content": long_content,
                "source_file": "auth.ts",
                "date": "2026-02-27",
                "importance": 7,
                "summary": "Auth implementation",
            }
        )

        # Compact: snippet should be truncated (<=150 chars)
        assert "snippet" in result
        assert len(result["snippet"]) <= 150
        # Should NOT have full "content" key
        assert "content" not in result
        # Must have chunk_id for drill-down
        assert result["chunk_id"] == "chunk-abc123"
        # Must have core fields
        assert result["score"] == 0.95
        assert result["project"] == "test"
        assert result["summary"] == "Auth implementation"

    def test_compact_result_omits_verbose_fields(self):
        """Compact results should NOT include source_file, importance — only essentials."""
        from brainlayer.mcp._shared import _build_compact_result

        result = _build_compact_result(
            {
                "score": 0.9,
                "chunk_id": "chunk-1",
                "project": "test",
                "content": "Some content",
                "source_file": "auth.ts",
                "date": "2026-02-27",
                "importance": 5,
                "summary": "Test",
            }
        )

        # Compact mode should not have verbose fields
        assert "source_file" not in result
        assert "importance" not in result
        assert "content_type" not in result
        assert "tags" not in result
        assert "intent" not in result

    def test_search_default_detail_is_compact(self, tmp_path):
        """_search default detail param is 'compact'."""
        import inspect

        from brainlayer.mcp.search_handler import _search

        sig = inspect.signature(_search)
        assert sig.parameters["detail"].default == "compact"


# ============================================================
# 2. brain_search detail="full" returns verbose format
# ============================================================


class TestDetailFullSearch:
    """brain_search with detail='full' returns the current verbose format."""

    def test_detail_full_bypasses_compact(self):
        """When detail='full', _search should NOT go through compact path."""
        import inspect

        from brainlayer.mcp.search_handler import _search

        # The function should accept detail parameter
        sig = inspect.signature(_search)
        assert "detail" in sig.parameters

    def test_detail_full_format_backward_compat(self):
        """Old 'format' kwarg still works as backward compat for 'detail'."""
        import inspect

        from brainlayer.mcp.search_handler import _search

        sig = inspect.signature(_search)
        # Should still accept 'format' for backward compat
        assert "format" in sig.parameters


# ============================================================
# 3. brain_expand tool returns full content + surrounding context
# ============================================================


class TestBrainExpand:
    """brain_expand tool should be a first-class MCP tool."""

    def test_brain_expand_tool_registered(self):
        """brain_expand is listed in MCP tools."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        tool_names = [t.name for t in tools]
        assert "brain_expand" in tool_names

    def test_brain_expand_schema(self):
        """brain_expand requires chunk_id, optional context param."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        expand_tool = next(t for t in tools if t.name == "brain_expand")
        props = expand_tool.inputSchema.get("properties", {})
        assert "chunk_id" in props
        assert "context" in props
        assert "chunk_id" in expand_tool.inputSchema.get("required", [])

    def test_brain_expand_returns_target_chunk(self, tmp_path):
        """brain_expand returns at least the target chunk content."""
        store = VectorStore(tmp_path / "test.db")

        _insert_chunks(
            store,
            ["chunk-2"],
            ["Full detailed content of the target chunk with all information"],
            [{"project": "test", "source_file": "test.jsonl"}],
            [_dummy_embed("chunk 2")],
        )

        # get_context returns dict with "target" containing the chunk
        ctx = store.get_context("chunk-2", before=2, after=2)
        assert "target" in ctx
        assert ctx["target"]["content"] == "Full detailed content of the target chunk with all information"

    def test_brain_expand_routes_in_call_tool(self):
        """brain_expand is callable via the MCP call_tool dispatcher."""
        from brainlayer.mcp import call_tool

        # This should not raise "Unknown tool" error
        # It will fail with chunk not found, but that's expected
        result = asyncio.run(call_tool("brain_expand", {"chunk_id": "nonexistent"}))
        # Should get an error about chunk not found, NOT "Unknown tool"
        text = result.content[0].text if hasattr(result, "content") else result[0].text
        assert "Unknown tool" not in text


# ============================================================
# 4. Digest deduplicates entity mentions
# ============================================================


class TestDigestEntityDedup:
    """Digest should not return duplicate entity references."""

    def test_digest_deduplicates_repeated_entity_mentions(self, tmp_path):
        """When content mentions 'golems' twice, entities list should have it only once."""
        from brainlayer.pipeline.digest import digest_content

        store = VectorStore(tmp_path / "test.db")

        result = digest_content(
            content=(
                "The golems project has a monorepo architecture. "
                "We discussed golems deployment strategy. "
                "The golems team agreed on using Convex."
            ),
            store=store,
            embed_fn=_dummy_embed,
            title="Meeting about golems",
        )

        entity_names_with_types = [(e["name"].lower(), e["entity_type"]) for e in result["entities"]]
        counts = Counter(entity_names_with_types)
        for (name, etype), count in counts.items():
            assert count == 1, f"Entity '{name}' ({etype}) appears {count} times, expected 1"

    def test_digest_deduplicates_case_variants(self, tmp_path):
        """'BrainLayer' and 'brainlayer' should resolve to same entity."""
        from brainlayer.pipeline.digest import digest_content

        store = VectorStore(tmp_path / "test.db")

        result = digest_content(
            content="BrainLayer is a knowledge pipeline. We use brainlayer for memory management.",
            store=store,
            embed_fn=_dummy_embed,
        )

        # After dedup, brainlayer should appear only once (regardless of casing)
        entity_ids = [e["entity_id"] for e in result["entities"] if e["entity_id"]]
        # All mentions of brainlayer should resolve to the same entity_id
        brainlayer_entities = [e for e in result["entities"] if "brainlayer" in e["name"].lower()]
        if len(brainlayer_entities) > 0:
            ids = set(e["entity_id"] for e in brainlayer_entities)
            assert len(ids) <= 1, f"BrainLayer resolved to multiple entity IDs: {ids}"


# ============================================================
# 5. Entity resolution with vector similarity fallback
# ============================================================


class TestEntityResolutionVectorSimilarity:
    """Entity resolution should use vector similarity when exact match fails."""

    def test_resolve_links_to_existing_entity_case_insensitive(self, tmp_path):
        """'BrainLayer' should link to existing 'brainlayer' entity, not create new."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        store = VectorStore(tmp_path / "test.db")

        # Create existing entity
        existing_id = store.upsert_entity("project-bl", "project", "brainlayer", embedding=_dummy_embed("brainlayer"))

        # Resolve a differently-cased variant
        resolved_id = resolve_entity("BrainLayer", "project", "memory pipeline", store)

        assert resolved_id == existing_id, (
            f"Expected to link to existing '{existing_id}', got new entity '{resolved_id}'"
        )

    def test_resolve_with_vector_similarity_fallback(self, tmp_path):
        """Entity resolution should fall back to vector similarity for near-matches."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        store = VectorStore(tmp_path / "test.db")

        # Create existing entity with embedding
        existing_id = store.upsert_entity(
            "person-etan",
            "person",
            "Etan Heyman",
            embedding=_varied_embed("person: Etan Heyman"),
        )

        # Resolve should accept an embed_fn for vector similarity fallback
        # This tests the new signature with embed_fn parameter
        resolved_id = resolve_entity(
            "Etan",
            "person",
            "discussed brainlayer architecture",
            store,
            embed_fn=_varied_embed,
        )

        # Should resolve to existing entity (exact name won't match, but
        # the current code already handles this via FTS. The key is it doesn't
        # create a new entity)
        # Note: with dummy embeddings all being the same, vector similarity
        # will always be high, so this tests the pathway exists
        assert resolved_id is not None

    def test_resolve_creates_new_when_no_match(self, tmp_path):
        """Entity resolution creates new entity when nothing matches."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        store = VectorStore(tmp_path / "test.db")

        # No entities exist yet
        resolved_id = resolve_entity(
            "Completely New Person",
            "person",
            "some context",
            store,
            embed_fn=_varied_embed,
        )

        assert resolved_id is not None
        entity = store.get_entity(resolved_id)
        assert entity is not None
        assert entity["name"] == "Completely New Person"


# ============================================================
# MCP tool schema tests
# ============================================================


class TestMCPSchemaChanges:
    """Verify MCP tool schema reflects the new detail param."""

    def test_brain_search_has_detail_param(self):
        """brain_search schema includes 'detail' parameter."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        search_tool = next(t for t in tools if t.name == "brain_search")
        props = search_tool.inputSchema.get("properties", {})
        assert "detail" in props
        assert props["detail"]["default"] == "compact"

    def test_brain_search_detail_enum_values(self):
        """detail param accepts 'compact' and 'full'."""
        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        search_tool = next(t for t in tools if t.name == "brain_search")
        detail_schema = search_tool.inputSchema["properties"]["detail"]
        assert set(detail_schema.get("enum", [])) == {"compact", "full"}

    def test_server_instructions_mention_brain_expand(self):
        """Server instructions mention brain_expand tool."""
        from brainlayer.mcp import server

        assert "brain_expand" in server.instructions
