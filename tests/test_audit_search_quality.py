"""Tests for audit-driven search quality fixes.

Issue 1: hybrid_search has no KG signal in fused ranking
Issue 2: MCP entity_type enum doesn't match canonical ENTITY_TYPES
Batch enrichment is retired; its no-store/no-factory contract is covered by
test_retired_controller_batch.py (all phases, empty and nonempty candidates).
"""


class TestKGSignalInHybridSearch:
    """hybrid_search should include a KG leg in RRF fusion."""

    def test_hybrid_search_accepts_kg_boost_param(self):
        """hybrid_search should accept a kg_boost parameter to enable KG-linked chunk boosting."""
        import inspect

        from brainlayer.search_repo import SearchMixin

        sig = inspect.signature(SearchMixin.hybrid_search)
        assert "kg_boost" in sig.parameters, "hybrid_search must accept kg_boost param for KG-linked chunk boosting"

    def test_kg_linked_chunks_get_score_boost(self, tmp_path):
        """Chunks linked to entities via kg_entity_chunks should get a score boost in hybrid_search."""
        import random

        from brainlayer._helpers import serialize_f32
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)

        cursor = store.conn.cursor()

        # Insert two chunks directly
        for cid, content in [
            ("chunk-linked", "BrainLayer architecture uses sqlite-vec for vector storage"),
            ("chunk-unlinked", "BrainLayer architecture uses sqlite-vec for vector storage copy"),
        ]:
            cursor.execute(
                """INSERT INTO chunks (id, content, metadata, source_file, project, content_type)
                   VALUES (?, ?, '{}', 'test.py', 'test', 'note')""",
                (cid, content),
            )

        # Create entity and link one chunk
        cursor.execute(
            "INSERT INTO kg_entities (id, name, entity_type, canonical_name) VALUES (?, ?, ?, ?)",
            ("ent-1", "BrainLayer", "project", "brainlayer"),
        )
        cursor.execute(
            "INSERT INTO kg_entity_chunks (entity_id, chunk_id, mention_type) VALUES (?, ?, ?)",
            ("ent-1", "chunk-linked", "direct"),
        )

        # Create embeddings for both chunks
        random.seed(42)
        embedding = [random.random() for _ in range(1024)]
        emb_bytes = serialize_f32(embedding)
        cursor.execute("INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)", ("chunk-linked", emb_bytes))
        cursor.execute("INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)", ("chunk-unlinked", emb_bytes))

        # Search with KG boost enabled
        results = store.hybrid_search(
            query_embedding=embedding,
            query_text="BrainLayer architecture",
            n_results=10,
            kg_boost=True,
        )

        ids = results["ids"][0]
        assert "chunk-linked" in ids, "KG-linked chunk should appear in results"

        # The linked chunk should rank higher due to KG boost
        if "chunk-linked" in ids and "chunk-unlinked" in ids:
            linked_idx = ids.index("chunk-linked")
            unlinked_idx = ids.index("chunk-unlinked")
            assert linked_idx <= unlinked_idx, "KG-linked chunk should rank at least as high as unlinked chunk"

        store.close()


class TestEntityTypeEnumAlignment:
    """MCP entity_type enum must match canonical ENTITY_TYPES from kg/__init__.py."""

    BRAIN_ENTITY_TYPES = [
        "person",
        "agent",
        "golem",
        "tool",
        "platform",
        "project",
        "technology",
        "library",
        "organization",
        "company",
        "topic",
        "concept",
        "workflow",
        "skill",
        "decision",
        "protocol",
        "health_metric",
        "community",
        "device",
        "event",
        "location",
        "source",
    ]

    async def test_mcp_recall_enum_matches_canonical(self):
        """brain_recall entity_type enum must match kg/__init__.py ENTITY_TYPES."""
        from brainlayer.kg import ENTITY_TYPES
        from brainlayer.mcp import list_tools

        tools = await list_tools()
        recall_tool = next(t for t in tools if t.name == "brain_recall")
        schema = recall_tool.input_schema
        mcp_enum = schema["properties"]["entity_type"]["enum"]

        assert sorted(mcp_enum) == sorted(ENTITY_TYPES), (
            f"MCP brain_recall entity_type enum {mcp_enum} does not match canonical ENTITY_TYPES {ENTITY_TYPES}"
        )

    async def test_mcp_entity_enum_matches_canonical(self):
        """brain_entity entity_type enum must match the MCP schema contract."""
        from brainlayer.mcp import _full_tool_definitions

        tools = _full_tool_definitions()
        entity_tool = next(t for t in tools if t.name == "brain_entity")
        schema = entity_tool.input_schema
        mcp_enum = schema["properties"]["entity_type"]["enum"]

        assert sorted(mcp_enum) == sorted(self.BRAIN_ENTITY_TYPES), (
            f"MCP brain_entity entity_type enum {mcp_enum} does not match expected taxonomy {self.BRAIN_ENTITY_TYPES}"
        )
