"""Tests for audit-driven search quality fixes.

Issue 1: hybrid_search has no KG signal in fused ranking
Issue 2: MCP entity_type enum doesn't match canonical ENTITY_TYPES
Issue 3: enrich_batch returns enriched=0 without processing
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
    ]

    async def test_mcp_recall_enum_matches_canonical(self):
        """brain_recall entity_type enum must match kg/__init__.py ENTITY_TYPES."""
        from brainlayer.kg import ENTITY_TYPES
        from brainlayer.mcp import list_tools

        tools = await list_tools()
        recall_tool = next(t for t in tools if t.name == "brain_recall")
        schema = recall_tool.inputSchema
        mcp_enum = schema["properties"]["entity_type"]["enum"]

        assert sorted(mcp_enum) == sorted(ENTITY_TYPES), (
            f"MCP brain_recall entity_type enum {mcp_enum} does not match canonical ENTITY_TYPES {ENTITY_TYPES}"
        )

    async def test_mcp_entity_enum_matches_canonical(self):
        """brain_entity entity_type enum must match the MCP schema contract."""
        from brainlayer.mcp import list_tools

        tools = await list_tools()
        entity_tool = next(t for t in tools if t.name == "brain_entity")
        schema = entity_tool.inputSchema
        mcp_enum = schema["properties"]["entity_type"]["enum"]

        assert sorted(mcp_enum) == sorted(self.BRAIN_ENTITY_TYPES), (
            f"MCP brain_entity entity_type enum {mcp_enum} does not match expected taxonomy {self.BRAIN_ENTITY_TYPES}"
        )


class TestEnrichBatchNotStub:
    """enrich_batch must actually process chunks, not just count and return 0."""

    def test_enrich_batch_processes_candidates(self, tmp_path, monkeypatch):
        """enrich_batch should attempt to enrich unenriched chunks, not return enriched=0."""
        from brainlayer import enrichment_controller
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)

        # Insert an unenriched chunk directly (char_count >= 50 required by get_enrichment_candidates)
        content = "This is test content that should be enriched with summary and tags for quality improvement"
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count)
               VALUES (?, ?, '{}', 'test.py', 'test', 'note', ?)""",
            ("unenriched-1", content, len(content)),
        )

        # Mock Gemini client to avoid real API calls
        mock_enrichment = {
            "summary": "Test content for enrichment",
            "tags": ["test"],
            "importance": 5,
            "intent": "testing",
        }

        def mock_get_gemini_client():
            class MockClient:
                class models:
                    @staticmethod
                    def generate_content(model, contents, config):
                        import json

                        class MockResponse:
                            text = json.dumps(mock_enrichment)

                        return MockResponse()

            return MockClient()

        monkeypatch.setattr(enrichment_controller, "_get_gemini_client", mock_get_gemini_client)
        monkeypatch.setattr(enrichment_controller, "AUTO_ENRICH_ENABLED", True)

        result = enrichment_controller.enrich_batch(store, limit=10)
        assert result.enriched > 0 or result.attempted > 0, (
            f"enrich_batch returned enriched={result.enriched}, attempted={result.attempted}. "
            "It should process unenriched chunks, not be a stub."
        )

        store.close()
