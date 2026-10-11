"""Tests for KG hybrid retrieval — vector + KG via Reciprocal Rank Fusion.

Tests cover:
1. kg_search: structured fact retrieval from kg_relations
2. kg_hybrid_search: combined vector + KG results via RRF
3. Entity auto-resolution in search
4. Fact filtering (current facts only, relation_type filter)
5. RRF scoring correctness
"""

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embedding():
    """Generate a deterministic 1024-dim embedding from text."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


@pytest.fixture
def populated_kg(store, mock_embedding):
    """Store with KG entities, relations, and linked chunks."""
    # Create chunks
    chunks = [
        {
            "id": "chunk-1",
            "content": "Etan discussed the brainlayer architecture with Delta at Cantaloupe",
            "metadata": {},
            "source_file": "test.jsonl",
            "project": "brainlayer",
            "content_type": "user_message",
            "value_type": "HIGH",
            "char_count": 67,
        },
        {
            "id": "chunk-2",
            "content": "The weekly standup covered API design and deployment plans",
            "metadata": {},
            "source_file": "test.jsonl",
            "project": "brainlayer",
            "content_type": "user_message",
            "value_type": "HIGH",
            "char_count": 57,
        },
        {
            "id": "chunk-3",
            "content": "Railway deployment for the coach package was completed",
            "metadata": {},
            "source_file": "test.jsonl",
            "project": "golems",
            "content_type": "ai_code",
            "value_type": "HIGH",
            "char_count": 54,
        },
    ]
    embs = [mock_embedding(c["content"]) for c in chunks]
    store.upsert_chunks(chunks, embs)

    # Create KG entities
    store.upsert_entity("person-etan", "person", "Etan Heyman", canonical_name="etan_heyman", description="Developer")
    store.upsert_entity("person-delta", "person", "Delta Cohen", canonical_name="yuval_cohen")
    store.upsert_entity("org-cantaloupe", "organization", "Cantaloupe", canonical_name="cantaloupe")
    store.upsert_entity("project-brainlayer", "project", "brainlayer", canonical_name="brainlayer")
    store.upsert_entity("meeting-standup", "meeting", "Weekly Standup", canonical_name="weekly_standup")

    # Create relations
    store.add_relation(
        "rel-1", "person-etan", "org-cantaloupe", "works_at", fact="Etan works at Cantaloupe", importance=0.9
    )
    store.add_relation(
        "rel-2",
        "person-etan",
        "project-brainlayer",
        "builds",
        fact="Etan builds brainlayer",
        importance=0.8,
        source_chunk_id="chunk-1",
    )
    store.add_relation(
        "rel-3",
        "person-etan",
        "meeting-standup",
        "attended",
        fact="Etan attended the weekly standup",
        source_chunk_id="chunk-2",
    )
    store.add_relation("rel-4", "person-delta", "org-cantaloupe", "works_at", fact="Delta works at Cantaloupe")

    # Link entities to chunks
    store.link_entity_chunk("person-etan", "chunk-1", relevance=0.95, mention_type="explicit")
    store.link_entity_chunk("person-delta", "chunk-1", relevance=0.9, mention_type="explicit")
    store.link_entity_chunk("org-cantaloupe", "chunk-1", relevance=0.85, mention_type="explicit")
    store.link_entity_chunk("meeting-standup", "chunk-2", relevance=0.9, mention_type="explicit")

    return store


# ── kg_search tests ────────────────────────────────────────


class TestKGSearch:
    """Test structured KG fact retrieval."""

    def test_search_by_entity_name(self, populated_kg):
        results = populated_kg.kg_search("Etan")
        assert len(results) > 0
        # Should find facts about Etan
        facts = [r["fact"] for r in results if r.get("fact")]
        assert any("Etan" in f for f in facts)

    def test_search_by_entity_name_returns_relations(self, populated_kg):
        results = populated_kg.kg_search("Etan")
        assert any(r["relation_type"] == "works_at" for r in results)
        assert any(r["relation_type"] == "builds" for r in results)

    def test_search_with_relation_type_filter(self, populated_kg):
        results = populated_kg.kg_search("Etan", relation_type="works_at")
        assert len(results) >= 1
        assert all(r["relation_type"] == "works_at" for r in results)

    def test_search_excludes_expired(self, populated_kg):
        populated_kg.soft_close_relation("rel-1")
        results = populated_kg.kg_search("Etan", relation_type="works_at")
        assert len(results) == 0  # works_at was soft-closed

    def test_search_returns_entity_info(self, populated_kg):
        results = populated_kg.kg_search("Cantaloupe")
        assert len(results) > 0
        # Should include target entity details
        for r in results:
            assert "source_entity" in r
            assert "target_entity" in r

    def test_search_not_found(self, populated_kg):
        results = populated_kg.kg_search("nonexistent_xyzzy_12345")
        assert len(results) == 0


# ── kg_hybrid_search tests ────────────────────────────────


class TestKGHybridSearch:
    """Test combined vector + KG retrieval with RRF fusion."""

    def test_hybrid_returns_both_chunks_and_facts(self, populated_kg, mock_embedding):
        query_emb = mock_embedding("Etan brainlayer architecture")
        results = populated_kg.kg_hybrid_search(
            query_embedding=query_emb,
            query_text="Etan brainlayer",
            n_results=10,
        )
        # Should have chunk results
        assert len(results["chunks"]) > 0
        # Should have KG fact results
        assert len(results["facts"]) > 0

    def test_hybrid_facts_have_scores(self, populated_kg, mock_embedding):
        query_emb = mock_embedding("Etan works at Cantaloupe")
        results = populated_kg.kg_hybrid_search(
            query_embedding=query_emb,
            query_text="Etan Cantaloupe",
            n_results=10,
        )
        for fact in results["facts"]:
            assert "rrf_score" in fact
            assert fact["rrf_score"] > 0

    def test_hybrid_chunks_use_standard_format(self, populated_kg, mock_embedding):
        query_emb = mock_embedding("brainlayer")
        results = populated_kg.kg_hybrid_search(
            query_embedding=query_emb,
            query_text="brainlayer",
            n_results=5,
        )
        assert "documents" in results["chunks"]
        assert "metadatas" in results["chunks"]

    def test_hybrid_with_entity_filter(self, populated_kg, mock_embedding):
        query_emb = mock_embedding("brainlayer architecture")
        results = populated_kg.kg_hybrid_search(
            query_embedding=query_emb,
            query_text="architecture",
            n_results=10,
            entity_name="Etan",
        )
        # Facts should be about Etan
        for fact in results["facts"]:
            assert fact["source_entity"]["name"] == "Etan Heyman" or fact["target_entity"]["name"] == "Etan Heyman"

    def test_hybrid_empty_kg_still_returns_chunks(self, store, mock_embedding):
        """Even with no KG data, vector search should still work."""
        chunks = [
            {
                "id": "chunk-1",
                "content": "Some test content about architecture",
                "metadata": {},
                "source_file": "test.jsonl",
                "project": "test",
                "content_type": "user_message",
                "value_type": "HIGH",
                "char_count": 36,
            },
        ]
        store.upsert_chunks(chunks, [mock_embedding("architecture")])

        results = store.kg_hybrid_search(
            query_embedding=mock_embedding("architecture"),
            query_text="architecture",
            n_results=5,
        )
        assert len(results["chunks"]["documents"][0]) > 0
        assert len(results["facts"]) == 0


# ── RRF scoring tests ────────────────────────────────────


class TestRRFScoring:
    """Test Reciprocal Rank Fusion scoring correctness."""

    def test_rrf_score_decreases_with_rank(self, populated_kg, mock_embedding):
        query_emb = mock_embedding("Etan")
        results = populated_kg.kg_hybrid_search(
            query_embedding=query_emb,
            query_text="Etan",
            n_results=10,
        )
        if len(results["facts"]) >= 2:
            scores = [f["rrf_score"] for f in results["facts"]]
            assert scores == sorted(scores, reverse=True)
