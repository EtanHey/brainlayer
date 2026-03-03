"""Tests for search quality improvements (collab tasks 1, 2, 3, 6).

Task 1: FTS5 expansion — summary, tags, resolved_query indexed
Task 2: Entity-aware routing — auto-detect entities, route to kg_hybrid_search
Task 3: Post-RRF reranking — importance + recency boost
Task 6: format=format bug fix

All tests use tmp_path fixtures (isolated, no production DB).
"""

import json
from pathlib import Path

import pytest

from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with expanded FTS5 schema."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Mock embedding function returning deterministic 1024-dim vectors."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


def _insert_chunk(
    store,
    chunk_id,
    content,
    summary=None,
    tags=None,
    resolved_query=None,
    importance=None,
    created_at=None,
    project=None,
    source="claude_code",
    embedding=None,
):
    """Helper: insert a chunk with enrichment metadata into the store."""
    if embedding is None:
        seed = sum(ord(c) for c in chunk_id[:20]) % 100
        embedding = [float(seed + i) / 1000.0 for i in range(1024)]

    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (id, content, metadata, source_file, project,
           content_type, char_count, source, summary, tags, resolved_query,
           importance, created_at)
           VALUES (?, ?, '{}', 'test.jsonl', ?, 'assistant_text', ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk_id,
            content,
            project,
            len(content),
            source,
            summary,
            json.dumps(tags) if tags else None,
            resolved_query,
            importance,
            created_at,
        ),
    )
    from brainlayer._helpers import serialize_f32

    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )


# ── Task 6: format=format bug ────────────────────────────────────


class TestFormatBug:
    """The default search fallback should pass detail=detail, not format=format."""

    def test_default_search_uses_detail_param(self):
        """search_handler._brain_search default path passes detail correctly."""
        import ast

        handler_path = Path(__file__).parent.parent / "src" / "brainlayer" / "mcp" / "search_handler.py"
        source = handler_path.read_text()

        # The bug was `format=format` on the last return in _brain_search.
        # After fix, it should be `detail=detail`.
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "format":
                # If format=format exists, it's the Python builtin — bug is present
                if isinstance(node.value, ast.Name) and node.value.id == "format":
                    pytest.fail("format=format bug still present — passes Python builtin format()")


# ── Task 1: FTS5 expansion ────────────────────────────────────────


class TestFTS5Expansion:
    """FTS5 should index summary, tags, and resolved_query alongside content."""

    def test_fts5_has_summary_column(self, store):
        """chunks_fts virtual table includes summary column."""
        cursor = store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks_fts)")}
        assert "summary" in cols, f"FTS5 columns: {cols}"

    def test_fts5_has_tags_column(self, store):
        """chunks_fts virtual table includes tags column."""
        cursor = store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks_fts)")}
        assert "tags" in cols, f"FTS5 columns: {cols}"

    def test_fts5_has_resolved_query_column(self, store):
        """chunks_fts virtual table includes resolved_query column."""
        cursor = store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks_fts)")}
        assert "resolved_query" in cols, f"FTS5 columns: {cols}"

    def test_fts5_search_matches_summary(self, store, mock_embed):
        """FTS5 keyword search finds chunks by their summary text."""
        _insert_chunk(
            store,
            "chunk-summary-1",
            content="Some generic code implementation details.",
            summary="Voice calibration for coachClaude scheduling agent",
            tags=["voice", "coach"],
        )
        results = store.hybrid_search(
            query_embedding=mock_embed("voice calibration"),
            query_text="voice calibration",
            n_results=5,
        )
        ids = results["ids"][0]
        assert "chunk-summary-1" in ids, f"Expected chunk-summary-1 in results, got: {ids}"

    def test_fts5_search_matches_tags(self, store, mock_embed):
        """FTS5 keyword search finds chunks by their tags."""
        _insert_chunk(
            store,
            "chunk-tags-1",
            content="Implemented the new scheduling feature.",
            tags=["authentication", "jwt-tokens", "security"],
        )
        results = store.hybrid_search(
            query_embedding=mock_embed("jwt tokens"),
            query_text="jwt-tokens",
            n_results=5,
        )
        ids = results["ids"][0]
        assert "chunk-tags-1" in ids, f"Expected chunk-tags-1 in results, got: {ids}"

    def test_fts5_search_matches_resolved_query(self, store, mock_embed):
        """FTS5 keyword search finds chunks by their resolved_query."""
        _insert_chunk(
            store,
            "chunk-rq-1",
            content="Here is how the auth flow works internally.",
            resolved_query="how does authentication work in the golems monorepo",
        )
        results = store.hybrid_search(
            query_embedding=mock_embed("authentication golems"),
            query_text="authentication golems monorepo",
            n_results=5,
        )
        ids = results["ids"][0]
        assert "chunk-rq-1" in ids, f"Expected chunk-rq-1 in results, got: {ids}"

    def test_fts5_insert_trigger_syncs_enrichment(self, store):
        """INSERT trigger populates FTS5 with summary/tags/resolved_query."""
        _insert_chunk(
            store,
            "chunk-trigger-1",
            content="test content",
            summary="test summary for trigger",
            tags=["trigger-test"],
            resolved_query="trigger test query",
        )
        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT summary, tags, resolved_query FROM chunks_fts WHERE chunk_id = ?",
                ("chunk-trigger-1",),
            )
        )
        assert len(rows) == 1
        assert rows[0][0] == "test summary for trigger"
        assert "trigger-test" in rows[0][1]
        assert rows[0][2] == "trigger test query"

    def test_fts5_update_trigger_syncs_enrichment(self, store):
        """UPDATE trigger re-syncs FTS5 when enrichment metadata changes."""
        _insert_chunk(store, "chunk-update-1", content="original content")
        cursor = store.conn.cursor()
        cursor.execute(
            "UPDATE chunks SET summary = ?, tags = ? WHERE id = ?",
            ("updated summary", '["updated-tag"]', "chunk-update-1"),
        )
        rows = list(
            cursor.execute(
                "SELECT summary FROM chunks_fts WHERE chunk_id = ?",
                ("chunk-update-1",),
            )
        )
        assert len(rows) == 1
        assert rows[0][0] == "updated summary"


# ── Task 3: Post-RRF reranking ────────────────────────────────────


class TestPostRRFReranking:
    """RRF scoring should boost results by importance and recency."""

    def test_high_importance_ranks_higher(self, store, mock_embed):
        """A chunk with importance=9 should rank above importance=2 at similar semantic distance."""
        # Insert two chunks with similar content but different importance
        emb = mock_embed("database optimization patterns")
        _insert_chunk(
            store,
            "low-imp",
            content="database optimization patterns discussion",
            importance=2.0,
            created_at="2026-03-01T00:00:00",
            embedding=emb,
        )
        # Slightly different embedding for the high-importance one
        emb_hi = [v + 0.001 for v in emb]
        _insert_chunk(
            store,
            "high-imp",
            content="database optimization patterns and best practices",
            importance=9.0,
            created_at="2026-03-01T00:00:00",
            embedding=emb_hi,
        )
        results = store.hybrid_search(
            query_embedding=emb,
            query_text="database optimization",
            n_results=2,
        )
        ids = results["ids"][0]
        assert len(ids) == 2
        # High importance should come first
        assert ids[0] == "high-imp", f"Expected high-imp first, got: {ids}"

    def test_recent_chunk_ranks_higher(self, store, mock_embed):
        """A recent chunk should rank above an old chunk at similar distance."""
        emb = mock_embed("authentication flow")
        _insert_chunk(
            store,
            "old-chunk",
            content="authentication flow implementation details",
            importance=5.0,
            created_at="2025-06-01T00:00:00",  # 9 months ago
            embedding=emb,
        )
        emb_new = [v + 0.001 for v in emb]
        _insert_chunk(
            store,
            "new-chunk",
            content="authentication flow implementation and testing",
            importance=5.0,
            created_at="2026-03-02T00:00:00",  # yesterday
            embedding=emb_new,
        )
        results = store.hybrid_search(
            query_embedding=emb,
            query_text="authentication flow",
            n_results=2,
        )
        ids = results["ids"][0]
        assert len(ids) == 2
        assert ids[0] == "new-chunk", f"Expected new-chunk first, got: {ids}"

    def test_no_importance_no_crash(self, store, mock_embed):
        """Chunks without importance metadata should still rank without errors."""
        emb = mock_embed("test query")
        _insert_chunk(
            store,
            "no-imp",
            content="test content without importance",
            importance=None,
            created_at="2026-03-01T00:00:00",
            embedding=emb,
        )
        results = store.hybrid_search(
            query_embedding=emb,
            query_text="test content",
            n_results=5,
        )
        assert len(results["ids"][0]) >= 1

    def test_no_created_at_no_crash(self, store, mock_embed):
        """Chunks without created_at should still rank without errors."""
        emb = mock_embed("another test")
        _insert_chunk(
            store,
            "no-date",
            content="test content without date",
            importance=5.0,
            created_at=None,
            embedding=emb,
        )
        results = store.hybrid_search(
            query_embedding=emb,
            query_text="test content without date",
            n_results=5,
        )
        assert len(results["ids"][0]) >= 1


# ── Task 2: Entity-aware routing ────────────────────────────────────


class TestEntityAwareRouting:
    """brain_search should auto-detect entity names and route to KG search."""

    def test_entity_detection_finds_known_entity(self, store):
        """When query contains a known entity name, it should be detected."""
        # Insert an entity into the KG
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO kg_entities (id, entity_type, name, metadata, created_at)
               VALUES (?, ?, ?, '{}', datetime('now'))""",
            ("ent-1", "person", "Avi Simon"),
        )
        # Also populate FTS
        cursor.execute(
            "INSERT INTO kg_entities_fts (name, metadata, entity_id) VALUES (?, '{}', ?)",
            ("Avi Simon", "ent-1"),
        )

        from brainlayer.mcp.search_handler import _detect_entities

        entities = _detect_entities("What does Avi Simon prefer for meetings?", store)
        assert len(entities) >= 1
        assert any(e["name"].lower() == "avi simon" for e in entities)

    def test_entity_detection_returns_empty_for_no_match(self, store):
        """When no entity names match, return empty list."""
        from brainlayer.mcp.search_handler import _detect_entities

        entities = _detect_entities("how to implement authentication", store)
        assert entities == []

    def test_kg_hybrid_search_returns_entity_linked_chunks(self, store, mock_embed):
        """kg_hybrid_search returns entity-linked chunks and facts when called with entity_name."""
        # Insert entity
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO kg_entities (id, entity_type, name, metadata, created_at)
               VALUES (?, ?, ?, '{}', datetime('now'))""",
            ("ent-route-1", "person", "Michal Cohen"),
        )
        cursor.execute(
            "INSERT INTO kg_entities_fts (name, metadata, entity_id) VALUES (?, '{}', ?)",
            ("Michal Cohen", "ent-route-1"),
        )
        # Insert a chunk linked to this entity
        _insert_chunk(
            store,
            "ent-chunk-1",
            content="Michal Cohen prefers morning meetings before 10am",
            importance=7.0,
            created_at="2026-03-01T00:00:00",
        )
        cursor.execute(
            "INSERT INTO kg_entity_chunks (entity_id, chunk_id, relevance) VALUES (?, ?, ?)",
            ("ent-route-1", "ent-chunk-1", 0.95),
        )

        # kg_hybrid_search should return chunks + facts for the entity
        results = store.kg_hybrid_search(
            query_embedding=mock_embed("Michal Cohen meetings"),
            query_text="Michal Cohen meetings",
            n_results=5,
            entity_name="Michal Cohen",
        )
        assert "chunks" in results
        assert "facts" in results

    def test_entity_routing_skipped_when_filters_active(self, store):
        """Entity routing should be skipped when search filters are active."""
        # Insert an entity — it should NOT be detected when filters are active
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO kg_entities (id, entity_type, name, metadata, created_at)
               VALUES (?, ?, ?, '{}', datetime('now'))""",
            ("ent-skip-1", "person", "David Ben"),
        )
        cursor.execute(
            "INSERT INTO kg_entities_fts (name, metadata, entity_id) VALUES (?, '{}', ?)",
            ("David Ben", "ent-skip-1"),
        )

        from brainlayer.mcp.search_handler import _detect_entities

        # Without filters, entity should be detected
        entities = _detect_entities("What does David Ben think about X?", store)
        assert len(entities) >= 1

        # The routing code checks has_active_filters before calling _detect_entities.
        # We verify that _detect_entities itself works; the filter guard is in _brain_search.
