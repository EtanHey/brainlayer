"""Tests for KG rebuild pipeline — tag extraction + Groq NER batch.

Covers:
- Tag-based entity extraction from enrichment metadata
- Groq-backed NER caller
- Batch processing with multi-chunk grouping
- Enrichment hook fix (seed entities + LLM enabled)
"""

import json

import pytest

from brainlayer.pipeline.entity_extraction import (
    extract_entities_combined,
    extract_entities_from_tags,
)
from brainlayer.vector_store import VectorStore


class TestTagExtraction:
    """Extract entities from enrichment tags without API calls."""

    def test_technology_tags(self):
        tags = ["python", "typescript", "react"]
        entities = extract_entities_from_tags(tags)
        names = {e.text for e in entities}
        assert "python" in names
        assert "typescript" in names
        assert "react" in names
        assert all(e.entity_type == "technology" for e in entities)

    def test_project_tags(self):
        tags = ["brainlayer", "voicelayer"]
        entities = extract_entities_from_tags(tags)
        names = {e.text for e in entities}
        assert "brainlayer" in names
        assert "voicelayer" in names
        assert all(e.entity_type == "project" for e in entities)

    def test_mixed_tags(self):
        """Tags with both projects and technologies."""
        tags = ["brainlayer", "python", "planning", "architecture-design"]
        entities = extract_entities_from_tags(tags)
        names = {e.text for e in entities}
        assert "brainlayer" in names
        assert "python" in names
        # Non-entity tags should be skipped
        assert "planning" not in names

    def test_unknown_tags_skipped(self):
        tags = ["planning", "architecture-design", "workflow"]
        entities = extract_entities_from_tags(tags)
        assert len(entities) == 0

    def test_tag_confidence(self):
        tags = ["python"]
        entities = extract_entities_from_tags(tags)
        assert entities[0].confidence >= 0.7
        assert entities[0].confidence <= 0.9

    def test_empty_tags(self):
        entities = extract_entities_from_tags([])
        assert entities == []

    def test_project_takes_priority_over_tech(self):
        """If a tag matches both project and tech, project wins."""
        # 'domica' is a project, not a technology
        tags = ["domica"]
        entities = extract_entities_from_tags(tags)
        assert entities[0].entity_type == "project"

    def test_non_string_tags_skipped(self):
        """Non-string items in tags list should be silently skipped."""
        tags = ["python", 42, None, True, "react"]
        entities = extract_entities_from_tags(tags)
        names = {e.text for e in entities}
        assert "python" in names
        assert "react" in names
        assert len(entities) == 2

    def test_dot_normalization(self):
        """Tags with dots (node.js, next.js) should match normalized forms."""
        tags = ["node.js", "next.js", "vue.js"]
        entities = extract_entities_from_tags(tags)
        names = {e.text for e in entities}
        # nodejs, nextjs, vuejs are in KNOWN_TECH_TAGS
        assert "nodejs" in names or "node.js" in names  # depends on canonical form
        assert "nextjs" in names or "next.js" in names


# ── Groq NER Caller ──


class TestGroqNERCaller:
    """Test the Groq-backed NER extraction."""

    def test_groq_ner_returns_entities(self):
        """Mock Groq response should be parsed into entities."""
        from brainlayer.pipeline.entity_extraction import parse_llm_ner_response

        groq_response = json.dumps(
            {
                "entities": [
                    {"text": "FastAPI", "type": "tool"},
                    {"text": "Railway", "type": "tool"},
                ],
                "relations": [
                    {"source": "FastAPI", "target": "Railway", "type": "uses"},
                ],
            }
        )
        source_text = "Deploy FastAPI to Railway for production."
        entities, relations = parse_llm_ner_response(groq_response, source_text)
        assert len(entities) == 2
        assert len(relations) == 1

    def test_multi_chunk_ner_prompt(self):
        """Multi-chunk NER prompt should include all chunk contents."""
        from brainlayer.pipeline.kg_extraction_groq import build_multi_chunk_ner_prompt

        chunks = [
            {"id": "c1", "content": "Etan uses brainlayer."},
            {"id": "c2", "content": "Dor builds Domica."},
        ]
        prompt = build_multi_chunk_ner_prompt(chunks)
        assert "Etan uses brainlayer" in prompt
        assert "Dor builds Domica" in prompt
        assert "CHUNK c1" in prompt or "c1" in prompt

    def test_parse_multi_chunk_response(self):
        """Multi-chunk response should map entities to chunk IDs."""
        from brainlayer.pipeline.kg_extraction_groq import parse_multi_chunk_response

        response = json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "entities": [{"text": "Etan", "type": "person"}],
                        "relations": [],
                    },
                    {
                        "chunk_id": "c2",
                        "entities": [{"text": "Domica", "type": "company"}],
                        "relations": [{"source": "Dor", "target": "Domica", "type": "works_at"}],
                    },
                ]
            }
        )
        results = parse_multi_chunk_response(response)
        assert len(results) == 2
        assert results[0]["chunk_id"] == "c1"
        assert len(results[0]["entities"]) == 1
        assert results[1]["chunk_id"] == "c2"
        assert len(results[1]["relations"]) == 1


# ── Enrichment Hook Fix ──


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


class TestVectorStoreClose:
    """close() should properly clean up thread-local connections."""

    def test_close_clears_read_conn_reference(self, store):
        """After close(), _local.read_conn should be None, not a closed connection."""
        # Force creation of a thread-local read connection
        _ = store._read_cursor()
        assert getattr(store._local, "read_conn", None) is not None
        store.close()
        # Should be cleared to None, not left as closed connection
        assert getattr(store._local, "read_conn", None) is None


class TestEnrichmentHookFix:
    """The enrichment hook should pass seed entities and enable LLM extraction."""

    def test_seed_entities_not_empty(self):
        """DEFAULT_SEED_ENTITIES should have all expected types."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

        assert len(DEFAULT_SEED_ENTITIES) > 0
        assert "person" in DEFAULT_SEED_ENTITIES
        assert "project" in DEFAULT_SEED_ENTITIES
        assert "company" in DEFAULT_SEED_ENTITIES

    def test_extract_with_seeds_finds_entities(self):
        """With real seed entities, extraction finds known names."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

        text = "Etan Heyman is building brainlayer for memory management."
        result = extract_entities_combined(text, DEFAULT_SEED_ENTITIES, use_llm=False)
        names = {e.text.lower() for e in result.entities}
        assert "etan heyman" in names
        assert "brainlayer" in names

    def test_kg_from_chunk_with_seeds(self, store):
        """extract_kg_from_chunk with seeds should create entities in KG."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES
        from brainlayer.pipeline.kg_extraction import extract_kg_from_chunk

        # Insert a test chunk
        store.conn.cursor().execute(
            "INSERT INTO chunks (id, content, source, project, metadata, source_file) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "test-chunk-1",
                "Etan Heyman is working on brainlayer with Dor Zohar at Domica.",
                "test",
                "test",
                "{}",
                "test.jsonl",
            ),
        )

        stats = extract_kg_from_chunk(
            store=store,
            chunk_id="test-chunk-1",
            seed_entities=DEFAULT_SEED_ENTITIES,
            use_llm=False,
        )

        assert stats["entities_created"] >= 3  # Etan, brainlayer, Dor, Domica
        assert stats["chunks_linked"] >= 3

        # Verify entities are in KG
        cursor = store._read_cursor()
        entities = list(cursor.execute("SELECT name, entity_type FROM kg_entities"))
        entity_names = {e[0].lower() for e in entities}
        assert "etan heyman" in entity_names
        assert "brainlayer" in entity_names

    def test_explicit_mention_not_downgraded(self, store):
        """An 'explicit' mention_type must not be overwritten by 'inferred'."""
        # Create an entity
        entity_id = store.upsert_entity("test-ent-1", "technology", "TestEntity")
        # Insert a chunk
        store.conn.cursor().execute(
            "INSERT INTO chunks (id, content, source, project, metadata, source_file) VALUES (?, ?, ?, ?, ?, ?)",
            ("mention-chunk-1", "Test content", "test", "test", "{}", "test.jsonl"),
        )
        # Link with explicit mention
        store.link_entity_chunk(entity_id, "mention-chunk-1", mention_type="explicit")
        # Now try to overwrite with inferred (simulates tag extraction after seed)
        store.link_entity_chunk(entity_id, "mention-chunk-1", mention_type="inferred")
        # Verify explicit is preserved
        cursor = store._read_cursor()
        row = list(
            cursor.execute(
                "SELECT mention_type FROM kg_entity_chunks WHERE entity_id = ? AND chunk_id = ?",
                (entity_id, "mention-chunk-1"),
            )
        )
        assert row[0][0] == "explicit"

    def test_kg_with_llm_mock(self, store):
        """extract_kg_from_chunk with mock LLM should extract more entities."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES
        from brainlayer.pipeline.kg_extraction import extract_kg_from_chunk

        store.conn.cursor().execute(
            "INSERT INTO chunks (id, content, source, project, metadata, source_file) VALUES (?, ?, ?, ?, ?, ?)",
            ("test-chunk-2", "Etan Heyman deploys FastAPI to Railway.", "test", "test", "{}", "test.jsonl"),
        )

        def mock_llm(prompt):
            return json.dumps(
                {
                    "entities": [
                        {"text": "FastAPI", "type": "tool"},
                        {"text": "Railway", "type": "tool"},
                    ],
                    "relations": [
                        {"source": "Etan Heyman", "target": "FastAPI", "type": "uses"},
                    ],
                }
            )

        stats = extract_kg_from_chunk(
            store=store,
            chunk_id="test-chunk-2",
            seed_entities=DEFAULT_SEED_ENTITIES,
            use_llm=True,
            llm_caller=mock_llm,
        )

        # Should find seed entities + LLM entities
        assert stats["entities_created"] >= 2
        # Should have at least one relation
        assert stats["relations_created"] >= 1
