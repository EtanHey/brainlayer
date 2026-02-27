"""Tests for KG extraction pipeline — wires entity extraction to KG tables.

Tests cover:
1. extract_kg_from_chunk: entity extraction + resolution + linking
2. Seed entity matching into KG
3. Relation creation from extraction results
4. Entity dedup via resolution
5. mention_type set correctly
6. Integration with VectorStore KG methods
"""

import pytest

from brainlayer.pipeline.entity_extraction import ExtractedEntity, ExtractedRelation, ExtractionResult
from brainlayer.pipeline.kg_extraction import extract_kg_from_chunk, process_extraction_result
from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with a few test chunks."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def store_with_chunks(store):
    """Store with sample chunks for KG extraction testing."""
    chunks = [
        {
            "id": "chunk-1",
            "content": "Etan discussed the brainlayer architecture with Claude",
            "metadata": "{}",
            "source_file": "test.jsonl",
            "project": "brainlayer",
            "content_type": "user_message",
            "value_type": "HIGH",
            "char_count": 55,
        },
        {
            "id": "chunk-2",
            "content": "The meeting with Cantaloupe was about the new API design",
            "metadata": "{}",
            "source_file": "test.jsonl",
            "project": "brainlayer",
            "content_type": "user_message",
            "value_type": "HIGH",
            "char_count": 56,
        },
    ]
    embs = [[0.1] * 1024 for _ in chunks]
    store.upsert_chunks(chunks, embs)
    return store


# ── process_extraction_result tests ────────────────────────────


class TestProcessExtractionResult:
    """Test converting extraction results into KG entities + relations."""

    def test_entities_created(self, store_with_chunks):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan",
                    entity_type="person",
                    start=0,
                    end=4,
                    confidence=0.95,
                    source="seed",
                ),
                ExtractedEntity(
                    text="brainlayer",
                    entity_type="project",
                    start=19,
                    end=29,
                    confidence=0.9,
                    source="seed",
                ),
            ],
            relations=[],
            chunk_id="chunk-1",
        )

        stats = process_extraction_result(store_with_chunks, result)
        assert stats["entities_created"] >= 2
        assert stats["chunks_linked"] >= 2

        # Verify entities exist in KG
        entity = store_with_chunks.resolve_entity("Etan")
        assert entity is not None
        assert entity["entity_type"] == "person"

    def test_relations_created(self, store_with_chunks):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan",
                    entity_type="person",
                    start=0,
                    end=4,
                    confidence=0.95,
                    source="seed",
                ),
                ExtractedEntity(
                    text="Cantaloupe",
                    entity_type="organization",
                    start=17,
                    end=27,
                    confidence=0.8,
                    source="llm",
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Etan",
                    target_text="Cantaloupe",
                    relation_type="works_at",
                    confidence=0.7,
                ),
            ],
            chunk_id="chunk-2",
        )

        stats = process_extraction_result(store_with_chunks, result)
        assert stats["relations_created"] >= 1

        # Verify relation exists
        entity = store_with_chunks.resolve_entity("Etan")
        rels = store_with_chunks.get_entity_relations(entity["id"], direction="outgoing")
        assert any(r["relation_type"] == "works_at" for r in rels)

    def test_chunk_linking_with_mention_type(self, store_with_chunks):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan",
                    entity_type="person",
                    start=0,
                    end=4,
                    confidence=0.95,
                    source="seed",
                ),
            ],
            relations=[],
            chunk_id="chunk-1",
        )

        process_extraction_result(store_with_chunks, result)

        entity = store_with_chunks.resolve_entity("Etan")
        chunks = store_with_chunks.get_entity_chunks(entity["id"])
        assert len(chunks) >= 1
        assert chunks[0]["mention_type"] == "explicit"

    def test_entity_dedup_via_resolution(self, store_with_chunks):
        """Second extraction of same entity shouldn't create duplicate."""
        result1 = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan",
                    entity_type="person",
                    start=0,
                    end=4,
                    confidence=0.95,
                    source="seed",
                ),
            ],
            relations=[],
            chunk_id="chunk-1",
        )
        result2 = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan",
                    entity_type="person",
                    start=0,
                    end=4,
                    confidence=0.95,
                    source="seed",
                ),
            ],
            relations=[],
            chunk_id="chunk-2",
        )

        process_extraction_result(store_with_chunks, result1)
        process_extraction_result(store_with_chunks, result2)

        stats = store_with_chunks.kg_stats()
        person_count = stats["entity_types"].get("person", 0)
        assert person_count == 1  # Only one "Etan" entity

    def test_empty_extraction_no_crash(self, store_with_chunks):
        result = ExtractionResult(entities=[], relations=[], chunk_id="chunk-1")
        stats = process_extraction_result(store_with_chunks, result)
        assert stats["entities_created"] == 0
        assert stats["relations_created"] == 0

    def test_source_chunk_id_on_relations(self, store_with_chunks):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="Etan", entity_type="person", start=0, end=4, confidence=0.95, source="seed"),
                ExtractedEntity(
                    text="brainlayer", entity_type="project", start=19, end=29, confidence=0.9, source="seed"
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Etan",
                    target_text="brainlayer",
                    relation_type="builds",
                    confidence=0.8,
                ),
            ],
            chunk_id="chunk-1",
        )

        process_extraction_result(store_with_chunks, result)

        entity = store_with_chunks.resolve_entity("Etan")
        rels = store_with_chunks.get_entity_relations(entity["id"], direction="outgoing")
        builds_rel = [r for r in rels if r["relation_type"] == "builds"]
        assert len(builds_rel) == 1
        assert builds_rel[0]["source_chunk_id"] == "chunk-1"


# ── extract_kg_from_chunk tests ────────────────────────────


class TestExtractKGFromChunk:
    """Test the full extraction flow for a single chunk."""

    def test_seed_extraction(self, store_with_chunks):
        seed_entities = {
            "person": ["Etan"],
            "project": ["brainlayer"],
        }

        stats = extract_kg_from_chunk(
            store=store_with_chunks,
            chunk_id="chunk-1",
            seed_entities=seed_entities,
            use_llm=False,
        )
        assert stats["entities_created"] >= 1

    def test_missing_chunk_returns_empty(self, store_with_chunks):
        stats = extract_kg_from_chunk(
            store=store_with_chunks,
            chunk_id="nonexistent",
            seed_entities={},
            use_llm=False,
        )
        assert stats["entities_created"] == 0

    def test_llm_extraction_with_mock(self, store_with_chunks):
        """Test LLM extraction with a mock caller that returns structured JSON."""
        mock_response = '{"entities": [{"text": "Etan", "type": "person"}], "relations": []}'

        def mock_llm(prompt):
            return mock_response

        stats = extract_kg_from_chunk(
            store=store_with_chunks,
            chunk_id="chunk-1",
            seed_entities={},
            use_llm=True,
            llm_caller=mock_llm,
        )
        assert stats["entities_created"] >= 1


# ── Confidence and importance propagation ────────────────────


class TestConfidencePropagation:
    """Test that extraction confidence flows to KG entities."""

    def test_seed_entity_gets_high_confidence(self, store_with_chunks):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan",
                    entity_type="person",
                    start=0,
                    end=4,
                    confidence=0.95,
                    source="seed",
                ),
            ],
            relations=[],
            chunk_id="chunk-1",
        )

        process_extraction_result(store_with_chunks, result)
        entity = store_with_chunks.resolve_entity("Etan")
        # Seed entities should have high confidence
        assert entity["confidence"] >= 0.9

    def test_llm_entity_gets_lower_confidence(self, store_with_chunks):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="SomeThing",
                    entity_type="project",
                    start=0,
                    end=9,
                    confidence=0.6,
                    source="llm",
                ),
            ],
            relations=[],
            chunk_id="chunk-1",
        )

        process_extraction_result(store_with_chunks, result)
        entity = store_with_chunks.resolve_entity("SomeThing")
        # LLM entities inherit their extraction confidence
        assert entity["confidence"] <= 0.8
