"""Tests for batch entity extraction pipeline.

Covers:
- Processing multiple chunks through extraction
- Storing extracted entities to KG via entity_resolution
- Storing extracted relations to KG
- Linking entities to source chunks (provenance)
- Seed entity loading
"""

import json

import pytest

from brainlayer.pipeline.batch_extraction import (
    DEFAULT_SEED_ENTITIES,
    process_batch,
    process_chunk,
    store_extraction_result,
)
from brainlayer.pipeline.entity_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with KG tables."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _mock_llm(prompt):
    """Mock LLM that returns entities based on prompt content."""
    text = prompt.split("Text:\n")[-1] if "Text:\n" in prompt else ""
    entities = []
    relations = []

    if "Railway" in text:
        entities.append({"text": "Railway", "type": "tool"})
    if "FastAPI" in text:
        entities.append({"text": "FastAPI", "type": "tool"})
    if "Supabase" in text:
        entities.append({"text": "Supabase", "type": "tool"})
    if "Railway" in text and "FastAPI" in text:
        relations.append({"source": "FastAPI", "target": "Railway", "type": "uses"})

    return json.dumps({"entities": entities, "relations": relations})


# ── Seed Entities ──


class TestSeedEntities:
    """Default seed entities should cover Etan's ecosystem."""

    def test_default_person_seeds_are_empty(self):
        assert DEFAULT_SEED_ENTITIES["person"] == []

    def test_optional_person_seed_config(self, tmp_path, monkeypatch):
        from brainlayer.pipeline.batch_extraction import _load_person_seed_entities

        monkeypatch.delenv("BRAINLAYER_PERSON_SEED_ENTITIES_PATH", raising=False)
        assert _load_person_seed_entities() == []

        config = tmp_path / "person-seeds.json"
        config.write_text('["Alex Example"]')
        monkeypatch.setenv("BRAINLAYER_PERSON_SEED_ENTITIES_PATH", str(config))
        assert _load_person_seed_entities() == ["Alex Example"]

    def test_malformed_person_seed_config_fails_soft(self, tmp_path, monkeypatch, caplog):
        from brainlayer.pipeline.batch_extraction import _load_person_seed_entities

        config = tmp_path / "person-seeds.json"
        config.write_text('{"private-marker": 42}')
        monkeypatch.setenv("BRAINLAYER_PERSON_SEED_ENTITIES_PATH", str(config))
        assert _load_person_seed_entities() == []
        assert "ValueError" in caplog.text
        assert "private-marker" not in caplog.text
        caplog.clear()
        monkeypatch.setenv("BRAINLAYER_PERSON_SEED_ENTITIES_PATH", str(tmp_path / "missing.json"))
        assert _load_person_seed_entities() == []
        assert "FileNotFoundError" in caplog.text
        assert str(tmp_path) not in caplog.text

    def test_seed_has_companies(self):
        assert "company" in DEFAULT_SEED_ENTITIES
        names = DEFAULT_SEED_ENTITIES["company"]
        assert "Domica" in names
        assert "Cantaloupe AI" in names

    def test_seed_has_projects(self):
        assert "project" in DEFAULT_SEED_ENTITIES
        names = DEFAULT_SEED_ENTITIES["project"]
        assert "brainlayer" in names
        assert "golems" in names

    def test_seed_has_agents(self):
        assert "agent" in DEFAULT_SEED_ENTITIES
        names = DEFAULT_SEED_ENTITIES["agent"]
        assert "brainClaude" in names


# ── Single Chunk Processing ──


class TestProcessChunk:
    """Process a single chunk through extraction."""

    def test_process_returns_result(self):
        chunk = {"id": "chunk-1", "content": "Person Alpha is building Example Corp."}
        result = process_chunk(chunk, llm_caller=_mock_llm)
        assert isinstance(result, ExtractionResult)
        assert result.chunk_id == "chunk-1"

    def test_process_finds_seed_entities(self):
        chunk = {"id": "chunk-1", "content": "Person Alpha leads product at Example Corp."}
        result = process_chunk(
            chunk,
            seed_entities={
                "person": ["Person Alpha"],
                "company": ["Example Corp"],
                "project": ["brainlayer", "Sample Project"],
            },
            llm_caller=_mock_llm,
        )
        names = {e.text for e in result.entities}
        assert "Person Alpha" in names
        assert "Example Corp" in names

    def test_process_finds_llm_entities(self):
        chunk = {"id": "chunk-2", "content": "Deploy to Railway using FastAPI."}
        result = process_chunk(chunk, llm_caller=_mock_llm)
        names = {e.text for e in result.entities}
        assert "Railway" in names
        assert "FastAPI" in names

    def test_process_finds_relations(self):
        chunk = {"id": "chunk-3", "content": "FastAPI deployed to Railway."}
        result = process_chunk(chunk, llm_caller=_mock_llm)
        assert len(result.relations) >= 1
        rel = result.relations[0]
        assert rel.source_text == "FastAPI"
        assert rel.target_text == "Railway"


# ── Store Extraction Result ──


class TestStoreExtractionResult:
    """Store extraction results into the KG."""

    def test_stores_entities(self, store):
        result = ExtractionResult(
            entities=[
                ExtractedEntity("Person Alpha", "person", 0, 11, 0.95, "seed"),
                ExtractedEntity("Example Corp", "company", 25, 31, 0.95, "seed"),
            ],
            relations=[],
            chunk_id="chunk-1",
        )
        entity_ids = store_extraction_result(result, store)
        assert len(entity_ids) == 2

        # Entities should exist in KG
        for eid in entity_ids.values():
            entity = store.get_entity(eid)
            assert entity is not None

    def test_stores_relations(self, store):
        result = ExtractionResult(
            entities=[
                ExtractedEntity("Person Alpha", "person", 0, 9, 0.95, "seed"),
                ExtractedEntity("Example Corp", "company", 20, 26, 0.95, "seed"),
            ],
            relations=[
                ExtractedRelation("Person Alpha", "Example Corp", "works_at", 0.8),
            ],
            chunk_id="chunk-1",
        )
        entity_ids = store_extraction_result(result, store)

        # Relation should exist
        dor_id = entity_ids["Person Alpha"]
        rels = store.get_entity_relations(dor_id, direction="outgoing")
        assert len(rels) >= 1
        assert rels[0]["relation_type"] == "works_at"

    def test_links_entities_to_chunks(self, store):
        result = ExtractionResult(
            entities=[
                ExtractedEntity("brainlayer", "project", 0, 10, 0.95, "seed"),
            ],
            relations=[],
            chunk_id="chunk-42",
        )
        store_extraction_result(result, store)

        # Entity should be linked to chunk
        cursor = store.conn.cursor()
        links = list(cursor.execute("SELECT chunk_id FROM kg_entity_chunks WHERE entity_id LIKE 'project%'"))
        assert any(l[0] == "chunk-42" for l in links)

    def test_dedup_across_chunks(self, store):
        """Same entity from different chunks should resolve to one KG entity."""
        result1 = ExtractionResult(
            entities=[
                ExtractedEntity("Person Alpha", "person", 0, 11, 0.95, "seed"),
            ],
            relations=[],
            chunk_id="chunk-1",
        )
        result2 = ExtractionResult(
            entities=[
                ExtractedEntity("Person Alpha", "person", 0, 11, 0.95, "seed"),
            ],
            relations=[],
            chunk_id="chunk-2",
        )
        ids1 = store_extraction_result(result1, store)
        ids2 = store_extraction_result(result2, store)

        # Should resolve to same entity
        assert ids1["Person Alpha"] == ids2["Person Alpha"]

    def test_confidence_stored_as_relevance(self, store):
        """Entity-chunk link relevance should reflect extraction confidence."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity("golems", "project", 0, 6, 0.85, "llm"),
            ],
            relations=[],
            chunk_id="chunk-1",
        )
        store_extraction_result(result, store)

        cursor = store.conn.cursor()
        links = list(cursor.execute("SELECT relevance FROM kg_entity_chunks WHERE chunk_id = 'chunk-1'"))
        assert len(links) == 1
        assert abs(links[0][0] - 0.85) < 0.01


# ── Batch Processing ──


class TestBatchProcessing:
    """Process multiple chunks in batch."""

    def test_batch_processes_all(self, store):
        chunks = [
            {"id": "c1", "content": "Person Alpha works on brainlayer."},
            {"id": "c2", "content": "Person Alpha leads Example Corp product."},
            {"id": "c3", "content": "Deploy FastAPI to Railway."},
        ]
        stats = process_batch(chunks, store, llm_caller=_mock_llm)
        assert stats["chunks_processed"] == 3
        assert stats["entities_found"] > 0

    def test_batch_populates_kg(self, store):
        chunks = [
            {"id": "c1", "content": "Person Alpha founded Example Corp with Person Alpha."},
        ]
        process_batch(
            chunks,
            store,
            seed_entities={
                "person": ["Person Alpha"],
                "company": ["Example Corp"],
                "project": ["brainlayer", "Sample Project"],
            },
            llm_caller=_mock_llm,
        )

        # KG should have entities
        cursor = store.conn.cursor()
        count = cursor.execute("SELECT COUNT(*) FROM kg_entities").fetchone()[0]
        assert count >= 2  # At least Etan and Dor

    def test_batch_handles_empty_chunks(self, store):
        chunks = [
            {"id": "c1", "content": ""},
            {"id": "c2", "content": "   "},
        ]
        stats = process_batch(chunks, store, llm_caller=_mock_llm)
        assert stats["chunks_processed"] == 2
        assert stats["entities_found"] == 0
