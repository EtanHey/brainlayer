"""Tests for KG relation extraction — co-occurrence based."""

from brainlayer.pipeline.entity_extraction import (
    ExtractedEntity,
    extract_cooccurrence_relations,
    extract_entities_combined,
)


class TestCooccurrenceRelations:
    """Rule-based relation extraction from co-occurring entities."""

    def test_two_entities_produce_relation(self):
        """Two entities in the same text should produce a co-occurrence relation."""
        entities = [
            ExtractedEntity(text="BrainLayer", entity_type="project", start=0, end=10, confidence=0.9, source="seed"),
            ExtractedEntity(text="SQLite", entity_type="technology", start=20, end=26, confidence=0.8, source="seed"),
        ]
        relations = extract_cooccurrence_relations(entities)
        assert len(relations) >= 1
        rel = relations[0]
        assert rel.source_text == "BrainLayer"
        assert rel.target_text == "SQLite"
        assert rel.relation_type == "co_occurs_with"
        assert 0 < rel.confidence <= 1.0

    def test_no_relations_for_single_entity(self):
        """A single entity can't have co-occurrence relations."""
        entities = [
            ExtractedEntity(text="BrainLayer", entity_type="project", start=0, end=10, confidence=0.9, source="seed"),
        ]
        relations = extract_cooccurrence_relations(entities)
        assert len(relations) == 0

    def test_no_duplicate_relations(self):
        """Same entity pair should produce at most one relation."""
        entities = [
            ExtractedEntity(text="Foo", entity_type="project", start=0, end=3, confidence=0.9, source="seed"),
            ExtractedEntity(text="Bar", entity_type="technology", start=10, end=13, confidence=0.8, source="seed"),
        ]
        relations = extract_cooccurrence_relations(entities)
        pairs = [(r.source_text, r.target_text) for r in relations]
        assert len(pairs) == len(set(pairs))

    def test_same_type_entities_not_related(self):
        """Entities of the same type shouldn't get co-occurrence relations (too noisy)."""
        entities = [
            ExtractedEntity(text="Foo", entity_type="project", start=0, end=3, confidence=0.9, source="seed"),
            ExtractedEntity(text="Bar", entity_type="project", start=10, end=13, confidence=0.8, source="seed"),
        ]
        relations = extract_cooccurrence_relations(entities)
        assert len(relations) == 0


class TestCombinedExtractsRelations:
    """extract_entities_combined should produce relations even without LLM."""

    def test_combined_produces_cooccurrence_relations(self):
        """Combined extraction should include co-occurrence relations from entities."""
        seed = {
            "project": ["BrainLayer"],
            "technology": ["SQLite", "Python"],
        }
        text = "BrainLayer uses SQLite for storage and Python for the CLI."
        result = extract_entities_combined(text, seed, llm_caller=None, use_llm=False)
        assert len(result.entities) >= 2
        assert len(result.relations) >= 1
        rel_types = {r.relation_type for r in result.relations}
        assert "co_occurs_with" in rel_types
