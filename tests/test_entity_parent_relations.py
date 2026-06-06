"""Tests for entity parent relations and canonical relation type expansion."""

import pytest

from brainlayer.pipeline.entity_extraction import ExtractedEntity, ExtractedRelation, ExtractionResult
from brainlayer.pipeline.kg_extraction import (
    _RELATION_TYPE_ALIASES,
    CANONICAL_RELATION_TYPES,
    validate_extraction_result,
)
from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def test_parent_id_column_exists(store):
    """kg_entities should expose parent_id in schema."""
    cursor = store._read_cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
    assert "parent_id" in cols


def test_upsert_entity_with_parent(store):
    """upsert_entity should persist parent_id."""
    store.upsert_entity("company-1", "company", "OpenAI")
    store.upsert_entity("person-1", "person", "Sam Altman", parent_id="company-1")

    cursor = store._read_cursor()
    row = cursor.execute("SELECT parent_id FROM kg_entities WHERE id = ?", ("person-1",)).fetchone()
    assert row is not None
    assert row[0] == "company-1"


def test_upsert_entity_with_subtype(store):
    """upsert_entity should persist entity_subtype."""
    store.upsert_entity("source-1", "source", "t3.gg", entity_subtype="channel")

    cursor = store._read_cursor()
    row = cursor.execute("SELECT entity_type, entity_subtype FROM kg_entities WHERE id = ?", ("source-1",)).fetchone()
    assert row is not None
    assert row[0] == "source"
    assert row[1] == "channel"


def test_source_type_hierarchy_seeded(store):
    """Source should be seeded as a top-level entity child."""
    cursor = store._read_cursor()
    row = cursor.execute(
        "SELECT parent_type, description FROM entity_type_hierarchy WHERE child_type = 'source'"
    ).fetchone()

    assert row is not None
    assert row[0] == "entity"
    assert row[1] == "External content source: channel, podcast, blog, newsletter"


def test_get_entity_children(store):
    """Children should be returned in the expected shape."""
    store.upsert_entity("company-1", "company", "OpenAI", description="AI company", importance=0.9)
    store.upsert_entity("person-1", "person", "Sam Altman", parent_id="company-1", importance=0.8)
    store.upsert_entity("person-2", "person", "Mira Murati", parent_id="company-1", importance=0.7)

    children = store.get_entity_children("company-1")

    assert [child["id"] for child in children] == ["person-1", "person-2"]
    assert children[0]["entity_type"] == "person"
    assert children[0]["name"] == "Sam Altman"


def test_get_entity_parent(store):
    """Parent should be returned in the expected shape."""
    store.upsert_entity("company-1", "company", "OpenAI", description="AI company", importance=0.9)
    store.upsert_entity("person-1", "person", "Sam Altman", parent_id="company-1")

    parent = store.get_entity_parent("person-1")

    assert parent is not None
    assert parent["id"] == "company-1"
    assert parent["entity_type"] == "company"
    assert parent["name"] == "OpenAI"


def test_get_entity_children_empty(store):
    """Entities without children should return an empty list."""
    store.upsert_entity("company-1", "company", "OpenAI")

    assert store.get_entity_children("company-1") == []


def test_get_entity_parent_none(store):
    """Entities without a parent should return None."""
    store.upsert_entity("person-1", "person", "Sam Altman")

    assert store.get_entity_parent("person-1") is None


def test_set_entity_parent(store):
    """Parent should be assignable after entity creation."""
    store.upsert_entity("company-1", "company", "OpenAI")
    store.upsert_entity("person-1", "person", "Sam Altman")

    store.set_entity_parent("person-1", "company-1")

    parent = store.get_entity_parent("person-1")
    assert parent is not None
    assert parent["id"] == "company-1"


def test_canonical_relation_types_expanded():
    """New canonical relation types should be accepted directly."""
    expected = {
        "depends_on",
        "spawns",
        "created",
        "lives_in",
        "leads",
        "freelances_for",
        "hosts",
        "appears_on",
    }
    assert expected.issubset(CANONICAL_RELATION_TYPES)


def test_relation_type_aliases():
    """Known legacy aliases should normalize to canonical relation types."""
    assert _RELATION_TYPE_ALIASES["ceo_of"] == "leads"
    assert _RELATION_TYPE_ALIASES["worked_at"] == "works_at"
    assert _RELATION_TYPE_ALIASES["framework_for"] == "depends_on"
    assert _RELATION_TYPE_ALIASES["contact_at"] == "affiliated_with"
    assert _RELATION_TYPE_ALIASES["host_of"] == "hosts"
    assert _RELATION_TYPE_ALIASES["guest_on"] == "appears_on"
    assert _RELATION_TYPE_ALIASES["appeared_on"] == "appears_on"

    result = ExtractionResult(
        entities=[
            ExtractedEntity("Etan Heyman", "person", 0, 11, 0.9, "seed"),
            ExtractedEntity("BrainLayer", "project", 12, 22, 0.9, "seed"),
            ExtractedEntity("Cantaloupe", "company", 23, 33, 0.9, "seed"),
        ],
        relations=[
            ExtractedRelation("Etan Heyman", "BrainLayer", "framework_for", 0.8, {}),
            ExtractedRelation("Etan Heyman", "Cantaloupe", "worked_at", 0.8, {}),
            ExtractedRelation("Etan Heyman", "Cantaloupe", "ceo_of", 0.8, {}),
            ExtractedRelation("Etan Heyman", "BrainLayer", "host_of", 0.8, {}),
        ],
        chunk_id="chunk-1",
    )

    validated = validate_extraction_result(result)
    relation_types = [rel.relation_type for rel in validated.relations]

    assert relation_types == ["depends_on", "works_at", "leads", "hosts"]


def test_validation_coerces_command_shaped_people_to_agents():
    """Slash commands and agent-shaped names must not remain person/project."""
    result = ExtractionResult(
        entities=[
            ExtractedEntity("/pr-loop", "person", 0, 8, 0.8, "llm"),
            ExtractedEntity("brainlayerCodex", "project", 9, 23, 0.8, "llm"),
            ExtractedEntity("yash", "person", 24, 28, 0.8, "llm"),
        ],
        relations=[],
        chunk_id="chunk-1",
    )

    validated = validate_extraction_result(result)

    assert [entity.entity_type for entity in validated.entities] == ["agent", "agent", "topic"]


def test_validation_coerces_youtube_channel_to_source_with_subtype():
    """Channel URLs and source-like suffixes should become Source facets."""
    result = ExtractionResult(
        entities=[
            ExtractedEntity("youtube.com/@t3dotgg", "project", 0, 19, 0.8, "llm"),
            ExtractedEntity("Lex Fridman Podcast", "company", 20, 39, 0.8, "llm"),
            ExtractedEntity("BrainLayer Newsletter", "topic", 40, 62, 0.8, "llm"),
        ],
        relations=[],
        chunk_id="chunk-1",
    )

    validated = validate_extraction_result(result)

    assert [(entity.entity_type, entity.entity_subtype) for entity in validated.entities] == [
        ("source", "channel"),
        ("source", "podcast"),
        ("source", "newsletter"),
    ]
