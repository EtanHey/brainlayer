"""Tests for KG Standard Tables — standardized spec matching Convex kgSpec.ts.

Tests cover:
1. New columns on kg_entities (canonical_name, description, confidence, importance, valid_from/until, group_id)
2. New columns on kg_relations (fact, importance, valid_from/until, expired_at, source_chunk_id)
3. New column on kg_entity_chunks (mention_type)
4. kg_current_facts VIEW
5. Soft-close relations (expired_at)
6. 2-hop graph traversal via recursive CTE
7. Entity resolution (exact alias → fuzzy name)
8. effective_score decay function
9. Shared constants match spec
10. Backward compatibility — existing CRUD still works
"""

import pytest

from brainlayer.kg import DECAY_CONSTANTS, ENTITY_TYPES, RELATION_TYPES, effective_score
from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with a small graph: 3 entities, 2 relations."""
    store.upsert_entity(
        "person-1",
        "person",
        "Etan Heyman",
        canonical_name="etan_heyman",
        description="Software developer",
        confidence=0.95,
        importance=0.8,
        group_id="default",
    )
    store.upsert_entity(
        "org-1",
        "organization",
        "Cantaloupe",
        canonical_name="cantaloupe",
        description="Tech company",
    )
    store.upsert_entity(
        "meeting-1",
        "meeting",
        "Weekly Standup",
        canonical_name="weekly_standup",
        valid_from="2026-02-27T09:00:00Z",
        valid_until="2026-02-27T10:00:00Z",
    )
    store.add_relation(
        "rel-1",
        "person-1",
        "org-1",
        "works_at",
        fact="Etan works at Cantaloupe",
        importance=0.9,
    )
    store.add_relation(
        "rel-2",
        "person-1",
        "meeting-1",
        "attended",
        fact="Etan attended the weekly standup",
        source_chunk_id="chunk-abc",
    )
    return store


# ── Constants Tests ────────────────────────────────────────


class TestKGConstants:
    """Verify shared constants match the spec."""

    def test_entity_types_complete(self):
        assert len(ENTITY_TYPES) == 7
        assert "person" in ENTITY_TYPES
        assert "constraint" in ENTITY_TYPES
        assert "preference" in ENTITY_TYPES
        assert "life_event" in ENTITY_TYPES
        assert "meeting" in ENTITY_TYPES
        assert "location" in ENTITY_TYPES
        assert "organization" in ENTITY_TYPES

    def test_relation_types_complete(self):
        assert len(RELATION_TYPES) == 9
        assert "has_constraint" in RELATION_TYPES
        assert "knows" in RELATION_TYPES
        assert "supersedes" in RELATION_TYPES

    def test_decay_constants_keys(self):
        assert "constraint" in DECAY_CONSTANTS
        assert "preference" in DECAY_CONSTANTS
        assert "life_event" in DECAY_CONSTANTS
        assert "casual" in DECAY_CONSTANTS
        assert "meeting" in DECAY_CONSTANTS

    def test_life_event_no_decay(self):
        assert DECAY_CONSTANTS["life_event"] == 0


# ── effective_score Tests ────────────────────────────────────


class TestEffectiveScore:
    """Test time-decayed scoring function."""

    def test_perfect_score_at_zero_age(self):
        score = effective_score(1.0, 1.0, 0, "constraint")
        assert score == pytest.approx(1.0)

    def test_half_life_constraint(self):
        # constraint half-life ~365 days: at 365 days, score ≈ 0.5
        score = effective_score(1.0, 1.0, 365, "constraint")
        assert 0.45 < score < 0.55

    def test_half_life_casual(self):
        # casual half-life ~30 days: at 30 days, score ≈ 0.5
        score = effective_score(1.0, 1.0, 30, "casual")
        assert 0.45 < score < 0.55

    def test_life_event_no_decay(self):
        score = effective_score(1.0, 1.0, 10000, "life_event")
        assert score == pytest.approx(1.0)

    def test_confidence_importance_multiply(self):
        score = effective_score(0.5, 0.5, 0, "constraint")
        assert score == pytest.approx(0.25)

    def test_unknown_type_uses_default(self):
        # Unknown type defaults to preference rate (0.0077)
        score = effective_score(1.0, 1.0, 90, None)
        assert 0.45 < score < 0.55


# ── New Column Tests ────────────────────────────────────────


class TestKGStandardColumns:
    """Verify new standard columns exist on KG tables."""

    def test_kg_entities_new_columns(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        for col in (
            "canonical_name",
            "description",
            "confidence",
            "importance",
            "valid_from",
            "valid_until",
            "group_id",
        ):
            assert col in cols, f"Missing column: {col}"

    def test_kg_relations_new_columns(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_relations)")}
        for col in ("fact", "importance", "valid_from", "valid_until", "expired_at", "source_chunk_id"):
            assert col in cols, f"Missing column: {col}"

    def test_kg_entity_chunks_mention_type(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
        assert "mention_type" in cols

    def test_kg_current_facts_view_exists(self, store):
        cursor = store._read_cursor()
        views = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        assert "kg_current_facts" in views

    def test_canonical_name_index(self, store):
        cursor = store._read_cursor()
        indexes = {row[1] for row in cursor.execute("PRAGMA index_list(kg_entities)")}
        assert "idx_kg_entities_canonical" in indexes

    def test_validity_index_on_entities(self, store):
        cursor = store._read_cursor()
        indexes = {row[1] for row in cursor.execute("PRAGMA index_list(kg_entities)")}
        assert "idx_kg_entities_valid" in indexes

    def test_validity_index_on_relations(self, store):
        cursor = store._read_cursor()
        indexes = {row[1] for row in cursor.execute("PRAGMA index_list(kg_relations)")}
        assert "idx_kg_relations_validity" in indexes


# ── Entity CRUD with Standard Fields ────────────────────────


class TestEntityStandardCRUD:
    """Test entity upsert/get with new standard fields."""

    def test_upsert_with_standard_fields(self, populated_store):
        entity = populated_store.get_entity("person-1")
        assert entity is not None
        assert entity["canonical_name"] == "etan_heyman"
        assert entity["description"] == "Software developer"
        assert entity["confidence"] == 0.95
        assert entity["importance"] == 0.8
        assert entity["group_id"] == "default"

    def test_upsert_with_validity(self, populated_store):
        entity = populated_store.get_entity("meeting-1")
        assert entity["valid_from"] == "2026-02-27T09:00:00Z"
        assert entity["valid_until"] == "2026-02-27T10:00:00Z"

    def test_upsert_defaults(self, store):
        """New fields default to sensible values."""
        store.upsert_entity("e-1", "person", "Test")
        entity = store.get_entity("e-1")
        assert entity["confidence"] == 1.0
        assert entity["importance"] == 0.5
        assert entity["canonical_name"] == "test"  # auto-lowered from name
        assert entity["description"] is None
        assert entity["valid_from"] is None
        assert entity["valid_until"] is None
        assert entity["group_id"] is None

    def test_backward_compat_metadata_still_works(self, store):
        """Existing callers using metadata= kwarg still work."""
        store.upsert_entity("e-1", "person", "Test", metadata={"key": "value"})
        entity = store.get_entity("e-1")
        assert entity["metadata"]["key"] == "value"


# ── Relation CRUD with Standard Fields ────────────────────────


class TestRelationStandardCRUD:
    """Test relation CRUD with new standard fields."""

    def test_relation_with_fact(self, populated_store):
        rels = populated_store.get_entity_relations("person-1", direction="outgoing")
        works_at = [r for r in rels if r["relation_type"] == "works_at"][0]
        assert works_at["fact"] == "Etan works at Cantaloupe"
        assert works_at["importance"] == 0.9

    def test_relation_with_source_chunk(self, populated_store):
        rels = populated_store.get_entity_relations("person-1", direction="outgoing")
        attended = [r for r in rels if r["relation_type"] == "attended"][0]
        assert attended["source_chunk_id"] == "chunk-abc"

    def test_relation_defaults(self, store):
        store.upsert_entity("e-1", "person", "A")
        store.upsert_entity("e-2", "person", "B")
        store.add_relation("r-1", "e-1", "e-2", "knows")
        rels = store.get_entity_relations("e-1", direction="outgoing")
        assert rels[0]["fact"] is None
        assert rels[0]["importance"] == 0.5
        assert rels[0]["expired_at"] is None
        assert rels[0]["source_chunk_id"] is None


# ── Soft-Close Relations ────────────────────────────────────


class TestSoftCloseRelation:
    """Test soft-closing (expiring) relations."""

    def test_soft_close_sets_expired_at(self, populated_store):
        populated_store.soft_close_relation("rel-1")
        rels = populated_store.get_entity_relations("person-1", direction="outgoing")
        works_at = [r for r in rels if r["relation_type"] == "works_at"][0]
        assert works_at["expired_at"] is not None

    def test_current_facts_excludes_expired(self, populated_store):
        # Both relations visible before soft-close
        facts = populated_store.get_current_facts("person-1")
        assert len(facts) == 2

        # Soft-close one
        populated_store.soft_close_relation("rel-1")

        # Only non-expired visible
        facts = populated_store.get_current_facts("person-1")
        assert len(facts) == 1
        assert facts[0]["relation_type"] == "attended"

    def test_current_facts_view_filters_expired(self, populated_store):
        populated_store.soft_close_relation("rel-1")
        cursor = populated_store._read_cursor()
        rows = list(cursor.execute("SELECT * FROM kg_current_facts WHERE source_id = ?", ("person-1",)))
        assert len(rows) == 1


# ── 2-Hop Traversal ────────────────────────────────────────


class TestGraphTraversal:
    """Test multi-hop graph traversal via recursive CTE."""

    def test_1hop_traversal(self, populated_store):
        result = populated_store.traverse(entity_id="person-1", max_depth=1)
        # person-1 → org-1, person-1 → meeting-1
        entity_ids = {r["entity_id"] for r in result}
        assert "org-1" in entity_ids
        assert "meeting-1" in entity_ids

    def test_2hop_traversal(self, store):
        # Build A → B → C chain
        store.upsert_entity("a", "person", "Alice")
        store.upsert_entity("b", "person", "Bob")
        store.upsert_entity("c", "person", "Carol")
        store.add_relation("r1", "a", "b", "knows")
        store.add_relation("r2", "b", "c", "knows")

        result = store.traverse(entity_id="a", max_depth=2)
        entity_ids = {r["entity_id"] for r in result}
        assert "b" in entity_ids  # 1 hop
        assert "c" in entity_ids  # 2 hops

    def test_traversal_depth_limit(self, store):
        # A → B → C, but max_depth=1 should not reach C
        store.upsert_entity("a", "person", "Alice")
        store.upsert_entity("b", "person", "Bob")
        store.upsert_entity("c", "person", "Carol")
        store.add_relation("r1", "a", "b", "knows")
        store.add_relation("r2", "b", "c", "knows")

        result = store.traverse(entity_id="a", max_depth=1)
        entity_ids = {r["entity_id"] for r in result}
        assert "b" in entity_ids
        assert "c" not in entity_ids

    def test_traversal_no_cycles(self, store):
        # A ↔ B (bidirectional) should not loop
        store.upsert_entity("a", "person", "Alice")
        store.upsert_entity("b", "person", "Bob")
        store.add_relation("r1", "a", "b", "knows")
        store.add_relation("r2", "b", "a", "knows")

        result = store.traverse(entity_id="a", max_depth=3)
        assert len(result) == 1  # Only b reachable


# ── Entity Resolution ────────────────────────────────────────


class TestEntityResolution:
    """Test entity resolution: exact alias → fuzzy name."""

    def test_resolve_by_exact_alias(self, store):
        store.upsert_entity("person-1", "person", "Etan Heyman")
        store.add_entity_alias("EtanHey", "person-1", alias_type="nickname")

        result = store.resolve_entity("EtanHey")
        assert result is not None
        assert result["id"] == "person-1"

    def test_resolve_by_exact_name(self, store):
        store.upsert_entity("person-1", "person", "Etan Heyman")

        result = store.resolve_entity("Etan Heyman")
        assert result is not None
        assert result["id"] == "person-1"

    def test_resolve_by_fuzzy_name(self, store):
        store.upsert_entity("person-1", "person", "Etan Heyman")

        result = store.resolve_entity("Etan")
        assert result is not None
        assert result["id"] == "person-1"

    def test_resolve_not_found(self, store):
        result = store.resolve_entity("nonexistent_xyzzy_12345")
        assert result is None

    def test_resolve_by_canonical_name(self, store):
        store.upsert_entity(
            "person-1",
            "person",
            "Etan Heyman",
            canonical_name="etan_heyman",
        )
        result = store.resolve_entity("etan_heyman")
        assert result is not None
        assert result["id"] == "person-1"


# ── Link Entity Chunk with mention_type ────────────────────


class TestEntityChunkMentionType:
    """Test linking entities to chunks with mention_type."""

    def test_link_with_mention_type(self, store):
        store.upsert_entity("e-1", "person", "Test")
        # We need a chunk to link to — use a minimal insert
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, value_type, char_count)
               VALUES (?, ?, '{}', 'test.jsonl', 'test', 'user_message', 'HIGH', 10)""",
            ("chunk-1", "test content"),
        )
        emb = [0.1] * 1024
        from brainlayer.vector_store import serialize_f32

        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            ("chunk-1", serialize_f32(emb)),
        )

        store.link_entity_chunk("e-1", "chunk-1", relevance=0.9, mention_type="explicit")

        chunks = store.get_entity_chunks("e-1")
        assert chunks[0]["mention_type"] == "explicit"

    def test_link_default_mention_type(self, store):
        store.upsert_entity("e-1", "person", "Test")
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, value_type, char_count)
               VALUES (?, ?, '{}', 'test.jsonl', 'test', 'user_message', 'HIGH', 10)""",
            ("chunk-1", "test content"),
        )
        emb = [0.1] * 1024
        from brainlayer.vector_store import serialize_f32

        cursor.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            ("chunk-1", serialize_f32(emb)),
        )

        store.link_entity_chunk("e-1", "chunk-1")
        chunks = store.get_entity_chunks("e-1")
        assert chunks[0]["mention_type"] is None  # Not set by default


# ── Migration Backward Compatibility ────────────────────────


class TestBackwardCompatibility:
    """Verify existing schema + data survives the migration."""

    def test_existing_kg_tests_entity_types_still_work(self, store):
        """The old entity types (golem, skill, project, topic) still work."""
        store.upsert_entity("golem-1", "golem", "brainClaude")
        store.upsert_entity("skill-1", "skill", "railway-deploy")
        entity = store.get_entity("golem-1")
        assert entity is not None
        assert entity["entity_type"] == "golem"

    def test_existing_relation_type_field(self, store):
        """relation_type column still works (spec calls it 'relation')."""
        store.upsert_entity("a", "person", "A")
        store.upsert_entity("b", "person", "B")
        store.add_relation("r1", "a", "b", "knows")
        rels = store.get_entity_relations("a", direction="outgoing")
        assert rels[0]["relation_type"] == "knows"

    def test_double_init_with_new_columns(self, tmp_path):
        """Opening same DB twice doesn't crash with new columns."""
        db_path = tmp_path / "test.db"
        s1 = VectorStore(db_path)
        s1.upsert_entity("e-1", "person", "Test", confidence=0.9)
        s1.close()

        s2 = VectorStore(db_path)
        entity = s2.get_entity("e-1")
        assert entity["confidence"] == 0.9
        s2.close()
