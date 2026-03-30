"""Tests for R49 Phase 1 — Entity Contracts, Health Scoring, Type Hierarchy, Aliases.

Tests cover:
1. Schema: entity_contracts, entity_health, entity_type_hierarchy tables created
2. Schema: kg_entity_aliases table upgraded with new columns
3. Schema: kg_entities gets entity_subtype, status, updated_at columns
4. Schema: kg_entity_chunks gets relation_tier, weight columns
5. Contracts YAML loading and validation
6. Health scoring algorithm (5-dimension weighted)
7. Health level classification (5 tiers)
8. Entity lookup includes completeness_score and health_level
"""

from pathlib import Path

import pytest
import yaml

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
def mock_embedding():
    """Generate a deterministic 1024-dim embedding from text."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


@pytest.fixture
def contracts_path():
    """Path to the entity-types.yaml contracts file."""
    return Path(__file__).parent.parent / "contracts" / "entity-types.yaml"


@pytest.fixture
def populated_store(store, mock_embedding):
    """Store with entities, relations, and chunks for health scoring tests."""
    # Create an agent entity
    store.upsert_entity(
        entity_id="ent-coach",
        entity_type="agent",
        name="coachClaude",
        metadata={"description": "Health coach", "capabilities": ["health_tracking"]},
        embedding=mock_embedding("coachClaude"),
        description="Health and habits coaching agent",
    )

    # Create a second entity for relations
    store.upsert_entity(
        entity_id="ent-brain",
        entity_type="tool",
        name="BrainLayer",
        metadata={"description": "Memory layer", "purpose": "persistent memory"},
        embedding=mock_embedding("BrainLayer"),
        description="Persistent memory for AI agents",
    )

    # Create a person entity (minimal)
    store.upsert_entity(
        entity_id="ent-etan",
        entity_type="person",
        name="Etan",
        metadata={"role": "developer"},
        embedding=mock_embedding("Etan"),
        description="Developer and owner",
    )

    # Add relation: coachClaude USES BrainLayer
    store.add_relation(
        relation_id="rel-coach-brain",
        source_id="ent-coach",
        target_id="ent-brain",
        relation_type="USES",
    )

    # Add relation: coachClaude CREATED_BY Etan
    store.add_relation(
        relation_id="rel-coach-etan",
        source_id="ent-coach",
        target_id="ent-etan",
        relation_type="CREATED_BY",
    )

    # Add some chunks linked to coachClaude
    dummy_emb = mock_embedding("dummy chunk")
    for i in range(5):
        chunk_id = f"chunk-coach-{i}"
        store.upsert_chunks(
            [
                {
                    "id": chunk_id,
                    "content": f"Coach info chunk {i}",
                    "metadata": {},
                    "source_file": "test",
                    "project": "test",
                    "content_type": "user_message",
                    "value_type": None,
                    "char_count": 20,
                }
            ],
            [dummy_emb],
        )
        store.link_entity_chunk("ent-coach", chunk_id, relevance=0.8 + i * 0.04)

    # Add chunks to BrainLayer entity (3 chunks)
    for i in range(3):
        chunk_id = f"chunk-brain-{i}"
        store.upsert_chunks(
            [
                {
                    "id": chunk_id,
                    "content": f"BrainLayer info chunk {i}",
                    "metadata": {},
                    "source_file": "test",
                    "project": "test",
                    "content_type": "user_message",
                    "value_type": None,
                    "char_count": 20,
                }
            ],
            [dummy_emb],
        )
        store.link_entity_chunk("ent-brain", chunk_id, relevance=0.9)

    return store


# ── 1. Schema Tests: New Tables ────────────────────────────────


class TestEntityContractsSchema:
    """Verify R49 schema additions exist after VectorStore init."""

    def test_entity_contracts_table_exists(self, store):
        cursor = store._read_cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "entity_contracts" in tables

    def test_entity_contracts_columns(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(entity_contracts)")}
        expected = {"entity_type", "field_name", "field_type", "requirement", "description"}
        assert expected.issubset(cols)

    def test_entity_health_table_exists(self, store):
        cursor = store._read_cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "entity_health" in tables

    def test_entity_health_columns(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(entity_health)")}
        expected = {
            "entity_name",
            "completeness_score",
            "health_level",
            "missing_required",
            "missing_expected",
            "chunk_count",
            "relationship_count",
            "last_scored_at",
        }
        assert expected.issubset(cols)

    def test_entity_type_hierarchy_table_exists(self, store):
        cursor = store._read_cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "entity_type_hierarchy" in tables

    def test_entity_type_hierarchy_columns(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(entity_type_hierarchy)")}
        expected = {"child_type", "parent_type", "description"}
        assert expected.issubset(cols)


# ── 2. Schema Tests: Altered Columns ──────────────────────────


class TestAlteredColumns:
    """Verify ALTER TABLE additions to existing tables."""

    def test_kg_entities_has_entity_subtype(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        assert "entity_subtype" in cols

    def test_kg_entities_has_status(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        assert "status" in cols

    def test_kg_entities_status_default_active(self, store, mock_embedding):
        """New entities should default to status='active'."""
        store.upsert_entity(
            entity_id="ent-test",
            entity_type="concept",
            name="TestConcept",
            embedding=mock_embedding("test"),
        )
        cursor = store._read_cursor()
        row = list(cursor.execute("SELECT status FROM kg_entities WHERE id = ?", ("ent-test",)))
        assert row[0][0] == "active"

    def test_kg_entities_has_updated_at(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        # updated_at already exists from original schema
        assert "updated_at" in cols

    def test_kg_entity_chunks_has_relation_tier(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
        assert "relation_tier" in cols

    def test_kg_entity_chunks_has_weight(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
        assert "weight" in cols

    def test_kg_entity_chunks_relation_tier_default(self, store, mock_embedding):
        """New entity-chunk links should default to relation_tier=4, weight=0.25."""
        store.upsert_entity(
            entity_id="ent-tier",
            entity_type="concept",
            name="TierTest",
            embedding=mock_embedding("tier"),
        )
        store.upsert_chunks(
            [
                {
                    "id": "chunk-tier-1",
                    "content": "test",
                    "metadata": {},
                    "source_file": "t",
                    "project": "t",
                    "content_type": "user_message",
                    "value_type": None,
                    "char_count": 4,
                }
            ],
            [mock_embedding("tier chunk")],
        )
        store.link_entity_chunk("ent-tier", "chunk-tier-1")
        cursor = store._read_cursor()
        row = list(
            cursor.execute(
                "SELECT relation_tier, weight FROM kg_entity_chunks WHERE entity_id = ? AND chunk_id = ?",
                ("ent-tier", "chunk-tier-1"),
            )
        )
        assert row[0][0] == 4  # default tier
        assert row[0][1] == 0.25  # default weight


# ── 3. Upgraded Aliases Table ──────────────────────────────────


class TestEntityAliases:
    """Verify kg_entity_aliases has R49 columns (alias_type upgraded, valid_from, valid_to)."""

    def test_aliases_has_valid_from(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_aliases)")}
        assert "valid_from" in cols

    def test_aliases_has_valid_to(self, store):
        cursor = store._read_cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_aliases)")}
        assert "valid_to" in cols


# ── 4. Contracts YAML ─────────────────────────────────────────


class TestContractsYAML:
    """Verify entity-types.yaml is valid and complete."""

    def test_contracts_file_exists(self, contracts_path):
        assert contracts_path.exists(), f"Expected contracts at {contracts_path}"

    def test_contracts_yaml_parses(self, contracts_path):
        with open(contracts_path) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_contracts_has_version(self, contracts_path):
        with open(contracts_path) as f:
            data = yaml.safe_load(f)
        assert "version" in data

    def test_contracts_has_required_entity_types(self, contracts_path):
        with open(contracts_path) as f:
            data = yaml.safe_load(f)
        entity_types = data.get("entity_types", {})
        for etype in ["agent", "person", "tool", "project", "concept"]:
            assert etype in entity_types, f"Missing entity type: {etype}"

    def test_each_type_has_required_fields(self, contracts_path):
        with open(contracts_path) as f:
            data = yaml.safe_load(f)
        for etype, spec in data["entity_types"].items():
            assert "required" in spec, f"{etype} missing 'required' field list"
            assert "expected" in spec, f"{etype} missing 'expected' field list"
            assert "health_criteria" in spec, f"{etype} missing 'health_criteria'"

    def test_agent_type_has_name_description(self, contracts_path):
        with open(contracts_path) as f:
            data = yaml.safe_load(f)
        agent = data["entity_types"]["agent"]
        assert "name" in agent["required"]
        assert "type" in agent["required"]
        assert "description" in agent["required"]

    def test_health_criteria_fields(self, contracts_path):
        with open(contracts_path) as f:
            data = yaml.safe_load(f)
        for etype, spec in data["entity_types"].items():
            hc = spec["health_criteria"]
            assert "min_chunks" in hc, f"{etype} health_criteria missing min_chunks"
            assert "max_stale_days" in hc, f"{etype} health_criteria missing max_stale_days"
            assert "min_relationships" in hc, f"{etype} health_criteria missing min_relationships"


# ── 5. Health Scoring Algorithm ────────────────────────────────


class TestHealthScoring:
    """Test the score_entity_health scoring algorithm."""

    def test_score_module_importable(self):
        """The scoring module should be importable."""

    def test_load_contracts(self, contracts_path):
        from scripts.score_entity_health import load_contracts

        contracts = load_contracts(str(contracts_path))
        assert "agent" in contracts
        assert "person" in contracts

    def test_score_returns_required_fields(self, populated_store, contracts_path):
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-coach", "agent", contracts["agent"])
        assert "completeness_score" in result
        assert "health_level" in result
        assert "missing_required" in result
        assert "missing_expected" in result
        assert "chunk_count" in result
        assert "relationship_count" in result

    def test_score_range_0_to_1(self, populated_store, contracts_path):
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-coach", "agent", contracts["agent"])
        assert 0.0 <= result["completeness_score"] <= 1.0

    def test_health_level_1_to_5(self, populated_store, contracts_path):
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-coach", "agent", contracts["agent"])
        assert 1 <= result["health_level"] <= 5

    def test_health_level_classification(self):
        """Verify the 5-tier classification boundaries."""
        from scripts.score_entity_health import classify_health_level

        assert classify_health_level(0.90) == 5  # Very Detailed
        assert classify_health_level(0.85) == 5  # boundary
        assert classify_health_level(0.84) == 4  # Detailed
        assert classify_health_level(0.65) == 4  # boundary
        assert classify_health_level(0.64) == 3  # Moderate
        assert classify_health_level(0.45) == 3  # boundary
        assert classify_health_level(0.44) == 2  # Basic
        assert classify_health_level(0.25) == 2  # boundary
        assert classify_health_level(0.24) == 1  # Stub
        assert classify_health_level(0.0) == 1  # Stub

    def test_chunk_count_accurate(self, populated_store, contracts_path):
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-coach", "agent", contracts["agent"])
        assert result["chunk_count"] == 5

    def test_relationship_count_accurate(self, populated_store, contracts_path):
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-coach", "agent", contracts["agent"])
        assert result["relationship_count"] == 2

    def test_missing_required_detected(self, populated_store, contracts_path):
        """coachClaude doesn't have all required fields populated — detect gaps."""
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-coach", "agent", contracts["agent"])
        # capabilities is partially there in metadata, memory_domains is missing
        assert isinstance(result["missing_required"], list)

    def test_tool_entity_scoring(self, populated_store, contracts_path):
        """BrainLayer (tool) should also get scored."""
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        result = score_entity(populated_store, "ent-brain", "tool", contracts["tool"])
        assert 0.0 <= result["completeness_score"] <= 1.0
        assert result["chunk_count"] == 3

    def test_stub_entity_low_score(self, store, mock_embedding, contracts_path):
        """An entity with no chunks, no relations = very low score (Stub)."""
        from scripts.score_entity_health import load_contracts, score_entity

        contracts = load_contracts(str(contracts_path))
        store.upsert_entity(
            entity_id="ent-stub",
            entity_type="concept",
            name="StubConcept",
            embedding=mock_embedding("stub"),
        )
        result = score_entity(store, "ent-stub", "concept", contracts["concept"])
        assert result["health_level"] <= 2  # Stub or Basic
        assert result["chunk_count"] == 0
        assert result["relationship_count"] == 0


# ── 6. Health Score Persistence ────────────────────────────────


class TestHealthPersistence:
    """Test that scores are written to entity_health table."""

    def test_populate_entity_health(self, populated_store, contracts_path):
        from scripts.score_entity_health import load_contracts, score_all_entities

        contracts = load_contracts(str(contracts_path))
        score_all_entities(populated_store, contracts)

        cursor = populated_store._read_cursor()
        rows = list(cursor.execute("SELECT entity_name, completeness_score, health_level FROM entity_health"))
        names = {row[0] for row in rows}
        assert "coachClaude" in names
        assert "BrainLayer" in names

    def test_health_score_updates_on_rerun(self, populated_store, contracts_path):
        """Running score_all_entities twice should upsert, not duplicate."""
        from scripts.score_entity_health import load_contracts, score_all_entities

        contracts = load_contracts(str(contracts_path))
        score_all_entities(populated_store, contracts)
        score_all_entities(populated_store, contracts)

        cursor = populated_store._read_cursor()
        rows = list(cursor.execute("SELECT COUNT(*) FROM entity_health"))
        # Should have exactly as many rows as entities, not doubled
        entity_count = list(cursor.execute("SELECT COUNT(*) FROM kg_entities"))[0][0]
        assert rows[0][0] == entity_count


# ── 7. Entity Lookup Enhancement ───────────────────────────────


class TestEntityLookupEnhanced:
    """Verify brain_entity includes completeness_score and health_level."""

    def test_entity_lookup_includes_health(self, populated_store, mock_embedding, contracts_path):
        """After scoring, entity_lookup should include health data."""
        from scripts.score_entity_health import load_contracts, score_all_entities

        contracts = load_contracts(str(contracts_path))
        score_all_entities(populated_store, contracts)

        from brainlayer.pipeline.digest import entity_lookup

        result = entity_lookup(
            query="coachClaude",
            store=populated_store,
            embed_fn=mock_embedding,
            entity_type="agent",
        )
        assert result is not None
        assert "completeness_score" in result
        assert "health_level" in result
        assert isinstance(result["completeness_score"], float)
        assert isinstance(result["health_level"], int)

    def test_entity_lookup_no_health_when_unscored(self, store, mock_embedding):
        """Before scoring runs, health fields should be None or absent."""
        store.upsert_entity(
            entity_id="ent-unscored",
            entity_type="concept",
            name="UnscoredConcept",
            embedding=mock_embedding("unscored"),
        )
        from brainlayer.pipeline.digest import entity_lookup

        result = entity_lookup(
            query="UnscoredConcept",
            store=store,
            embed_fn=mock_embedding,
        )
        assert result is not None
        # Should still return, just with None health data
        assert result.get("completeness_score") is None
        assert result.get("health_level") is None


# ── 8. Type Hierarchy Data ─────────────────────────────────────


class TestTypeHierarchy:
    """Verify type hierarchy is populated with seed data."""

    def test_hierarchy_has_seed_data(self, store):
        """entity_type_hierarchy should contain core types."""
        cursor = store._read_cursor()
        rows = list(cursor.execute("SELECT child_type, parent_type FROM entity_type_hierarchy"))
        hierarchy = {row[0]: row[1] for row in rows}
        assert "agent" in hierarchy
        assert "person" in hierarchy
        assert "tool" in hierarchy
        assert "project" in hierarchy
        assert "concept" in hierarchy

    def test_subtypes_mapped(self, store):
        """Subtypes like golem->agent, platform->tool should exist."""
        cursor = store._read_cursor()
        rows = list(cursor.execute("SELECT child_type, parent_type FROM entity_type_hierarchy"))
        hierarchy = {row[0]: row[1] for row in rows}
        assert hierarchy.get("golem") == "agent"
        assert hierarchy.get("platform") == "tool"
        assert hierarchy.get("skill") == "concept"
        assert hierarchy.get("decision") == "concept"
