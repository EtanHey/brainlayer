"""Tests for Gate 2: Entity dedup/healing.

Covers:
- kg_entity_aliases table creation
- Alias CRUD operations
- Entity resolution (seed → alias → fuzzy → semantic)
- Merge operation
- Hebrew prefix stripping
"""

import json
from pathlib import Path

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with KG tables."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _upsert(store, etype, name, eid=None, metadata=None):
    """Helper: upsert entity with auto-generated ID."""
    if eid is None:
        eid = f"{etype}-{name.lower().replace(' ', '-')}"
    return store.upsert_entity(eid, etype, name, metadata=metadata or {})


# ── Alias table exists ──


class TestAliasTableSchema:
    """kg_entity_aliases table should be created by _init_db."""

    def test_alias_table_exists(self, store):
        """The alias table should exist after init."""
        cursor = store.conn.cursor()
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        assert "kg_entity_aliases" in tables

    def test_alias_table_columns(self, store):
        """Alias table should have alias, entity_id, alias_type, created_at."""
        cursor = store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_aliases)")}
        assert "alias" in cols
        assert "entity_id" in cols
        assert "alias_type" in cols
        assert "created_at" in cols


# ── Alias CRUD ──


class TestAliasCRUD:
    """CRUD operations for entity aliases."""

    def test_add_alias(self, store):
        """Adding an alias should be retrievable."""
        entity_id = _upsert(store, "person", "Etan Heyman")
        store.add_entity_alias("Etan", entity_id, alias_type="name")

        result = store.get_entity_by_alias("Etan")
        assert result is not None
        assert result["id"] == entity_id
        assert result["name"] == "Etan Heyman"

    def test_add_multiple_aliases(self, store):
        """Multiple aliases should all resolve to the same entity."""
        entity_id = _upsert(store, "person", "Etan Heyman")
        store.add_entity_alias("Etan", entity_id, alias_type="name")
        store.add_entity_alias("etanheyman", entity_id, alias_type="handle")
        store.add_entity_alias("@EtanHey", entity_id, alias_type="handle")

        for alias in ["Etan", "etanheyman", "@EtanHey"]:
            result = store.get_entity_by_alias(alias)
            assert result is not None
            assert result["id"] == entity_id

    def test_unknown_alias_returns_none(self, store):
        """Looking up a non-existent alias should return None."""
        result = store.get_entity_by_alias("nonexistent-alias")
        assert result is None

    def test_case_insensitive_alias_lookup(self, store):
        """Alias lookup should be case-insensitive."""
        entity_id = _upsert(store, "person", "Dor Zohar")
        store.add_entity_alias("dor", entity_id, alias_type="name")

        # Should match regardless of case
        assert store.get_entity_by_alias("Dor") is not None
        assert store.get_entity_by_alias("DOR") is not None
        assert store.get_entity_by_alias("dor") is not None

    def test_get_aliases_for_entity(self, store):
        """Should list all aliases for a given entity."""
        entity_id = _upsert(store, "person", "Etan Heyman")
        store.add_entity_alias("Etan", entity_id, alias_type="name")
        store.add_entity_alias("etanheyman", entity_id, alias_type="handle")

        aliases = store.get_entity_aliases(entity_id)
        alias_names = {a["alias"] for a in aliases}
        assert "Etan" in alias_names
        assert "etanheyman" in alias_names

    def test_duplicate_alias_ignored(self, store):
        """Adding the same alias twice should not error (idempotent)."""
        entity_id = _upsert(store, "person", "Etan Heyman")
        store.add_entity_alias("Etan", entity_id, alias_type="name")
        store.add_entity_alias("Etan", entity_id, alias_type="name")  # No error

        aliases = store.get_entity_aliases(entity_id)
        assert len(aliases) == 1


# ── Entity Resolution ──


class TestEntityResolution:
    """resolve_entity should find existing entities before creating new ones."""

    def test_exact_name_match(self, store):
        """Exact name match should resolve to existing entity."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        entity_id = _upsert(store, "person", "Dor Zohar")

        resolved = resolve_entity("Dor Zohar", "person", "", store)
        assert resolved == entity_id

    def test_alias_match(self, store):
        """Alias should resolve to the aliased entity."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        entity_id = _upsert(store, "person", "Etan Heyman")
        store.add_entity_alias("etanheyman", entity_id, alias_type="handle")

        resolved = resolve_entity("etanheyman", "person", "", store)
        assert resolved == entity_id

    def test_new_entity_created(self, store):
        """Unrecognized name should create a new entity."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        resolved = resolve_entity("Unknown Person", "person", "", store)
        assert resolved is not None

        entity = store.get_entity(resolved)
        assert entity is not None
        assert entity["name"] == "Unknown Person"

    def test_case_insensitive_resolution(self, store):
        """Resolution should be case-insensitive for both name and alias."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        entity_id = _upsert(store, "company", "Cantaloupe AI")

        # Different casing should resolve to same entity
        resolved = resolve_entity("cantaloupe ai", "company", "", store)
        assert resolved == entity_id


# ── Hebrew Prefix Stripping ──


class TestHebrewPrefixStripping:
    """Hebrew morphological prefixes should be stripped for matching."""

    def test_strip_prefix_bet(self):
        """Strip ב prefix."""
        from brainlayer.pipeline.entity_resolution import strip_hebrew_prefix

        assert strip_hebrew_prefix("בדומיקה") == "דומיקה"

    def test_strip_prefix_lamed(self):
        """Strip ל prefix."""
        from brainlayer.pipeline.entity_resolution import strip_hebrew_prefix

        assert strip_hebrew_prefix("לדור") == "דור"

    def test_strip_prefix_mem(self):
        """Strip מ prefix."""
        from brainlayer.pipeline.entity_resolution import strip_hebrew_prefix

        assert strip_hebrew_prefix("מדור") == "דור"

    def test_no_strip_short_words(self):
        """Don't strip if result would be too short (< 2 chars)."""
        from brainlayer.pipeline.entity_resolution import strip_hebrew_prefix

        # Single char after strip - don't strip
        assert strip_hebrew_prefix("בד") == "בד"

    def test_no_strip_non_hebrew(self):
        """Don't strip from non-Hebrew text."""
        from brainlayer.pipeline.entity_resolution import strip_hebrew_prefix

        assert strip_hebrew_prefix("Domica") == "Domica"

    def test_hebrew_alias_resolves(self, store):
        """Hebrew alias should resolve to the entity."""
        from brainlayer.pipeline.entity_resolution import resolve_entity

        entity_id = _upsert(store, "person", "Dor Zohar")
        store.add_entity_alias("דור זוהר", entity_id, alias_type="hebrew")

        resolved = resolve_entity("דור זוהר", "person", "", store)
        assert resolved == entity_id


# ── Merge Operation ──


class TestMergeEntities:
    """Merging duplicate entities should preserve all links."""

    def test_merge_moves_chunk_links(self, store):
        """After merge, all chunk links should point to the kept entity."""
        from brainlayer.pipeline.entity_resolution import merge_entities

        keep_id = _upsert(store, "person", "Etan Heyman")
        merge_id = _upsert(store, "person", "Etan")

        # Link chunks to the entity-to-be-merged
        store.link_entity_chunk(merge_id, "chunk-1", relevance=0.9)
        store.link_entity_chunk(merge_id, "chunk-2", relevance=0.8)
        store.link_entity_chunk(keep_id, "chunk-3", relevance=0.7)

        merge_entities(store, keep_id, merge_id)

        # All links should now point to keep_id
        cursor = store.conn.cursor()
        links = list(cursor.execute(
            "SELECT entity_id, chunk_id FROM kg_entity_chunks ORDER BY chunk_id"
        ))
        entity_ids = {l[0] for l in links}
        assert entity_ids == {keep_id}
        assert len(links) == 3

    def test_merge_moves_relations(self, store):
        """After merge, all relations should reference the kept entity."""
        from brainlayer.pipeline.entity_resolution import merge_entities

        keep_id = _upsert(store, "person", "Etan Heyman")
        merge_id = _upsert(store, "person", "Etan")
        company_id = _upsert(store, "company", "Domica")

        store.add_relation("rel-1", merge_id, company_id, "works_at")

        merge_entities(store, keep_id, merge_id)

        # Relation should now reference keep_id
        cursor = store.conn.cursor()
        rels = list(cursor.execute(
            "SELECT source_id, target_id, relation_type FROM kg_relations"
        ))
        assert len(rels) == 1
        assert rels[0][0] == keep_id
        assert rels[0][1] == company_id

    def test_merge_stores_alias(self, store):
        """Merged entity's name should become an alias of the kept entity."""
        from brainlayer.pipeline.entity_resolution import merge_entities

        keep_id = _upsert(store, "person", "Etan Heyman")
        merge_id = _upsert(store, "person", "Etan")

        merge_entities(store, keep_id, merge_id)

        # "Etan" should now be an alias
        result = store.get_entity_by_alias("Etan")
        assert result is not None
        assert result["id"] == keep_id

    def test_merge_deletes_duplicate(self, store):
        """The merged entity should be deleted."""
        from brainlayer.pipeline.entity_resolution import merge_entities

        keep_id = _upsert(store, "person", "Etan Heyman")
        merge_id = _upsert(store, "person", "Etan")

        merge_entities(store, keep_id, merge_id)

        assert store.get_entity(merge_id) is None
        assert store.get_entity(keep_id) is not None
