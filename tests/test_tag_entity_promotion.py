"""Tests for tag-to-entity promotion pipeline."""

import json

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _insert_chunk_with_tags(store, chunk_id, tags):
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, tags, created_at
        ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'assistant_text', ?, 'tests', ?, datetime('now'))""",
        (
            chunk_id,
            f"content for {chunk_id}",
            len(chunk_id),
            json.dumps(tags),
        ),
    )


class TestTagPromotionHeuristics:
    def test_classify_tag_entity_type_uses_spec_heuristics(self):
        from brainlayer.pipeline.tag_entity_promotion import classify_tag_entity_type

        assert classify_tag_entity_type("telegram") == "technology"
        assert classify_tag_entity_type("andrew-huberman") == "person"
        assert classify_tag_entity_type("neuroscience") == "topic"
        assert classify_tag_entity_type("hebrew-writing") == "topic"
        assert classify_tag_entity_type("founders-community") == "community"
        assert classify_tag_entity_type("morning-routine") == "topic"


class TestTagPromotionCandidates:
    def test_find_candidates_skips_existing_and_activity_tags(self, store):
        from brainlayer.pipeline.tag_entity_promotion import find_promotion_candidates

        _insert_chunk_with_tags(store, "chunk-1", ["telegram", "debugging", "existing-topic"])
        _insert_chunk_with_tags(store, "chunk-2", ["telegram", "debugging", "existing-topic"])
        store.upsert_entity("existing-topic", "topic", "existing-topic")

        candidates = find_promotion_candidates(store, min_count=2)

        assert [candidate["tag"] for candidate in candidates] == ["telegram"]


class TestTagPromotionExecution:
    def test_promote_tag_candidates_creates_entities_and_links_chunks(self, store):
        from brainlayer.pipeline.tag_entity_promotion import promote_tag_entities

        _insert_chunk_with_tags(store, "chunk-1", ["telegram", "feature-dev"])
        _insert_chunk_with_tags(store, "chunk-2", ["telegram"])
        _insert_chunk_with_tags(store, "chunk-3", ["neuroscience"])
        _insert_chunk_with_tags(store, "chunk-4", ["neuroscience"])

        stats = promote_tag_entities(store, min_count=2)

        assert stats["candidates"] == 2
        assert stats["entities_created"] == 2
        assert stats["links_created"] == 4

        cursor = store._read_cursor()
        entities = {
            row[0]: row[1]
            for row in cursor.execute(
                "SELECT name, entity_type FROM kg_entities WHERE id LIKE 'auto-tag-%'"
            )
        }
        assert entities["telegram"] == "technology"
        assert entities["neuroscience"] == "topic"

        links = list(
            cursor.execute(
                "SELECT entity_id, chunk_id, mention_type FROM kg_entity_chunks WHERE entity_id LIKE 'auto-tag-%'"
            )
        )
        assert len(links) == 4
        assert {row[2] for row in links} == {"tag"}

    def test_vector_store_seeds_new_entity_types(self, store):
        cursor = store._read_cursor()
        rows = list(
            cursor.execute(
                "SELECT child_type, parent_type FROM entity_type_hierarchy WHERE child_type IN (?, ?, ?, ?, ?, ?)",
                ("topic", "protocol", "community", "health_metric", "workflow", "device"),
            )
        )

        assert dict(rows) == {
            "topic": "concept",
            "protocol": "topic",
            "community": "entity",
            "health_metric": "topic",
            "workflow": "concept",
            "device": "entity",
        }
