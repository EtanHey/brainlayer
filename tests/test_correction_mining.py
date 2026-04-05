"""Tests for correction mining pipeline."""

from __future__ import annotations

import json

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


def _insert_chunk(store: VectorStore, chunk_id: str, content: str, tags: list[str]) -> None:
    store.conn.cursor().execute(
        """
        INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, tags, created_at
        ) VALUES (?, ?, '{}', 'test.jsonl', 'brainlayer', 'note', ?, 'tests', ?, datetime('now'))
        """,
        (
            chunk_id,
            content,
            len(content),
            json.dumps(tags),
        ),
    )


def test_extract_identity():
    from brainlayer.pipeline.correction_mining import extract_corrections

    matches = extract_corrections("Avi Simon is a developer.")

    assert len(matches) == 1
    assert matches[0].pattern_type == "identity"
    assert matches[0].entity_name == "Avi Simon"
    assert matches[0].attribute == "developer"
    assert matches[0].old_value is None
    assert matches[0].new_value == "developer"


def test_extract_negation():
    from brainlayer.pipeline.correction_mining import extract_corrections

    matches = extract_corrections("Avi is not from Wix.")

    assert len(matches) == 1
    assert matches[0].pattern_type == "negation"
    assert matches[0].entity_name == "Avi"
    assert matches[0].old_value == "from Wix"
    assert matches[0].new_value is None


def test_extract_association():
    from brainlayer.pipeline.correction_mining import extract_corrections

    matches = extract_corrections("Avi works at Lightricks.")

    assert len(matches) == 1
    assert matches[0].pattern_type == "association"
    assert matches[0].entity_name == "Avi"
    assert matches[0].attribute == "works_at"
    assert matches[0].new_value == "Lightricks"


def test_extract_alias():
    from brainlayer.pipeline.correction_mining import extract_corrections

    matches = extract_corrections("EtanHey aka Etan Heyman")

    assert len(matches) == 1
    assert matches[0].pattern_type == "alias"
    assert matches[0].entity_name == "Etan Heyman"
    assert matches[0].old_value == "EtanHey"
    assert matches[0].new_value == "Etan Heyman"


def test_extract_hebrew():
    from brainlayer.pipeline.correction_mining import extract_corrections

    matches = extract_corrections("אבי הוא מפתח")

    assert len(matches) == 1
    assert matches[0].pattern_type == "hebrew_identity"
    assert matches[0].entity_name == "אבי"
    assert matches[0].new_value == "מפתח"


def test_mine_corrections_from_chunks(store):
    from brainlayer.pipeline.correction_mining import mine_corrections, promote_corrections

    _insert_chunk(store, "chunk-1", "Avi Simon is a developer.", ["correction"])
    _insert_chunk(store, "chunk-2", "Avi works at Lightricks.", ["clarification"])
    _insert_chunk(store, "chunk-3", "EtanHey aka Etan Heyman", ["user-correction"])
    _insert_chunk(store, "chunk-4", "Avi is not from Wix.", ["correction"])

    stats = mine_corrections(store)

    assert stats["chunks_processed"] == 4
    assert stats["pairs_extracted"] == 4
    assert stats["by_pattern_type"] == {
        "alias": 1,
        "association": 1,
        "identity": 1,
        "negation": 1,
    }

    promotion = promote_corrections(store, min_confidence=0.8)

    assert promotion["entities_upserted"] >= 2
    assert promotion["relations_created"] == 1
    assert promotion["aliases_created"] == 1
    assert promotion["manual_review"] == 1

    person = store.get_entity_by_name("person", "Avi Simon")
    assert person is not None
    assert person["metadata"]["identity"] == "developer"

    relations = store.get_entity_relations(person["id"], direction="outgoing")
    assert any(rel["relation_type"] == "works_at" and rel["target_name"] == "Lightricks" for rel in relations)

    alias_rows = list(
        store._read_cursor().execute(
            "SELECT alias, alias_type FROM kg_entity_aliases WHERE entity_id = ?",
            (store.get_entity_by_name("person", "Etan Heyman")["id"],),
        )
    )
    assert ("EtanHey", "correction") in alias_rows


def test_correction_pairs_table(store):
    cols = {row[1]: row[2] for row in store._read_cursor().execute("PRAGMA table_info(correction_pairs)")}

    assert cols == {
        "id": "INTEGER",
        "chunk_id": "TEXT",
        "pattern_type": "TEXT",
        "entity_name": "TEXT",
        "attribute": "TEXT",
        "old_value": "TEXT",
        "new_value": "TEXT",
        "confidence": "REAL",
        "created_at": "TEXT",
    }


def test_no_false_positives():
    from brainlayer.pipeline.correction_mining import extract_corrections

    assert extract_corrections("this is great") == []
    assert extract_corrections("it is not ideal") == []
    assert extract_corrections("working at home today") == []
