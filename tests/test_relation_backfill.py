"""Relation coverage must not mistake entity links for completed extraction."""

import json
import sqlite3

import pytest

from brainlayer.pipeline.relation_backfill import backfill


@pytest.fixture
def db(tmp_path):
    conn = sqlite3.connect(tmp_path / "relations.db")
    conn.executescript("""
        CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT, created_at TEXT,
            archived_at TEXT, superseded_by TEXT, aggregated_into TEXT);
        CREATE TABLE kg_entities (id TEXT PRIMARY KEY, name TEXT, entity_type TEXT);
        CREATE TABLE kg_entity_chunks (entity_id TEXT, chunk_id TEXT);
        CREATE TABLE kg_relations (id TEXT PRIMARY KEY, source_id TEXT, target_id TEXT,
            relation_type TEXT, properties TEXT, confidence REAL, fact TEXT,
            source_chunk_id TEXT, expired_at TEXT, importance REAL,
            UNIQUE(source_id, target_id, relation_type));
        INSERT INTO chunks VALUES ('c1', 'Atlas uses SQLite for storage.', '2026-01-01', NULL, NULL, NULL);
        INSERT INTO kg_entities VALUES ('p', 'Atlas', 'project'), ('t', 'SQLite', 'technology');
        INSERT INTO kg_entity_chunks VALUES ('p', 'c1'), ('t', 'c1');
    """)
    yield conn
    conn.close()


def response(relations=None):
    return json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "relations": relations
                    if relations is not None
                    else [
                        {"source_id": "p", "target_id": "t", "type": "uses", "quote": "Atlas uses SQLite for storage."}
                    ],
                }
            ]
        }
    )


def test_linked_chunk_gets_grounded_relation_and_resume_skips_it(db):
    before = db.execute("SELECT * FROM chunks").fetchall()
    result = backfill(db, lambda _: response(), limit=5)
    assert result["relations_added"] == 1
    row = db.execute(
        "SELECT source_id, target_id, relation_type, source_chunk_id, fact, properties FROM kg_relations"
    ).fetchone()
    assert row[:5] == ("p", "t", "uses", "c1", "Atlas uses SQLite for storage.")
    assert json.loads(row[5])["evidence_quote"] == row[4]
    assert db.execute("SELECT * FROM chunks").fetchall() == before
    assert backfill(db, lambda _: pytest.fail("completed chunk retried"), limit=5)["chunks_processed"] == 0


def test_valid_empty_is_completed_but_failed_response_is_retryable(db):
    with pytest.raises(ValueError):
        backfill(db, lambda _: '{"chunks": []}', limit=1)
    assert db.execute("SELECT count(*) FROM kg_relation_backfill").fetchone()[0] == 0
    assert backfill(db, lambda _: response([]), limit=1)["chunks_processed"] == 1
    assert backfill(db, lambda _: pytest.fail("empty extraction retried"), limit=1)["chunks_processed"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        {"source_id": "unknown"},
        {"target_id": "p"},
        {"quote": "fabricated evidence"},
        {"quote": "SQLite"},
        {"type": "co_occurs_with"},
        {"type": "imagined_type"},
    ],
)
def test_invalid_relations_never_commit_or_mark_complete(db, mutation):
    rel = json.loads(response())["chunks"][0]["relations"][0] | mutation
    with pytest.raises(ValueError):
        backfill(db, lambda _: response([rel]), limit=1)
    assert db.execute("SELECT count(*) FROM kg_relations").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM kg_relation_backfill").fetchone()[0] == 0


def test_existing_expired_relation_is_never_revived_or_overwritten(db):
    db.execute("INSERT INTO kg_relations VALUES ('old','p','t','uses','{}',1,'old fact','old-source','2026-01-01',0.5)")
    db.commit()
    before = db.execute("SELECT * FROM kg_relations").fetchall()
    assert backfill(db, lambda _: response(), limit=1)["relations_added"] == 0
    assert db.execute("SELECT * FROM kg_relations").fetchall() == before


def test_changed_source_reprocessed_and_archived_source_excluded(db):
    backfill(db, lambda _: response([]), limit=1)
    db.execute("UPDATE chunks SET content = content || ' It is deployed.'")
    db.commit()
    assert backfill(db, lambda _: response(), limit=1)["relations_added"] == 1
    db.execute("DELETE FROM kg_relation_backfill")
    db.execute("UPDATE chunks SET archived_at='2026-02-01'")
    db.commit()
    assert backfill(db, lambda _: pytest.fail("archived source processed"), limit=1)["chunks_processed"] == 0


def test_source_change_during_call_rolls_back_relation_and_completion(db):
    def caller(_):
        db.execute("UPDATE chunks SET content='Atlas no longer uses SQLite.'")
        db.commit()
        return response()

    with pytest.raises(ValueError, match="changed"):
        backfill(db, caller, limit=1)
    assert db.execute("SELECT count(*) FROM kg_relations").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM kg_relation_backfill").fetchone()[0] == 0


def test_long_source_tail_is_processed_and_failed_window_is_retryable(db):
    text = "Atlas uses SQLite for storage." + " filler" * 1400 + " Atlas uses SQLite for storage."
    db.execute("UPDATE chunks SET content=?", (text,))
    db.commit()
    calls = []

    def caller(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError("second window failed")
        return response()

    with pytest.raises(RuntimeError):
        backfill(db, caller, limit=1)
    assert len(calls) == 2
    assert db.execute("SELECT count(*) FROM kg_relations").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM kg_relation_backfill").fetchone()[0] == 0
    result = backfill(db, lambda _: response(), limit=1)
    assert result["relations_added"] == 1
    assert result["windows_processed"] == 2


def test_semantically_invalid_endpoint_types_rejected_despite_exact_quote(db):
    db.execute("UPDATE kg_entities SET entity_type='concept' WHERE id='p'")
    db.commit()
    with pytest.raises(ValueError, match="supported endpoints"):
        backfill(db, lambda _: response(), limit=1)
    assert db.execute("SELECT count(*) FROM kg_relations").fetchone()[0] == 0


def test_entity_type_correction_invalidates_completion_without_source_change(db):
    db.execute("UPDATE kg_entities SET entity_type='concept' WHERE id='p'")
    db.commit()
    backfill(db, lambda _: response([]), limit=1)
    db.execute("UPDATE kg_entities SET entity_type='project' WHERE id='p'")
    db.commit()
    assert backfill(db, lambda _: response(), limit=1)["relations_added"] == 1


def test_type_change_during_inference_rolls_back(db):
    def caller(_):
        db.execute("UPDATE kg_entities SET entity_type='concept' WHERE id='p'")
        db.commit()
        return response()

    with pytest.raises(ValueError, match="Entities changed"):
        backfill(db, caller, limit=1)
    assert db.execute("SELECT count(*) FROM kg_relations").fetchone()[0] == 0


def test_entity_name_substring_is_not_a_mention(db):
    db.execute("UPDATE kg_entities SET name='Ann', entity_type='person' WHERE id='p'")
    db.execute("UPDATE chunks SET content='Anna uses SQLite.'")
    db.commit()
    assert backfill(db, lambda _: pytest.fail("Ann is not mentioned"), limit=1)["chunks_processed"] == 0
