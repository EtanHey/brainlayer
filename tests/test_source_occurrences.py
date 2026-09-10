import hashlib
import json

import pytest

from brainlayer.vector_store import VectorStore


def _chunk(chunk_id, content, source_file, line, pointer):
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {"session_id": "session-a", "source_line": line, "source_pointer": pointer},
        "source_file": source_file,
        "project": "brainlayer",
        "content_type": "assistant_text",
        "value_type": "HIGH",
        "char_count": len(content),
        "source": "codex_cli",
        "created_at": "2026-09-10T10:00:00Z",
    }


def _occurrences(store):
    return list(
        store.conn.cursor().execute(
            "SELECT canonical_chunk_id, source_file, source_line, source_pointer, incoming_content_hash "
            "FROM chunk_source_occurrences ORDER BY source_file, source_line"
        )
    )


def test_duplicate_retains_two_source_occurrences_and_first_survivor_metadata(tmp_path):
    store = VectorStore(tmp_path / "brainlayer.db")
    content = "The identical content should retain both physical source occurrence coordinates."
    first = _chunk("first", content, "one.jsonl", 2, "/payload/item/content/0/text")
    second = _chunk("second", content, "two.jsonl", 3, "/payload/content")

    store.upsert_chunks([first, second], [[0.1] * 1024, [0.2] * 1024])

    assert store.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE archived_at IS NULL").fetchone()[0] == 1
    assert _occurrences(store) == [
        ("first", "one.jsonl", 2, "/payload/item/content/0/text", hashlib.sha256(content.encode()).hexdigest()),
        ("first", "two.jsonl", 3, "/payload/content", hashlib.sha256(content.encode()).hexdigest()),
    ]
    survivor_metadata = store.conn.cursor().execute("SELECT metadata FROM chunks WHERE id = 'first'").fetchone()[0]
    assert json.loads(survivor_metadata) == first["metadata"]


def test_occurrence_retry_is_idempotent_and_changed_content_at_same_coordinates_is_distinct(tmp_path):
    store = VectorStore(tmp_path / "brainlayer.db")
    first = _chunk("one", "A unique first content statement for exact coordinate handling.", "same.jsonl", 7, "/p")
    changed = _chunk("two", "A materially changed second statement at the same physical coordinate.", "same.jsonl", 7, "/p")

    store.upsert_chunks([first], [[0.1] * 1024])
    store.upsert_chunks([first], [[0.1] * 1024])
    store.upsert_chunks([changed], [[0.2] * 1024])

    assert len(_occurrences(store)) == 2
    assert {row[4] for row in _occurrences(store)} == {
        hashlib.sha256(first["content"].encode()).hexdigest(),
        hashlib.sha256(changed["content"].encode()).hexdigest(),
    }


def test_archived_canonical_can_retain_same_coordinate_for_new_canonical_and_hard_delete_cascades(tmp_path):
    store = VectorStore(tmp_path / "brainlayer.db")
    content = "The same coordinate may later identify a new canonical chunk after archive."
    first = _chunk("old", content, "same.jsonl", 7, "/p")
    second = _chunk("new", content, "same.jsonl", 7, "/p")

    store.upsert_chunks([first], [[0.1] * 1024])
    assert store.archive_chunk("old")
    store.upsert_chunks([second], [[0.2] * 1024])

    assert {row[0] for row in _occurrences(store)} == {"old", "new"}
    store.conn.cursor().execute("DELETE FROM chunks WHERE id = 'new'")
    assert {row[0] for row in _occurrences(store)} == {"old"}


def test_same_id_new_source_records_second_occurrence_and_incomplete_coordinates_skip(tmp_path):
    store = VectorStore(tmp_path / "brainlayer.db")
    content = "A same ID repost should retain its new source coordinate without a new chunk."
    first = _chunk("stable", content, "one.jsonl", 1, "/first")
    second = _chunk("stable", content, "two.jsonl", 9, "/second")
    incomplete = _chunk("incomplete", "Incomplete coordinates must not be invented for storage.", "missing.jsonl", None, "")

    store.upsert_chunks([first], [[0.1] * 1024])
    store.upsert_chunks([second], [[0.1] * 1024])
    store.upsert_chunks([incomplete], [[0.1] * 1024])

    assert [(row[1], row[2]) for row in _occurrences(store)] == [("one.jsonl", 1), ("two.jsonl", 9)]


def test_occurrence_failure_rolls_back_its_canonical_insert(tmp_path, monkeypatch):
    store = VectorStore(tmp_path / "brainlayer.db")
    chunk = _chunk("atomic", "Occurrence failure must roll back the chunk transaction as well.", "atomic.jsonl", 4, "/p")

    def fail_occurrence(*_args):
        raise RuntimeError("synthetic occurrence failure")

    monkeypatch.setattr(store, "_record_source_occurrence", fail_occurrence)
    with pytest.raises(RuntimeError, match="synthetic occurrence failure"):
        store.upsert_chunks([chunk], [[0.1] * 1024])

    assert store.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert _occurrences(store) == []
