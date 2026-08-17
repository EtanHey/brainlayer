"""Repair (c): archive is archived_at (time or NULL). No flag/status/value_type twin."""

from datetime import datetime

import pytest

from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


def _store_chunk(store, mock_embed, content, **kwargs):
    return store_memory(
        store=store,
        embed_fn=mock_embed,
        content=content,
        memory_type=kwargs.pop("memory_type", "learning"),
        **kwargs,
    )


def _row(store, chunk_id):
    return (
        store.conn.cursor()
        .execute(
            "SELECT archived_at, archived, status, value_type FROM chunks WHERE id = ?",
            (chunk_id,),
        )
        .fetchone()
    )


class TestArchiveWriterSpeaksArchivedAtOnly:
    def test_archive_chunk_sets_timestamp_without_flag_status_or_value_type(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "Collapse archive writer")
        before = _row(store, result["id"])
        assert before[0] is None
        assert before[1] in (0, None)
        assert before[2] in ("active", None)
        assert (before[3] or "").lower() != "archived"

        store.archive_chunk(result["id"])

        after = _row(store, result["id"])
        assert after[0]
        datetime.fromisoformat(after[0])
        assert after[1] in (0, None)
        assert after[2] != "archived"
        assert (after[3] or "").lower() != "archived"
        assert after[3] == before[3]

    def test_get_chunk_hides_archived_at_even_when_other_flags_are_clear(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "Hidden only by archived_at")
        store.archive_chunk(result["id"])
        store.conn.cursor().execute(
            "UPDATE chunks SET archived = 0, status = 'active', value_type = 'high' WHERE id = ?",
            (result["id"],),
        )
        assert store.get_chunk(result["id"]) is None
        assert store.get_chunk(result["id"], include_archived=True) is not None


class TestDefaultSearchUsesArchivedAtOnly:
    def test_archived_at_is_hidden_and_include_archived_shows_it(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "UniqueCollapseToken archived_at hide")
        store.archive_chunk(result["id"])
        hidden = store.search(query_text="UniqueCollapseToken")
        shown = store.search(query_text="UniqueCollapseToken", include_archived=True)
        hidden_ids = hidden["ids"][0] if hidden["ids"] else []
        shown_ids = shown["ids"][0] if shown["ids"] else []
        assert result["id"] not in hidden_ids
        assert result["id"] in shown_ids

    def test_flag_without_timestamp_is_not_lifecycle_managed(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "UniqueFlagOnlyToken still searchable")
        store.conn.cursor().execute(
            "UPDATE chunks SET archived = 1, archived_at = NULL, status = 'active' WHERE id = ?",
            (result["id"],),
        )
        results = store.search(query_text="UniqueFlagOnlyToken")
        ids = results["ids"][0] if results["ids"] else []
        assert result["id"] in ids

    def test_status_archived_without_timestamp_is_not_lifecycle_managed(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "UniqueStatusOnlyToken still searchable")
        store.conn.cursor().execute(
            "UPDATE chunks SET status = 'archived', archived = 0, archived_at = NULL WHERE id = ?",
            (result["id"],),
        )
        results = store.search(query_text="UniqueStatusOnlyToken")
        ids = results["ids"][0] if results["ids"] else []
        assert result["id"] in ids

    def test_value_type_archived_without_timestamp_is_not_lifecycle_managed(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "UniqueValueOnlyToken still searchable")
        store.conn.cursor().execute(
            "UPDATE chunks SET value_type = 'ARCHIVED', archived = 0, archived_at = NULL, status = 'active' WHERE id = ?",
            (result["id"],),
        )
        results = store.search(query_text="UniqueValueOnlyToken")
        ids = results["ids"][0] if results["ids"] else []
        assert result["id"] in ids
