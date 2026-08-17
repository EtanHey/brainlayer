"""Default search lifecycle filter is archived_at + lineage, not flag/status twins."""

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


def _store_chunk(store, mock_embed, content):
    return store_memory(
        store=store,
        embed_fn=mock_embed,
        content=content,
        memory_type="learning",
    )


def test_archived_at_is_hidden_and_include_archived_shows_it(store, mock_embed):
    result = _store_chunk(store, mock_embed, "UniqueCollapseToken archived_at hide")
    store.archive_chunk(result["id"])
    hidden = store.search(query_text="UniqueCollapseToken")
    shown = store.search(query_text="UniqueCollapseToken", include_archived=True)
    hidden_ids = hidden["ids"][0] if hidden["ids"] else []
    shown_ids = shown["ids"][0] if shown["ids"] else []
    assert result["id"] not in hidden_ids
    assert result["id"] in shown_ids


def test_flag_without_timestamp_is_not_lifecycle_managed(store, mock_embed):
    result = _store_chunk(store, mock_embed, "UniqueFlagOnlyToken still searchable")
    store.conn.cursor().execute(
        "UPDATE chunks SET archived = 1, archived_at = NULL, status = 'active' WHERE id = ?",
        (result["id"],),
    )
    results = store.search(query_text="UniqueFlagOnlyToken")
    ids = results["ids"][0] if results["ids"] else []
    assert result["id"] in ids


def test_status_archived_without_timestamp_is_not_lifecycle_managed(store, mock_embed):
    result = _store_chunk(store, mock_embed, "UniqueStatusOnlyToken still searchable")
    store.conn.cursor().execute(
        "UPDATE chunks SET status = 'archived', archived = 0, archived_at = NULL WHERE id = ?",
        (result["id"],),
    )
    results = store.search(query_text="UniqueStatusOnlyToken")
    ids = results["ids"][0] if results["ids"] else []
    assert result["id"] in ids


def test_value_type_archived_without_timestamp_is_not_lifecycle_managed(store, mock_embed):
    result = _store_chunk(store, mock_embed, "UniqueValueOnlyToken still searchable")
    store.conn.cursor().execute(
        "UPDATE chunks SET value_type = 'ARCHIVED', archived = 0, archived_at = NULL, status = 'active' WHERE id = ?",
        (result["id"],),
    )
    results = store.search(query_text="UniqueValueOnlyToken")
    ids = results["ids"][0] if results["ids"] else []
    assert result["id"] in ids
