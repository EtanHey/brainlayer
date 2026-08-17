"""Default search lifecycle filter is archived_at + lineage, not flag/status twins."""

from datetime import datetime, timedelta, timezone

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


def _embed(text: str) -> list[float]:
    seed = sum(ord(c) for c in text[:50]) % 100
    return [float(seed + i) / 1000.0 for i in range(1024)]


def test_value_type_archived_survives_store_reopen(tmp_path):
    db = tmp_path / "t.db"
    s = VectorStore(db)
    cid = store_memory(store=s, embed_fn=_embed, content="ReopenProbeToken", memory_type="learning")["id"]
    s.conn.cursor().execute(
        "UPDATE chunks SET value_type='ARCHIVED', archived=0, archived_at=NULL, status='active' WHERE id=?", (cid,)
    )
    s.close()
    s2 = VectorStore(db)  # startup backfill runs here
    row = list(s2.conn.cursor().execute("SELECT archived FROM chunks WHERE id=?", (cid,)))[0]
    ids = s2.search(query_text="ReopenProbeToken")["ids"]
    assert row[0] in (0, None)
    assert cid in (ids[0] if ids else [])
    s2.close()


def test_hybrid_search_fts_hides_archived_at_and_include_archived_shows_it(store, mock_embed):
    result = _store_chunk(store, mock_embed, "HybridFtsArchiveToken unique fts hide")
    store.archive_chunk(result["id"])
    hidden = store.hybrid_search(
        query_embedding=None,
        query_text="HybridFtsArchiveToken",
        n_results=10,
        filter_meta_noise=False,
    )
    shown = store.hybrid_search(
        query_embedding=None,
        query_text="HybridFtsArchiveToken",
        n_results=10,
        include_archived=True,
        filter_meta_noise=False,
    )
    hidden_ids = hidden["ids"][0] if hidden["ids"] else []
    shown_ids = shown["ids"][0] if shown["ids"] else []
    assert result["id"] not in hidden_ids
    assert result["id"] in shown_ids


def test_hybrid_search_recent_fallback_hides_archived_at(store, mock_embed):
    now = datetime.now(timezone.utc)
    archived = _store_chunk(
        store,
        mock_embed,
        "HybridRecentHiddenToken stays out of recency",
        created_at=(now - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
    )
    active = _store_chunk(
        store,
        mock_embed,
        "HybridRecentVisibleToken stays in recency",
        created_at=(now - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
    )
    store.archive_chunk(archived["id"])
    hidden = store.hybrid_search(
        query_embedding=None,
        query_text="today",
        recency_rerank=True,
        n_results=25,
        filter_meta_noise=False,
    )
    shown = store.hybrid_search(
        query_embedding=None,
        query_text="today",
        recency_rerank=True,
        n_results=25,
        include_archived=True,
        filter_meta_noise=False,
    )
    hidden_ids = hidden["ids"][0] if hidden["ids"] else []
    shown_ids = shown["ids"][0] if shown["ids"] else []
    assert active["id"] in hidden_ids
    assert archived["id"] not in hidden_ids
    assert archived["id"] in shown_ids
