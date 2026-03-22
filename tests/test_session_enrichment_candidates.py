"""Tests for SessionMixin.get_enrichment_candidates()."""

from datetime import datetime, timedelta, timezone

from brainlayer.vector_store import VectorStore


def _dummy_embed(text):  # noqa: ARG001
    return [0.1] * 1024


def _insert_chunk(
    store: VectorStore,
    chunk_id: str,
    content: str,
    *,
    created_at: str | None = None,
    source: str = "claude_code",
    content_type: str = "assistant_text",
) -> None:
    store.upsert_chunks(
        [
            {
                "id": chunk_id,
                "content": content,
                "metadata": {},
                "source_file": "test.jsonl",
                "project": "brainlayer",
                "content_type": content_type,
                "char_count": len(content),
                "source": source,
                "created_at": created_at,
            }
        ],
        [_dummy_embed(content)],
    )


def test_get_enrichment_candidates_returns_only_unenriched_chunks(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    _insert_chunk(store, "c1", "a" * 80)
    _insert_chunk(store, "c2", "b" * 80)

    results = store.get_enrichment_candidates(limit=10)

    assert {row["id"] for row in results} == {"c1", "c2"}


def test_get_enrichment_candidates_skips_already_enriched_chunks(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    _insert_chunk(store, "c1", "a" * 80)
    _insert_chunk(store, "c2", "b" * 80)
    store.update_enrichment("c2", summary="done", tags=["python"])

    results = store.get_enrichment_candidates(limit=10)

    assert [row["id"] for row in results] == ["c1"]


def test_get_enrichment_candidates_skips_short_chunks(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    _insert_chunk(store, "short", "tiny")
    _insert_chunk(store, "long", "x" * 80)

    results = store.get_enrichment_candidates(limit=10, min_content_length=50)

    assert [row["id"] for row in results] == ["long"]


def test_get_enrichment_candidates_honors_since_hours_filter(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    recent = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    _insert_chunk(store, "old", "o" * 80, created_at=old)
    _insert_chunk(store, "recent", "r" * 80, created_at=recent)

    results = store.get_enrichment_candidates(limit=10, since_hours=24)

    assert [row["id"] for row in results] == ["recent"]


def test_get_enrichment_candidates_honors_explicit_chunk_ids_filter(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    _insert_chunk(store, "c1", "a" * 80)
    _insert_chunk(store, "c2", "b" * 80)
    _insert_chunk(store, "c3", "c" * 80)

    results = store.get_enrichment_candidates(limit=10, chunk_ids=["c1", "c3"])

    assert {row["id"] for row in results} == {"c1", "c3"}


def test_get_enrichment_candidates_respects_limit(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    for idx in range(5):
        _insert_chunk(store, f"c{idx}", "x" * 80)

    results = store.get_enrichment_candidates(limit=2)

    assert len(results) == 2


def test_get_enrichment_candidates_returns_empty_list_when_nothing_needs_enrichment(tmp_path):
    store = VectorStore(tmp_path / "test.db")
    _insert_chunk(store, "c1", "a" * 80)
    store.update_enrichment("c1", summary="done", tags=["python"])

    results = store.get_enrichment_candidates(limit=10)

    assert results == []
