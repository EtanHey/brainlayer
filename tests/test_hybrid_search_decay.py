import apsw
import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.search_repo import _hybrid_cache
from brainlayer.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clear_hybrid_cache():
    _hybrid_cache.clear()
    yield
    _hybrid_cache.clear()


@pytest.fixture
def store(tmp_path):
    database = VectorStore(tmp_path / "hybrid-decay.db")
    yield database
    database.close()


def _embed(text: str) -> list[float]:
    seed = (sum(ord(c) for c in text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    embedding: list[float],
    decay_score: float = 1.0,
    half_life_days: float = 30.0,
    retrieval_count: int = 0,
):
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, decay_score, half_life_days, retrieval_count, created_at
        ) VALUES (?, ?, '{}', 'test.jsonl', 'hybrid-decay', 'assistant_text', ?, 'claude_code', ?, ?, ?, '2026-04-05T00:00:00Z')""",
        (chunk_id, content, len(content), decay_score, half_life_days, retrieval_count),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )


def test_hybrid_search_decay_score_changes_ranking(store):
    query_embedding = _embed("decay ranking query")
    _insert_chunk(
        store,
        chunk_id="fresh",
        content="decay ranking query fresh result",
        embedding=query_embedding,
        decay_score=1.0,
    )
    _insert_chunk(
        store,
        chunk_id="stale",
        content="decay ranking query stale result",
        embedding=[value + 0.00001 for value in query_embedding],
        decay_score=0.05,
    )

    results = store.hybrid_search(
        query_embedding=query_embedding,
        query_text="decay ranking query",
        n_results=2,
    )

    assert results["ids"][0][0] == "fresh"


def test_hybrid_search_queues_retrieval_strengthening_until_flush_threshold(store):
    store._retrieval_strengthening_flush_threshold = 1000
    query_embedding = _embed("strengthen later")
    _insert_chunk(store, chunk_id="target", content="strengthen later result", embedding=query_embedding)

    store.hybrid_search(
        query_embedding=query_embedding,
        query_text="strengthen later",
        n_results=1,
    )

    row = (
        store.conn.cursor()
        .execute(
            "SELECT retrieval_count, last_retrieved, half_life_days FROM chunks WHERE id = ?",
            ("target",),
        )
        .fetchone()
    )
    assert row == (0, None, 30.0)
    assert store._retrieval_strengthening_pending["target"]["retrieval_count_delta"] == 1


def test_flush_retrieval_strengthening_updates_chunk_state(store):
    store._retrieval_strengthening_flush_threshold = 1000
    query_embedding = _embed("strengthen now")
    _insert_chunk(store, chunk_id="target", content="strengthen now result", embedding=query_embedding)

    store.hybrid_search(
        query_embedding=query_embedding,
        query_text="strengthen now",
        n_results=1,
    )
    store.flush_retrieval_strengthening_updates(now=1_234_567.0)

    row = (
        store.conn.cursor()
        .execute(
            "SELECT retrieval_count, last_retrieved, half_life_days FROM chunks WHERE id = ?",
            ("target",),
        )
        .fetchone()
    )
    assert row == (1, 1_234_567.0, 33.0)


def test_flush_retrieval_strengthening_keeps_queue_on_busy_error(store, monkeypatch):
    query_embedding = _embed("busy queue")
    _insert_chunk(store, chunk_id="target", content="busy queue result", embedding=query_embedding)
    store.hybrid_search(
        query_embedding=query_embedding,
        query_text="busy queue",
        n_results=1,
    )

    monkeypatch.setattr(
        store,
        "_apply_retrieval_strengthening_updates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(apsw.BusyError("database is locked")),
    )

    store.flush_retrieval_strengthening_updates(now=1_234_567.0)

    assert "target" in store._retrieval_strengthening_pending


def test_retrieval_strengthening_updates_top_10_only(store):
    store._retrieval_strengthening_flush_threshold = 1
    query_embedding = _embed("top ten query")

    for index in range(12):
        _insert_chunk(
            store,
            chunk_id=f"chunk-{index}",
            content=f"top ten query candidate {index}",
            embedding=[value + (index * 0.00001) for value in query_embedding],
        )

    store.hybrid_search(
        query_embedding=query_embedding,
        query_text="top ten query",
        n_results=12,
    )

    strengthened = {
        row[0]: row[1] for row in store.conn.cursor().execute("SELECT id, retrieval_count FROM chunks ORDER BY id")
    }
    assert sum(1 for value in strengthened.values() if value == 1) == 10
