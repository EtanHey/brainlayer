import concurrent.futures
import threading
import time

import apsw
import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.mcp import _shared, search_handler
from brainlayer.search_repo import _hybrid_cache
from brainlayer.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clear_hybrid_cache():
    _shared._close_search_vector_store()
    _shared._vector_store = None
    _hybrid_cache.clear()
    yield
    _shared._close_search_vector_store()
    if _shared._vector_store is not None:
        _shared._vector_store.close()
    _shared._search_vector_store = None
    _shared._vector_store = None
    _hybrid_cache.clear()


def _embed(text: str) -> list[float]:
    seed = (sum(ord(c) for c in text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


class FakeEmbeddingModel:
    def embed_query(self, _query: str) -> list[float]:
        return _embed("pooled handler search")


def _minimal_search_results(chunk_id: str) -> dict:
    return {
        "ids": [[chunk_id]],
        "documents": [["pooled handler search result"]],
        "metadatas": [[{"source_file": "pooled.md", "project": "brainlayer"}]],
        "distances": [[0.25]],
    }


def _create_vector_db(db_path):
    store = VectorStore(db_path)
    store.close()


def _insert_chunk(
    store: VectorStore,
    *,
    chunk_id: str,
    content: str,
    embedding: list[float],
):
    cursor = store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type,
            char_count, source, decay_score, half_life_days, retrieval_count, created_at
        ) VALUES (?, ?, '{}', 'readonly.jsonl', 'readonly', 'assistant_text', ?, 'claude_code', 1.0, 30.0, 0, '2026-04-05T00:00:00Z')""",
        (chunk_id, content, len(content)),
    )
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )


def test_open_readonly_skips_init_retry(tmp_path, monkeypatch):
    db_path = tmp_path / "readonly.db"
    _create_vector_db(db_path)

    init_retry_calls = []

    def fail_init_retry(self):
        init_retry_calls.append(self.db_path)
        raise AssertionError("_init_db_with_retry should not run for readonly stores")

    monkeypatch.setattr(VectorStore, "_init_db_with_retry", fail_init_retry)

    store = VectorStore(db_path, readonly=True)
    try:
        assert store._readonly is True
        assert init_retry_calls == []
    finally:
        store.close()


def test_readonly_rejects_writes(tmp_path):
    db_path = tmp_path / "readonly.db"
    _create_vector_db(db_path)

    store = VectorStore(db_path, readonly=True)
    try:
        with pytest.raises(apsw.ReadOnlyError):
            store.conn.cursor().execute(
                "INSERT INTO chunks (id, content, metadata, source_file) VALUES ('x', 'x', '{}', 'x')"
            )
    finally:
        store.close()


def test_readonly_skips_strengthening(tmp_path):
    db_path = tmp_path / "readonly.db"
    query_embedding = _embed("readonly strengthening")

    writer = VectorStore(db_path)
    try:
        _insert_chunk(
            writer,
            chunk_id="target",
            content="readonly strengthening result",
            embedding=query_embedding,
        )
    finally:
        writer.close()

    store = VectorStore(db_path, readonly=True)
    try:
        store._retrieval_strengthening_flush_threshold = 1
        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="readonly strengthening",
            n_results=1,
        )
        assert results["ids"][0] == ["target"]
    finally:
        store.close()

    inspector = VectorStore(db_path)
    try:
        row = (
            inspector.conn.cursor()
            .execute("SELECT retrieval_count, last_retrieved FROM chunks WHERE id = 'target'")
            .fetchone()
        )
        assert row == (0, None)
    finally:
        inspector.close()


def test_readonly_busyerror_resilience(tmp_path, monkeypatch):
    db_path = tmp_path / "readonly.db"
    _create_vector_db(db_path)

    def fail_init_retry(self):
        raise AssertionError("_init_db_with_retry should not run while opening readonly under write contention")

    monkeypatch.setattr(VectorStore, "_init_db_with_retry", fail_init_retry)

    writer_conn = apsw.Connection(str(db_path))
    writer_cursor = writer_conn.cursor()
    writer_cursor.execute("BEGIN IMMEDIATE")
    try:
        store = VectorStore(db_path, readonly=True)
        try:
            assert store._readonly is True
            assert store.conn.cursor().execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0] > 0
        finally:
            store.close()
    finally:
        writer_cursor.execute("ROLLBACK")
        writer_conn.close()


def test_explicit_readonly_does_not_create_parent_directory(tmp_path, monkeypatch):
    db_path = tmp_path / "missing-parent" / "readonly.db"

    def fake_init_readonly(self):
        self._local = None

    monkeypatch.setattr(VectorStore, "_init_readonly_db", fake_init_readonly)

    VectorStore(db_path, readonly=True)

    assert not db_path.parent.exists()


def test_search_vector_store_bootstraps_missing_db_then_reopens_readonly(tmp_path, monkeypatch):
    db_path = tmp_path / "fresh" / "brainlayer.db"
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "2")

    store = _shared._get_search_vector_store()
    try:
        assert db_path.exists()
        assert store._readonly is True
        assert store.count() == 0
        with pytest.raises(apsw.ReadOnlyError):
            store.conn.cursor().execute(
                "INSERT INTO chunks (id, content, metadata, source_file) VALUES ('x', 'x', '{}', 'x')"
            )
    finally:
        _shared._close_search_vector_store()


def test_search_vector_store_bootstraps_stale_schema_then_reopens_readonly(tmp_path, monkeypatch):
    db_path = tmp_path / "stale.db"
    conn = apsw.Connection(str(db_path))
    conn.cursor().execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT NOT NULL,
            source_file TEXT NOT NULL,
            project TEXT,
            content_type TEXT,
            value_type TEXT,
            char_count INTEGER,
            source TEXT,
            sender TEXT,
            language TEXT,
            conversation_id TEXT,
            position INTEGER,
            context_summary TEXT,
            chunk_origin TEXT DEFAULT 'unknown'
        )
        """
    )
    conn.close()
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "2")

    store = _shared._get_search_vector_store()
    try:
        columns = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        assert {"status", "archived", "summary", "resolved_queries", "chunk_origin"}.issubset(columns)
        assert store._readonly is True
        with pytest.raises(apsw.ReadOnlyError):
            store.conn.cursor().execute(
                "INSERT INTO chunks (id, content, metadata, source_file) VALUES ('x', 'x', '{}', 'x')"
            )
    finally:
        _shared._close_search_vector_store()


def test_search_store_bootstrap_required_for_partial_kg_schema(tmp_path):
    """regression-guard: readonly search must bootstrap when only one KG table exists."""
    db_path = tmp_path / "partial-kg.db"
    store = VectorStore(db_path)
    store.close()

    conn = apsw.Connection(str(db_path))
    try:
        conn.cursor().execute("DROP TABLE kg_relations")
    finally:
        conn.close()

    assert _shared._search_store_needs_bootstrap(db_path) is True


def test_search_store_pool_preopens_fixed_readonly_handles(tmp_path, monkeypatch):
    db_path = tmp_path / "pooled.db"
    _create_vector_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "3")

    first = _shared._get_search_vector_store()

    assert first._readonly is True
    assert _shared._search_vector_store is first
    assert len(_shared._search_vector_store_pool_handles) == 3
    assert {id(store) for store in _shared._search_vector_store_pool_handles} == {
        id(store) for store in _shared._search_vector_store_pool.queue
    }
    assert all(store._readonly for store in _shared._search_vector_store_pool_handles)


def test_search_store_checkout_deserializes_slow_reads(tmp_path, monkeypatch):
    db_path = tmp_path / "parallel.db"
    _create_vector_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "3")

    start_gate = threading.Barrier(4)
    release_gate = threading.Event()
    seen_store_ids: set[int] = set()
    lock = threading.Lock()

    def slow_read() -> None:
        with _shared._search_store_checkout() as store:
            with lock:
                seen_store_ids.add(id(store))
            start_gate.wait(timeout=1.0)
            assert release_gate.wait(timeout=1.0)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(slow_read) for _ in range(3)]
        start_gate.wait(timeout=1.0)
        elapsed_before_release = time.perf_counter() - started
        release_gate.set()
        for future in futures:
            future.result(timeout=1.0)

    assert len(seen_store_ids) == 3
    assert elapsed_before_release < 0.5


def test_search_handler_uses_distinct_pool_handles_for_concurrent_slow_reads(tmp_path, monkeypatch):
    db_path = tmp_path / "handler-pool.db"
    _create_vector_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "3")
    monkeypatch.setattr(search_handler, "_get_embedding_model", lambda: FakeEmbeddingModel())

    _shared._get_search_vector_store()
    start_gate = threading.Barrier(4)
    release_gate = threading.Event()
    seen_store_ids: set[int] = set()
    lock = threading.Lock()

    for handle in _shared._search_vector_store_pool_handles:
        handle.count = lambda: 1
        handle.enrich_results_with_session_context = lambda results: results

        def slow_search(*, _handle=handle, **_kwargs):
            with lock:
                seen_store_ids.add(id(_handle))
            start_gate.wait(timeout=1.0)
            assert release_gate.wait(timeout=1.0)
            return _minimal_search_results(f"chunk-{id(_handle)}")

        handle.hybrid_search = slow_search

    def run_search() -> None:
        import asyncio

        asyncio.run(search_handler._search(query="pooled handler search"))

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_search) for _ in range(3)]
        start_gate.wait(timeout=1.0)
        elapsed_before_release = time.perf_counter() - started
        release_gate.set()
        for future in futures:
            future.result(timeout=1.0)

    assert len(seen_store_ids) == 3
    assert elapsed_before_release < 0.5


def test_search_store_checkout_beyond_pool_blocks_then_raises(tmp_path, monkeypatch):
    db_path = tmp_path / "bounded.db"
    _create_vector_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "2")
    monkeypatch.setenv("BRAINLAYER_READ_BUSY_TIMEOUT_MS", "120")

    with _shared._search_store_checkout(), _shared._search_store_checkout():
        started = time.perf_counter()
        with pytest.raises(apsw.BusyError, match="read pool"):
            with _shared._search_store_checkout():
                pass
        elapsed = time.perf_counter() - started

    assert elapsed >= 0.10
    assert len(_shared._search_vector_store_pool_handles) == 2


def test_search_store_ram_clamp_rejects_oversized_pool(tmp_path, monkeypatch):
    db_path = tmp_path / "clamped.db"
    _create_vector_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "13")
    monkeypatch.setenv("BRAINLAYER_READ_CACHE_KB", "64000")

    with pytest.raises(ValueError, match="read pool RAM clamp"):
        _shared._get_search_vector_store()


def test_writer_completes_while_read_pool_handles_are_checked_out(tmp_path, monkeypatch):
    db_path = tmp_path / "writer-progress.db"
    _create_vector_db(db_path)
    monkeypatch.setenv("BRAINLAYER_DB", str(db_path))
    monkeypatch.setenv("BRAINLAYER_READ_POOL_SIZE", "2")

    with _shared._search_store_checkout() as reader_a, _shared._search_store_checkout() as reader_b:
        for reader in (reader_a, reader_b):
            reader.conn.cursor().execute("BEGIN")
            reader.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone()
        writer = VectorStore(db_path)
        try:
            started = time.perf_counter()
            _insert_chunk(
                writer,
                chunk_id="writer-progress",
                content="writer completes while readers are checked out",
                embedding=_embed("writer completes while readers are checked out"),
            )
            writer.conn.cursor().execute("PRAGMA wal_checkpoint(PASSIVE)")
            elapsed = time.perf_counter() - started
        finally:
            writer.close()
            for reader in (reader_a, reader_b):
                reader.conn.cursor().execute("ROLLBACK")

    inspector = VectorStore(db_path, readonly=True)
    try:
        count = (
            inspector.conn.cursor().execute("SELECT COUNT(*) FROM chunks WHERE id = 'writer-progress'").fetchone()[0]
        )
    finally:
        inspector.close()

    assert count == 1
    assert elapsed < 1.0
