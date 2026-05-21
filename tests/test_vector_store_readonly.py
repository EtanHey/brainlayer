import apsw
import pytest

from brainlayer._helpers import serialize_f32
from brainlayer.mcp import _shared
from brainlayer.search_repo import _hybrid_cache
from brainlayer.vector_store import VectorStore


@pytest.fixture(autouse=True)
def clear_hybrid_cache():
    _shared._search_vector_store = None
    _shared._vector_store = None
    _hybrid_cache.clear()
    yield
    for store in (_shared._search_vector_store, _shared._vector_store):
        if store is not None:
            store.close()
    _shared._search_vector_store = None
    _shared._vector_store = None
    _hybrid_cache.clear()


def _embed(text: str) -> list[float]:
    seed = (sum(ord(c) for c in text[:40]) % 97) / 1000.0
    return [seed + (i / 10000.0) for i in range(1024)]


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
        store.close()
        _shared._search_vector_store = None


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
        store.close()
        _shared._search_vector_store = None
