from __future__ import annotations

import apsw
import pytest

import brainlayer.vector_store as vector_store
from brainlayer.vector_store import VectorStore


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_WRITER_PIDFILE_DIR", str(tmp_path / "pidfiles"))
    store = VectorStore(tmp_path / "brainlayer.db")
    try:
        yield store
    finally:
        store.close()


def _chunk(chunk_id: str, content: str | None = None) -> dict:
    body = content or f"Unique bounded transaction content for {chunk_id}"
    return {
        "id": chunk_id,
        "content": body,
        "metadata": {"test": "bound-index-write-txn"},
        "source_file": "isolated-upsert.jsonl",
        "project": "txn-batch-test",
        "content_type": "note",
        "char_count": len(body),
        "source": "test",
        "created_at": "2026-07-06T00:00:00Z",
    }


def _embedding(seed: int) -> list[float]:
    return [float(seed) / 1000.0] * 1024


def _is_insert_chunk_statement(statement: str) -> bool:
    return "INSERT INTO CHUNKS" in " ".join(statement.upper().split())


def test_upsert_chunks_commits_large_batches_in_bounded_transactions(isolated_store, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", "2")
    statements: list[str] = []

    def trace(_cursor, statement, _bindings):
        statements.append(str(statement).strip().upper())
        return True

    isolated_store.conn.setexectrace(trace)

    processed = isolated_store.upsert_chunks(
        [_chunk(f"chunk-{index}") for index in range(5)],
        [_embedding(index) for index in range(5)],
    )

    assert processed == 5
    assert [statement for statement in statements if statement == "BEGIN IMMEDIATE"] == [
        "BEGIN IMMEDIATE",
        "BEGIN IMMEDIATE",
        "BEGIN IMMEDIATE",
    ]
    assert [statement for statement in statements if statement == "COMMIT"] == ["COMMIT", "COMMIT", "COMMIT"]
    assert isolated_store.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone() == (5,)


def test_index_txn_batch_env_uses_dedicated_cap(monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", str(vector_store._MAX_INDEX_TXN_BATCH + 1))

    assert vector_store._index_txn_batch_size() == vector_store._DEFAULT_INDEX_TXN_BATCH


def test_upsert_chunks_preserves_dedupe_and_repeat_upsert_shape(isolated_store, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", "2")
    duplicate_content = "Repeat bounded transaction dedupe content should collapse"
    chunks = [
        _chunk("canonical", duplicate_content),
        _chunk("duplicate", duplicate_content),
        _chunk("distinct", "A distinct bounded transaction memory with unique final state"),
    ]

    assert isolated_store.upsert_chunks(chunks, [_embedding(1), _embedding(2), _embedding(3)]) == 3
    assert isolated_store.upsert_chunks(chunks, [_embedding(1), _embedding(2), _embedding(3)]) == 3

    cursor = isolated_store.conn.cursor()
    active_rows = cursor.execute("SELECT id FROM chunks WHERE COALESCE(archived, 0) = 0 ORDER BY id").fetchall()
    aliases = cursor.execute(
        "SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias ORDER BY old_chunk_id"
    ).fetchall()
    vector_rows = cursor.execute("SELECT chunk_id FROM chunk_vectors ORDER BY chunk_id").fetchall()

    assert active_rows == [("canonical",), ("distinct",)]
    assert aliases == [("duplicate", "canonical")]
    assert vector_rows == [("canonical",), ("distinct",)]


def test_busy_sub_batch_retries_without_replaying_committed_sub_batches(isolated_store, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", "2")
    insert_attempts: dict[str, int] = {}
    failed_once = False

    def trace(_cursor, statement, bindings):
        nonlocal failed_once
        if _is_insert_chunk_statement(str(statement)) and bindings:
            chunk_id = str(bindings[0])
            insert_attempts[chunk_id] = insert_attempts.get(chunk_id, 0) + 1
            if chunk_id == "chunk-2" and not failed_once:
                failed_once = True
                raise apsw.BusyError("simulated transient second sub-batch busy")
        return True

    isolated_store.conn.setexectrace(trace)

    processed = isolated_store.upsert_chunks(
        [_chunk(f"chunk-{index}") for index in range(4)],
        [_embedding(index) for index in range(4)],
    )

    assert processed == 4
    assert insert_attempts == {
        "chunk-0": 1,
        "chunk-1": 1,
        "chunk-2": 2,
        "chunk-3": 1,
    }
    assert isolated_store.conn.cursor().execute("SELECT COUNT(*) FROM chunks").fetchone() == (4,)


def test_failed_later_sub_batch_invalidates_after_prior_commit(isolated_store, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_INDEX_TXN_BATCH", "2")
    cleared_paths: list[object] = []
    invalidated_filtered_counts = 0

    def fake_clear_hybrid_search_cache(db_path=None):
        cleared_paths.append(db_path)

    def fake_invalidate_filtered_count_caches():
        nonlocal invalidated_filtered_counts
        invalidated_filtered_counts += 1

    def trace(_cursor, statement, bindings):
        if _is_insert_chunk_statement(str(statement)) and bindings and bindings[0] == "chunk-2":
            raise RuntimeError("simulated permanent second sub-batch failure")
        return True

    monkeypatch.setattr("brainlayer.search_repo.clear_hybrid_search_cache", fake_clear_hybrid_search_cache)
    monkeypatch.setattr(isolated_store, "_invalidate_filtered_count_caches", fake_invalidate_filtered_count_caches)
    isolated_store.conn.setexectrace(trace)

    with pytest.raises(RuntimeError, match="permanent second sub-batch failure"):
        isolated_store.upsert_chunks(
            [_chunk(f"chunk-{index}") for index in range(4)],
            [_embedding(index) for index in range(4)],
        )

    cursor = isolated_store.conn.cursor()
    assert cursor.execute("SELECT id FROM chunks ORDER BY id").fetchall() == [("chunk-0",), ("chunk-1",)]
    assert cleared_paths == [isolated_store.db_path]
    assert invalidated_filtered_counts == 1
