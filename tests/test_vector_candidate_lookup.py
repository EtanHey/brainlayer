"""Candidate reranking must fetch vec0 rows by primary-key point lookup."""

import struct

import apsw
import numpy as np
import pytest
import sqlite_vec

from brainlayer.search_repo import SearchMixin


class CandidateStore(SearchMixin):
    def __init__(self, conn):
        self.conn = conn

    def _read_cursor(self):
        return self.conn.cursor()


@pytest.mark.parametrize("operation", ["rerank", "mmr_load"])
def test_candidate_vectors_have_bounded_sql_work(tmp_path, operation):
    conn = apsw.Connection(str(tmp_path / "vectors.db"))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    conn.execute("CREATE VIRTUAL TABLE chunk_vectors USING vec0(chunk_id TEXT PRIMARY KEY, embedding FLOAT[4])")
    conn.executemany(
        "INSERT INTO chunk_vectors VALUES (?, ?)",
        ((f"v-{i}", struct.pack("4f", i, 1, 2, 3)) for i in range(6000)),
    )
    store = CandidateStore(conn)
    ids = ["v-5999", "missing", "v-1", "v-1"]
    semantic = {
        "ids": [ids],
        "documents": [["far", "missing", "near", "duplicate"]],
        "metadatas": [[{} for _ in ids]],
        "distances": [[10.0, 2.0, 10.0, 10.0]],
    }
    steps = 0

    def budget():
        nonlocal steps
        steps += 100
        assert steps < 10000, "candidate lookup scanned vec0 instead of using its primary key"
        return False

    conn.set_progress_handler(budget, 100)
    try:
        if operation == "rerank":
            result = store._rerank_binary_results_with_float([0.0] * 4, semantic)
            assert result["ids"][0] == ["missing", "v-1", "v-1", "v-5999"]
            assert result["distances"][0] == [2.0, 15.0, 15.0, 5999**2 + 14.0]
        else:
            result = store._load_chunk_embeddings(ids)
            assert set(result) == {"v-1", "v-5999"}
            np.testing.assert_array_equal(result["v-1"], np.array([1, 1, 2, 3], dtype=np.float32))
        assert store._load_chunk_embeddings([]) == {}
    finally:
        conn.set_progress_handler(None)
        conn.close()
