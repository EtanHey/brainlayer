import json

import apsw

from brainlayer.search_repo import SearchMixin, clear_hybrid_search_cache


class RecordingCursor:
    def __init__(self, rows_by_table=None, interrupt_fts: bool = False):
        self.calls = []
        self.rows_by_table = rows_by_table or {}
        self.interrupt_fts = interrupt_fts

    def execute(self, sql, params=()):
        self.calls.append((sql, list(params)))
        if "FROM chunks_fts " in sql and self.interrupt_fts:
            raise apsw.InterruptError("interrupted")
        if "FROM chunks_fts_trigram" in sql:
            return list(self.rows_by_table.get("chunks_fts_trigram", []))
        if "FROM chunks_fts " in sql:
            return list(self.rows_by_table.get("chunks_fts", []))
        return []

    def get_connection(self):
        return None


class RecordingBinaryStore(SearchMixin):
    db_path = "<recording-binary-store>"
    _has_chunk_origin = False
    _chunk_tags_available = False

    def __init__(self):
        self.cursor = RecordingCursor()
        self.effective_k_calls = []

    def _read_cursor(self):
        return self.cursor

    def _audit_recursion_exclusion_sql(self, *_args):
        return "1=1"

    def _checkpoint_exclusion_clause(self, *_args):
        return None

    def _effective_knn_k(self, *args, **kwargs):
        self.effective_k_calls.append((args, kwargs))
        return 2000 if kwargs.get("cap_filtered", True) else 4096


def _semantic_result(chunk_id="semantic-1"):
    return {
        "ids": [[chunk_id]],
        "documents": [["semantic content"]],
        "metadatas": [[{"source_file": "semantic.md", "project": "brainlayer"}]],
        "distances": [[0.1]],
    }


def _fts_row(chunk_id):
    return (
        chunk_id,
        0,
        f"{chunk_id} content",
        json.dumps({}),
        f"{chunk_id}.md",
        "brainlayer",
        "assistant_text",
        "memory",
        20,
        None,
        None,
        7,
        None,
        "2026-05-31T00:00:00Z",
        "claude_code",
        None,
        None,
        1.0,
        "unknown",
    )


class RecordingHybridStore(SearchMixin):
    db_path = "<recording-hybrid-store>"
    _binary_index_available = True
    _trigram_fts_available = True
    _has_chunk_origin = False
    _chunk_tags_available = False

    def __init__(self, cursor):
        self.cursor = cursor
        self.binary_kwargs = None
        self.queued_ids = None

    def _read_cursor(self):
        return self.cursor

    def _audit_recursion_exclusion_sql(self, *_args):
        return "1=1"

    def _checkpoint_exclusion_clause(self, *_args):
        return None

    def _binary_search(self, **kwargs):
        self.binary_kwargs = kwargs
        return _semantic_result()

    def _rerank_binary_results_with_float(self, _query_embedding, semantic_results):
        return semantic_results

    def _queue_retrieval_strengthening(self, chunk_ids):
        self.queued_ids = chunk_ids

    def get_agent_profile(self, _agent_id):
        return None


def test_brainbar_fast_profile_binary_search_uses_fixed_k_without_count_expansion_or_retry():
    store = RecordingBinaryStore()

    results = store._binary_search(
        query_embedding=[0.1, 0.2, 0.3],
        n_results=50,
        brainbar_helper_fast_profile=True,
    )

    assert results["ids"] == [[]]
    assert len(store.cursor.calls) == 1
    assert store.cursor.calls[0][1][1] == 400
    assert store.effective_k_calls == []


def test_brainbar_fast_profile_uses_primary_fts_only_and_threads_binary_flag():
    clear_hybrid_search_cache()
    cursor = RecordingCursor(
        rows_by_table={
            "chunks_fts": [_fts_row("fts-1")],
            "chunks_fts_trigram": [_fts_row("trigram-1")],
        }
    )
    store = RecordingHybridStore(cursor)

    store.hybrid_search(
        query_embedding=[0.1, 0.2, 0.3],
        query_text="helper fast profile",
        n_results=5,
        brainbar_helper_fast_profile=True,
    )

    executed_sql = "\n".join(sql for sql, _params in cursor.calls)
    assert "FROM chunks_fts f" in executed_sql
    assert "chunks_fts_trigram" not in executed_sql
    assert store.binary_kwargs["brainbar_helper_fast_profile"] is True


def test_brainbar_fast_profile_returns_semantic_results_when_primary_fts_is_timeboxed():
    clear_hybrid_search_cache()
    cursor = RecordingCursor(interrupt_fts=True)
    store = RecordingHybridStore(cursor)

    results = store.hybrid_search(
        query_embedding=[0.1, 0.2, 0.3],
        query_text="slow helper fts",
        n_results=5,
        brainbar_helper_fast_profile=True,
    )

    assert results["ids"][0] == ["semantic-1"]
