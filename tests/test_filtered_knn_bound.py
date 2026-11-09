"""Filtered candidate sizing must not scan the content corpus."""

import time

import apsw
import pytest

from brainlayer.search_repo import SearchMixin


class PopulationStore(SearchMixin):
    _has_chunk_origin = True
    _has_source_class = False
    _chunk_tags_available = True

    def __init__(self, conn):
        self.conn = conn

    def _read_cursor(self):
        return self.conn.cursor()


@pytest.mark.parametrize("include_checkpoints,include_audit", [(False, True), (True, False), (False, False)])
def test_knn_estimate_has_fixed_sql_work_budget(tmp_path, include_checkpoints, include_audit):
    conn = apsw.Connection(str(tmp_path / "large-content.db"))
    conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, content TEXT, chunk_origin TEXT, tags TEXT)")
    conn.execute("CREATE TABLE chunk_tags(chunk_id TEXT, tag TEXT, PRIMARY KEY(chunk_id, tag))")
    # 96 MiB with no excluded rows: the old exact count must inspect it all.
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, 'unknown', '[]')",
        ((str(i), "ordinary memory " + "x" * 16384) for i in range(6000)),
    )
    conn.execute("COMMIT")
    store = PopulationStore(conn)
    callbacks = 0

    def enforce_budget():
        nonlocal callbacks
        callbacks += 1
        assert callbacks < 70, "candidate sizing scanned the corpus instead of a bounded covering index"
        return False

    # Deterministic SQL work limit is the regression gate; elapsed time is diagnostic.
    conn.set_progress_handler(enforce_budget, 1000)
    started = time.perf_counter()
    try:
        k = store._effective_knn_k(50, False, include_checkpoints, include_audit)
        assert k == 2000
        assert store._effective_knn_k(50, False, include_checkpoints, include_audit, cap_filtered=False) == 4096
        print(f"bounded k sizing: {time.perf_counter() - started:.6f}s, {callbacks * 1000} VM steps maximum")
    finally:
        conn.set_progress_handler(None)
        conn.close()


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize(
    "include_checkpoints,include_audit", [(False, False), (False, True), (True, False), (True, True)]
)
def test_population_bound_preserves_filtered_result_sets(
    tmp_path, monkeypatch, binary, tied, include_checkpoints, include_audit
):
    import json

    import numpy as np

    from brainlayer.search_repo import _precompact_content_exclusion_sql, clear_hybrid_search_cache
    from brainlayer.vector_store import VectorStore

    store = VectorStore(tmp_path / "correctness.db")
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(160, 1024)).astype(np.float32).tolist()
    query = rng.normal(size=1024).astype(np.float32).tolist()
    if tied:
        vectors = [query] * len(vectors)
    chunks = []
    for i in range(160):
        # Include legacy content-only checkpoints, tag-only and whitespace-wrapped
        # recursive MCP output, plus ordinary rows and source-class exclusions.
        content = "hydra retrieval ordinary memory"
        if i % 11 == 0:
            content = "[PreCompact checkpoint] hydra retrieval"
        elif i % 13 == 0:
            content = '  { "jsonrpc" : "2.0", "result": "hydra retrieval" }'
        chunks.append(
            dict(
                id=f"chunk-{i}",
                content=content,
                metadata={},
                source_file="test.jsonl",
                project="brainlayer",
                content_type="user_message",
                char_count=len(content),
                source="claude_code",
                tags=json.dumps(["audit"] if i % 7 == 0 else []),
                source_class="desktop" if i % 17 == 0 else "cli",
            )
        )
    try:
        store.upsert_chunks(chunks, vectors)
        store.conn.execute("UPDATE chunks SET chunk_origin = 'unknown' WHERE content LIKE '[PreCompact%'")
        store.conn.execute("UPDATE chunks SET source_class = 'desktop' WHERE CAST(substr(id, 7) AS INTEGER) % 17 = 0")
        cp = store.conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE chunk_origin = 'precompact_checkpoint' "
            f"OR NOT ({_precompact_content_exclusion_sql('content')})"
        ).fetchone()[0]
        audit = store.conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE NOT ({store._audit_recursion_exclusion_sql('id', 'tags', 'content')})"
        ).fetchone()[0]
        bounded = store._effective_knn_k

        def legacy_k(
            n_results,
            needs_overfetch,
            include_checkpoints,
            include_audit,
            include_hidden_source_classes=False,
            *,
            cap_filtered=True,
        ):
            k = max(n_results, min(n_results * 10, 1000)) if needs_overfetch else n_results
            for count, included in [(cp, include_checkpoints), (audit, include_audit)]:
                if not included:
                    k = min(k + count, max(k, 2000)) if cap_filtered else k + count
            return min(
                store._source_class_filtered_knn_k(k, include_hidden_source_classes, cap_filtered=cap_filtered), 4096
            )

        def search():
            kwargs = dict(
                query_embedding=query,
                n_results=10,
                include_checkpoints=include_checkpoints,
                include_audit=include_audit,
            )
            if binary:
                return store.hybrid_search(query_text="hydra retrieval", **kwargs)
            return store.search(**kwargs)

        monkeypatch.setattr(store, "_effective_knn_k", legacy_k)
        before = search()
        clear_hybrid_search_cache(store.db_path)
        monkeypatch.setattr(store, "_effective_knn_k", bounded)
        after = search()
        assert before == after
    finally:
        clear_hybrid_search_cache(store.db_path)
        store.close()
