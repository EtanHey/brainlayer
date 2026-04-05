"""Tests for the Ranx-based search evaluation framework."""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("IR_DATASETS_HOME", "/tmp/ir_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytest
from ranx import Run

from brainlayer.vector_store import VectorStore


@pytest.fixture
def qrels_file(tmp_path: Path) -> Path:
    path = tmp_path / "qrels.json"
    path.write_text(
        json.dumps(
            {
                "q1": {"doc-a": 3, "doc-b": 1},
                "q2": {"doc-c": 2, "doc-d": 0},
            }
        )
    )
    return path


@pytest.fixture
def eval_store(tmp_path: Path):
    store = VectorStore(tmp_path / "eval.db")
    yield store
    store.close()


def test_benchmark_loads_qrels(qrels_file: Path):
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))

    assert set(benchmark.qrels.keys()) == {"q1", "q2"}


def test_benchmark_runs_pipeline(qrels_file: Path):
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE, SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    queries = [query for query in DEFAULT_QUERY_SUITE if query[0] in {"q1", "q2"}]

    run = benchmark.run_pipeline(
        lambda query_text: [("doc-a", 2.0), ("doc-b", 1.0)] if "architecture" in query_text else [("doc-c", 3.0)],
        queries,
    )

    assert set(run.keys()) == {"q1", "q2"}
    assert run["q1"]["doc-a"] == 2.0


def test_benchmark_evaluates(qrels_file: Path):
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE, SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    queries = [query for query in DEFAULT_QUERY_SUITE if query[0] in {"q1", "q2"}]
    run = benchmark.run_pipeline(
        lambda query_text: [("doc-a", 4.0)] if "architecture" in query_text else [("doc-c", 5.0)],
        queries,
    )

    scores = benchmark.evaluate_pipeline(run, metrics=["ndcg@10"])

    assert "ndcg@10" in scores
    assert 0.0 <= scores["ndcg@10"] <= 1.0


def test_benchmark_evaluates_with_mismatched_query_ids(qrels_file: Path):
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    run = Run(run={"q1": {"doc-a": 4.0}})

    scores = benchmark.evaluate_pipeline(run, metrics=["ndcg@10"])

    assert "ndcg@10" in scores
    assert 0.0 <= scores["ndcg@10"] <= 1.0


def test_benchmark_compares(qrels_file: Path):
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE, SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    queries = [query for query in DEFAULT_QUERY_SUITE if query[0] in {"q1", "q2"}]

    better = benchmark.run_pipeline(
        lambda query_text: [("doc-a", 4.0)] if "architecture" in query_text else [("doc-c", 4.0)],
        queries,
    )
    worse = benchmark.run_pipeline(
        lambda query_text: [("doc-b", 4.0)] if "architecture" in query_text else [("doc-d", 4.0)],
        queries,
    )

    comparison = benchmark.compare_pipelines({"better": better, "worse": worse}, metrics=["ndcg@10"])

    assert "better" in comparison
    assert "worse" in comparison


def test_qrels_format_valid():
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE

    qrels_path = Path(__file__).parent / "eval_qrels.json"
    qrels = json.loads(qrels_path.read_text())
    suite_query_ids = {query_id for query_id, _query_text in DEFAULT_QUERY_SUITE}

    assert len(qrels) == len(suite_query_ids) - 2, f"Expected {len(suite_query_ids) - 2} qrels, found {len(qrels)}"
    for query_id, judgments in qrels.items():
        assert judgments, f"{query_id} has no judgments"
        assert 1 <= len(judgments) <= 40, f"{query_id} should have 1-40 judgments, found {len(judgments)}"
        for chunk_id, grade in judgments.items():
            assert isinstance(chunk_id, str) and chunk_id
            assert grade in {0, 1, 2, 3}, f"{query_id}:{chunk_id} has invalid grade {grade}"


def test_all_suite_queries_have_qrels():
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE

    qrels_path = Path(__file__).parent / "eval_qrels.json"
    qrels = json.loads(qrels_path.read_text())

    missing = [
        query_id
        for query_id, _query_text in DEFAULT_QUERY_SUITE
        if query_id not in {"q1", "q2"} and query_id not in qrels
    ]
    assert missing == []


def test_frustration_queries_in_suite():
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE

    qrels_path = Path(__file__).parent / "eval_qrels.json"
    qrels = json.loads(qrels_path.read_text())
    suite_map = dict(DEFAULT_QUERY_SUITE)
    frustration_query_ids = sorted(query_id for query_id in qrels if query_id.startswith("frustration_"))

    assert frustration_query_ids, "Expected frustration queries in eval_qrels.json"
    for query_id in frustration_query_ids:
        assert query_id in suite_map, f"{query_id} missing from DEFAULT_QUERY_SUITE"
        assert suite_map[query_id].strip(), f"{query_id} has empty query text"


def test_pooled_qrels_cover_both_pipelines(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "build_qrels.py"
    spec = importlib.util.spec_from_file_location("build_qrels_module", module_path)
    assert spec and spec.loader
    build_qrels_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_qrels_module)

    class FakeCursor:
        def execute(self, query, params):
            chunk_id = params[0]
            rows = {
                "fts-only": ("fts content", "fts summary", json.dumps(["fts"])),
                "shared": ("shared content", "shared summary", json.dumps(["shared"])),
                "hybrid-only": ("hybrid content", "hybrid summary", json.dumps(["hybrid"])),
            }
            row = rows.get(chunk_id)
            return iter([] if row is None else [row])

    class FakeStore:
        def _read_cursor(self):
            return FakeCursor()

    monkeypatch.setattr(build_qrels_module, "DEFAULT_QUERY_SUITE", [("pooled_q", "pooled query")])
    monkeypatch.setattr(
        build_qrels_module,
        "pipeline_fts5_only",
        lambda store, query, n_results=20: [("fts-only", 1.0), ("shared", 0.5)],
    )
    monkeypatch.setattr(
        build_qrels_module,
        "pipeline_hybrid_rrf",
        lambda store, query, n_results=20: [("hybrid-only", 1.0), ("shared", 0.7)],
    )

    qrels = build_qrels_module.build_qrels(FakeStore(), n_results=20, mode="pool")

    assert set(qrels["pooled_q"]) == {"fts-only", "shared", "hybrid-only"}


def test_pipeline_fts5_returns_results(eval_store: VectorStore):
    from brainlayer.eval.benchmark import pipeline_fts5_only

    cursor = eval_store.conn.cursor()
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, summary, tags
        ) VALUES (?, ?, '{}', 'eval.jsonl', 'brainlayer', 'note', ?, ?, ?)""",
        (
            "chunk-fts-1",
            "BrainLayer architecture uses sqlite FTS5 and vector search.",
            len("BrainLayer architecture uses sqlite FTS5 and vector search."),
            "Architecture notes for BrainLayer",
            json.dumps(["search", "architecture"]),
        ),
    )
    cursor.execute(
        """INSERT INTO chunks (
            id, content, metadata, source_file, project, content_type, char_count, summary, tags
        ) VALUES (?, ?, '{}', 'eval.jsonl', 'brainlayer', 'note', ?, ?, ?)""",
        (
            "chunk-fts-2",
            "Sleep optimization protocol with Huberman notes.",
            len("Sleep optimization protocol with Huberman notes."),
            "Sleep notes",
            json.dumps(["health"]),
        ),
    )

    results = pipeline_fts5_only(eval_store, "BrainLayer architecture", n_results=5)

    assert results, "Expected FTS5 pipeline to return ranked results"
    assert results[0][0] == "chunk-fts-1"
    assert isinstance(results[0][1], float)


def test_pipeline_hybrid_rrf_returns_rank_scores():
    from brainlayer.eval.benchmark import pipeline_hybrid_rrf

    class FakeStore:
        def hybrid_search(self, query_embedding, query_text, n_results=20):
            assert query_embedding == [0.1, 0.2]
            assert query_text == "hybrid query"
            assert n_results == 3
            return {"ids": [["chunk-a", "chunk-b", "chunk-c"]]}

    results = pipeline_hybrid_rrf(FakeStore(), "hybrid query", n_results=3, embed_fn=lambda query: [0.1, 0.2])

    assert results == [("chunk-a", 1.0), ("chunk-b", 0.5), ("chunk-c", 1.0 / 3.0)]
