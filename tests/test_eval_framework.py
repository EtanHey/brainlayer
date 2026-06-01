"""Tests for the Ranx-based search evaluation framework."""

import importlib.util
import json
import math
import os
import sys
import types
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
    from brainlayer.eval.benchmark import DEFAULT_QUERY_SUITE, DEFAULT_RUN_METRICS, SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    queries = [query for query in DEFAULT_QUERY_SUITE if query[0] in {"q1", "q2"}]
    run = benchmark.run_pipeline(
        lambda query_text: [("doc-a", 4.0)] if "architecture" in query_text else [("doc-c", 5.0)],
        queries,
    )

    scores = benchmark.evaluate_pipeline(run, metrics=["ndcg@3", "precision@3", "ndcg@10"])

    assert "ndcg@3" in DEFAULT_RUN_METRICS
    assert "precision@3" in DEFAULT_RUN_METRICS
    assert "ndcg@3" in scores
    assert "precision@3" in scores
    assert "ndcg@10" in scores
    assert 0.0 <= scores["ndcg@10"] <= 1.0


def test_benchmark_evaluates_with_mismatched_query_ids(qrels_file: Path):
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    run = Run(run={"q1": {"doc-a": 4.0}})

    scores = benchmark.evaluate_pipeline(run, metrics=["ndcg@10"])

    assert "ndcg@10" in scores
    assert 0.0 <= scores["ndcg@10"] <= 1.0


def test_manual_fallback_averages_only_qrels_queries(qrels_file: Path):
    """regression-guard: manual fallback should match Ranx's qrels-query averaging."""
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    run_dict = {
        "q1": {"doc-a": 4.0},
        "q-extra": {"doc-z": 9.0},
    }

    scores = benchmark._evaluate_without_ranx(run_dict, ["recall@10"])

    assert scores["recall@10"] == pytest.approx(0.25)


def test_manual_fallback_map_at_k_uses_full_relevant_count(tmp_path: Path):
    """regression-guard: map@k fallback must match ranx full-relevance denominator semantics."""
    from brainlayer.eval.benchmark import SearchBenchmark

    qrels_path = tmp_path / "qrels.json"
    qrels_path.write_text(json.dumps({"q1": {f"doc-{idx}": 1 for idx in range(20)}}))
    benchmark = SearchBenchmark(str(qrels_path))
    run_dict = {"q1": {f"doc-{idx}": 20 - idx for idx in range(10)}}

    scores = benchmark._evaluate_without_ranx(run_dict, ["map@10"])

    assert scores["map@10"] == pytest.approx(0.5)


def test_manual_fallback_rejects_non_positive_metric_cutoff(qrels_file: Path):
    """regression-guard: metric cutoffs must fail clearly before AP denominator math."""
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))

    with pytest.raises(ValueError, match="Metric cutoff must be > 0"):
        benchmark._evaluate_without_ranx({"q1": {"doc-a": 1.0}}, ["map@0"])


def test_manual_fallback_ndcg_uses_linear_gain(qrels_file: Path):
    """regression-guard: ndcg fallback must match ranx ndcg's linear-gain semantics."""
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    run_dict = {"q1": {"doc-b": 2.0, "doc-a": 1.0}}

    scores = benchmark._evaluate_without_ranx(run_dict, ["ndcg@10"])

    expected_q1 = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    expected = (expected_q1 + 0.0) / 2
    assert scores["ndcg@10"] == pytest.approx(expected)


def test_manual_fallback_precision_at_k(qrels_file: Path):
    from brainlayer.eval.benchmark import SearchBenchmark

    benchmark = SearchBenchmark(str(qrels_file))
    run_dict = {"q1": {"doc-a": 3.0, "doc-z": 2.0, "doc-b": 1.0}}

    scores = benchmark._evaluate_without_ranx(run_dict, ["precision@3", "p@3"])

    expected = ((2 / 3) + 0.0) / 2
    assert scores["precision@3"] == pytest.approx(expected)
    assert scores["p@3"] == pytest.approx(expected)


def test_pr3_relevance_suite_has_labeled_conceptual_queries():
    from brainlayer.eval.benchmark import PR3_RELEVANCE_QUERY_SUITE, canonical_eval_doc_id

    qrels_path = Path(__file__).parent / "eval_pr3_relevance_qrels.json"
    qrels = json.loads(qrels_path.read_text())
    suite_map = dict(PR3_RELEVANCE_QUERY_SUITE)

    assert "pr3_knowledge_stale" in suite_map
    assert "keep knowledge base from going stale" == suite_map["pr3_knowledge_stale"]
    assert set(qrels) == set(suite_map)
    for query_id, judgments in qrels.items():
        assert judgments, f"{query_id} has no judgments"
        assert all("/Users/" not in doc_id for doc_id in judgments)
        assert all(grade in {0, 1, 2, 3} for grade in judgments.values())
    assert (
        canonical_eval_doc_id("/Users/example/.claude/projects/-Users-example-Gits-brainlayer/session.jsonl:7")
        == "claude-project:Gits-brainlayer/session.jsonl:7"
    )
    assert (
        canonical_eval_doc_id("/Users/john-smith/.claude/projects/-Users-john-smith-Gits-brainlayer/session.jsonl:7")
        == "claude-project:Gits-brainlayer/session.jsonl:7"
    )


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


def test_prewarm_benchmark_embedder_loads_model_once(monkeypatch):
    from brainlayer.eval.benchmark import prewarm_benchmark_embedder

    class FakeModel:
        def __init__(self):
            self.load_count = 0
            self.queries = []

        def _load_model(self):
            self.load_count += 1
            return object()

        def embed_query(self, query):
            self.queries.append(query)
            return [float(len(query))]

    fake_model = FakeModel()
    fake_embeddings = types.SimpleNamespace(
        DEFAULT_MODEL="fake-model",
        get_embedding_model=lambda model_name: fake_model,
    )
    monkeypatch.setitem(sys.modules, "brainlayer.embeddings", fake_embeddings)

    embed_fn = prewarm_benchmark_embedder()

    assert fake_model.load_count == 1
    assert embed_fn("alpha") == [5.0]
    assert embed_fn("beta") == [4.0]
    assert fake_model.load_count == 1
    assert fake_model.queries == ["alpha", "beta"]


def test_run_benchmark_reuses_prewarmed_embedder_for_hybrid_rrf(monkeypatch, tmp_path: Path):
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_module", module_path)
    assert spec and spec.loader
    run_benchmark_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_benchmark_module)

    prewarm_calls = []
    embedded_queries = []

    def fake_prewarm_benchmark_embedder():
        prewarm_calls.append("prewarm")

        def embed(query):
            embedded_queries.append(query)
            return [0.1, 0.2]

        return embed

    class FakeStore:
        def __init__(self, db_path):
            self.db_path = db_path

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class FakeRun:
        pass

    class FakeBenchmark:
        def __init__(self, qrels_path):
            self.qrels_path = qrels_path

        def queries_in_qrels(self, queries):
            return [("q1", "first query"), ("q2", "second query")]

        def run_pipeline(self, pipeline_fn, queries):
            for _query_id, query_text in queries:
                pipeline_fn(query_text)
            return FakeRun()

        def evaluate_pipeline(self, run):
            return {"ndcg@3": 1.0, "precision@3": 1.0, "recall@20": 1.0}

    def fake_pipeline_hybrid_rrf(store, query, n_results=20, *, embed_fn=None):
        assert embed_fn is not None
        assert embed_fn(query) == [0.1, 0.2]
        return [(f"doc-{query}", 1.0)]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--pipeline", "hybrid_rrf"])
    monkeypatch.setattr(run_benchmark_module, "SearchBenchmark", FakeBenchmark)
    monkeypatch.setattr(run_benchmark_module, "VectorStore", FakeStore)
    monkeypatch.setattr(run_benchmark_module, "prewarm_benchmark_embedder", fake_prewarm_benchmark_embedder)
    monkeypatch.setattr(run_benchmark_module, "pipeline_hybrid_rrf", fake_pipeline_hybrid_rrf)

    run_benchmark_module.main()

    assert prewarm_calls == ["prewarm"]
    assert embedded_queries == ["first query", "second query"]
