"""Tests for the Ranx-based search evaluation framework."""

import json
from pathlib import Path

import pytest

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
    qrels_path = Path(__file__).parent / "eval_qrels.json"
    qrels = json.loads(qrels_path.read_text())

    assert len(qrels) >= 25, f"Expected at least 25 queries, found {len(qrels)}"
    for query_id, judgments in qrels.items():
        assert judgments, f"{query_id} has no judgments"
        assert 1 <= len(judgments) <= 20, f"{query_id} should have 1-20 judgments, found {len(judgments)}"
        for chunk_id, grade in judgments.items():
            assert isinstance(chunk_id, str) and chunk_id
            assert grade in {0, 1, 2, 3}, f"{query_id}:{chunk_id} has invalid grade {grade}"


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
