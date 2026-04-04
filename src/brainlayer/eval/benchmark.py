"""Ranx-based benchmark helpers for BrainLayer search evaluation."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("IR_DATASETS_HOME", "/tmp/ir_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ranx import Qrels, Run, compare, evaluate

from brainlayer._helpers import _escape_fts5_query

DEFAULT_RUN_METRICS = ["ndcg@10", "recall@20", "map@10", "mrr"]
DEFAULT_COMPARE_METRICS = ["ndcg@10", "recall@20"]
DEFAULT_QUERY_SUITE: list[tuple[str, str]] = [
    ("q1", "BrainLayer architecture"),
    ("q2", "sleep optimization"),
    ("known_entity_t3_code", "T3 Code"),
    ("known_entity_theo_browne", "Theo Browne"),
    ("known_entity_brainlayer_architecture", "BrainLayer architecture"),
    ("known_entity_avi_simon", "Avi Simon"),
    ("known_entity_voicelayer", "VoiceLayer"),
    ("health_dopamine", "dopamine"),
    ("health_huberman_protocol", "Huberman protocol"),
    ("health_sleep_optimization", "sleep optimization"),
    ("health_vo2_max", "VO2 max"),
    ("cross_language_boker_routine", "בוקר morning routine"),
    ("cross_language_ivrit_writing_style", "Hebrew writing style em dash"),
    ("cross_language_mehayom_sprint_payment", "MeHayom sprint payment"),
    ("cross_language_deploy_hebrew", "deploy פריסה"),
    ("temporal_recent_job_search", "recent job search"),
    ("temporal_this_week", "what happened this week"),
    ("temporal_recent_brainlayer_work", "recent BrainLayer work"),
    ("conceptual_morning_routine", "morning routine"),
    ("conceptual_deployment_strategy", "deployment strategy"),
    ("conceptual_search_quality", "search quality evaluation"),
    ("conceptual_agent_memory", "agent memory"),
    ("frustration_expectation_failure", "expectation failure"),
    ("frustration_wrong_assumption", "wrong assumption"),
    ("frustration_db_locking", "DB locking"),
    ("frustration_search_recall_miss", "search recall miss"),
    ("frustration_context_injection_failure", "context injection failure"),
]


class ReadOnlyBenchmarkStore:
    """Minimal readonly store wrapper for FTS-only benchmark access."""

    def __init__(self, db_path: str | Path):
        path = Path(db_path).expanduser()
        self.db_path = path
        self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)

    def _read_cursor(self):
        return self.conn.cursor()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SearchBenchmark:
    """Benchmarks search pipelines against graded relevance judgments."""

    def __init__(self, qrels_path: str):
        self.qrels_path = Path(qrels_path)
        self.qrels = self._load_qrels(self.qrels_path)
        self.ranx_qrels = Qrels.from_dict(self.qrels)

    def _load_qrels(self, qrels_path: Path) -> dict[str, dict[str, int]]:
        payload = json.loads(qrels_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Qrels JSON must be a dict of query_id -> {doc_id: grade}")
        return payload

    def queries_in_qrels(self, queries: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return [query for query in queries if query[0] in self.qrels]

    def run_pipeline(
        self,
        pipeline_fn: Callable[[str], list[tuple[str, float]]],
        queries: Iterable[tuple[str, str]],
    ) -> Run:
        run_dict: dict[str, dict[str, float]] = {}
        for query_id, query_text in queries:
            results = pipeline_fn(query_text)
            run_dict[query_id] = {chunk_id: float(score) for chunk_id, score in results}
        return Run(run=run_dict)

    def evaluate_pipeline(self, run: Run, metrics: list[str] | None = None) -> dict[str, float]:
        metric_list = metrics or DEFAULT_RUN_METRICS
        scores = evaluate(self.ranx_qrels, run, metric_list)
        if isinstance(scores, dict):
            return scores
        if len(metric_list) == 1:
            return {metric_list[0]: float(scores)}
        raise TypeError(f"Unexpected Ranx evaluate() result type: {type(scores)!r}")

    def compare_pipelines(self, runs: dict[str, Run], metrics: list[str] | None = None) -> str:
        named_runs = [Run(name=name, run=run.to_dict()) for name, run in runs.items()]
        report = compare(
            self.ranx_qrels,
            runs=named_runs,
            metrics=metrics or DEFAULT_COMPARE_METRICS,
            max_p=0.05,
            rounding_digits=4,
        )
        return str(report)


def pipeline_fts5_only(store, query: str, n_results: int = 20) -> list[tuple[str, float]]:
    """FTS5-only search using BM25 rank from the chunks_fts table."""
    fts_query = _escape_fts5_query(query)
    if not fts_query:
        return []

    cursor = store._read_cursor()
    rows = list(
        cursor.execute(
            """
            SELECT f.chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts f
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, n_results),
        )
    )
    return [(chunk_id, float(-score)) for chunk_id, score in rows]


def pipeline_hybrid_rrf(store, query: str, n_results: int = 20) -> list[tuple[str, float]]:
    """Hybrid search placeholder that uses store.hybrid_search when available."""
    if not hasattr(store, "hybrid_search"):
        return pipeline_fts5_only(store, query, n_results=n_results)
    raise NotImplementedError("Hybrid RRF benchmark pipeline depends on query embeddings and P0 search wiring.")


def pipeline_hybrid_entity(store, query: str, n_results: int = 20) -> list[tuple[str, float]]:
    """Future hybrid + entity benchmark placeholder."""
    raise NotImplementedError("Entity-boosted benchmark pipeline is reserved for future search work.")
