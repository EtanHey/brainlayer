"""Ranx-based benchmark helpers for BrainLayer search evaluation."""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("IR_DATASETS_HOME", "/tmp/ir_datasets")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from ranx import Qrels, Run, compare, evaluate

from brainlayer._helpers import _escape_fts5_query

DEFAULT_RUN_METRICS = ["ndcg@3", "precision@3", "ndcg@10", "recall@20", "map@10", "mrr"]
DEFAULT_COMPARE_METRICS = ["ndcg@3", "precision@3", "ndcg@10", "recall@20"]
_RANX_RUN_FALLBACK_PATCHED = False
_CLAUDE_PROJECT_DOC_ID_RE = re.compile(r"^/Users/([^/]+)/\.claude/projects/(.+)$")


def canonical_eval_doc_id(doc_id: str) -> str:
    """Normalize machine-local doc ids into portable benchmark ids."""
    match = _CLAUDE_PROJECT_DOC_ID_RE.match(doc_id)
    if match:
        username, project_doc_id = match.groups()
        local_prefix = f"-Users-{username}-"
        if project_doc_id.startswith(local_prefix):
            project_doc_id = project_doc_id[len(local_prefix) :]
        return f"claude-project:{project_doc_id}"
    return doc_id


MINED_FRUSTRATION_QUERY_SUITE: list[tuple[str, str]] = [
    ("frustration_001", "The formatter changed my hand-written comment; restore it without changing the code."),
    ("frustration_002", "The test passed locally but failed in CI. Find the environment difference."),
    ("frustration_003", "A search result omitted the most relevant document. Check the ranking inputs."),
    ("frustration_004", "The migration ran twice and duplicated a row. Make it safe to retry."),
    ("frustration_005", "The button is clipped at narrow widths. Fix the responsive layout."),
    ("frustration_006", "A timeout produced an empty success response. Preserve the error."),
    ("frustration_007", "The cache returned stale data after an update. Trace invalidation."),
    ("frustration_008", "The CLI accepted an unknown flag. Report invalid arguments clearly."),
    ("frustration_009", "The export dropped Unicode text. Preserve the original characters."),
    ("frustration_010", "The retry loop kept running after cancellation. Stop promptly."),
    ("frustration_011", "The parser treated a quoted comma as a delimiter. Correct the field boundary."),
    ("frustration_012", "The review missed a concurrent write. Add a focused race test."),
    ("frustration_013", "The progress indicator reached complete before the file was written."),
    ("frustration_014", "The sort order changed between runs. Make ties deterministic."),
    ("frustration_015", "The file watcher skipped a rename event. Reconcile the new path."),
    ("frustration_016", "A malformed input caused a stack trace. Return a useful validation error."),
    ("frustration_017", "The config override was ignored after restart. Load it from the selected path."),
    ("frustration_018", "The keyboard shortcut activated twice. Remove the duplicate handler."),
    ("frustration_019", "The report counted archived records. Exclude them from active totals."),
    ("frustration_020", "The output included a secret-like fixture. Redact it before logging."),
    ("frustration_021", "The backup check reported success before verifying the archive."),
    ("frustration_022", "The import accepted a duplicate identifier. Reject it with the source row number."),
    ("frustration_023", "The displayed timestamp used the wrong time zone. Apply the selected locale."),
    ("frustration_024", "The dry run modified the target file. Ensure it performs no writes."),
]
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
    *MINED_FRUSTRATION_QUERY_SUITE,
    ("frustration_expectation_failure", "expectation failure"),
    ("frustration_wrong_assumption", "wrong assumption"),
    ("frustration_db_locking", "DB locking"),
    ("frustration_search_recall_miss", "search recall miss"),
    ("frustration_context_injection_failure", "context injection failure"),
]
PR3_RELEVANCE_QUERY_SUITE: list[tuple[str, str]] = [
    ("pr3_knowledge_stale", "keep knowledge base from going stale"),
    ("pr3_memory_decay_refresh", "decay refresh stale memories"),
    ("pr3_status_pollution", "status notes polluting search results"),
    ("pr3_durable_decisions", "durable decisions over status updates"),
    ("pr3_rrf_relevance_design", "BrainLayer search relevance RRF design"),
    ("pr3_sqlite_vector_decision", "why did we choose sqlite vector storage"),
    ("pr3_telegram_socket_research", "research telegram socket updates"),
    ("pr3_operational_status_opt_in", "operational heartbeat status"),
    ("pr3_test_eval_opt_in", "ad-hoc eval test query"),
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


def _patch_ranx_run_fallback() -> None:
    global _RANX_RUN_FALLBACK_PATCHED
    if _RANX_RUN_FALLBACK_PATCHED:
        return

    def _init(self, name: str | None = None, run: dict[str, dict[str, Any]] | None = None, **_kwargs):
        self.name = name
        self.run = run or {}

    def _to_dict(self):
        return self.run

    def _keys(self):
        return self.run.keys()

    def _getitem(self, key: str):
        return self.run[key]

    Run.__init__ = _init
    Run.to_dict = _to_dict
    Run.keys = _keys
    Run.__getitem__ = _getitem
    _RANX_RUN_FALLBACK_PATCHED = True


def _run_from_dict(run_dict: dict[str, dict[str, float]]) -> Run:
    try:
        return Run(run=run_dict)
    except Exception:
        _patch_ranx_run_fallback()
        return Run(run=run_dict)


class SearchBenchmark:
    """Benchmarks search pipelines against graded relevance judgments."""

    def __init__(self, qrels_path: str):
        self.qrels_path = Path(qrels_path)
        self.qrels = self._load_qrels(self.qrels_path)
        self.ranx_qrels = self._build_ranx_qrels()

    def _load_qrels(self, qrels_path: Path) -> dict[str, dict[str, int]]:
        payload = json.loads(qrels_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError("Qrels JSON must be a dict of query_id -> {doc_id: grade}")
        qrels: dict[str, dict[str, int]] = {}
        for query_id, judgments in payload.items():
            normalized: dict[str, int] = {}
            for doc_id, grade in judgments.items():
                normalized_id = canonical_eval_doc_id(str(doc_id))
                normalized[normalized_id] = max(int(grade), normalized.get(normalized_id, 0))
            qrels[query_id] = normalized
        return qrels

    def _build_ranx_qrels(self) -> Qrels | None:
        try:
            return Qrels.from_dict(self.qrels)
        except Exception:
            _patch_ranx_run_fallback()
            return None

    def queries_in_qrels(self, queries: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return [query for query in queries if query[0] in self.qrels and query[1].strip()]

    def run_pipeline(
        self,
        pipeline_fn: Callable[[str], list[tuple[str, float]]],
        queries: Iterable[tuple[str, str]],
    ) -> Run:
        run_dict: dict[str, dict[str, float]] = {}
        for query_id, query_text in queries:
            results = pipeline_fn(query_text)
            run_dict[query_id] = {}
            for chunk_id, score in results:
                normalized_id = canonical_eval_doc_id(str(chunk_id))
                run_dict[query_id][normalized_id] = max(float(score), run_dict[query_id].get(normalized_id, 0.0))
        return _run_from_dict(run_dict)

    def evaluate_pipeline(self, run: Run, metrics: list[str] | None = None) -> dict[str, float]:
        metric_list = metrics or DEFAULT_RUN_METRICS
        if self.ranx_qrels is not None:
            try:
                scores = evaluate(self.ranx_qrels, run, metric_list, make_comparable=True)
                if isinstance(scores, dict):
                    return scores
                if len(metric_list) == 1:
                    return {metric_list[0]: float(scores)}
                raise TypeError(f"Unexpected Ranx evaluate() result type: {type(scores)!r}")
            except Exception:
                pass
        return self._evaluate_without_ranx(run.to_dict(), metric_list)

    def compare_pipelines(self, runs: dict[str, Run], metrics: list[str] | None = None) -> str:
        metric_list = metrics or DEFAULT_COMPARE_METRICS
        if self.ranx_qrels is not None:
            try:
                named_runs = [Run(name=name, run=run.to_dict()) for name, run in runs.items()]
                report = compare(
                    self.ranx_qrels,
                    runs=named_runs,
                    metrics=metric_list,
                    max_p=0.05,
                    rounding_digits=4,
                )
                return str(report)
            except Exception:
                pass
        lines = []
        for name, run in runs.items():
            scores = self._evaluate_without_ranx(run.to_dict(), metric_list)
            metrics_text = ", ".join(f"{metric}={score:.4f}" for metric, score in scores.items())
            lines.append(f"{name}: {metrics_text}")
        return "\n".join(lines)

    def _evaluate_without_ranx(self, run_dict: dict[str, dict[str, Any]], metrics: list[str]) -> dict[str, float]:
        return {metric: self._manual_metric(run_dict, metric) for metric in metrics}

    def _manual_metric(self, run_dict: dict[str, dict[str, Any]], metric: str) -> float:
        name, cutoff = self._parse_metric(metric)
        if name == "ndcg":
            return self._mean_query_score(run_dict, lambda qid: self._ndcg(run_dict.get(qid, {}), qid, cutoff))
        if name == "recall":
            return self._mean_query_score(run_dict, lambda qid: self._recall(run_dict.get(qid, {}), qid, cutoff))
        if name in {"precision", "p"}:
            return self._mean_query_score(run_dict, lambda qid: self._precision(run_dict.get(qid, {}), qid, cutoff))
        if name == "map":
            return self._mean_query_score(
                run_dict, lambda qid: self._average_precision(run_dict.get(qid, {}), qid, cutoff)
            )
        if name == "mrr":
            return self._mean_query_score(
                run_dict, lambda qid: self._reciprocal_rank(run_dict.get(qid, {}), qid, cutoff)
            )
        raise ValueError(f"Unsupported metric without ranx: {metric}")

    def _parse_metric(self, metric: str) -> tuple[str, int | None]:
        name, sep, cutoff_text = metric.partition("@")
        if not sep:
            return name, None
        cutoff = int(cutoff_text)
        if cutoff <= 0:
            raise ValueError(f"Metric cutoff must be > 0: {metric}")
        return name, cutoff

    def _mean_query_score(self, run_dict: dict[str, dict[str, Any]], score_fn: Callable[[str], float]) -> float:
        query_ids = sorted(self.qrels)
        if not query_ids:
            return 0.0
        return sum(score_fn(query_id) for query_id in query_ids) / len(query_ids)

    def _ranked_docs(self, run: dict[str, Any], cutoff: int | None) -> list[str]:
        ranked = [doc_id for doc_id, _score in sorted(run.items(), key=lambda item: float(item[1]), reverse=True)]
        return ranked[:cutoff] if cutoff is not None else ranked

    def _relevant_docs(self, query_id: str) -> dict[str, int]:
        return {doc_id: grade for doc_id, grade in self.qrels.get(query_id, {}).items() if grade > 0}

    def _ndcg(self, run: dict[str, Any], query_id: str, cutoff: int | None) -> float:
        ranked = self._ranked_docs(run, cutoff)
        gains = self.qrels.get(query_id, {})
        dcg = sum(gains.get(doc_id, 0) / math.log2(rank + 2) for rank, doc_id in enumerate(ranked))
        ideal_grades = sorted((grade for grade in gains.values() if grade > 0), reverse=True)
        if cutoff is not None:
            ideal_grades = ideal_grades[:cutoff]
        idcg = sum(grade / math.log2(rank + 2) for rank, grade in enumerate(ideal_grades))
        return dcg / idcg if idcg else 0.0

    def _recall(self, run: dict[str, Any], query_id: str, cutoff: int | None) -> float:
        relevant = self._relevant_docs(query_id)
        if not relevant:
            return 0.0
        retrieved = set(self._ranked_docs(run, cutoff))
        return len(retrieved & set(relevant)) / len(relevant)

    def _precision(self, run: dict[str, Any], query_id: str, cutoff: int | None) -> float:
        if cutoff is None:
            cutoff = len(run)
        if cutoff <= 0:
            return 0.0
        relevant = set(self._relevant_docs(query_id))
        if not relevant:
            return 0.0
        ranked = self._ranked_docs(run, cutoff)
        return len(set(ranked) & relevant) / cutoff

    def _average_precision(self, run: dict[str, Any], query_id: str, cutoff: int | None) -> float:
        relevant = set(self._relevant_docs(query_id))
        if not relevant:
            return 0.0
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(self._ranked_docs(run, cutoff), start=1):
            if doc_id not in relevant:
                continue
            hits += 1
            precision_sum += hits / rank
        return precision_sum / len(relevant)

    def _reciprocal_rank(self, run: dict[str, Any], query_id: str, cutoff: int | None) -> float:
        relevant = set(self._relevant_docs(query_id))
        for rank, doc_id in enumerate(self._ranked_docs(run, cutoff), start=1):
            if doc_id in relevant:
                return 1 / rank
        return 0.0


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


def pipeline_hybrid_rrf(
    store,
    query: str,
    n_results: int = 20,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    kg_boost: bool = False,
) -> list[tuple[str, float]]:
    """Hybrid search benchmark using rank-based scores from RRF ordering."""
    if not hasattr(store, "hybrid_search"):
        return pipeline_fts5_only(store, query, n_results=n_results)

    if embed_fn is None:
        from brainlayer.embeddings import embed_query

        embed_fn = embed_query

    query_embedding = embed_fn(query)
    search_kwargs = {
        "query_embedding": query_embedding,
        "query_text": query,
        "n_results": n_results,
    }
    if kg_boost:
        search_kwargs["kg_boost"] = True
    results = store.hybrid_search(
        **search_kwargs,
    )
    chunk_ids = results.get("ids", [[]])[0]
    return [(chunk_id, 1.0 / (rank + 1)) for rank, chunk_id in enumerate(chunk_ids)]


def prewarm_benchmark_embedder(model_name: str | None = None) -> Callable[[str], list[float]]:
    """Create one warmed query embedder for an entire benchmark run."""
    from brainlayer.embeddings import DEFAULT_MODEL, get_embedding_model

    model = get_embedding_model(model_name or DEFAULT_MODEL)
    model._load_model()
    return model.embed_query


def pipeline_hybrid_entity(
    store,
    query: str,
    n_results: int = 20,
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[tuple[str, float]]:
    """Hybrid RRF search with KG entity-linked chunk boosting enabled."""
    return pipeline_hybrid_rrf(store, query, n_results=n_results, embed_fn=embed_fn, kg_boost=True)
