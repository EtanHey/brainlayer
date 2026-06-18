#!/usr/bin/env python3
"""Benchmark before/after DB-shrink snapshots and persist the gate result."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brainlayer.db_shrink import dbstat_sizes
from brainlayer.eval.benchmark import (
    DEFAULT_QUERY_SUITE,
    ReadOnlyBenchmarkStore,
    SearchBenchmark,
    _run_from_dict,
    canonical_eval_doc_id,
    pipeline_fts5_only,
    prewarm_benchmark_embedder,
)
from brainlayer.vector_store import VectorStore


def _run_pipeline(
    db_path: Path,
    qrels_path: str,
    pipeline: str,
    n_results: int,
    *,
    query_timeout_ms: int,
) -> dict[str, Any]:
    benchmark = SearchBenchmark(qrels_path)
    queries = benchmark.queries_in_qrels(DEFAULT_QUERY_SUITE)
    if not queries:
        raise SystemExit("No benchmark queries have both qrels and query text.")

    timed_out_queries: list[str] = []

    def _run_with_progress(pipeline_fn):
        run_dict: dict[str, dict[str, float]] = {}
        total = len(queries)
        for index, (query_id, query_text) in enumerate(queries, start=1):
            print(f"[{pipeline}] {db_path.name} query {index}/{total}: {query_id}", flush=True)
            results = pipeline_fn(query_id, query_text)
            run_dict[query_id] = {}
            for chunk_id, score in results:
                normalized_id = canonical_eval_doc_id(str(chunk_id))
                run_dict[query_id][normalized_id] = max(float(score), run_dict[query_id].get(normalized_id, 0.0))
        return _run_from_dict(run_dict)

    if pipeline == "fts5":
        with ReadOnlyBenchmarkStore(db_path) as store:
            def _fts_query(query_id: str, query_text: str):
                deadline = time.monotonic() + (query_timeout_ms / 1000.0)
                store.conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 1000)
                try:
                    return pipeline_fts5_only(store, query_text, n_results=n_results)
                except sqlite3.OperationalError as exc:
                    if "interrupted" not in str(exc).lower():
                        raise
                    timed_out_queries.append(query_id)
                    return []
                finally:
                    store.conn.set_progress_handler(None, 0)

            run = _run_with_progress(_fts_query)
    elif pipeline == "hybrid_rrf":
        embed_fn = prewarm_benchmark_embedder()
        with VectorStore(db_path, readonly=True) as store:
            run = _run_with_progress(
                lambda _query_id, query_text: [
                    (chunk_id, 1.0 / (rank + 1))
                    for rank, chunk_id in enumerate(
                        store.hybrid_search(
                            query_embedding=embed_fn(query_text),
                            query_text=query_text,
                            n_results=n_results,
                            brainbar_helper_fast_profile=True,
                        ).get("ids", [[]])[0]
                    )
                ]
            )
    else:
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    return {
        "metrics": benchmark.evaluate_pipeline(run),
        "query_count": len(queries),
        "query_timeout_ms": query_timeout_ms,
        "timed_out_queries": timed_out_queries,
    }


def _gate(before: dict[str, Any], after: dict[str, Any], *, max_regression: float) -> dict[str, Any]:
    gated_metrics = ("ndcg@3", "ndcg@10", "recall@20")
    regressions: list[dict[str, Any]] = []
    for pipeline, before_result in before.items():
        after_result = after[pipeline]
        for metric in gated_metrics:
            before_value = float(before_result["metrics"].get(metric, 0.0))
            after_value = float(after_result["metrics"].get(metric, 0.0))
            delta = after_value - before_value
            if delta < -max_regression:
                regressions.append(
                    {
                        "pipeline": pipeline,
                        "metric": metric,
                        "before": before_value,
                        "after": after_value,
                        "delta": delta,
                    }
                )
    return {
        "verdict": "pass" if not regressions else "fail",
        "max_regression": max_regression,
        "regressions": regressions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-db", required=True)
    parser.add_argument("--after-db", required=True)
    parser.add_argument("--qrels-path", default="tests/eval_qrels.json")
    parser.add_argument("--pipeline", action="append", choices=["fts5", "hybrid_rrf"], dest="pipelines")
    parser.add_argument("--n-results", type=int, default=20)
    parser.add_argument("--max-regression", type=float, default=0.005)
    parser.add_argument("--query-timeout-ms", type=int, default=10_000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pipelines = args.pipelines or ["fts5", "hybrid_rrf"]
    before_db = Path(args.before_db).expanduser().resolve()
    after_db = Path(args.after_db).expanduser().resolve()
    before_results = {
        pipeline: _run_pipeline(
            before_db,
            args.qrels_path,
            pipeline,
            args.n_results,
            query_timeout_ms=args.query_timeout_ms,
        )
        for pipeline in pipelines
    }
    after_results = {
        pipeline: _run_pipeline(
            after_db,
            args.qrels_path,
            pipeline,
            args.n_results,
            query_timeout_ms=args.query_timeout_ms,
        )
        for pipeline in pipelines
    }
    gate = _gate(before_results, after_results, max_regression=args.max_regression)
    before_bytes = before_db.stat().st_size
    after_bytes = after_db.stat().st_size

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = Path(args.output) if args.output else Path("eval_results") / "db-shrink-2026-06-01.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "before_db": str(before_db),
        "after_db": str(after_db),
        "before_bytes": before_bytes,
        "after_bytes": after_bytes,
        "reclaimed_bytes": max(0, before_bytes - after_bytes),
        "pipelines": pipelines,
        "before": before_results,
        "after": after_results,
        "gate": gate,
        "dbstat_top_before": dict(list(dbstat_sizes(before_db).items())[:20]),
        "dbstat_top_after": dict(list(dbstat_sizes(after_db).items())[:20]),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved DB-shrink eval to {output_path}")
    return 0 if gate["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
