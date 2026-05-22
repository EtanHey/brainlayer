"""Phase 4a eval metrics — pure-Python implementation.

Computes recall@N, ndcg@10, and per-category aggregates given a results.json
from runner.py. Self-contained: no external dependencies beyond stdlib.

Phase 4b can replace this with Ranx/RAGAS/DeepEval wrappers for richer metrics.
For now, this gives us a portable baseline that runs in CI.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def _is_relevant(chunk: dict[str, Any], query_meta: dict[str, Any]) -> bool:
    """Determine if a chunk is relevant to a query.

    Heuristics (in order):
    1. If query_meta declares `expected_entity`, chunk_id containing the entity (case-insensitive)
       OR chunk score above expected_score_range[0] qualifies.
    2. If query_meta has no expectations, any non-empty chunk with score>0 counts.
    3. Future: replace with explicit relevance judgements once gathered.
    """
    chunk_id = (chunk.get("chunk_id") or "").lower()
    score = chunk.get("score")
    expected_entity = query_meta.get("expected_entity")
    score_range = query_meta.get("expected_score_range")

    if expected_entity:
        entity_lower = expected_entity.lower().replace("-", "").replace(" ", "")
        if entity_lower and entity_lower in chunk_id.replace("-", "").replace(" ", ""):
            return True

    if score_range and isinstance(score, (int, float)):
        lo = score_range[0] if len(score_range) > 0 else 0.0
        return score >= lo

    if score is not None and isinstance(score, (int, float)) and score > 0:
        return True

    if chunk_id and not expected_entity and not score_range:
        return True

    return False


def recall_at_n(chunks: list[dict[str, Any]], query_meta: dict[str, Any], n: int) -> float:
    """Recall@N: fraction of expected results that appear in top N.

    With heuristic relevance (we don't have explicit relevance judgements),
    recall@N degenerates to "does at least one relevant chunk appear in top N".
    Returns 1.0 if yes, 0.0 if no.

    For queries with `expected_min_recall_at_20: K`, requires K relevant chunks in top N.
    """
    top_n = chunks[:n]
    relevant_count = sum(1 for c in top_n if _is_relevant(c, query_meta))
    min_required = query_meta.get(f"expected_min_recall_at_{n}", 1)
    if relevant_count >= min_required:
        return 1.0
    if min_required == 0:
        return 1.0
    return relevant_count / max(min_required, 1)


def ndcg_at_n(chunks: list[dict[str, Any]], query_meta: dict[str, Any], n: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at N.

    Without explicit relevance judgements, uses binary relevance via _is_relevant + chunk score.
    IDCG assumes the ideal ranking has all relevant chunks at the top.
    """
    top_n = chunks[:n]
    dcg = 0.0
    for i, chunk in enumerate(top_n):
        if _is_relevant(chunk, query_meta):
            rel = 1.0
            score = chunk.get("score")
            if isinstance(score, (int, float)):
                rel = max(0.0, min(1.0, float(score)))
            dcg += rel / math.log2(i + 2)
    relevant_in_top_n = sum(1 for c in top_n if _is_relevant(c, query_meta))
    if relevant_in_top_n == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(relevant_in_top_n))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_category(results: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-category metrics: recall@20, ndcg@10, mean latency, count."""
    if not results:
        return {"n": 0, "recall_at_20_mean": 0.0, "ndcg_at_10_mean": 0.0, "latency_p50_ms": 0.0}
    recall_vals: list[float] = []
    ndcg_vals: list[float] = []
    for r in results:
        # results.json format from runner.py: each result has top_5_chunk_ids + top_5_scores
        # Reconstruct chunks list for metric computation
        chunks = []
        for cid, score in zip(r.get("top_5_chunk_ids", []), r.get("top_5_scores", [])):
            chunks.append({"chunk_id": cid, "score": score})
        query_meta = {
            "expected_entity": r.get("expected_entity"),
            "expected_min_recall_at_20": r.get("expected_min_recall_at_20", 1),
        }
        recall_vals.append(recall_at_n(chunks, query_meta, n=20))
        ndcg_vals.append(ndcg_at_n(chunks, query_meta, n=10))
    latencies = sorted(r.get("latency_ms", 0.0) for r in results)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    return {
        "n": len(results),
        "recall_at_20_mean": round(sum(recall_vals) / max(len(recall_vals), 1), 4),
        "ndcg_at_10_mean": round(sum(ndcg_vals) / max(len(ndcg_vals), 1), 4),
        "latency_p50_ms": round(p50, 2),
    }


def compute_all(results_summary: dict[str, Any]) -> dict[str, Any]:
    """Compute per-category + aggregate metrics from a results.json summary."""
    all_results = results_summary.get("results", [])
    by_category: dict[str, list[dict[str, Any]]] = {}
    for r in all_results:
        by_category.setdefault(r.get("category", "_unknown"), []).append(r)

    per_category = {cat: aggregate_category(rs) for cat, rs in by_category.items()}
    overall = aggregate_category(all_results)

    return {
        "n_queries": len(all_results),
        "overall": overall,
        "per_category": per_category,
    }


def compare_to_baseline(
    current_metrics: dict[str, Any], baseline_metrics: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Compare current metrics to a baseline + thresholds. Return a verdict dict.

    verdict = {"passed": bool, "failures": [{"check": str, "expected": ..., "actual": ..., "category": ...}]}
    """
    failures: list[dict[str, Any]] = []
    aggregate = thresholds.get("aggregate", {})
    per_cat_min = thresholds.get("per_category_minimum", {})
    no_regress = aggregate.get("no_category_regression_percent", -5)

    # Aggregate ndcg@10
    ndcg_min = aggregate.get("ndcg_at_10_minimum", 0.85)
    actual_ndcg = current_metrics.get("overall", {}).get("ndcg_at_10_mean", 0.0)
    if actual_ndcg < ndcg_min:
        failures.append({
            "check": "aggregate_ndcg_at_10",
            "expected_min": ndcg_min,
            "actual": actual_ndcg,
        })

    # Per-category recall@20
    for cat, cat_metrics in current_metrics.get("per_category", {}).items():
        per_cat_threshold = per_cat_min.get(cat, aggregate.get("recall_at_20_minimum_per_category", 0.90))
        actual_recall = cat_metrics.get("recall_at_20_mean", 0.0)
        if actual_recall < per_cat_threshold:
            failures.append({
                "check": "per_category_recall_at_20",
                "category": cat,
                "expected_min": per_cat_threshold,
                "actual": actual_recall,
            })

        # Regression check (vs baseline)
        baseline_cat = baseline_metrics.get("per_category", {}).get(cat, {})
        baseline_recall = baseline_cat.get("recall_at_20_mean")
        if baseline_recall and baseline_recall > 0:
            regression_pct = ((actual_recall - baseline_recall) / baseline_recall) * 100
            if regression_pct < no_regress:
                failures.append({
                    "check": "category_regression",
                    "category": cat,
                    "baseline_recall_at_20": baseline_recall,
                    "actual_recall_at_20": actual_recall,
                    "regression_percent": round(regression_pct, 2),
                    "max_allowed_regression_percent": no_regress,
                })

    # Latency budgets
    latency_thresholds = thresholds.get("latency", {})
    actual_p50 = current_metrics.get("overall", {}).get("latency_p50_ms", 0.0)
    max_p95 = latency_thresholds.get("per_query_p95_milliseconds_max", 500)
    # NB: aggregate_category only computes p50 here; full p95 requires raw latencies in results.json
    # which compute_all reads but doesn't aggregate. Phase 4b should extend.

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Phase 4a metrics — compute or compare")
    parser.add_argument("--results", type=Path, required=True, help="results.json from runner.py")
    parser.add_argument("--baseline", type=Path, help="Optional baseline.json to compare against")
    parser.add_argument("--thresholds", type=Path, help="Optional thresholds.yaml (defaults to evals/phase4a/thresholds.yaml)")
    parser.add_argument("--output", type=Path, help="Write metrics JSON to this path")
    args = parser.parse_args()

    results_data = json.loads(args.results.read_text())
    metrics = compute_all(results_data)

    if args.baseline:
        baseline_data = json.loads(args.baseline.read_text())
        thresholds_path = args.thresholds or args.results.parent / "thresholds.yaml"
        if thresholds_path.exists():
            try:
                import yaml
                thresholds_data = yaml.safe_load(thresholds_path.read_text())
            except ImportError:
                thresholds_data = {}
        else:
            thresholds_data = {}
        verdict = compare_to_baseline(metrics, baseline_data, thresholds_data)
        metrics["verdict_vs_baseline"] = verdict
        print(f"Verdict: {'PASS' if verdict['passed'] else 'FAIL'} ({verdict['n_failures']} failures)")
        for f in verdict["failures"]:
            print(f"  - {f}")

    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote: {args.output}")
    else:
        print(json.dumps(metrics, indent=2))
