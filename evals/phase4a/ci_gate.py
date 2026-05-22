"""Phase 4a CI gate — `bl-eval smoke` / `bl-eval full --compare-to baseline.json`.

Wraps runner.py + metrics.py for CI invocation. Returns exit codes:
- 0 PASS (all checks satisfied)
- 1 FAIL (verdict failed OR runner errored)
- 2 WARN (partial — e.g., baseline missing but eval ran)

Usage:
    python evals/phase4a/ci_gate.py smoke
        # Runs sentinel.yaml, asserts every query returns non-empty within latency budget

    python evals/phase4a/ci_gate.py full --baseline evals/phase4a/baseline.json
        # Runs all 80 queries, compares to baseline, asserts no category regresses >5%
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Module-relative imports
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from runner import run_eval  # noqa: E402
from metrics import compute_all, compare_to_baseline  # noqa: E402


def smoke(args: argparse.Namespace) -> int:
    queries_path = args.queries or (HERE / "sentinel.yaml")
    output_path = args.output or Path("/tmp/phase4a-smoke-results.json")
    print(f"=== smoke: running {queries_path.name} ===")

    summary = run_eval(queries_path, output_path, n_results=args.num or 5)

    # Smoke gates:
    # 1. Wall-clock total < 30s
    # 2. Every query returns at least 1 chunk OR has explicit `expected_min_recall_at_20: 0`
    # 3. No query timed out
    elapsed_s = summary.get("elapsed_total_seconds", 0)
    if elapsed_s > 30:
        print(f"FAIL: smoke total wall-clock {elapsed_s:.1f}s > 30s budget")
        return 1

    n_failures = 0
    for r in summary.get("results", []):
        qid = r.get("id", "?")
        if r.get("error"):
            print(f"FAIL: {qid} errored: {r['error'][:120]}")
            n_failures += 1
            continue
        if r.get("n_returned", 0) == 0:
            print(f"FAIL: {qid} returned no results (query: {r.get('query', '')[:50]})")
            n_failures += 1

    if n_failures:
        print(f"=== smoke FAILED: {n_failures} sentinel queries failed ===")
        return 1

    lat = summary.get("aggregate_latency", {})
    print(
        f"=== smoke PASSED: {summary['n_queries']} queries in {elapsed_s:.1f}s "
        f"(p50={lat.get('p50_ms', 0):.0f}ms p95={lat.get('p95_ms', 0):.0f}ms max={lat.get('max_ms', 0):.0f}ms) ==="
    )
    return 0


def full(args: argparse.Namespace) -> int:
    queries_path = args.queries or (HERE / "queries.yaml")
    output_path = args.output or Path("/tmp/phase4a-full-results.json")
    print(f"=== full: running {queries_path.name} ===")

    summary = run_eval(queries_path, output_path, n_results=args.num or 20)
    metrics = compute_all(summary)

    # If a baseline is supplied, compare
    baseline_path = args.baseline
    thresholds_path = args.thresholds or (HERE / "thresholds.yaml")
    if baseline_path and baseline_path.exists() and thresholds_path.exists():
        try:
            import yaml
        except ImportError:
            print("WARN: PyYAML not available — cannot load thresholds; skipping comparison")
            return 2

        baseline_data = json.loads(baseline_path.read_text())
        thresholds_data = yaml.safe_load(thresholds_path.read_text())
        # If baseline is itself a metrics output (compute_all shape), use directly
        baseline_metrics = baseline_data if "per_category" in baseline_data else compute_all(baseline_data)
        verdict = compare_to_baseline(metrics, baseline_metrics, thresholds_data)

        # Persist current run + verdict
        if args.output:
            args.output.write_text(json.dumps({
                "summary": summary,
                "metrics": metrics,
                "verdict": verdict,
            }, indent=2, ensure_ascii=False))

        if verdict["passed"]:
            print(f"=== full PASSED ({metrics['n_queries']} queries) ===")
            print(f"  overall: recall@20={metrics['overall']['recall_at_20_mean']} "
                  f"ndcg@10={metrics['overall']['ndcg_at_10_mean']}")
            return 0
        else:
            print(f"=== full FAILED: {verdict['n_failures']} threshold breaches ===")
            for f in verdict["failures"]:
                print(f"  - {f}")
            return 1

    # No baseline: just emit metrics for review (WARN exit)
    print(f"=== full eval run COMPLETE (no baseline for comparison; WARN exit) ===")
    print(json.dumps(metrics["overall"], indent=2))
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bl-eval", description="Phase 4a CI gate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke", help="Run sentinel queries (target <30s)")
    p_smoke.add_argument("--queries", type=Path, help="Override sentinel.yaml path")
    p_smoke.add_argument("--output", type=Path, help="Override output JSON path")
    p_smoke.add_argument("--num", type=int, help="Override num_results per query")

    p_full = sub.add_parser("full", help="Run all 80 queries + optional baseline compare")
    p_full.add_argument("--queries", type=Path, help="Override queries.yaml path")
    p_full.add_argument("--output", type=Path, help="Override output JSON path")
    p_full.add_argument("--baseline", type=Path, help="baseline.json for regression comparison")
    p_full.add_argument("--thresholds", type=Path, help="Override thresholds.yaml path")
    p_full.add_argument("--num", type=int, help="Override num_results per query")

    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        return smoke(args)
    if args.cmd == "full":
        return full(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
