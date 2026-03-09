#!/usr/bin/env python3
"""Run all eval cases and save scored results to tests/eval_baselines.json.

Usage:
    python scripts/run_evals.py               # run + save + print summary
    python scripts/run_evals.py --no-save     # run + print, don't save
    python scripts/run_evals.py --diff        # compare to saved baseline

This script calls run_baseline() and run_hook_baseline() from test_eval_baselines.py
and writes the combined results to tests/eval_baselines.json.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src is importable
src = Path(__file__).parent.parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

tests_dir = Path(__file__).parent.parent / "tests"
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))

from test_eval_baselines import run_baseline, run_hook_baseline

BASELINE_FILE = Path(__file__).parent.parent / "tests" / "eval_baselines.json"


def main():
    parser = argparse.ArgumentParser(description="Run BrainLayer eval suite")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    parser.add_argument("--diff", action="store_true", help="Compare to saved baseline")
    args = parser.parse_args()

    # Load previous baseline for diff
    prev = None
    if args.diff and BASELINE_FILE.exists():
        prev = json.loads(BASELINE_FILE.read_text())

    print("Running search quality evals...")
    search_results = run_baseline()
    print("Running hook entity injection evals...")
    hook_results = run_hook_baseline()

    combined = {
        "search": search_results,
        "hook": hook_results,
        "combined_score_pct": round(
            (search_results["pass_count"] + hook_results["pass_count"])
            / (search_results["total"] + hook_results["total"]) * 100,
            1,
        ),
    }

    # Print summary
    print(f"\n=== brain_search quality ===")
    print(f"Score: {search_results['pass_count']}/{search_results['total']} ({search_results['score_pct']}%)")
    if prev:
        prev_pct = prev.get("search", {}).get("score_pct", 0)
        delta = search_results["score_pct"] - prev_pct
        print(f"Delta vs baseline: {delta:+.1f}%")
    for case in search_results["cases"]:
        status = "✓" if case["passed"] else "✗"
        rank = f"rank={case['actual_rank']}" if case["actual_rank"] else "not found"
        print(f"  {status} [{case['name']}] {rank}")
        if not case["passed"]:
            print(f"       top: {case['top_snippet'][:70]!r}")

    print(f"\n=== hook entity injection ===")
    print(f"Score: {hook_results['pass_count']}/{hook_results['total']} ({hook_results['score_pct']}%)")
    if prev:
        prev_hook_pct = prev.get("hook", {}).get("score_pct", 0)
        hook_delta = hook_results["score_pct"] - prev_hook_pct
        print(f"Delta vs baseline: {hook_delta:+.1f}%")
    for case in hook_results["cases"]:
        status = "✓" if case["passed"] else "✗"
        print(f"  {status} [{case['name']}]")
        if not case["passed"]:
            print(f"       output: {case['output_preview'][:80]!r}")

    print(f"\nCombined: {combined['combined_score_pct']}%")
    if prev:
        prev_combined = prev.get("combined_score_pct", 0)
        print(f"Delta vs baseline: {combined['combined_score_pct'] - prev_combined:+.1f}%")

    if not args.no_save:
        BASELINE_FILE.write_text(json.dumps(combined, indent=2))
        print(f"\nSaved to {BASELINE_FILE}")


if __name__ == "__main__":
    main()
