"""Phase 4a eval runner — invokes brain_search via the `brainlayer search` CLI.

This uses subprocess rather than the DaemonClient Python API because the CLI is
the canonical user-facing entry point and avoids daemon-startup latency in batch runs.

Captures per-query latency + result chunk IDs (parsed from CLI text output). Writes JSON.

Usage:
    python evals/phase4a/runner.py \\
        --queries evals/phase4a/queries.yaml \\
        --output evals/phase4a/results.json

    # Or for the sentinel subset (fast pre-commit smoke):
    python evals/phase4a/runner.py \\
        --queries evals/phase4a/sentinel.yaml \\
        --output /tmp/sentinel-results.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.stderr.write("ERROR: PyYAML required. Install: pip install pyyaml\n")
    sys.exit(1)


CHUNK_ID_RE = re.compile(r"\b([a-z0-9]+-[a-f0-9]{6,}|[a-z0-9_-]{10,}-[a-f0-9]{8,})\b", re.IGNORECASE)
SCORE_RE = re.compile(r"score[:\s=]+([0-9.]+)", re.IGNORECASE)


def _load_queries(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load query set from queries.yaml (categorized) or sentinel.yaml (flat sentinel: [...] key)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected dict at top level, got {type(data).__name__}")
    if "sentinel" in data and isinstance(data["sentinel"], list):
        return {"sentinel": data["sentinel"]}
    return {cat: qlist for cat, qlist in data.items() if isinstance(qlist, list)}


def _resolve_brainlayer_bin() -> str:
    """Find brainlayer CLI executable."""
    bin_path = shutil.which("brainlayer")
    if bin_path:
        return bin_path
    candidates = [
        Path.home() / ".local" / "bin" / "brainlayer",
        Path("/Library/Frameworks/Python.framework/Versions/3.13/bin/brainlayer"),
        Path("/usr/local/bin/brainlayer"),
    ]
    for p in candidates:
        if p.is_file() and p.is_absolute():
            return str(p)
    sys.stderr.write("ERROR: brainlayer CLI not found in PATH or known locations\n")
    sys.exit(1)


def _run_query_via_cli(
    bl_bin: str, query: str, n_results: int, timeout_s: float = 30.0
) -> tuple[str, float, str | None]:
    """Run `brainlayer search` via subprocess. Returns (stdout, wall_ms, error or None)."""
    t0 = time.perf_counter_ns()
    try:
        result = subprocess.run(
            [bl_bin, "search", query, "--num", str(n_results)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        if result.returncode != 0:
            return result.stdout or "", elapsed_ms, f"exit={result.returncode}: {result.stderr[:200]}"
        return result.stdout, elapsed_ms, None
    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return "", elapsed_ms, f"timeout after {timeout_s}s"
    except Exception as exc:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return "", elapsed_ms, f"{type(exc).__name__}: {exc}"


def _parse_cli_output(stdout: str, max_chunks: int = 20) -> list[dict[str, Any]]:
    """Best-effort parse of `brainlayer search` text output.

    The CLI uses Rich formatting; we extract chunk IDs and scores via regex.
    For Phase 4a runner this is sufficient — exact chunk matching is what
    eval cares about. Phase 4b should use a structured output flag if available.
    """
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line in stdout.splitlines():
        cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", line)
        for match in CHUNK_ID_RE.finditer(cleaned):
            chunk_id = match.group(1)
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            score_match = SCORE_RE.search(cleaned)
            chunks.append({
                "chunk_id": chunk_id,
                "score": float(score_match.group(1)) if score_match else None,
            })
            if len(chunks) >= max_chunks:
                return chunks
    return chunks


def run_eval(
    queries_path: Path,
    output_path: Path,
    n_results: int = 20,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    bl_bin = _resolve_brainlayer_bin()
    queries_by_category = _load_queries(queries_path)

    started_at = time.time()
    per_query_results: list[dict[str, Any]] = []
    per_category_latencies: dict[str, list[float]] = {}

    for cat, qlist in queries_by_category.items():
        per_category_latencies[cat] = []
        for q in qlist:
            qid = q.get("id", f"{cat}-{len(per_query_results):03d}")
            query_text = q.get("query", "")
            if not query_text:
                continue
            stdout, elapsed_ms, err = _run_query_via_cli(bl_bin, query_text, n_results, timeout_s)
            chunks = _parse_cli_output(stdout, max_chunks=n_results)
            per_category_latencies[cat].append(elapsed_ms)
            per_query_results.append({
                "id": qid,
                "category": cat,
                "query": query_text,
                "expected_entity": q.get("expected_entity"),
                "n_returned": len(chunks),
                "latency_ms": round(elapsed_ms, 2),
                "top_5_chunk_ids": [c["chunk_id"] for c in chunks[:5]],
                "top_5_scores": [c.get("score") for c in chunks[:5]],
                "error": err,
            })

    elapsed_total_s = time.time() - started_at

    def _stats(latencies: list[float]) -> dict[str, Any]:
        if not latencies:
            return {"n": 0}
        sorted_l = sorted(latencies)
        n = len(sorted_l)
        return {
            "n": n,
            "min_ms": round(sorted_l[0], 2),
            "p50_ms": round(sorted_l[n // 2], 2),
            "p95_ms": round(sorted_l[min(int(n * 0.95), n - 1)], 2),
            "max_ms": round(sorted_l[-1], 2),
        }

    all_latencies = [r["latency_ms"] for r in per_query_results if r.get("latency_ms") is not None]

    summary = {
        "started_at": started_at,
        "elapsed_total_seconds": round(elapsed_total_s, 2),
        "n_queries": len(per_query_results),
        "n_categories": len(per_category_latencies),
        "aggregate_latency": _stats(all_latencies),
        "per_category_latency": {cat: _stats(lats) for cat, lats in per_category_latencies.items()},
        "results": per_query_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 4a eval runner")
    parser.add_argument("--queries", type=Path, required=True, help="Path to queries.yaml or sentinel.yaml")
    parser.add_argument("--output", type=Path, required=True, help="Path to JSON output file")
    parser.add_argument("--num", "-n", type=int, default=20, help="Number of results per query (default: 20)")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-query timeout in seconds (default: 30)")
    args = parser.parse_args(argv)

    if not args.queries.exists():
        sys.stderr.write(f"ERROR: queries file not found: {args.queries}\n")
        return 1

    summary = run_eval(args.queries, args.output, n_results=args.num, timeout_s=args.timeout)
    lat = summary["aggregate_latency"]
    print(f"Phase 4a eval complete: {summary['n_queries']} queries in {summary['elapsed_total_seconds']:.1f}s")
    print(f"  aggregate latency: p50={lat.get('p50_ms', 'n/a')}ms p95={lat.get('p95_ms', 'n/a')}ms max={lat.get('max_ms', 'n/a')}ms")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
