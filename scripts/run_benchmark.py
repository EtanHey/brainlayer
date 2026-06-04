"""Run search quality benchmarks across pipeline configurations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from brainlayer.eval.benchmark import (
    DEFAULT_QUERY_SUITE,
    ReadOnlyBenchmarkStore,
    SearchBenchmark,
    pipeline_fts5_only,
    pipeline_hybrid_entity,
    pipeline_hybrid_rrf,
    prewarm_benchmark_embedder,
)
from brainlayer.paths import get_db_path
from brainlayer.vector_store import VectorStore


def _open_readonly_vector_store(db_path: Path):
    try:
        return VectorStore(db_path, readonly=True)
    except TypeError:
        return VectorStore(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=str(get_db_path()), help="Path to the BrainLayer SQLite DB")
    parser.add_argument("--qrels-path", default="tests/eval_qrels.json", help="Path to the qrels JSON file")
    parser.add_argument("--pipeline", default="fts5", choices=["fts5", "hybrid_rrf", "hybrid_entity"])
    parser.add_argument("--n-results", type=int, default=20)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to evaluate, e.g. --metrics recall@5 precision@5",
    )
    args = parser.parse_args()

    benchmark = SearchBenchmark(args.qrels_path)
    queries = benchmark.queries_in_qrels(DEFAULT_QUERY_SUITE)
    if not queries:
        raise SystemExit("No benchmark queries have both qrels and query text.")

    pipeline_name = args.pipeline
    pipeline_fn_map = {
        "fts5": pipeline_fts5_only,
        "hybrid_rrf": pipeline_hybrid_rrf,
        "hybrid_entity": pipeline_hybrid_entity,
    }

    store_factory = ReadOnlyBenchmarkStore if pipeline_name == "fts5" else _open_readonly_vector_store
    hybrid_embed_fn = prewarm_benchmark_embedder() if pipeline_name in {"hybrid_rrf", "hybrid_entity"} else None
    with store_factory(Path(args.db_path)) as store:
        run = benchmark.run_pipeline(
            lambda query_text: pipeline_fn_map[pipeline_name](
                store,
                query_text,
                n_results=args.n_results,
                **({"embed_fn": hybrid_embed_fn} if hybrid_embed_fn is not None else {}),
            ),
            queries,
        )

    if args.metrics:
        scores = benchmark.evaluate_pipeline(run, metrics=args.metrics)
    else:
        scores = benchmark.evaluate_pipeline(run)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_payload = {
        "pipeline": pipeline_name,
        "timestamp": timestamp,
        "metrics": scores,
        "query_count": len(queries),
        "qrels_path": args.qrels_path,
    }

    output_dir = Path("tests/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pipeline_name}_{timestamp}.json"
    output_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result_payload, indent=2, sort_keys=True))
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    main()
