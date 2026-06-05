#!/usr/bin/env python3
"""Entity-context judge CLI for KG flag-batch clusters.

Default workflow for the overnight run:
  1. --emit-prompts DIR writes self-contained prompt files for Cursor workers.
  2. --collect DIR validates worker verdict JSONs and emits merged reports.

The direct LLM path is optional fallback only: --judge groq.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.kg_judge import (  # noqa: E402
    collect_worker_verdicts,
    emit_prompt_files,
    judge_clusters_with_backend,
    load_flag_batch_clusters,
    write_verdict_outputs,
)
from brainlayer.paths import get_db_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", default=None, help="Comma-separated flag-batch categories to process")
    parser.add_argument("--flag-batch", type=Path, help="Flag-batch JSON from scripts/kg_flag_batch.py")
    parser.add_argument("--out", type=Path, default=None, help="Merged verdict JSON output path")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N clusters")
    parser.add_argument("--markdown", type=Path, default=None, help="Optional human review markdown table path")
    parser.add_argument("--db", type=Path, default=get_db_path(), help="BrainLayer SQLite DB path")
    parser.add_argument("--gits-root", type=Path, default=Path.home() / "Gits", help="Local repos root")
    parser.add_argument("--emit-prompts", type=Path, default=None, help="Write Cursor-worker judge prompts to this dir")
    parser.add_argument(
        "--collect", type=Path, default=None, help="Collect worker verdict JSON/JSONL files from this dir"
    )
    parser.add_argument("--judge", choices=["groq"], default=None, help="Optional direct LLM fallback backend")
    args = parser.parse_args()

    modes = sum(value is not None for value in (args.emit_prompts, args.collect, args.judge))
    if modes != 1:
        raise SystemExit("Choose exactly one mode: --emit-prompts DIR, --collect DIR, or --judge groq")

    if args.collect is not None:
        verdicts = collect_worker_verdicts(args.collect)
        out_path = write_verdict_outputs(verdicts, out_json=args.out, markdown_path=args.markdown, mode="collect")
        print(f"COLLECTED {len(verdicts)} verdicts to {out_path}")
        return

    if args.flag_batch is None:
        raise SystemExit("--flag-batch is required for --emit-prompts and --judge")

    clusters = load_flag_batch_clusters(args.flag_batch, categories=args.categories, limit=args.limit)
    if args.emit_prompts is not None:
        written = emit_prompt_files(clusters, args.emit_prompts, db_path=args.db, gits_root=args.gits_root)
        print(f"EMITTED {len(written)} prompts to {args.emit_prompts}")
        return

    if args.judge is not None:
        verdicts = judge_clusters_with_backend(clusters, backend=args.judge, db_path=args.db, gits_root=args.gits_root)
        out_path = write_verdict_outputs(
            verdicts, out_json=args.out, markdown_path=args.markdown, mode=f"judge:{args.judge}"
        )
        print(f"WROTE {len(verdicts)} verdicts to {out_path}")


if __name__ == "__main__":
    main()
