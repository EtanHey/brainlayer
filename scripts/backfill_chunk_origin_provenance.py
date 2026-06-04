#!/usr/bin/env python3
"""Backfill chunks.chunk_origin provenance for legacy unknown rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.chunk_origin_backfill import backfill_chunk_origin_provenance  # noqa: E402
from brainlayer.paths import get_db_path  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill chunks.chunk_origin for legacy unknown rows. Default mode is dry-run. "
            "Before --apply on the live DB, stop enrichment workers or coordinate a quiet window with BL-LEAD."
        )
    )
    parser.add_argument("--db", type=Path, default=get_db_path(), help="BrainLayer DB path")
    parser.add_argument("--apply", action="store_true", help="Write inferred origins to the DB")
    parser.add_argument("--batch-size", type=int, default=5000, help="Rows to scan per transaction batch")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Run WAL checkpoint every N applied batches")
    args = parser.parse_args(argv)

    result = backfill_chunk_origin_provenance(
        args.db.expanduser(),
        apply=args.apply,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )
    print(
        json.dumps(
            {
                "db": str(args.db.expanduser()),
                "mode": "apply" if args.apply else "dry-run",
                "scanned": result.scanned,
                "updated": result.updated,
                "batches": result.batches,
                "checkpoints": result.checkpoints,
                "inferred": result.inferred,
                "next": (
                    "verify chunk_origin distribution"
                    if args.apply
                    else "rerun with --apply after stopping enrichment workers or coordinating with BL-LEAD"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
