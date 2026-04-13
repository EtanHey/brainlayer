#!/usr/bin/env python3
"""Weekly decay maintenance job for BrainLayer."""

from __future__ import annotations

import argparse
import json
import sys

from brainlayer.decay_job import run_decay_job
from brainlayer.paths import get_db_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BrainLayer decay maintenance")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    parser.add_argument("--quiet", action="store_true", help="Silent unless error")
    parser.add_argument("--dry-run", action="store_true", help="Compute changes without writing")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Rows per decay batch")
    args = parser.parse_args()

    db_path = get_db_path()

    try:
        stats = run_decay_job(db_path, dry_run=args.dry_run, batch_size=args.batch_size)
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "db": str(db_path)}))
        elif not args.quiet:
            print(f"Decay job failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({"db": str(db_path), **stats}))
    elif not args.quiet:
        print(
            "Decay job complete: "
            f"rows={stats['rows_processed']} archived={stats['archived_rows']} "
            f"pinned={stats['pinned_rows']} avg_decay={stats['average_decay']:.4f} "
            f"duration_s={stats['duration_seconds']:.2f} dry_run={stats['dry_run']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
