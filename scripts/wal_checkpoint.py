#!/usr/bin/env python3
"""SQLite WAL checkpoint for BrainLayer database.

Runs PRAGMA wal_checkpoint(TRUNCATE) to compact the WAL file.
Safe to run while the DB is in use (WAL mode supports concurrent readers).

Usage:
    python3 scripts/wal_checkpoint.py           # checkpoint + report
    python3 scripts/wal_checkpoint.py --json     # JSON output for hooks
    python3 scripts/wal_checkpoint.py --quiet    # silent unless error
"""

import argparse
import json
import sys
from pathlib import Path

# Allow direct execution from a source checkout without requiring installation.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.wal_checkpoint import checkpoint, format_size, get_wal_size, resolve_db_path, run_wal_checkpoint


def main():
    parser = argparse.ArgumentParser(description="BrainLayer WAL checkpoint")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--quiet", action="store_true", help="Silent unless error")
    parser.add_argument("--mode", default="TRUNCATE",
                        choices=["PASSIVE", "FULL", "RESTART", "TRUNCATE"],
                        help="Checkpoint mode (default: TRUNCATE)")
    args = parser.parse_args()

    try:
        result = run_wal_checkpoint(args.mode)
    except FileNotFoundError:
        if args.json:
            print(json.dumps({"error": "no database found"}))
        else:
            print("No BrainLayer database found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Checkpoint failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result))
    elif not args.quiet:
        print(f"DB: {result['db']}")
        print(f"WAL: {result['wal_before']} → {result['wal_after']}")
        print(
            f"Checkpoint ({result['mode']}): {result['checkpointed_pages']}/{result['log_pages']} pages"
            f"{' (busy)' if result['busy'] else ''}"
        )

    # Exit non-zero if checkpoint was incomplete (busy)
    sys.exit(1 if result["busy"] else 0)


if __name__ == "__main__":
    main()
