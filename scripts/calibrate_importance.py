#!/usr/bin/env python3
"""Phase 1: Importance calibration — heuristic SQL fix.

Usage:
    python scripts/calibrate_importance.py --dry-run   # Report only
    python scripts/calibrate_importance.py              # Apply changes
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brainlayer.calibrate import calibrate_importance
from brainlayer.paths import get_db_path
from brainlayer.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Calibrate importance scores via heuristic SQL fix")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report distribution changes without modifying DB",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Override DB path (default: canonical brainlayer.db)",
    )
    args = parser.parse_args()

    db_path = args.db or get_db_path()
    print(f"DB: {db_path}")

    store = VectorStore(db_path)
    try:
        result = calibrate_importance(store, dry_run=args.dry_run)
        _print_report(result, args.dry_run)
    finally:
        store.close()


def _print_report(result: dict, dry_run: bool):
    before = result["before"]
    after_key = "after_simulated" if dry_run else "after"
    after = result[after_key]

    print(f"\n{'=== DRY RUN ===' if dry_run else '=== APPLIED ==='}")
    print(f"\nBEFORE ({before['total']:,} chunks):")
    for bucket, cnt in sorted(before["buckets"].items()):
        pct = cnt / before["total"] * 100 if before["total"] else 0
        print(f"  {bucket:>5}: {cnt:>8,} ({pct:5.1f}%)")
    print(f"  >= 7: {before['high_count']:>8,} ({before['high_pct']:5.1f}%)")

    print(f"\nAFTER ({after['total']:,} chunks):")
    for bucket, cnt in sorted(after["buckets"].items()):
        pct = cnt / after["total"] * 100 if after["total"] else 0
        print(f"  {bucket:>5}: {cnt:>8,} ({pct:5.1f}%)")
    print(f"  >= 7: {after['high_count']:>8,} ({after['high_pct']:5.1f}%)")

    delta = after["high_pct"] - before["high_pct"]
    print(f"\nDelta: {delta:+.1f}pp (target: <20%)")


if __name__ == "__main__":
    main()
