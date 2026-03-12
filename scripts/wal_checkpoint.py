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
import os
import sqlite3
import sys

_CANONICAL_DB = os.path.expanduser("~/.local/share/brainlayer/brainlayer.db")


def get_db_path():
    env = os.environ.get("BRAINLAYER_DB")
    if env and os.path.exists(env):
        return env
    if os.path.exists(_CANONICAL_DB):
        return _CANONICAL_DB
    return None


def get_wal_size(db_path):
    """Return WAL file size in bytes, or 0 if no WAL."""
    wal_path = db_path + "-wal"
    try:
        return os.path.getsize(wal_path)
    except OSError:
        return 0


def format_size(size_bytes):
    """Human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}TB"


def checkpoint(db_path, mode="TRUNCATE"):
    """Run WAL checkpoint. Returns (busy, log_pages, checkpointed_pages)."""
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        # Ensure WAL mode
        conn.execute("PRAGMA journal_mode=WAL")
        result = conn.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        return result  # (busy, log, checkpointed)
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="BrainLayer WAL checkpoint")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--quiet", action="store_true", help="Silent unless error")
    parser.add_argument("--mode", default="TRUNCATE",
                        choices=["PASSIVE", "FULL", "RESTART", "TRUNCATE"],
                        help="Checkpoint mode (default: TRUNCATE)")
    args = parser.parse_args()

    db_path = get_db_path()
    if not db_path:
        if args.json:
            print(json.dumps({"error": "no database found"}))
        elif not args.quiet:
            print("No BrainLayer database found.", file=sys.stderr)
        sys.exit(1)

    wal_before = get_wal_size(db_path)

    try:
        busy, log_pages, checkpointed = checkpoint(db_path, args.mode)
    except sqlite3.Error as e:
        if args.json:
            print(json.dumps({"error": str(e), "db": db_path}))
        else:
            print(f"Checkpoint failed: {e}", file=sys.stderr)
        sys.exit(1)

    wal_after = get_wal_size(db_path)

    if args.json:
        print(json.dumps({
            "db": db_path,
            "mode": args.mode,
            "wal_before_bytes": wal_before,
            "wal_after_bytes": wal_after,
            "wal_before": format_size(wal_before),
            "wal_after": format_size(wal_after),
            "busy": busy,
            "log_pages": log_pages,
            "checkpointed_pages": checkpointed,
        }))
    elif not args.quiet:
        print(f"DB: {db_path}")
        print(f"WAL: {format_size(wal_before)} → {format_size(wal_after)}")
        print(f"Checkpoint ({args.mode}): {checkpointed}/{log_pages} pages"
              f"{' (busy)' if busy else ''}")

    # Exit non-zero if checkpoint was incomplete (busy)
    sys.exit(1 if busy else 0)


if __name__ == "__main__":
    main()
