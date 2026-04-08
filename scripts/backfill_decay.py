#!/usr/bin/env python3

import argparse

from brainlayer.decay_backfill import backfill_decay_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill decay fields for existing chunks.")
    parser.add_argument("--db", required=True, help="Path to the BrainLayer SQLite database")
    parser.add_argument("--dry-run", action="store_true", help="Compute updates without writing them")
    args = parser.parse_args()

    stats = backfill_decay_fields(args.db, dry_run=args.dry_run)
    print(
        f"backfill_decay updated_rows={stats['updated_rows']} dry_run={args.dry_run} db={args.db}"
    )


if __name__ == "__main__":
    main()
