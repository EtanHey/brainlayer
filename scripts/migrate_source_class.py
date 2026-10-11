#!/usr/bin/env python3
"""Run the source_class migration on a copy or a supervised, gated live database."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainlayer.source_class_migration import migrate_source_class, reclassify_t3_source_class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path, required=True, help="copy or canonical DB (writes require the supervised gate)"
    )
    parser.add_argument("--git-sha", required=True, help="exact 40-character migration commit SHA")
    parser.add_argument("--actor", required=True, help="operator identity for the ledger")
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--t3-only", action="store_true", help="correct desktop T3 rows; read-only dry-run by default")
    parser.add_argument(
        "--apply", action="store_true", help="apply --t3-only; canonical writes require the supervised gate"
    )
    args = parser.parse_args()
    if args.apply and not args.t3_only:
        parser.error("--apply requires --t3-only; the legacy supervised migration already applies")
    migrate = reclassify_t3_source_class if args.t3_only else migrate_source_class
    receipt = migrate(
        args.db,
        git_sha=args.git_sha,
        actor=args.actor,
        batch_size=args.batch_size,
        **({"apply": args.apply} if args.t3_only else {}),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
