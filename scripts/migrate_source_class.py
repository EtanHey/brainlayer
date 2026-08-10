#!/usr/bin/env python3
"""Run the supervised source_class migration on an offline database copy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainlayer.source_class_migration import migrate_source_class


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="offline database copy")
    parser.add_argument("--git-sha", required=True, help="exact 40-character migration commit SHA")
    parser.add_argument("--actor", required=True, help="operator identity for the ledger")
    parser.add_argument("--batch-size", type=int, default=5_000)
    args = parser.parse_args()
    receipt = migrate_source_class(
        args.db,
        git_sha=args.git_sha,
        actor=args.actor,
        batch_size=args.batch_size,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
