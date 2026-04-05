#!/usr/bin/env python3
"""Mine frustration chunks into Ranx qrels entries."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.frustration_mining import (
    append_qrels,
    generate_qrels,
    mine_frustration_pairs,
    write_qrels_file,
)


def load_existing_qrels(path: Path) -> dict[str, dict[str, int]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid qrels payload in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default=str(get_db_path()), help="Path to the BrainLayer SQLite DB")
    parser.add_argument("--output", default="tests/eval_qrels.json", help="Qrels file to append entries to")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of mined pairs")
    parser.add_argument("--json", action="store_true", help="Emit stats as JSON")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify the qrels file")
    args = parser.parse_args()

    with sqlite3.connect(args.db_path) as conn:
        pairs = mine_frustration_pairs(conn)

    if args.limit > 0:
        pairs = pairs[: args.limit]

    new_qrels = generate_qrels(pairs)
    output_path = Path(args.output)
    existing_qrels = load_existing_qrels(output_path)
    merged_qrels = append_qrels(existing_qrels, new_qrels)

    stats = {
        "db_path": str(args.db_path),
        "pairs_mined": len(pairs),
        "new_queries": len(new_qrels),
        "total_queries": len(merged_qrels),
        "output": str(output_path),
    }

    if not args.dry_run:
        write_qrels_file(output_path, merged_qrels)

    if args.json:
        print(json.dumps(stats, indent=2, sort_keys=True))
        return

    print(f"Mined {stats['pairs_mined']} frustration pairs")
    print(f"Added {stats['new_queries']} qrels entries")
    print(f"Total qrels queries: {stats['total_queries']}")
    print(f"Output: {stats['output']}")


if __name__ == "__main__":
    main()
