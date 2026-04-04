#!/usr/bin/env python3
"""Promote high-frequency concept tags into KG entities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.tag_entity_promotion import promote_tag_entities
from brainlayer.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote high-frequency chunk tags into KG entities")
    parser.add_argument("--min-count", type=int, default=500, help="Minimum tagged chunk count to promote")
    parser.add_argument("--limit", type=int, default=None, help="Optional candidate limit")
    parser.add_argument("--dry-run", action="store_true", help="Show candidates without writing")
    args = parser.parse_args()

    store = None
    try:
        store = VectorStore(get_db_path())
        stats = promote_tag_entities(
            store,
            min_count=args.min_count,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
