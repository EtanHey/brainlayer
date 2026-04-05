#!/usr/bin/env python3
"""Mine correction-tagged chunks into correction pairs and promote them into the KG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.paths import get_db_path
from brainlayer.pipeline.correction_mining import mine_corrections, promote_corrections
from brainlayer.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine correction chunks into KG entities and aliases")
    parser.add_argument("--min-confidence", type=float, default=0.8, help="Minimum confidence to promote into KG")
    parser.add_argument("--skip-promotion", action="store_true", help="Only stage correction pairs")
    args = parser.parse_args()

    store = None
    try:
        store = VectorStore(get_db_path())
        stats = {"mining": mine_corrections(store)}
        if not args.skip_promotion:
            stats["promotion"] = promote_corrections(store, min_confidence=args.min_confidence)
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
