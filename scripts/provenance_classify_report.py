#!/usr/bin/env python3
"""Read-only provenance classifier report for BrainLayer chunks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import apsw

from brainlayer.agent_provenance import classify_provenance, effective_visibility
from brainlayer.paths import DEFAULT_DB_PATH


def build_report(db_path: str | Path) -> dict[str, Any]:
    """Classify chunks from a read-only DB connection and return JSON-safe counts."""
    db_path = Path(db_path).expanduser()
    tags: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    visibility: Counter[str] = Counter()
    total = 0

    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        cursor = conn.cursor()
        for _chunk_id, source_file, content_class in cursor.execute(
            "SELECT id, source_file, content_class FROM chunks"
        ):
            decision = classify_provenance(str(source_file or ""), content_class)
            tags[decision.provenance_tag] += 1
            policies[decision.search_policy] += 1
            visibility[effective_visibility(decision, content_class)] += 1
            total += 1
    finally:
        conn.close()

    return {
        "dry_run": True,
        "db_path": str(db_path),
        "total_chunks": total,
        "tags": dict(sorted(tags.items())),
        "policies": {policy: policies.get(policy, 0) for policy in ("KEEP", "ISOLATE", "OUT")},
        "effective_visibility": {bucket: visibility.get(bucket, 0) for bucket in ("cold", "default", "operational")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="BrainLayer SQLite DB path")
    parser.add_argument("--dry-run", action="store_true", default=True, help="No-op flag; report is always read-only")
    args = parser.parse_args()

    print(json.dumps(build_report(args.db), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
