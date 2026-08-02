#!/usr/bin/env python3
"""Read-only provenance classifier report for BrainLayer chunks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import apsw

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.agent_provenance import classify_provenance, effective_visibility
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.t3_provenance import DEFAULT_T3_STATE_DB, t3_app_codex_session_ids


def build_report(db_path: str | Path) -> dict[str, Any]:
    """Classify chunks from a read-only DB connection and return JSON-safe counts."""
    db_path = Path(db_path).expanduser()
    tags: Counter[str] = Counter()
    policies: Counter[str] = Counter()
    visibility: Counter[str] = Counter()
    total = 0
    t3_state_db = Path(os.environ.get("BRAINLAYER_T3_STATE_DB", DEFAULT_T3_STATE_DB)).expanduser()
    linked_t3_session_ids = t3_app_codex_session_ids(t3_state_db) if t3_state_db.exists() else set()

    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        cursor = conn.cursor()
        for _chunk_id, source_file, content_class in cursor.execute(
            "SELECT id, source_file, content_class FROM chunks"
        ):
            decision = classify_provenance(
                str(source_file or ""), content_class, t3_linked_session_ids=linked_t3_session_ids
            )
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
