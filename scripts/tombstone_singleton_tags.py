#!/usr/bin/env python3
"""Dry-run/apply singleton tag tombstoning against a BrainLayer DB."""

from __future__ import annotations

import argparse
from pathlib import Path

import apsw
import sqlite_vec

from brainlayer.paths import get_db_path
from brainlayer.tag_normalization import tombstone_singleton_tags


def _connect(db_path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(db_path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_db_path(), help="BrainLayer SQLite DB path")
    parser.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run rollback.")
    args = parser.parse_args()

    conn = _connect(args.db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        result = tombstone_singleton_tags(conn)
        if args.apply:
            conn.execute("COMMIT")
            mode = "APPLIED"
        else:
            conn.execute("ROLLBACK")
            mode = "DRY_RUN"
        print(f"{mode} tombstoned={result.tombstoned} updated_chunks={result.updated_chunks}")
        return 0
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
