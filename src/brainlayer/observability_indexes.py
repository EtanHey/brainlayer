"""Explicit install-time index build; runtime opens never migrate these indexes."""

import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

from .disk_space import measure_disk_space

INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_observability_time ON chunks (datetime(created_at) DESC, id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_observability_source ON chunks (source_class, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_observability_author ON chunks "
    "(archived_at, provenance_class, source_class, source_file, created_at)",
)
REQUIRED_COLUMNS = frozenset(("id", "created_at", "source_class", "archived_at", "provenance_class", "source_file"))


def ensure_observability_indexes(path: Path) -> dict:
    """Run with writers stopped; WAL readers may retain checkpoint frames."""
    started = time.monotonic()
    with closing(
        sqlite3.connect(path.resolve().as_uri() + "?mode=rw", uri=True, isolation_level=None, timeout=300)
    ) as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            return {"status": "skipped", "reason": "chunks columns missing: " + ", ".join(missing)}
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        if all(sql.split()[5] in names for sql in INDEXES):
            return {"status": "present", "elapsed_seconds": time.monotonic() - started}
        count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        # Sample key widths, with headroom for b-tree pages, expression/id duplication and sorting.
        width = db.execute(
            "SELECT COALESCE(MAX(n), 0) FROM (SELECT "
            "length(CAST(COALESCE(id,'') || COALESCE(created_at,'') || COALESCE(source_class,'') || "
            "COALESCE(archived_at,'') || COALESCE(provenance_class,'') || COALESCE(source_file,'') AS BLOB)) n "
            "FROM chunks LIMIT 4096)"
        ).fetchone()[0]
        estimated = max(512 * 1024**2, count * (width + 128) * 2)
        required = 2 * estimated + 2 * 1024**3
        space = measure_disk_space(path)
        if space.free_now_bytes < required:
            return {
                "status": "skipped",
                "reason": "insufficient free-now disk space",
                "required_bytes": required,
                "free_now_bytes": space.free_now_bytes,
                "estimated_index_bytes": estimated,
            }

        checkpoints = []

        def checkpoint(phase):
            try:
                result = db.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                state = "busy" if result[0] or result[1] != result[2] else "complete"
                record = {"phase": phase, "state": state, "result": list(result)}
            except sqlite3.Error as exc:
                record = {"phase": phase, "state": "error", "reason": str(exc)}
            checkpoints.append(record)
            print("observability checkpoint: " + json.dumps(record), file=sys.stderr)

        checkpoint("before")
        try:
            db.execute("BEGIN IMMEDIATE")
            for sql in INDEXES:
                db.execute(sql)
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            checkpoint("after")
    return {
        "status": "built",
        "checkpoints": checkpoints,
        "elapsed_seconds": time.monotonic() - started,
        "estimated_index_bytes": estimated,
        "required_bytes": required,
    }
