"""One-off repair for chunks whose project was derived from a date directory."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from brainlayer.vector_store import VectorStore
from brainlayer.watcher_bridge import _extract_project_from_session_file

NUMERIC_PROJECT_PREDICATE = "(project GLOB '[0-9]' OR project GLOB '[0-9][0-9]')"
_BATCH_SIZE_MIN = 5_000
_BATCH_SIZE_MAX = 10_000


@dataclass(frozen=True)
class ProjectBackfillResult:
    rows_updated: int
    rows_rederived: int
    rows_set_null: int
    rows_left_untouched: int


def _export_rollback(conn: sqlite3.Connection, rollback_path: Path) -> None:
    if rollback_path.exists():
        raise FileExistsError(f"rollback artifact already exists: {rollback_path}")
    rollback_path.parent.mkdir(parents=True, exist_ok=True)
    with rollback_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("id", "project"))
        writer.writerows(
            conn.execute(f"SELECT id, project FROM chunks WHERE {NUMERIC_PROJECT_PREDICATE} ORDER BY rowid")
        )


@contextmanager
def _operation_writer_lock(db_path: Path) -> Iterator[None]:
    """Acquire BrainLayer's standard pidfile without opening a second DB writer."""
    store = VectorStore.__new__(VectorStore)
    store.db_path = db_path
    store._writer_pidfile_acquired = False
    store._acquire_writer_pidfile()
    try:
        yield
    finally:
        store._release_writer_pidfile()


def backfill_numeric_projects(
    db_path: str | Path,
    *,
    rollback_path: str | Path | None = None,
    batch_size: int = _BATCH_SIZE_MIN,
) -> ProjectBackfillResult:
    """Update only one- and two-digit projects from recorded session metadata."""
    if not _BATCH_SIZE_MIN <= batch_size <= _BATCH_SIZE_MAX:
        raise ValueError("batch_size must be between 5,000 and 10,000")
    if rollback_path is None:
        raise ValueError("rollback_path is required before backfilling numeric projects")

    db_path = Path(db_path)
    with _operation_writer_lock(db_path):
        conn = sqlite3.connect(db_path, timeout=30)
        source_projects: dict[str, str | None] = {}
        try:
            rows_left_untouched = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM chunks WHERE COALESCE({NUMERIC_PROJECT_PREDICATE}, 0) = 0"
                ).fetchone()[0]
            )
            _export_rollback(conn, Path(rollback_path))

            conn.execute("PRAGMA wal_checkpoint(FULL)")
            last_rowid = 0
            batches = 0
            rows_updated = rows_rederived = rows_set_null = 0
            while True:
                rows = conn.execute(
                    f"""
                    SELECT rowid, id, source_file
                    FROM chunks
                    WHERE rowid > ? AND {NUMERIC_PROJECT_PREDICATE}
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (last_rowid, batch_size),
                ).fetchall()
                if not rows:
                    break

                projects_by_rowid: dict[int, str | None] = {}
                for rowid, _chunk_id, source_file in rows:
                    if source_file not in source_projects:
                        source_projects[source_file] = _extract_project_from_session_file(source_file)
                    projects_by_rowid[rowid] = source_projects[source_file]
                last_rowid = rows[-1][0]

                conn.execute("BEGIN IMMEDIATE")
                try:
                    for rowid, chunk_id, _source_file in rows:
                        project = projects_by_rowid[rowid]
                        update = conn.execute(
                            f"UPDATE chunks SET project = ? WHERE id = ? AND {NUMERIC_PROJECT_PREDICATE}",
                            (project, chunk_id),
                        )
                        if update.rowcount == 1:
                            rows_updated += 1
                            if project is None:
                                rows_set_null += 1
                            else:
                                rows_rederived += 1
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                batches += 1
                if batches % 3 == 0:
                    conn.execute("PRAGMA wal_checkpoint(FULL)")

            conn.execute("PRAGMA wal_checkpoint(FULL)")
            return ProjectBackfillResult(rows_updated, rows_rederived, rows_set_null, rows_left_untouched)
        finally:
            conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-derive numeric BrainLayer projects from session metadata")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--rollback-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=_BATCH_SIZE_MIN)
    args = parser.parse_args(argv)
    print(backfill_numeric_projects(args.db, rollback_path=args.rollback_path, batch_size=args.batch_size))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
