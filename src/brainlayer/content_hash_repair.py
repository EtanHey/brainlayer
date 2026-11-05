"""Bounded repair of merged rows whose stored content_hash went stale (#623).

Before #623 was fixed, ``merge_duplicate_chunk`` and ``merge_existing_chunk_content``
rewrote ``chunks.content`` into a ``[Merged duplicate originals]`` body without
rewriting ``content_hash``. The drain prefers a stored hash, so every fresh
enrichment event for such a row was rejected as stale, forever.

This tool rewrites ONLY that column, ONLY on those rows:

* **Explicit DB.** ``--db`` is required and must already exist. The live
  canonical DB (and whatever this process resolves as live) is refused outright;
  run it on a copy. Nothing calls this on startup.
* **Dry-run first.** Without ``--apply`` the DB is opened read-only and only the
  candidate count and ids are reported.
* **Narrow scope.** A row qualifies only when its content starts with the merge
  marker, its ``content_hash`` is non-NULL, and that hash differs from
  ``sha256(content.strip())``. NULL hashes and legacy-scheme hashes on ordinary
  rows are not this defect and are left alone.
* **Bounded.** At most ``--max-rows`` rows per run, written in ``--batch-size``
  transactions; the report says when more remain.
* **Identity-checked.** Each write is ``WHERE id = ? AND content = ? AND
  content_hash = ?`` with the values read at scan time, so a row changed in
  between is skipped and counted, never stamped with a hash of other text.
* **Idempotent.** A second run finds nothing to do.

Content, provenance, FTS rows and vectors are untouched: ``content_hash`` is not
an FTS trigger column, and the embedded text does not change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import apsw

from .chunk_origin_wipe import assert_not_live_db
from .chunk_write import canonical_content_hash
from .dedupe import _MERGED_MARKER

DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_ROWS = 5_000
SAMPLE_IDS = 10


def _find_candidates(conn: apsw.Connection, *, scan_batch: int, max_rows: int) -> tuple[list[tuple], int, bool]:
    candidates: list[tuple[str, str, str, str]] = []
    scanned = 0
    last_rowid = -1
    while len(candidates) < max_rows:
        rows = conn.execute(
            """
            SELECT rowid, id, content, content_hash FROM chunks
            WHERE rowid > ? AND content_hash IS NOT NULL AND substr(content, 1, ?) = ?
            ORDER BY rowid LIMIT ?
            """,
            (last_rowid, len(_MERGED_MARKER), _MERGED_MARKER, scan_batch),
        ).fetchall()
        if not rows:
            return candidates, scanned, False
        for rowid, chunk_id, content, stored_hash in rows:
            last_rowid = rowid
            scanned += 1
            expected = canonical_content_hash(content)
            if stored_hash != expected:
                candidates.append((str(chunk_id), content, stored_hash, expected))
                if len(candidates) == max_rows:
                    break
    more = (
        conn.execute(
            "SELECT 1 FROM chunks WHERE rowid > ? AND content_hash IS NOT NULL AND substr(content, 1, ?) = ? LIMIT 1",
            (last_rowid, len(_MERGED_MARKER), _MERGED_MARKER),
        ).fetchone()
        is not None
    )
    return candidates, scanned, more


def repair_merged_content_hashes(
    db_path: str | Path,
    *,
    apply: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> dict[str, Any]:
    if batch_size < 1 or max_rows < 1:
        raise ValueError("batch_size and max_rows must be positive")
    resolved = assert_not_live_db(Path(db_path))
    if not resolved.is_file():
        raise FileNotFoundError(f"no database at {resolved}; this tool never creates one")
    flags = apsw.SQLITE_OPEN_READWRITE if apply else apsw.SQLITE_OPEN_READONLY
    conn = apsw.Connection(str(resolved), flags=flags)
    try:
        conn.setbusytimeout(30_000)
        if "content_hash" not in {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}:
            raise RuntimeError("chunks.content_hash does not exist; nothing to repair")
        candidates, scanned, more = _find_candidates(conn, scan_batch=batch_size, max_rows=max_rows)
        repaired = skipped = batches = 0
        if apply:
            for start in range(0, len(candidates), batch_size):
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for chunk_id, content, stored_hash, expected in candidates[start : start + batch_size]:
                        conn.execute(
                            "UPDATE chunks SET content_hash = ? WHERE id = ? AND content = ? AND content_hash = ?",
                            (expected, chunk_id, content, stored_hash),
                        )
                        if conn.changes():
                            repaired += 1
                        else:
                            skipped += 1
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                batches += 1
    finally:
        conn.close()
    return {
        "mode": "apply" if apply else "dry-run",
        "merged_rows_scanned": scanned,
        "stale": len(candidates),
        "repaired": repaired,
        "skipped_identity_changed": skipped,
        "batches": batches,
        "more_remaining": more,
        "sample_ids": [candidate[0] for candidate in candidates[:SAMPLE_IDS]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m brainlayer.content_hash_repair",
        description="Repair stale content_hash on merged-duplicate rows (#623). Dry-run unless --apply; "
        "refuses the live DB.",
    )
    parser.add_argument("--db", type=Path, required=True, help="Path to a COPY of the database")
    parser.add_argument("--apply", action="store_true", help="Write the repair (default: dry-run)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    args = parser.parse_args(argv)
    result = repair_merged_content_hashes(
        args.db.expanduser(), apply=args.apply, batch_size=args.batch_size, max_rows=args.max_rows
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
