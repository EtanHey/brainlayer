#!/usr/bin/env python3
"""Hide historical T3 thinking without deleting content.

Lead runbook: quiesce ALL writers and take a fresh backup before --live --apply.
Rehearse on an APFS clone first. Dry-run is read-only and the default. Keep the
printed run ID: --undo --run-id ID restores byte-exact metadata and source_class.
The journal is committed in the same transaction as each 5,000-row batch; a
partial run can be undone by that ID. Later edits make undo refuse rather than
overwrite them. No content, embeddings, lifecycle or provenance is changed.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
import uuid
from collections import Counter
from contextlib import closing
from pathlib import Path

from brainlayer.ingest.t3 import T3Message, _is_thinking_message

CANONICAL_DB = Path.home() / ".local/share/brainlayer/brainlayer.db"
BATCH_SIZE = 5000
JOURNAL = "t3_thinking_reclassification"


def retry(operation):
    """Bounded SQLITE_BUSY retries, including checkpoint's busy result."""
    for attempt in range(6):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", 0)
            if code & 0xFF != sqlite3.SQLITE_BUSY or attempt == 5:
                raise
            time.sleep(0.1 * 2**attempt)


def checkpoint(c):
    def full():
        if c.execute("PRAGMA wal_checkpoint(FULL)").fetchone()[0]:
            exc = sqlite3.OperationalError("WAL checkpoint busy")
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc

    retry(full)


def candidates(c):
    result = []
    for row in c.execute(
        "SELECT id, source_class, metadata, source, sender FROM chunks "
        "WHERE source='t3' AND sender='assistant' "
        "AND source_class IS NOT 'agent-thinking'"
    ):
        metadata = json.loads(row[2])
        if not isinstance(metadata, dict):
            raise ValueError(f"Non-object metadata: {row[0]}")
        message_id = metadata.get("message_id")
        if isinstance(message_id, str) and _is_thinking_message(T3Message(message_id, "", row[4], "", "")):
            result.append(row)
    return result


def report(c, label):
    rows = candidates(c)
    groups = Counter((row[1], row[3]) for row in rows)
    print(
        json.dumps(
            {
                "phase": label,
                "eligible": len(rows),
                "t3_by_source_class": [
                    {"source_class": sc, "count": count}
                    for sc, count in c.execute(
                        "SELECT source_class, COUNT(*) FROM chunks WHERE source='t3' GROUP BY source_class"
                    )
                ],
                "by_source_class_and_source": [
                    {"source_class": sc, "source": source, "count": count}
                    for (sc, source), count in sorted(groups.items(), key=lambda item: str(item[0]))
                ],
            }
        )
    )
    return rows


def transaction(c, operation):
    def run():
        c.execute("BEGIN IMMEDIATE")
        try:
            operation()
            c.commit()
        except BaseException:
            c.rollback()
            raise

    retry(run)


def apply(c, rows, run_id):
    if not rows:
        return 0
    retry(
        lambda: c.execute(
            f"CREATE TABLE IF NOT EXISTS {JOURNAL} ("
            "run_id TEXT NOT NULL, chunk_id TEXT NOT NULL, original_class TEXT, "
            "original_metadata TEXT NOT NULL, applied_metadata TEXT NOT NULL, "
            "undone INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(run_id, chunk_id))"
        )
    )
    if c.execute(f"SELECT 1 FROM {JOURNAL} WHERE run_id=? LIMIT 1", (run_id,)).fetchone():
        raise ValueError("Run ID already used; omit --run-id to generate a fresh one")
    for start in range(0, len(rows), BATCH_SIZE):

        def batch():
            for chunk_id, sc, raw, _source, _sender in rows[start : start + BATCH_SIZE]:
                metadata = json.loads(raw)
                metadata.update(source_class="agent-thinking", thinking=True, thinking_source_class=sc)
                changed = json.dumps(metadata, ensure_ascii=False)
                cursor = c.execute(
                    "UPDATE chunks SET source_class='agent-thinking', metadata=? "
                    "WHERE id=? AND source_class IS ? AND metadata=?",
                    (changed, chunk_id, sc, raw),
                )
                if cursor.rowcount != 1:
                    raise ValueError(f"Row changed since census: {chunk_id}; quiesce writers")
                c.execute(f"INSERT INTO {JOURNAL} VALUES(?,?,?,?,?,0)", (run_id, chunk_id, sc, raw, changed))

        transaction(c, batch)
        if (start // BATCH_SIZE + 1) % 3 == 0:
            checkpoint(c)
    return len(rows)


def undo(c, run_id):
    if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (JOURNAL,)).fetchone():
        return 0
    if run_id is None:
        runs = c.execute(f"SELECT DISTINCT run_id FROM {JOURNAL} WHERE undone=0").fetchall()
        if not runs:
            return 0
        if len(runs) != 1:
            raise ValueError("Multiple active runs; specify --run-id")
        run_id = runs[0][0]
    rows = c.execute(
        f"SELECT chunk_id, original_class, original_metadata, applied_metadata "
        f"FROM {JOURNAL} WHERE run_id=? AND undone=0 ORDER BY chunk_id",
        (run_id,),
    ).fetchall()
    # Check the whole run before committing any undo batch.
    for chunk_id, _sc, _raw, changed in rows:
        if c.execute("SELECT source_class, metadata FROM chunks WHERE id=?", (chunk_id,)).fetchone() != (
            "agent-thinking",
            changed,
        ):
            raise ValueError(f"Row changed since migration: {chunk_id}; undo refused")
    for start in range(0, len(rows), BATCH_SIZE):

        def batch():
            for chunk_id, sc, raw, changed in rows[start : start + BATCH_SIZE]:
                cursor = c.execute(
                    "UPDATE chunks SET source_class=?, metadata=? WHERE id=? "
                    "AND source_class='agent-thinking' AND metadata=?",
                    (sc, raw, chunk_id, changed),
                )
                if cursor.rowcount != 1:
                    raise ValueError(f"Row changed since migration: {chunk_id}; undo refused")
                c.execute(f"UPDATE {JOURNAL} SET undone=1 WHERE run_id=? AND chunk_id=?", (run_id, chunk_id))

        transaction(c, batch)
        if (start // BATCH_SIZE + 1) % 3 == 0:
            checkpoint(c)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--live", action="store_true", help="Lead only, after backup and writer quiescence")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--apply", action="store_true")
    modes.add_argument("--undo", action="store_true")
    parser.add_argument("--run-id", help="Printed apply ID; omit on undo only when one active run exists")
    args = parser.parse_args(argv)
    path = args.db.expanduser().resolve()
    canonical = CANONICAL_DB.expanduser().resolve()
    if not args.live and (path == canonical or (canonical.exists() and path.samefile(canonical))):
        raise ValueError("Refusing canonical DB without --live (lead only)")
    path = path.resolve(strict=True)
    write = args.apply or args.undo
    with closing(
        sqlite3.connect(
            path.as_uri() + ("?mode=rw" if write else "?mode=ro"), uri=True, isolation_level=None, timeout=5
        )
    ) as c:
        rows = report(c, "before")
        if write:
            checkpoint(c)
            run_id = args.run_id or (str(uuid.uuid4()) if args.apply else None)
            print(json.dumps({"run_id": run_id}), flush=True)
            changed = apply(c, rows, run_id) if args.apply else undo(c, run_id)
            checkpoint(c)
            print(json.dumps({"changed": changed, "action": "apply" if args.apply else "undo"}))
        report(c, "after")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
