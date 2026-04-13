#!/usr/bin/env python3
"""Purge orphaned vectors from chunk_vectors and chunk_vectors_binary vec0 tables.

Orphans = vector entries whose chunk_id no longer exists in the chunks table.
Deletes in batches through the vec0 virtual table interface using APSW + sqlite-vec.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import apsw
import sqlite_vec

from brainlayer.paths import get_db_path

BATCH_SIZE = 2000
CHECKPOINT_EVERY = 5  # batches


def resolve_db_path(db_path: str | None = None) -> Path:
    return Path(db_path) if db_path else get_db_path()


def make_conn(db_path: Path):
    conn = apsw.Connection(str(db_path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def batched(values: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_orphan_ids(conn, vec_rowids_table: str) -> int:
    row = conn.execute(
        f"""
        SELECT COUNT(*) FROM {vec_rowids_table} vr
        WHERE vr.id NOT IN (SELECT id FROM chunks)
        """
    ).fetchone()
    return int(row[0]) if row else 0


def get_orphan_batch(conn, vec_rowids_table: str, limit: int = BATCH_SIZE) -> list[str]:
    """Fetch one bounded batch of orphan IDs from the vec shadow table."""
    return [
        row[0]
        for row in conn.execute(
            f"""
        SELECT vr.id FROM {vec_rowids_table} vr
        WHERE vr.id NOT IN (SELECT id FROM chunks)
        LIMIT ?
        """,
            (limit,),
        )
    ]


def purge_vec_table(conn, vec_table: str, vec_rowids_table: str, label: str):
    """Delete orphaned entries from a vec0 virtual table in batches."""
    print(f"\n{'=' * 60}")
    print(f"Purging orphans from {label}")
    print(f"{'=' * 60}", flush=True)

    total = count_orphan_ids(conn, vec_rowids_table)
    print(f"Found {total:,} orphaned vectors", flush=True)

    if total == 0:
        print("Nothing to purge.")
        return 0

    deleted = 0
    errors = 0
    batch_num = 0
    start = time.time()
    processed = 0

    while True:
        batch = get_orphan_batch(conn, vec_rowids_table, BATCH_SIZE)
        if not batch:
            break
        batch_num += 1
        processed += len(batch)

        for cid in batch:
            try:
                conn.execute(f"DELETE FROM {vec_table} WHERE chunk_id = ?", (cid,))
                deleted += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Error deleting {cid}: {e}")

        elapsed = time.time() - start
        rate = deleted / elapsed if elapsed > 0 else 0
        print(
            f"  Batch {batch_num}: {processed:,}/{total:,} processed, {deleted:,} deleted "
            f"({processed * 100 / total:.1f}%) — {rate:.0f}/s" + (f" [{errors} errors]" if errors else ""),
            flush=True,
        )

        if batch_num % CHECKPOINT_EVERY == 0:
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            print("  [WAL checkpoint]", flush=True)

        if errors:
            break

    elapsed = time.time() - start
    print(f"\nDone: {deleted:,} orphans purged from {label} in {elapsed:.1f}s")
    if errors:
        print(f"  ({errors} errors)")
        raise RuntimeError(f"Failed to purge {errors} orphaned vectors from {label}")
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", help="Path to the BrainLayer SQLite DB")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    print(f"Database: {db_path}")
    conn = make_conn(db_path)
    try:
        # Pre-check
        total_vecs = conn.execute("SELECT COUNT(*) FROM chunk_vectors_rowids").fetchone()[0]
        total_bins = conn.execute("SELECT COUNT(*) FROM chunk_vectors_binary_rowids").fetchone()[0]
        total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print(f"chunk_vectors entries: {total_vecs:,}")
        print(f"chunk_vectors_binary entries: {total_bins:,}")
        print(f"chunks table rows: {total_chunks:,}")

        deleted_float = purge_vec_table(conn, "chunk_vectors", "chunk_vectors_rowids", "chunk_vectors (float32)")
        deleted_binary = purge_vec_table(
            conn,
            "chunk_vectors_binary",
            "chunk_vectors_binary_rowids",
            "chunk_vectors_binary (bit)",
        )

        # Final checkpoint
        print("\nFinal WAL checkpoint...")
        conn.execute("PRAGMA wal_checkpoint(FULL)")

        # Post-check
        remaining_vecs = conn.execute("SELECT COUNT(*) FROM chunk_vectors_rowids").fetchone()[0]
        remaining_bins = conn.execute("SELECT COUNT(*) FROM chunk_vectors_binary_rowids").fetchone()[0]
        print("\nPost-purge:")
        print(f"  chunk_vectors: {remaining_vecs:,} (was {total_vecs:,})")
        print(f"  chunk_vectors_binary: {remaining_bins:,} (was {total_bins:,})")
        print(f"  Total deleted: {deleted_float + deleted_binary:,}")
        print("\nDone. Run VACUUM separately to reclaim disk space.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
