#!/usr/bin/env python3
"""Rebuild vec0 tables to reclaim space from orphaned/deleted vector slots.

vec0 doesn't shrink partitions on DELETE — it just marks slots invalid.
This script extracts valid vectors, drops the vec0 table (and all shadow tables),
recreates it, and re-inserts the vectors.

IMPORTANT: Stop all writers (enrichment, watcher) before running.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import apsw
import sqlite_vec

from brainlayer.paths import get_db_path

BATCH_SIZE = 5000


def resolve_db_path(db_path: str | None = None) -> Path:
    return Path(db_path) if db_path else get_db_path()


def make_conn(db_path: Path):
    conn = apsw.Connection(str(db_path))
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
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


def iter_valid_chunk_ids(conn, rowids_table: str) -> Iterator[str]:
    for row in conn.execute(
        f"SELECT id FROM {rowids_table} WHERE id IN (SELECT id FROM chunks)"
    ):
        yield row[0]


def read_embedding_or_raise(conn, vec_table: str, chunk_id: str):
    try:
        row = conn.execute(
            f"SELECT embedding FROM {vec_table} WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
    except Exception as exc:  # pragma: no cover - exercised via tests with fake conn
        raise RuntimeError(
            f"Failed to read embedding for {chunk_id} from {vec_table}"
        ) from exc
    if row is None:
        raise RuntimeError(f"Missing embedding for {chunk_id} in {vec_table}")
    return row[0]


def ensure_restore_succeeded(errors: int, backup_table: str, label: str) -> None:
    if errors:
        raise RuntimeError(
            f"Failed to restore {label} ({errors} errors); backup preserved in {backup_table}"
        )


def rebuild_float32(conn):
    print("\n" + "=" * 60)
    print("Rebuilding chunk_vectors (float32)")
    print("=" * 60, flush=True)

    # Count valid entries
    count = conn.execute(
        "SELECT COUNT(*) FROM chunk_vectors_rowids WHERE id IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    print(f"Valid vectors to preserve: {count:,}", flush=True)

    # Step 1: Extract valid vectors to a temp table
    print("Step 1: Extracting valid vectors to temp table...", flush=True)
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS _tmp_vec_backup")
    conn.execute("""
        CREATE TABLE _tmp_vec_backup (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)

    inserted = 0
    for batch_ids in batched(iter_valid_chunk_ids(conn, "chunk_vectors_rowids"), BATCH_SIZE):
        for cid in batch_ids:
            embedding = read_embedding_or_raise(conn, "chunk_vectors", cid)
            conn.execute(
                "INSERT INTO _tmp_vec_backup (chunk_id, embedding) VALUES (?, ?)",
                (cid, embedding),
            )
            inserted += 1

        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"  Extracted {inserted:,}/{count:,} ({rate:.0f}/s)", flush=True)

    if inserted != count:
        raise RuntimeError(f"Backed up {inserted} float vectors but expected {count}")

    print(f"Step 1 done: {inserted:,} vectors backed up in {time.time()-t0:.1f}s")

    # Step 2: Drop the vec0 table (drops all shadow tables)
    print("Step 2: Dropping old vec0 table...", flush=True)
    conn.execute("DROP TABLE IF EXISTS chunk_vectors")
    print("  Dropped.")

    # Step 3: Recreate
    print("Step 3: Recreating vec0 table...", flush=True)
    conn.execute("""
        CREATE VIRTUAL TABLE chunk_vectors USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding FLOAT[1024]
        )
    """)
    print("  Created.")

    # Step 4: Re-insert from backup
    print("Step 4: Re-inserting vectors...", flush=True)
    t0 = time.time()
    reinserted = 0
    errors = 0

    for cid, emb in conn.execute("SELECT chunk_id, embedding FROM _tmp_vec_backup"):
        try:
            conn.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (cid, emb),
            )
            reinserted += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Insert error for {cid}: {e}")

        if reinserted % 10000 == 0:
            elapsed = time.time() - t0
            rate = reinserted / elapsed if elapsed > 0 else 0
            print(f"  Inserted {reinserted:,}/{inserted:,} ({rate:.0f}/s)", flush=True)

    elapsed = time.time() - t0
    print(f"Step 4 done: {reinserted:,} re-inserted in {elapsed:.1f}s ({errors} errors)")

    ensure_restore_succeeded(errors, "_tmp_vec_backup", "chunk_vectors")

    # Step 5: Drop backup
    conn.execute("DROP TABLE _tmp_vec_backup")
    print("Backup table dropped.")

    return reinserted


def rebuild_binary(conn):
    print("\n" + "=" * 60)
    print("Rebuilding chunk_vectors_binary (bit)")
    print("=" * 60, flush=True)

    count = conn.execute(
        "SELECT COUNT(*) FROM chunk_vectors_binary_rowids WHERE id IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    print(f"Valid vectors to preserve: {count:,}", flush=True)

    # Step 1: Extract
    print("Step 1: Extracting valid vectors to temp table...", flush=True)
    t0 = time.time()
    conn.execute("DROP TABLE IF EXISTS _tmp_binvec_backup")
    conn.execute("""
        CREATE TABLE _tmp_binvec_backup (
            chunk_id TEXT PRIMARY KEY,
            embedding BLOB
        )
    """)

    inserted = 0
    for batch_ids in batched(
        iter_valid_chunk_ids(conn, "chunk_vectors_binary_rowids"), BATCH_SIZE
    ):
        for cid in batch_ids:
            embedding = read_embedding_or_raise(conn, "chunk_vectors_binary", cid)
            conn.execute(
                "INSERT INTO _tmp_binvec_backup (chunk_id, embedding) VALUES (?, ?)",
                (cid, embedding),
            )
            inserted += 1

        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"  Extracted {inserted:,}/{count:,} ({rate:.0f}/s)", flush=True)

    if inserted != count:
        raise RuntimeError(f"Backed up {inserted} binary vectors but expected {count}")

    print(f"Step 1 done: {inserted:,} vectors backed up in {time.time()-t0:.1f}s")

    # Step 2: Drop
    print("Step 2: Dropping old vec0 binary table...", flush=True)
    conn.execute("DROP TABLE IF EXISTS chunk_vectors_binary")
    print("  Dropped.")

    # Step 3: Recreate
    print("Step 3: Recreating vec0 binary table...", flush=True)
    conn.execute("""
        CREATE VIRTUAL TABLE chunk_vectors_binary USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding BIT[1024]
        )
    """)
    print("  Created.")

    # Step 4: Re-insert
    print("Step 4: Re-inserting vectors...", flush=True)
    t0 = time.time()
    reinserted = 0
    errors = 0

    for cid, emb in conn.execute("SELECT chunk_id, embedding FROM _tmp_binvec_backup"):
        try:
            conn.execute(
                "INSERT INTO chunk_vectors_binary (chunk_id, embedding) VALUES (?, ?)",
                (cid, emb),
            )
            reinserted += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Insert error for {cid}: {e}")

        if reinserted % 10000 == 0:
            elapsed = time.time() - t0
            rate = reinserted / elapsed if elapsed > 0 else 0
            print(f"  Inserted {reinserted:,}/{inserted:,} ({rate:.0f}/s)", flush=True)

    elapsed = time.time() - t0
    print(f"Step 4 done: {reinserted:,} re-inserted in {elapsed:.1f}s ({errors} errors)")

    ensure_restore_succeeded(errors, "_tmp_binvec_backup", "chunk_vectors_binary")

    # Step 5: Drop backup
    conn.execute("DROP TABLE _tmp_binvec_backup")
    print("Backup table dropped.")

    return reinserted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", help="Path to the BrainLayer SQLite DB")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)
    print(f"Database: {db_path}")
    conn = make_conn(db_path)

    # Pre-check
    db_pages = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    print(f"DB size: {db_pages * page_size / 1024**3:.2f} GB ({db_pages:,} pages)")

    float_count = rebuild_float32(conn)
    binary_count = rebuild_binary(conn)

    # Checkpoint
    print("\nFinal WAL checkpoint...", flush=True)
    conn.execute("PRAGMA wal_checkpoint(FULL)")

    # Post-check
    db_pages = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    print("\nPost-rebuild:")
    print(f"  Float vectors: {float_count:,}")
    print(f"  Binary vectors: {binary_count:,}")
    print(f"  DB pages: {db_pages:,} ({db_pages * page_size / 1024**3:.2f} GB)")
    print(f"  Freelist: {freelist:,} pages ({freelist * page_size / 1024**2:.1f} MB)")
    print("\nRun VACUUM to reclaim freelist space.")

    conn.close()


if __name__ == "__main__":
    main()
