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

    # Read from vec0 in batches via the rowids table
    all_ids = conn.execute(
        "SELECT id FROM chunk_vectors_rowids WHERE id IN (SELECT id FROM chunks)"
    ).fetchall()
    all_ids = [r[0] for r in all_ids]

    inserted = 0
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i : i + BATCH_SIZE]
        for cid in batch_ids:
            try:
                row = conn.execute(
                    "SELECT embedding FROM chunk_vectors WHERE chunk_id = ?", (cid,)
                ).fetchone()
                if row:
                    conn.execute(
                        "INSERT INTO _tmp_vec_backup (chunk_id, embedding) VALUES (?, ?)",
                        (cid, row[0]),
                    )
                    inserted += 1
            except Exception as e:
                print(f"  Warning: skip {cid}: {e}")

        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"  Extracted {inserted:,}/{count:,} ({rate:.0f}/s)", flush=True)

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

    rows = conn.execute("SELECT chunk_id, embedding FROM _tmp_vec_backup").fetchall()
    for cid, emb in rows:
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

    all_ids = conn.execute(
        "SELECT id FROM chunk_vectors_binary_rowids WHERE id IN (SELECT id FROM chunks)"
    ).fetchall()
    all_ids = [r[0] for r in all_ids]

    inserted = 0
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i : i + BATCH_SIZE]
        for cid in batch_ids:
            try:
                row = conn.execute(
                    "SELECT embedding FROM chunk_vectors_binary WHERE chunk_id = ?", (cid,)
                ).fetchone()
                if row:
                    conn.execute(
                        "INSERT INTO _tmp_binvec_backup (chunk_id, embedding) VALUES (?, ?)",
                        (cid, row[0]),
                    )
                    inserted += 1
            except Exception as e:
                print(f"  Warning: skip {cid}: {e}")

        elapsed = time.time() - t0
        rate = inserted / elapsed if elapsed > 0 else 0
        print(f"  Extracted {inserted:,}/{count:,} ({rate:.0f}/s)", flush=True)

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

    rows = conn.execute("SELECT chunk_id, embedding FROM _tmp_binvec_backup").fetchall()
    for cid, emb in rows:
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
