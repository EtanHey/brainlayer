#!/usr/bin/env python3
"""Backfill conversation_id and position columns for context view.

For all chunks: conversation_id = source_file, position = order within that file.

One-time script — no re-embedding needed, just SQL UPDATEs.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zikaron.vector_store import VectorStore

DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"


def backfill():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    store = VectorStore(DB_PATH)
    cursor = store.conn.cursor()

    # Check current state
    total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
    conv_filled = list(cursor.execute(
        "SELECT COUNT(*) FROM chunks WHERE conversation_id IS NOT NULL"
    ))[0][0]
    pos_filled = list(cursor.execute(
        "SELECT COUNT(*) FROM chunks WHERE position IS NOT NULL"
    ))[0][0]

    print(f"Total chunks: {total:,}", flush=True)
    print(f"conversation_id filled: {conv_filled:,}", flush=True)
    print(f"position filled: {pos_filled:,}", flush=True)

    # Step 1: Backfill conversation_id = source_file
    if conv_filled < total:
        print("\nBackfilling conversation_id...", flush=True)
        cursor.execute("""
            UPDATE chunks SET conversation_id = source_file
            WHERE conversation_id IS NULL
        """)
        conv_filled = list(cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE conversation_id IS NOT NULL"
        ))[0][0]
        print(f"  conversation_id filled: {conv_filled:,}/{total:,}", flush=True)

    # Step 2: Backfill position — process per source_file group
    if pos_filled < total:
        print("\nBackfilling position (per source_file)...", flush=True)

        # Get distinct source_files that need backfill
        source_files = list(cursor.execute("""
            SELECT DISTINCT source_file FROM chunks WHERE position IS NULL
        """))
        print(f"  {len(source_files)} source files to process", flush=True)

        updated = 0
        for i, (sf,) in enumerate(source_files):
            # Get chunk IDs ordered by rowid within this source_file
            chunk_ids = list(cursor.execute("""
                SELECT id FROM chunks
                WHERE source_file = ? AND position IS NULL
                ORDER BY rowid
            """, (sf,)))

            # Batch UPDATE with position
            for pos, (chunk_id,) in enumerate(chunk_ids):
                cursor.execute(
                    "UPDATE chunks SET position = ? WHERE id = ?",
                    (pos, chunk_id)
                )

            updated += len(chunk_ids)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(source_files)} files, {updated:,} chunks", flush=True)

        print(f"  Done: {updated:,} chunks positioned", flush=True)

    # Verify
    conv_final = list(cursor.execute(
        "SELECT COUNT(*) FROM chunks WHERE conversation_id IS NOT NULL"
    ))[0][0]
    pos_final = list(cursor.execute(
        "SELECT COUNT(*) FROM chunks WHERE position IS NOT NULL"
    ))[0][0]

    print(f"\nFinal state:")
    print(f"  conversation_id: {conv_final:,}/{total:,}")
    print(f"  position: {pos_final:,}/{total:,}")
    print("Done.", flush=True)

    store.close()


if __name__ == "__main__":
    backfill()
