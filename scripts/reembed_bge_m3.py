#!/usr/bin/env python3
"""Re-embed all chunks with BGE-M3 (multilingual, 1024 dims).

Replaces bge-large-en-v1.5 embeddings in chunk_vectors with BGE-M3 embeddings.
Both models output 1024 dimensions, so vec0 table schema stays the same.

Usage:
    python3 scripts/reembed_bge_m3.py [--batch-size 64] [--db-path PATH] [--dry-run]

Estimated time: ~70 minutes for 245K chunks on M1 Pro (MPS).
Peak memory: ~3 GB (model + batch of embeddings).
"""

import argparse
import logging
import struct
import sys
import time
from pathlib import Path

import apsw
import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
MAX_CHARS = 512  # Match existing truncation


def serialize_f32(vector) -> bytes:
    """Serialize float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def main():
    parser = argparse.ArgumentParser(description="Re-embed chunks with BGE-M3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--db-path", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--dry-run", action="store_true", help="Only count, don't embed")
    parser.add_argument("--start-offset", type=int, default=0, help="Resume from this row offset")
    args = parser.parse_args()

    db_path = args.db_path
    batch_size = args.batch_size

    # Connect to DB
    conn = apsw.Connection(db_path)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout = 5000")

    # Count total chunks
    total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
    logger.info(f"Total chunks in DB: {total}")

    if args.dry_run:
        vec_count = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors"))[0][0]
        logger.info(f"Existing vectors: {vec_count}")
        logger.info(f"Estimated time at 59 texts/sec: {total/59/60:.0f} minutes")
        return

    # Load model
    logger.info(f"Loading {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"Model loaded in {time.time()-t0:.1f}s, dim={model.get_sentence_embedding_dimension()}")

    # Process in batches using rowid ordering for deterministic resume
    processed = 0
    skipped = 0
    start_time = time.time()
    offset = args.start_offset

    while True:
        # Fetch batch of chunks
        rows = list(cursor.execute(
            "SELECT id, content FROM chunks ORDER BY rowid LIMIT ? OFFSET ?",
            (batch_size, offset),
        ))

        if not rows:
            break

        chunk_ids = [r[0] for r in rows]
        texts = []
        for _, content in rows:
            if content and len(content) > MAX_CHARS:
                content = content[:MAX_CHARS - 50] + "..."
            texts.append(content or "")

        # Embed batch
        try:
            embeddings = model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Failed to embed batch at offset {offset}: {e}")
            offset += batch_size
            skipped += len(rows)
            continue

        # Write to vec0 table — delete + insert (vec0 doesn't support REPLACE)
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            try:
                cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
                cursor.execute(
                    "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, serialize_f32(embedding.tolist())),
                )
            except Exception as e:
                logger.error(f"Failed to write vector for {chunk_id}: {e}")
                skipped += 1

        processed += len(rows)
        offset += batch_size

        # Progress logging every 10 batches
        if (processed // batch_size) % 10 == 0 or processed >= total:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed - args.start_offset) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {processed + args.start_offset}/{total} "
                f"({(processed + args.start_offset)/total*100:.1f}%) "
                f"| {rate:.0f} chunks/sec "
                f"| ETA: {eta/60:.0f} min "
                f"| skipped: {skipped}"
            )

    elapsed = time.time() - start_time
    logger.info(
        f"Done! Processed {processed} chunks in {elapsed/60:.1f} minutes "
        f"({processed/elapsed:.0f} chunks/sec). Skipped: {skipped}"
    )

    # Verify
    vec_count = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors"))[0][0]
    logger.info(f"Vector count in DB: {vec_count} (expected: {total})")

    conn.close()


if __name__ == "__main__":
    main()
