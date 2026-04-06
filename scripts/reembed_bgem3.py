#!/usr/bin/env python3
"""Re-embed all chunks using BAAI/bge-m3.

Reads chunks from the DB, generates new 1024-dim embeddings with BGE-M3,
and updates both chunk_vectors (float) and chunk_vectors_binary (quantized).

Resumable via checkpoint file. Use --test for a quick 100-chunk verification.

Usage:
    python scripts/reembed_bgem3.py --test                    # 100 chunks, verify
    python scripts/reembed_bgem3.py                           # full run, all chunks
    python scripts/reembed_bgem3.py --batch-size 64           # custom batch size
    python scripts/reembed_bgem3.py --checkpoint-every 500    # checkpoint frequency
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to path so we can import brainlayer.paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
DEFAULT_CHECKPOINT = Path.home() / ".local/share/brainlayer/reembed_bgem3_checkpoint.json"


def _get_default_db() -> Path:
    """Resolve DB path using brainlayer.paths if available, else fallback."""
    try:
        from brainlayer.paths import get_db_path

        return Path(get_db_path())
    except ImportError:
        return Path.home() / ".local/share/brainlayer/brainlayer.db"


def serialize_f32(vector: list[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def quantize_binary(embedding_bytes: bytes) -> bytes:
    """Binary-quantize a float32 embedding: each float -> 1 bit (positive=1, negative=0).

    Used for test DBs that don't have sqlite-vec's vec_quantize_binary().
    """
    n_floats = len(embedding_bytes) // 4
    floats = struct.unpack(f"{n_floats}f", embedding_bytes)
    bits = []
    for i in range(0, n_floats, 8):
        byte_val = 0
        for j in range(8):
            if i + j < n_floats and floats[i + j] > 0:
                byte_val |= 1 << j
        bits.append(byte_val)
    return bytes(bits)


def _open_apsw(db_path: str):
    """Open DB with APSW + sqlite-vec extension. Falls back to sqlite3 for test DBs."""
    try:
        import apsw
        import sqlite_vec

        conn = apsw.Connection(db_path)
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn, "apsw"
    except ImportError:
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn, "sqlite3"


def load_checkpoint(path: Path) -> dict:
    """Load checkpoint from file, or return empty state."""
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if data.get("model") == MODEL_NAME:
                return data
            print(f"Checkpoint model mismatch ({data.get('model')} != {MODEL_NAME}), starting fresh")
        except (json.JSONDecodeError, KeyError):
            pass
    return {"model": MODEL_NAME, "processed_ids": [], "last_updated": None}


def save_checkpoint(path: Path, state: dict) -> None:
    """Save checkpoint to file."""
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def get_chunks_to_process(db_path: str, already_done: set[str], limit: int | None = None) -> list[tuple[str, str]]:
    """Get (chunk_id, content) pairs that haven't been processed yet."""
    conn, backend = _open_apsw(db_path)
    try:
        query = "SELECT id, content FROM chunks WHERE content IS NOT NULL AND content != '' ORDER BY id"
        if limit:
            query += f" LIMIT {limit + len(already_done)}"
        if backend == "apsw":
            rows = list(conn.execute(query))
        else:
            rows = conn.execute(query).fetchall()
        # Filter out already-processed chunks
        result = [(cid, content) for cid, content in rows if cid not in already_done]
        if limit:
            result = result[:limit]
        return result
    finally:
        conn.close()


def load_model():
    """Load BGE-M3 model via sentence-transformers."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def _retry_on_busy(fn, max_attempts: int = 5, base_delay: float = 0.5):
    """Retry a callable on SQLITE_BUSY with exponential backoff."""
    import sqlite3 as _sqlite3

    for attempt in range(max_attempts):
        try:
            return fn()
        except (_sqlite3.OperationalError, Exception) as e:
            if "database is locked" not in str(e) and "BusyError" not in type(e).__name__:
                raise
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"  SQLITE_BUSY, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)


def reembed_batch(
    model,
    db_path: str,
    chunks: list[tuple[str, str]],
    checkpoint_state: dict,
    checkpoint_path: Path,
    checkpoint_every: int,
    batch_size: int,
) -> int:
    """Re-embed chunks in batches, updating DB and checkpointing."""
    conn, backend = _open_apsw(db_path)
    use_vec_quantize = backend == "apsw"  # sqlite-vec available

    # WAL checkpoint before bulk work
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

    total = len(chunks)
    processed = 0
    start_time = time.monotonic()

    try:
        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            ids = [cid for cid, _ in batch]
            texts = [content for _, content in batch]

            # Generate embeddings
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

            # Update DB with retry on SQLITE_BUSY
            def _write_batch():
                for chunk_id, embedding in zip(ids, embeddings):
                    emb_bytes = serialize_f32(embedding.tolist())

                    conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
                    conn.execute(
                        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                        (chunk_id, emb_bytes),
                    )
                    conn.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))
                    if use_vec_quantize:
                        conn.execute(
                            "INSERT INTO chunk_vectors_binary (chunk_id, embedding) VALUES (?, vec_quantize_binary(?))",
                            (chunk_id, emb_bytes),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO chunk_vectors_binary (chunk_id, embedding) VALUES (?, ?)",
                            (chunk_id, quantize_binary(emb_bytes)),
                        )
                if backend == "sqlite3":
                    conn.commit()

            _retry_on_busy(_write_batch)
            processed += len(batch)

            # Update checkpoint
            checkpoint_state["processed_ids"].extend(ids)
            if processed % checkpoint_every < batch_size or batch_start + batch_size >= total:
                save_checkpoint(checkpoint_path, checkpoint_state)

            # Progress
            elapsed = time.monotonic() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            print(
                f"  Processed {processed}/{total} chunks "
                f"({processed * 100 / total:.1f}%) "
                f"— {rate:.1f} chunks/s, ETA {eta:.0f}s"
            )
    finally:
        conn.close()

    return processed


def main():
    parser = argparse.ArgumentParser(description="Re-embed chunks with BAAI/bge-m3")
    parser.add_argument("--db", type=str, default=str(_get_default_db()), help="Path to brainlayer.db")
    parser.add_argument("--test", action="store_true", help="Test mode: process only 100 chunks")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size (default: 8, BGE-M3 needs more VRAM than bge-large)",
    )
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="Save checkpoint every N chunks")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to checkpoint file",
    )
    args = parser.parse_args()

    db_path = args.db
    checkpoint_path = Path(args.checkpoint_file)
    limit = 100 if args.test else None

    print(f"BGE-M3 Re-embedding {'(TEST MODE — 100 chunks)' if args.test else ''}")
    print(f"  DB: {db_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Batch size: {args.batch_size}")

    # Load checkpoint
    state = load_checkpoint(checkpoint_path)
    already_done = set(state["processed_ids"])
    if already_done:
        print(f"  Resuming — skipping {len(already_done)} already-processed chunks")

    # Get chunks to process
    chunks = get_chunks_to_process(db_path, already_done, limit=limit)
    if not chunks:
        print("No chunks to process. All done!")
        return

    print(f"  Chunks to process: {len(chunks)}")

    # Load model
    model = load_model()

    # Re-embed
    processed = reembed_batch(
        model=model,
        db_path=db_path,
        chunks=chunks,
        checkpoint_state=state,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
    )

    print(f"\nDone! Processed {processed} chunks with {MODEL_NAME}.")
    save_checkpoint(checkpoint_path, state)


if __name__ == "__main__":
    main()
