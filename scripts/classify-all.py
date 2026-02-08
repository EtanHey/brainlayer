#!/usr/bin/env python3
"""Classify all chunks using trained SetFit model.

Input: data/models/setfit-tagger/ (trained model)
Output: Updates tags + tag_confidence in chunks table

Takes ~20-40 min on M2 Mac for 226K chunks.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zikaron.vector_store import VectorStore

DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
MODEL_DIR = Path(__file__).parent.parent / "data" / "models" / "setfit-tagger"
BATCH_SIZE = 64


def main():
    if not MODEL_DIR.exists():
        print(f"No trained model found: {MODEL_DIR}")
        print("Run: python scripts/train-setfit.py")
        sys.exit(1)

    # Load model
    try:
        from setfit import SetFitModel
    except ImportError:
        print("Missing deps: pip install setfit")
        sys.exit(1)

    print("Loading model...", flush=True)
    model = SetFitModel.from_pretrained(str(MODEL_DIR))

    # Load metadata
    with open(MODEL_DIR / "training-meta.json") as f:
        meta = json.load(f)
    active_labels = meta["active_labels"]
    print(f"Active labels: {len(active_labels)}")

    # Open database
    store = VectorStore(DB_PATH)
    cursor = store.conn.cursor()

    # Ensure tags column exists
    try:
        cursor.execute("ALTER TABLE chunks ADD COLUMN tags TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE chunks ADD COLUMN tag_confidence REAL")
    except Exception:
        pass

    total = store.count()
    print(f"Total chunks: {total:,}")

    # Process in batches
    offset = 0
    classified = 0
    start_time = time.time()

    while offset < total:
        # Fetch batch
        rows = list(cursor.execute("""
            SELECT id, content FROM chunks
            ORDER BY rowid
            LIMIT ? OFFSET ?
        """, (BATCH_SIZE, offset)))

        if not rows:
            break

        ids = [r[0] for r in rows]
        texts = [r[1][:512] for r in rows]  # Truncate for model

        # Predict
        predictions = model.predict(texts)

        # Update database
        for chunk_id, preds in zip(ids, predictions):
            # preds is a list of 0/1 for each active label
            tags = []
            confidences = []

            if hasattr(preds, 'tolist'):
                preds = preds.tolist()

            if isinstance(preds, list):
                for i, pred in enumerate(preds):
                    if i < len(active_labels) and pred:
                        tags.append(active_labels[i])
                        confidences.append(0.75)  # SetFit doesn't give probabilities easily
            elif isinstance(preds, (int, float)):
                # Single label prediction
                if int(preds) < len(active_labels):
                    tags.append(active_labels[int(preds)])
                    confidences.append(0.75)

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            cursor.execute("""
                UPDATE chunks SET tags = ?, tag_confidence = ?
                WHERE id = ?
            """, (json.dumps(tags), avg_confidence, chunk_id))

        classified += len(rows)
        offset += BATCH_SIZE

        if classified % (BATCH_SIZE * 10) == 0:
            elapsed = time.time() - start_time
            rate = classified / elapsed if elapsed > 0 else 0
            remaining = (total - classified) / rate if rate > 0 else 0
            print(
                f"  {classified:,}/{total:,} "
                f"({classified*100//total}%) "
                f"— {rate:.0f} chunks/s "
                f"— ~{remaining/60:.1f} min remaining",
                flush=True
            )

    elapsed = time.time() - start_time
    print(f"\nDone: {classified:,} chunks classified in {elapsed/60:.1f} min")

    # Stats
    tagged = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND tags != '[]'"))[0][0]
    low_conf = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE tag_confidence < 0.6"))[0][0]
    print(f"Tagged (non-empty): {tagged:,}")
    print(f"Low confidence (<0.6): {low_conf:,}")
    print(f"\nNext: zikaron review")

    store.close()


if __name__ == "__main__":
    main()
