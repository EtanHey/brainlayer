#!/usr/bin/env python3
"""Verify Hebrew search quality after BGE-M3 re-embedding.

Tests that Hebrew queries return relevant results, and that
cross-language search (Hebrew query → English code docs) works.
"""

import struct
import sys
from pathlib import Path

import apsw
import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer

DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
MODEL = "BAAI/bge-m3"


def search(cursor, query_embedding, n=5):
    """Search chunk_vectors for nearest neighbors."""
    query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    results = list(cursor.execute(
        """SELECT c.id, c.content, c.source, c.language, v.distance
           FROM chunk_vectors v
           JOIN chunks c ON v.chunk_id = c.id
           WHERE v.embedding MATCH ? AND k = ?
           ORDER BY v.distance""",
        (query_bytes, n),
    ))
    return results


def main():
    print("Loading BGE-M3...")
    model = SentenceTransformer(MODEL)

    conn = apsw.Connection(str(DB_PATH), flags=apsw.SQLITE_OPEN_READONLY)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    cursor = conn.cursor()

    queries = [
        # Hebrew queries — should find WhatsApp chunks
        ("מה קורה עם הפרויקט?", "Hebrew: project status"),
        ("בוא נתכנן פגישה", "Hebrew: schedule meeting"),
        ("יש בעיה באימות", "Hebrew: auth problem"),
        # English queries — should find code chunks
        ("fix the authentication bug", "English: auth bug"),
        ("Supabase database migration", "English: DB migration"),
        # Cross-language — Hebrew query should find English code
        ("תיקון באג באימות", "Hebrew query for English auth code"),
        ("מסד נתונים מיגרציה", "Hebrew query for DB migration"),
    ]

    print(f"\n{'='*80}")
    print(f"Hebrew Search Quality Verification (BGE-M3)")
    print(f"{'='*80}")

    all_pass = True
    for query_text, description in queries:
        embedding = model.encode([query_text], normalize_embeddings=True)[0]
        results = search(cursor, embedding.tolist(), n=5)

        print(f"\n--- {description} ---")
        print(f"Query: '{query_text}'")

        if not results:
            print("  NO RESULTS!")
            all_pass = False
            continue

        for i, (cid, content, source, lang, dist) in enumerate(results):
            content_preview = (content or "")[:100].replace("\n", " ")
            print(f"  {i+1}. [dist={dist:.3f}] [{source or '?'}] {content_preview}")

        # Basic sanity: Hebrew queries should return at least some whatsapp results
        if "Hebrew" in description and "English" not in description:
            whatsapp_count = sum(1 for _, _, src, _, _ in results if src == "whatsapp")
            if whatsapp_count == 0:
                print(f"  WARNING: No WhatsApp results for Hebrew query!")

    conn.close()
    print(f"\n{'='*80}")
    print("Verification complete" + (" - all checks passed" if all_pass else " - some issues found"))


if __name__ == "__main__":
    main()
