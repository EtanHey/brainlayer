#!/usr/bin/env python3
"""Backfill source, sender, and language columns on existing chunks."""

import json
import re
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zikaron.vector_store import VectorStore

DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"

# Hebrew character range
HEBREW_RE = re.compile(r'[\u0590-\u05FF]')


def detect_source(project, content_type, metadata: dict) -> str:
    """Detect chunk source from metadata."""
    # WhatsApp: has chat_id with @s.whatsapp.net
    chat_id = metadata.get("chat_id", "")
    if chat_id and "@s.whatsapp.net" in chat_id:
        return "whatsapp"

    # YouTube: metadata has source=youtube
    if metadata.get("source") == "youtube":
        return "youtube"

    # Claude Code: has project or content_type
    if project or content_type:
        return "claude_code"

    return "unknown"


def detect_sender(source: str, content_type: str, metadata: dict) -> str:
    """Detect who sent this chunk."""
    if source == "claude_code":
        if content_type == "user_message":
            return "me"
        return "other"  # assistant_text, ai_code, stack_trace

    if source == "whatsapp":
        # Check if from owner's phone
        contact = metadata.get("contact_name", "")
        # Owner detection: if contact is empty or matches owner patterns
        # For now, mark based on relationship_tag if available
        rel = metadata.get("relationship_tag", "")
        if rel == "self" or contact in ("You", "אתה"):
            return "me"
        # Business/commercial detection
        if metadata.get("is_business") or (
            len(metadata.get("contact_name", "")) > 20  # Long business names
            and not HEBREW_RE.search(metadata.get("contact_name", ""))
        ):
            return "commercial"
        return "other"

    if source == "youtube":
        return "other"

    return "other"


def detect_language(content: str) -> str:
    """Detect language: en, he, or mixed."""
    if not content:
        return "en"
    hebrew_chars = len(HEBREW_RE.findall(content))
    total_alpha = len(re.findall(r'[a-zA-Z\u0590-\u05FF]', content))
    if total_alpha == 0:
        return "en"
    ratio = hebrew_chars / total_alpha
    if ratio > 0.3:
        return "he" if ratio > 0.7 else "mixed"
    return "en"


def main():
    print(f"Opening database: {DB_PATH}")
    vs = VectorStore(DB_PATH)
    conn = vs.conn

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Total chunks: {total}")

    if total == 0:
        print("Database is empty, nothing to backfill.")
        return

    # Check how many already tagged
    already = conn.execute("SELECT COUNT(*) FROM chunks WHERE source IS NOT NULL").fetchone()[0]
    if already > 0:
        print(f"Already tagged: {already} ({already*100//total}%)")
        if "--force" not in sys.argv:
            print("Use --force to re-tag all")
            return

    batch_size = 5000
    updated = 0

    for offset in range(0, total, batch_size):
        rows = conn.execute("""
            SELECT id, content, project, content_type, metadata
            FROM chunks
            LIMIT ? OFFSET ?
        """, (batch_size, offset)).fetchall()

        for row in rows:
            chunk_id, content, project, content_type, metadata_str = row
            try:
                metadata = json.loads(metadata_str) if metadata_str else {}
            except json.JSONDecodeError:
                metadata = {}

            source = detect_source(project, content_type, metadata)
            sender = detect_sender(source, content_type, metadata)
            language = detect_language(content or "")

            conn.execute("""
                UPDATE chunks SET source = ?, sender = ?, language = ?
                WHERE id = ?
            """, (source, sender, language, chunk_id))

        updated += len(rows)
        pct = updated * 100 // total
        print(f"\r  Processed {updated}/{total} ({pct}%)", end="", flush=True)

    print(f"\n\nDone. Tagged {updated} chunks.")

    # Show stats
    for col in ["source", "sender", "language"]:
        print(f"\n{col} distribution:")
        query = f"SELECT {col}, COUNT(*) FROM chunks GROUP BY {col} ORDER BY COUNT(*) DESC"
        for row in conn.execute(query):
            print(f"  {row[0] or 'NULL':20s} {row[1]:>8d}")


if __name__ == "__main__":
    main()
