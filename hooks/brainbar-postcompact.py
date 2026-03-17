#!/usr/bin/env python3
"""PostCompact hook — records chapter boundary in BrainLayer.

When Claude Code compacts context, this hook captures the compact_summary
and creates a chapter marker in the chapters table.

Target latency: <50ms.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path


def get_db_path() -> str:
    env_path = os.environ.get("BRAINLAYER_DB")
    if env_path:
        return env_path
    return str(Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db")


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    session_id = payload.get("session_id", "")
    compact_summary = payload.get("compact_summary", "")
    trigger = payload.get("trigger", "auto")

    if not session_id or not compact_summary:
        return

    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")

        # Create chapters table if not exists
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                chapter_index INTEGER NOT NULL,
                compact_summary TEXT NOT NULL,
                trigger TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, chapter_index)
            )
        """)

        # Get next chapter index
        row = conn.execute(
            "SELECT COALESCE(MAX(chapter_index), -1) + 1 FROM chapters WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_index = row[0] if row else 0

        conn.execute(
            "INSERT INTO chapters (session_id, chapter_index, compact_summary, trigger) VALUES (?, ?, ?, ?)",
            (session_id, next_index, compact_summary, trigger),
        )
        conn.commit()
        conn.close()

    except Exception:
        pass  # Best-effort — don't block Claude Code


if __name__ == "__main__":
    main()
