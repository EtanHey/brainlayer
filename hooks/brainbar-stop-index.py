#!/usr/bin/env python3
"""Stop hook — pairs assistant response with pending prompt and indexes to BrainLayer.

Reads last_assistant_message from stdin payload, retrieves pending prompt
(if any), and stores the paired chunk in BrainLayer's SQLite database.

Falls back to queue file if DB is locked or unavailable.
Target latency: <200ms.
"""

import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

PENDING_DIR = Path.home() / ".brainlayer" / "pending"
QUEUE_DIR = Path.home() / ".brainlayer" / "queue"

# Minimum response length worth indexing (skip trivial "ok" responses)
MIN_RESPONSE_LENGTH = 30

logger = logging.getLogger(__name__)


def get_db_path() -> str:
    """Resolve BrainLayer DB path."""
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
    response_text = payload.get("last_assistant_message", "")
    cwd = payload.get("cwd", "")

    if not session_id or not response_text:
        return

    # Skip trivial responses
    if len(response_text) < MIN_RESPONSE_LENGTH:
        return

    # Retrieve pending prompt (if UserPromptSubmit captured one)
    prompt_text = None
    pending_file = PENDING_DIR / f"{session_id}.txt"
    if pending_file.exists():
        try:
            prompt_text = pending_file.read_text()
            pending_file.unlink()
        except Exception:
            pass

    # Build paired content
    if prompt_text:
        content = f"User: {prompt_text}\n\nAssistant: {response_text}"
    else:
        content = f"Assistant: {response_text}"

    # Content hash for dedup
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    chunk_id = f"rt-{session_id[:8]}-{content_hash}"
    project = Path(cwd.rstrip("/")).name if cwd else "unknown"

    # Store in DB (matches production schema: id, content, metadata, source, etc.)
    db_path = get_db_path()
    metadata = json.dumps(
        {
            "session_id": session_id,
            "content_hash": content_hash,
            "cwd": cwd,
            "has_prompt": prompt_text is not None,
        }
    )

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout = 5000")

        transcript = payload.get("transcript_path", "realtime-hook")
        cursor = conn.execute(
            """INSERT OR IGNORE INTO chunks
               (id, content, metadata, source_file, source, project, content_type,
                char_count, conversation_id, importance)
               VALUES (?, ?, ?, ?, 'realtime', ?, 'assistant_text',
                ?, ?, 5)""",
            (chunk_id, content, metadata, transcript, project, len(content), session_id),
        )

        if cursor.rowcount > 0:
            conn.commit()

        conn.close()

    except Exception as exc:
        # Fallback: write to queue
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "chunk_id": chunk_id,
            "session_id": session_id,
            "content": content,
            "content_hash": content_hash,
            "project": project,
            "timestamp": time.time(),
        }
        queue_file = QUEUE_DIR / f"{session_id}.jsonl"
        with open(queue_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
