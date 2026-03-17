"""Real-time indexer for Claude Code hooks.

Correlates user prompts (UserPromptSubmit) with assistant responses (Stop)
and stores paired chunks in BrainLayer's SQLite database.

Architecture:
  - capture_prompt(): called by UserPromptSubmit hook, holds prompt in memory
  - index_response(): called by Stop hook, pairs with pending prompt, stores chunk
  - record_chapter(): called by PostCompact hook, creates chapter boundary
  - cleanup_stale_prompts(): evicts prompts older than max_age_seconds

Designed to swap backend from direct DB to BrainBar socket when it merges.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RealtimeIndexer:
    """Correlates prompts with responses and stores indexed chunks."""

    def __init__(self, db_path: str | None = None, queue_dir: str | None = None):
        self.db_path = db_path
        self.queue_dir = queue_dir
        self.pending_prompts: dict[str, dict] = {}
        self._db: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self):
        """Open database and ensure schema exists."""
        if not self.db_path:
            return
        try:
            self._db = sqlite3.connect(self.db_path)
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA busy_timeout = 5000")
            self._db.execute("PRAGMA synchronous = NORMAL")
            self._create_schema()
        except Exception as exc:
            logger.warning("Failed to open DB at %s: %s", self.db_path, exc)
            self._db = None

    def _create_schema(self):
        """Create tables if they don't exist."""
        if not self._db:
            return
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                project TEXT,
                content_type TEXT DEFAULT 'assistant_text',
                importance INTEGER DEFAULT 5,
                source TEXT DEFAULT 'realtime',
                chapter_id INTEGER,
                is_orphaned INTEGER NOT NULL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, content_hash)
            );

            CREATE TABLE IF NOT EXISTS chapters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                chapter_index INTEGER NOT NULL,
                compact_summary TEXT NOT NULL,
                trigger TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, chapter_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='rowid'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;
        """)
        self._db.commit()

    # MARK: - Prompt capture (UserPromptSubmit)

    def capture_prompt(self, session_id: str, prompt_text: str, cwd: str):
        """Store a pending prompt in memory, keyed by session_id."""
        self.pending_prompts[session_id] = {
            "text": prompt_text,
            "cwd": cwd,
            "timestamp": time.time(),
        }

    # MARK: - Response indexing (Stop)

    def index_response(
        self,
        session_id: str,
        response_text: str,
        cwd: str,
    ) -> str | None:
        """Pair response with pending prompt and store as a chunk.

        Returns chunk_id if stored, None if deduped or failed.
        """
        # Retrieve and clear pending prompt
        pending = self.pending_prompts.pop(session_id, None)
        prompt_text = pending["text"] if pending else None

        # Build content: prompt + response paired
        if prompt_text:
            content = f"User: {prompt_text}\n\nAssistant: {response_text}"
        else:
            content = f"Assistant: {response_text}"

        # Content hash for dedup (scoped to session via UNIQUE constraint)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        chunk_id = f"rt-{session_id[:8]}-{content_hash}"
        project = self._extract_project(cwd)

        # Try DB write
        if self._db:
            try:
                cursor = self._db.execute(
                    """INSERT OR IGNORE INTO chunks
                       (chunk_id, session_id, content, content_hash, project, source)
                       VALUES (?, ?, ?, ?, ?, 'realtime')""",
                    (chunk_id, session_id, content, content_hash, project),
                )
                if cursor.rowcount == 0:
                    # INSERT OR IGNORE skipped — duplicate
                    return None
                self._db.commit()
                return chunk_id
            except Exception as exc:
                logger.warning("DB write failed: %s", exc)
                self._write_to_queue(session_id, content, content_hash, project)
                return None
        else:
            self._write_to_queue(session_id, content, content_hash, project)
            return None

    # MARK: - Chapter boundaries (PostCompact)

    def record_chapter(
        self,
        session_id: str,
        compact_summary: str,
        trigger: str,
    ) -> int | None:
        """Create a chapter boundary from a PostCompact event.

        Returns chapter ID if created, None if failed.
        """
        if not self._db:
            return None

        # Get next chapter index for this session
        row = self._db.execute(
            "SELECT COALESCE(MAX(chapter_index), -1) + 1 FROM chapters WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        next_index = row[0] if row else 0

        try:
            cursor = self._db.execute(
                """INSERT INTO chapters (session_id, chapter_index, compact_summary, trigger)
                   VALUES (?, ?, ?, ?)""",
                (session_id, next_index, compact_summary, trigger),
            )
            self._db.commit()
            return cursor.lastrowid
        except Exception as exc:
            logger.warning("Chapter creation failed: %s", exc)
            return None

    # MARK: - Stale prompt cleanup

    def cleanup_stale_prompts(self, max_age_seconds: int = 300):
        """Evict pending prompts older than max_age_seconds."""
        now = time.time()
        stale = [sid for sid, data in self.pending_prompts.items() if now - data["timestamp"] > max_age_seconds]
        for sid in stale:
            del self.pending_prompts[sid]
        if stale:
            logger.info("Evicted %d stale prompts", len(stale))

    # MARK: - Helpers

    @staticmethod
    def _extract_project(cwd: str) -> str:
        """Extract project name from working directory path."""
        return Path(cwd.rstrip("/")).name

    def _write_to_queue(self, session_id: str, content: str, content_hash: str, project: str):
        """Fallback: write to queue directory when DB is unavailable."""
        if not self.queue_dir:
            return
        queue_path = Path(self.queue_dir)
        queue_path.mkdir(parents=True, exist_ok=True)
        entry = {
            "session_id": session_id,
            "content": content,
            "content_hash": content_hash,
            "project": project,
            "timestamp": time.time(),
        }
        queue_file = queue_path / f"{session_id}.jsonl"
        with open(queue_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def close(self):
        """Close the database connection."""
        if self._db:
            self._db.close()
            self._db = None
