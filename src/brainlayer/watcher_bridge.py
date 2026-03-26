"""Bridge between JSONLWatcher and BrainLayer's indexing pipeline.

Processes raw JSONL lines through classify → chunk → insert (deferred embedding).
Chunks are immediately searchable via FTS5; embeddings are backfilled by enrichment.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import get_db_path
from .pipeline.chunk import chunk_content
from .pipeline.classify import classify_content
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Cache project name normalization (same as cli/__init__.py)
_PROJECT_CACHE: dict[str, str] = {}


def _normalize_project_name(raw: str) -> str:
    """Convert encoded project path to human-readable name."""
    if raw in _PROJECT_CACHE:
        return _PROJECT_CACHE[raw]
    # Claude Code encodes paths: -Users-foo-Gits-myproject → myproject
    parts = raw.split("-")
    # Take last non-empty part
    name = parts[-1] if parts else raw
    _PROJECT_CACHE[raw] = name
    return name


def _extract_project_from_source(source_file: str) -> str | None:
    """Extract project name from the source file path."""
    p = Path(source_file)
    # ~/.claude/projects/{encoded-project-path}/{session}.jsonl
    parent_name = p.parent.name
    if parent_name and parent_name != "projects":
        return _normalize_project_name(parent_name)
    return None


def create_flush_callback(db_path: Path | None = None) -> callable:
    """Create an on_flush callback that processes JSONL lines into BrainLayer.

    Returns a callable that takes a list[dict] of raw JSONL entries and
    inserts them as chunks into the database (deferred embedding).
    """
    resolved_db = db_path or get_db_path()
    store = VectorStore(resolved_db)

    def flush_to_db(entries: list[dict[str, Any]]) -> None:
        """Process raw JSONL entries through pipeline and insert into DB."""
        cursor = store.conn.cursor()
        inserted = 0
        skipped = 0

        for entry in entries:
            source_file = entry.get("_source_file", "unknown")
            project = _extract_project_from_source(source_file)

            try:
                classified = classify_content(entry)
            except Exception:
                skipped += 1
                continue

            if classified is None:
                skipped += 1
                continue

            try:
                chunks = chunk_content(classified)
            except Exception:
                skipped += 1
                continue

            for chunk in chunks:
                content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()[:16]
                file_stem = Path(source_file).stem
                chunk_id = f"rt-{file_stem[:8]}-{content_hash}"

                # Extract timestamp from entry or use now
                created_at = entry.get("timestamp")
                if not created_at:
                    created_at = datetime.now(timezone.utc).isoformat()

                conversation_id = chunk.metadata.get("session_id") or file_stem

                try:
                    cursor.execute(
                        """INSERT OR IGNORE INTO chunks
                           (id, content, metadata, source_file, project,
                            content_type, value_type, char_count, source,
                            created_at, conversation_id, sender)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk_id,
                            chunk.content,
                            json.dumps(chunk.metadata),
                            source_file,
                            project,
                            chunk.content_type.value,
                            chunk.value.value,
                            chunk.char_count,
                            "realtime_watcher",
                            created_at,
                            conversation_id,
                            chunk.metadata.get("sender"),
                        ),
                    )
                    # APSW uses conn.changes() instead of cursor.rowcount
                    if store.conn.changes() > 0:
                        inserted += 1
                    else:
                        skipped += 1  # Duplicate
                except Exception as e:
                    logger.warning("Insert failed for %s: %s", chunk_id, e)
                    skipped += 1

        if inserted > 0:
            logger.info("Flushed %d chunks (%d skipped)", inserted, skipped)

    return flush_to_db
