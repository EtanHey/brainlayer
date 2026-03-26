"""Bridge between JSONLWatcher and BrainLayer's indexing pipeline.

Processes raw JSONL lines through pre-filter → classify → chunk → post-filter → insert.
Chunks are immediately searchable via FTS5; embeddings are backfilled by enrichment.

Filtering layers:
  1. Pre-classify: skip noise entry types, system-reminders, short messages
  2. classify_content: existing pipeline (skip tool JSON, acknowledgments, etc.)
  3. chunk_content: min-length by content type (80 for assistant, 15 for user)
  4. Post-chunk: strip system-reminder injections from content, skip file deletion diffs
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import get_db_path
from .pipeline.chunk import chunk_content
from .pipeline.classify import classify_content
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Pre-classify filters ─────────────────────────────────────────────────────

# Entry types to skip entirely (before classify_content sees them)
SKIP_ENTRY_TYPES = frozenset(
    {
        "progress",
        "queue-operation",
        "file-history-snapshot",
        "pr-link",
        "last-prompt",
        "system",
    }
)

# Allowed entry types (whitelist approach — anything not listed is skipped)
ALLOWED_ENTRY_TYPES = frozenset(
    {
        "user",
        "assistant",
        "whatsapp_message",
    }
)

# Minimum raw content length before even attempting classification
MIN_RAW_CONTENT_LENGTH = 20

# Regex for system-reminder blocks injected by hooks
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

# Regex for pure file deletion diffs (- lines only, no + lines)
_PURE_DELETION_DIFF_RE = re.compile(r"^```(?:diff)?\n(?:-[^\n]*\n)+```$", re.MULTILINE)


def _strip_system_reminders(text: str) -> str:
    """Remove system-reminder XML blocks from text content."""
    return _SYSTEM_REMINDER_RE.sub("", text).strip()


def _is_pure_deletion_diff(text: str) -> bool:
    """Check if text is just a file deletion diff with no added context."""
    stripped = text.strip()
    # Must contain diff markers
    if "---" not in stripped and "+++" not in stripped:
        return False
    lines = stripped.split("\n")
    diff_lines = [l for l in lines if l.startswith(("-", "+")) and not l.startswith(("---", "+++"))]
    if not diff_lines:
        return False
    # Pure deletion: all diff lines are removals, no additions
    additions = [l for l in diff_lines if l.startswith("+")]
    return len(additions) == 0


def _extract_raw_text(entry: dict) -> str:
    """Extract the raw text content from any JSONL entry type."""
    entry_type = entry.get("type", "")
    if entry_type == "user":
        raw = entry.get("message", {}).get("content", "")
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            parts = []
            for block in raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return " ".join(parts)
        return ""
    if entry_type == "assistant":
        blocks = entry.get("message", {}).get("content", [])
        parts = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def should_skip_entry(entry: dict) -> str | None:
    """Pre-classify filter. Returns skip reason or None to keep.

    This runs BEFORE classify_content to reject obvious noise early.
    """
    entry_type = entry.get("type", "")

    # Whitelist: only process known content types
    if entry_type not in ALLOWED_ENTRY_TYPES:
        return f"type:{entry_type}"

    # Extract raw text for content checks
    raw_text = _extract_raw_text(entry)

    # Skip very short content
    if len(raw_text.strip()) < MIN_RAW_CONTENT_LENGTH:
        return "too_short"

    # Skip if content is mostly system-reminder injection
    cleaned = _strip_system_reminders(raw_text)
    if len(cleaned.strip()) < MIN_RAW_CONTENT_LENGTH:
        return "system_reminder_only"

    return None


def should_skip_chunk_content(content: str) -> str | None:
    """Post-chunk filter. Returns skip reason or None to keep."""
    # Strip system-reminders from the final content
    cleaned = _strip_system_reminders(content)
    if len(cleaned.strip()) < MIN_RAW_CONTENT_LENGTH:
        return "system_reminder_residue"

    # Skip pure file deletion diffs
    if _is_pure_deletion_diff(cleaned):
        return "pure_deletion_diff"

    return None


# ── Project extraction ───────────────────────────────────────────────────────

_PROJECT_CACHE: dict[str, str] = {}


def _normalize_project_name(raw: str) -> str:
    """Convert encoded project path to human-readable name."""
    if raw in _PROJECT_CACHE:
        return _PROJECT_CACHE[raw]
    parts = raw.split("-")
    name = parts[-1] if parts else raw
    _PROJECT_CACHE[raw] = name
    return name


def _extract_project_from_source(source_file: str) -> str | None:
    """Extract project name from the source file path."""
    p = Path(source_file)
    parent_name = p.parent.name
    if parent_name and parent_name != "projects":
        return _normalize_project_name(parent_name)
    return None


# ── Flush callback ───────────────────────────────────────────────────────────


def create_flush_callback(db_path: Path | None = None) -> callable:
    """Create an on_flush callback that processes JSONL lines into BrainLayer.

    Returns a callable that takes a list[dict] of raw JSONL entries and
    inserts them as chunks into the database (deferred embedding).
    """
    resolved_db = db_path or get_db_path()
    store = VectorStore(resolved_db)

    def flush_to_db(entries: list[dict[str, Any]]) -> None:
        """Process raw JSONL entries through pipeline and insert into DB."""
        import time as _time

        flush_start = _time.monotonic()
        cursor = store.conn.cursor()
        inserted = 0
        skipped = 0
        source_files_seen: set[str] = set()

        for entry in entries:
            source_file = entry.get("_source_file", "unknown")
            source_files_seen.add(source_file)
            project = _extract_project_from_source(source_file)

            # Layer 1: Pre-classify filter
            skip_reason = should_skip_entry(entry)
            if skip_reason:
                skipped += 1
                continue

            # Layer 2: Pipeline classify
            try:
                classified = classify_content(entry)
            except Exception:
                skipped += 1
                continue

            if classified is None:
                skipped += 1
                continue

            # Layer 3: Pipeline chunk
            try:
                chunks = chunk_content(classified)
            except Exception:
                skipped += 1
                continue

            for chunk in chunks:
                # Layer 4: Post-chunk content filter
                clean_content = _strip_system_reminders(chunk.content)
                skip_reason = should_skip_chunk_content(clean_content)
                if skip_reason:
                    skipped += 1
                    continue

                content_hash = hashlib.sha256(clean_content.encode()).hexdigest()[:16]
                file_stem = Path(source_file).stem
                chunk_id = f"rt-{file_stem[:8]}-{content_hash}"

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
                            clean_content,
                            json.dumps(chunk.metadata),
                            source_file,
                            project,
                            chunk.content_type.value,
                            chunk.value.value,
                            len(clean_content),
                            "realtime_watcher",
                            created_at,
                            conversation_id,
                            chunk.metadata.get("sender"),
                        ),
                    )
                    if store.conn.changes() > 0:
                        inserted += 1
                    else:
                        skipped += 1
                except Exception as e:
                    logger.warning("Insert failed for %s: %s", chunk_id, e)
                    skipped += 1

        latency_ms = (_time.monotonic() - flush_start) * 1000

        if inserted > 0:
            logger.info(
                "Flushed %d chunks (%d skipped) in %.1fms",
                inserted,
                skipped,
                latency_ms,
            )

        try:
            from .telemetry import emit_watcher_flush

            emit_watcher_flush(
                chunks_indexed=inserted,
                chunks_skipped=skipped,
                latency_ms=latency_ms,
                source_files=list(source_files_seen),
            )
        except Exception:
            pass

    return flush_to_db
