"""Bridge between JSONLWatcher and BrainLayer's indexing pipeline.

Processes raw JSONL lines through pre-filter → classify → chunk → post-filter → insert.
Chunks are immediately searchable via FTS5; embeddings are backfilled by enrichment.

Filtering layers:
  1. Pre-classify: skip noise entry types, system-reminders, short messages
  2. classify_content: existing pipeline (skip tool JSON, acknowledgments, etc.)
  3. chunk_content: min-length by content type (80 for assistant, 15 for user)
  4. Post-chunk: strip system-reminder injections from content, skip file deletion diffs
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw

from .chunk_origin import detect_chunk_origin
from .claude_paths import extract_claude_conversation_id as _extract_claude_conversation_id
from .dedupe import find_duplicate, merge_duplicate_chunk, merge_existing_chunk_seen, normalized_exact_hash
from .ingest_guard import recursive_mcp_output_reason
from .paths import get_db_path
from .pipeline.chunk import chunk_content
from .pipeline.classify import classify_content
from .pipeline.correction_detection import build_correction_tags
from .queue_io import enqueue_watcher_chunk
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


def should_skip_entry(entry: dict, *, source_file: str | None = None) -> str | None:
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

    resolved_source_file = source_file or entry.get("_source_file")
    if recursive_mcp_output_reason(raw_text, source_file=resolved_source_file, reject_precompact=True):
        return "recursive_mcp_output"

    # Skip if content is mostly system-reminder injection
    cleaned = _strip_system_reminders(raw_text)
    if len(cleaned.strip()) < MIN_RAW_CONTENT_LENGTH:
        return "system_reminder_only"

    return None


def should_skip_chunk_content(
    content: str,
    *,
    chunk_id: str | None = None,
    source_file: str | None = None,
) -> str | None:
    """Post-chunk filter. Returns skip reason or None to keep."""
    # Strip system-reminders from the final content
    cleaned = _strip_system_reminders(content)
    if len(cleaned.strip()) < MIN_RAW_CONTENT_LENGTH:
        return "system_reminder_residue"

    if recursive_mcp_output_reason(cleaned, chunk_id=chunk_id, source_file=source_file, reject_precompact=True):
        return "recursive_mcp_output"

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

    if raw.startswith("-Users-") or raw.startswith("-home-"):
        parts = raw.split("-")
        markers = {"Gits", "Desktop", "projects", "config"}
        last_marker_idx = -1
        for i, part in enumerate(parts):
            if part in markers:
                last_marker_idx = i

        if last_marker_idx >= 0 and last_marker_idx < len(parts) - 1:
            repo_parts = [p for p in parts[last_marker_idx + 1 :] if p]
            name = "-".join(repo_parts) if repo_parts else raw
        else:
            name = raw
    else:
        name = raw

    _PROJECT_CACHE[raw] = name
    return name


def _extract_project_from_source(source_file: str) -> str | None:
    """Extract the project root from a watcher source path.

    For Claude Code transcripts the canonical project directory is the segment
    immediately under `.../projects/`, even when the JSONL lives under nested
    session folders like `subagents/` or `tool-results/`.
    """
    p = Path(source_file)
    parts = p.parts
    if "projects" in parts:
        project_index = parts.index("projects") + 1
        if project_index < len(parts):
            return _normalize_project_name(parts[project_index])

    parent_name = p.parent.name
    if parent_name:
        return _normalize_project_name(parent_name)
    return None


# ── Flush callback ───────────────────────────────────────────────────────────


def create_flush_callback(db_path: Path | None = None, *, arbitrated: bool | None = None) -> callable:
    """Create an on_flush callback that processes JSONL lines into BrainLayer.

    Returns a callable that takes a list[dict] of raw JSONL entries and
    inserts them as chunks into the database (deferred embedding).
    """
    if arbitrated is None:
        arbitrated = os.environ.get("BRAINLAYER_ARBITRATED") == "1"
    store = None if arbitrated else VectorStore(db_path or get_db_path())

    def flush_to_db(entries: list[dict[str, Any]]) -> int:
        """Process raw JSONL entries through pipeline and insert into DB."""
        import time as _time

        flush_start = _time.monotonic()
        cursor = None if store is None else store.conn.cursor()
        inserted = 0
        skipped = 0
        source_files_seen: set[str] = set()

        for entry in entries:
            source_file = entry.get("_source_file", "unknown")
            source_files_seen.add(source_file)
            project = _extract_project_from_source(source_file)
            claude_conversation_id = _extract_claude_conversation_id(source_file)

            # Layer 1: Pre-classify filter
            skip_reason = should_skip_entry(entry, source_file=source_file)
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
                clean_content = _strip_system_reminders(chunk.content)
                content_hash = normalized_exact_hash(clean_content)[:16]
                file_stem = Path(source_file).stem
                chunk_id = f"rt-{file_stem[:8]}-{content_hash}"

                # Layer 4: Post-chunk content filter
                skip_reason = should_skip_chunk_content(clean_content, chunk_id=chunk_id, source_file=source_file)
                if skip_reason:
                    skipped += 1
                    continue

                created_at = entry.get("timestamp")
                if not created_at:
                    created_at = datetime.now(timezone.utc).isoformat()

                conversation_id = chunk.metadata.get("session_id") or file_stem
                metadata = dict(chunk.metadata)
                if claude_conversation_id:
                    metadata["claude_conversation_id"] = claude_conversation_id
                tags = None
                if chunk.content_type.value == "user_message":
                    correction_tags = build_correction_tags(clean_content)
                    if correction_tags:
                        tags = json.dumps(correction_tags)
                chunk_origin = detect_chunk_origin(clean_content)

                try:
                    if arbitrated:
                        enqueue_watcher_chunk(
                            chunk_id=chunk_id,
                            content=clean_content,
                            metadata=metadata,
                            source_file=source_file,
                            project=project,
                            content_type=chunk.content_type.value,
                            value_type=chunk.value.value,
                            created_at=created_at,
                            conversation_id=conversation_id,
                            sender=metadata.get("sender"),
                            tags=json.loads(tags) if tags else None,
                            chunk_origin=chunk_origin,
                        )
                        inserted += 1
                    else:
                        assert cursor is not None and store is not None
                        for attempt in range(5):
                            transaction_started = False
                            try:
                                cursor.execute("BEGIN IMMEDIATE")
                                transaction_started = True
                                duplicate, dedupe_fields = find_duplicate(
                                    store.conn,
                                    chunk_id=chunk_id,
                                    content=clean_content,
                                    created_at=created_at,
                                    project=project,
                                    content_type=chunk.content_type.value,
                                )
                                if duplicate is not None:
                                    merge_duplicate_chunk(
                                        store.conn,
                                        canonical_id=duplicate.canonical_chunk_id,
                                        duplicate_id=chunk_id,
                                        incoming={
                                            "id": chunk_id,
                                            "content": clean_content,
                                            "tags": tags,
                                            "created_at": created_at,
                                            "last_seen_at": created_at,
                                        },
                                        mechanism=duplicate.mechanism,
                                        hamming_distance_value=duplicate.hamming_distance,
                                    )
                                    cursor.execute("COMMIT")
                                    transaction_started = False
                                    inserted += 1
                                    break
                                if merge_existing_chunk_seen(
                                    store.conn,
                                    chunk_id=chunk_id,
                                    incoming={
                                        "id": chunk_id,
                                        "content": clean_content,
                                        "tags": tags,
                                        "created_at": created_at,
                                        "last_seen_at": created_at,
                                    },
                                ):
                                    cursor.execute("COMMIT")
                                    transaction_started = False
                                    inserted += 1
                                    break
                                cursor.execute(
                                    """INSERT OR IGNORE INTO chunks
                                       (id, content, metadata, source_file, project,
                                        content_type, value_type, char_count, source,
                                        created_at, conversation_id, sender, tags, chunk_origin,
                                        seen_count, last_seen_at, dedupe_hash, simhash,
                                        simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3,
                                        ingested_at)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                               CAST(strftime('%s', 'now') AS INTEGER))""",
                                    (
                                        chunk_id,
                                        clean_content,
                                        json.dumps(metadata),
                                        source_file,
                                        project,
                                        chunk.content_type.value,
                                        chunk.value.value,
                                        len(clean_content),
                                        "realtime_watcher",
                                        created_at,
                                        conversation_id,
                                        metadata.get("sender"),
                                        tags,
                                        chunk_origin,
                                        1,
                                        created_at,
                                        dedupe_fields.dedupe_hash,
                                        dedupe_fields.simhash,
                                        dedupe_fields.bands[0],
                                        dedupe_fields.bands[1],
                                        dedupe_fields.bands[2],
                                        dedupe_fields.bands[3],
                                    ),
                                )
                                changed = store.conn.changes() > 0
                                cursor.execute("COMMIT")
                                transaction_started = False
                                if changed:
                                    inserted += 1
                                else:
                                    skipped += 1
                                break
                            except apsw.BusyError:
                                if transaction_started:
                                    cursor.execute("ROLLBACK")
                                if attempt == 4:
                                    raise
                                time.sleep(0.05 * (2**attempt))
                            except Exception:
                                if transaction_started:
                                    cursor.execute("ROLLBACK")
                                raise
                except Exception as e:
                    logger.warning("Queue/write failed for %s: %s", chunk_id, e)
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

        return inserted

    return flush_to_db
