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
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw

from .agent_provenance import classify_provenance, effective_visibility
from .alarm import BrainLayerAlarm
from .chunk_origin import detect_chunk_origin
from .claude_paths import extract_claude_conversation_id as _extract_claude_conversation_id
from .content_class import classify_content_class
from .dedupe import find_duplicate, merge_duplicate_chunk, merge_existing_chunk_seen, normalized_exact_hash
from .ingest_guard import recursive_mcp_output_reason
from .paths import get_db_path
from .pipeline.chunk import chunk_content
from .pipeline.classify import classify_content
from .pipeline.correction_detection import build_correction_tags
from .pipeline.secret_scrub import scrub_secrets
from .queue_io import enqueue_watcher_chunk
from .t3_provenance import DEFAULT_T3_STATE_DB, t3_app_codex_session_ids
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


class FlushWatermarks(dict[str, int]):
    """Confirmed per-source offsets returned by watcher flushes."""

    def __init__(self, *args: Any, inserted: int = 0, skipped: int = 0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.inserted = inserted
        self.skipped = skipped

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.inserted == other
        return super().__eq__(other)


def _nonnegative_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


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
        msg = entry.get("message")
        raw = msg if isinstance(msg, str) else (msg.get("content", "") if isinstance(msg, dict) else "")
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
        msg = entry.get("message")
        blocks = msg if isinstance(msg, str) else (msg.get("content", []) if isinstance(msg, dict) else [])
        if isinstance(blocks, str):
            return blocks
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


def _extract_workspace_project(entry: dict[str, Any] | None) -> str | None:
    """Return a project name from explicit session workspace metadata only."""
    if not isinstance(entry, dict):
        return None

    candidates = [entry]
    for key in ("payload", "metadata"):
        value = entry.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in candidates:
        for key in ("cwd", "workspace", "workdir"):
            value = candidate.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            name = _normalize_project_name(Path(value).name)
            if name and not name.isdigit():
                return name
    return None


def _extract_project_from_source(source_file: str, entry: dict[str, Any] | None = None) -> str | None:
    """Extract a project only from workspace metadata or Claude's project path.

    Codex, Cursor, and Gemini paths are date-partitioned and therefore cannot
    safely provide a project name. Claude Code encodes the workspace explicitly
    in its `projects/<workspace>` path, which remains a trusted fallback.
    """
    p = Path(source_file)
    parts = p.parts
    if "projects" in parts:
        project_index = parts.index("projects") + 1
        if project_index < len(parts):
            project = _normalize_project_name(parts[project_index])
            if not project.isdigit():
                return project
    return _extract_workspace_project(entry)


def _extract_project_from_session_file(source_file: str) -> str | None:
    """Derive a project from the source file's durable session metadata."""
    project = _extract_project_from_source(source_file)
    if project is not None:
        return project
    try:
        with Path(source_file).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                project = _extract_project_from_source(source_file, entry)
                if project is not None:
                    return project
    except OSError:
        return None
    return None


def _source_file_fingerprint(source_file: str) -> tuple[int, int] | None:
    """Return a version token for retrying an unresolved session file."""
    try:
        stat = Path(source_file).stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _source_file_identity(source_file: str) -> tuple[int, int, int] | None:
    """Return enough state to detect replacement or rewind without invalidating on append."""
    try:
        stat = Path(source_file).stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size


def _content_class_for_visibility(base_content_class: str, visibility: str) -> str:
    if visibility == "default":
        return base_content_class
    if visibility == "operational":
        return "operational"
    return "cold"


# ── Flush callback ───────────────────────────────────────────────────────────


def create_flush_callback(db_path: Path | None = None, *, arbitrated: bool | None = None) -> callable:
    """Create an on_flush callback that processes JSONL lines into BrainLayer.

    Returns a callable that takes a list[dict] of raw JSONL entries and
    inserts them as chunks into the database (deferred embedding).
    """
    if arbitrated is None:
        arbitrated = os.environ.get("BRAINLAYER_ARBITRATED") == "1"
    store = None if arbitrated else VectorStore(db_path or get_db_path())
    liveness_schema_ready = False
    source_projects: dict[str, str] = {}
    resolved_source_identities: dict[str, tuple[int, int, int]] = {}
    unresolved_source_versions: dict[str, tuple[int, int] | None] = {}

    def ensure_direct_liveness_schema() -> None:
        if liveness_schema_ready or store is None:
            return
        from .drain import _ensure_watcher_liveness_schema

        _ensure_watcher_liveness_schema(store.conn)

    def mark_direct_liveness_schema_ready() -> None:
        nonlocal liveness_schema_ready
        liveness_schema_ready = True

    def record_direct_liveness(chunk_id: str, ingested_at: int) -> None:
        if store is None:
            return
        from .drain import _record_watcher_liveness

        _record_watcher_liveness(store.conn, chunk_id, ingested_at)

    def flush_to_db(entries: list[dict[str, Any]]) -> FlushWatermarks:
        """Process raw JSONL entries through pipeline and insert into DB."""
        import time as _time

        flush_start = _time.monotonic()
        t3_state_db = Path(os.environ.get("BRAINLAYER_T3_STATE_DB", DEFAULT_T3_STATE_DB)).expanduser()
        t3_linkage_resolved = True
        try:
            linked_t3_session_ids = t3_app_codex_session_ids(t3_state_db) if t3_state_db.exists() else set()
        except BrainLayerAlarm:
            linked_t3_session_ids = set()
            t3_linkage_resolved = False
        cursor = None if store is None else store.conn.cursor()
        inserted = 0
        skipped = 0
        source_files_seen: set[str] = set()
        confirmed_offsets: dict[str, int] = {}

        def confirm_entry(entry: dict[str, Any], source_file: str) -> None:
            raw_offset = entry.get("_line_end_offset")
            if isinstance(raw_offset, int):
                confirmed_offsets[source_file] = max(confirmed_offsets.get(source_file, 0), raw_offset)

        def enqueue_chunk(
            *,
            chunk_id: str,
            clean_content: str,
            metadata: dict[str, Any],
            source_file: str,
            project: str | None,
            content_type: str,
            value_type: str,
            created_at: str,
            conversation_id: str,
            source_end_offset: int | None,
            sender: Any,
            tags: str | None,
            chunk_origin: str | None,
            content_class: str,
            provenance_class: str,
        ) -> None:
            enqueue_watcher_chunk(
                chunk_id=chunk_id,
                content=clean_content,
                metadata=metadata,
                source_file=source_file,
                project=project,
                content_type=content_type,
                value_type=value_type,
                created_at=created_at,
                conversation_id=conversation_id,
                source_end_offset=source_end_offset,
                sender=sender,
                tags=json.loads(tags) if tags else None,
                chunk_origin=chunk_origin,
                content_class=content_class,
                provenance_class=provenance_class,
            )

        for entry in entries:
            source_file = entry.get("_source_file", "unknown")
            source_files_seen.add(source_file)
            if not t3_linkage_resolved:
                source_only_provenance = classify_provenance(source_file, t3_linked_session_ids=set())
                if source_only_provenance.provenance_tag == "codex-session":
                    # Do not persist or confirm ambiguous Codex provenance; a later pass can replay this offset.
                    continue
            project = source_projects.get(source_file)
            source_identity = _source_file_identity(source_file)
            cached_identity = resolved_source_identities.get(source_file)
            if project is not None and source_identity is not None and cached_identity is not None:
                if source_identity[:2] != cached_identity[:2] or source_identity[2] < cached_identity[2]:
                    source_projects.pop(source_file, None)
                    resolved_source_identities.pop(source_file, None)
                    project = None
                else:
                    resolved_source_identities[source_file] = source_identity
            if project is None:
                source_version = _source_file_fingerprint(source_file)
                if (
                    source_file not in unresolved_source_versions
                    or unresolved_source_versions[source_file] != source_version
                ):
                    project = _extract_project_from_session_file(source_file)
                    if project is not None:
                        source_projects[source_file] = project
                        if source_identity is not None:
                            resolved_source_identities[source_file] = source_identity
                        unresolved_source_versions.pop(source_file, None)
                    else:
                        unresolved_source_versions[source_file] = source_version
            claude_conversation_id = _extract_claude_conversation_id(source_file)

            # Layer 1: Pre-classify filter
            skip_reason = should_skip_entry(entry, source_file=source_file)
            if skip_reason:
                skipped += 1
                confirm_entry(entry, source_file)
                continue

            # Layer 2: Pipeline classify
            try:
                classified = classify_content(entry)
            except Exception:
                logger.exception("Classification failed for watcher entry from %s", source_file)
                skipped += 1
                confirm_entry(entry, source_file)
                continue

            if classified is None:
                skipped += 1
                confirm_entry(entry, source_file)
                continue

            # Layer 3: Pipeline chunk
            try:
                chunks = chunk_content(classified)
            except Exception:
                logger.exception("Chunking failed for watcher entry from %s", source_file)
                skipped += 1
                confirm_entry(entry, source_file)
                continue

            entry_confirmed = True

            for chunk in chunks:
                clean_content = _strip_system_reminders(chunk.content)
                secret_scrub_result = scrub_secrets(clean_content)
                clean_content = secret_scrub_result.text
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
                if secret_scrub_result.redactions:
                    metadata["secret_scrub_redactions"] = sorted(
                        {redaction.provider for redaction in secret_scrub_result.redactions}
                    )
                if secret_scrub_result.quarantine:
                    metadata["secret_scrub_quarantine_count"] = len(secret_scrub_result.quarantine)
                if claude_conversation_id:
                    metadata["claude_conversation_id"] = claude_conversation_id
                tags = None
                if chunk.content_type.value == "user_message":
                    correction_tags = build_correction_tags(clean_content)
                    if correction_tags:
                        tags = json.dumps(correction_tags)
                chunk_origin = detect_chunk_origin(clean_content)
                base_content_class = classify_content_class(
                    clean_content,
                    content_type=chunk.content_type.value,
                    tags=json.loads(tags) if tags else None,
                    source="realtime_watcher",
                    source_file=source_file,
                    project=project,
                )
                provenance_decision = classify_provenance(
                    source_file,
                    base_content_class,
                    content=clean_content,
                    t3_linked_session_ids=linked_t3_session_ids,
                )
                visibility = effective_visibility(provenance_decision, base_content_class)
                content_class = _content_class_for_visibility(base_content_class, visibility)
                provenance_class = provenance_decision.provenance_tag
                metadata["provenance_tag"] = provenance_decision.provenance_tag
                metadata["provenance_search_policy"] = provenance_decision.search_policy
                metadata["provenance_effective_visibility"] = visibility

                if arbitrated:
                    try:
                        enqueue_chunk(
                            chunk_id=chunk_id,
                            clean_content=clean_content,
                            metadata=metadata,
                            source_file=source_file,
                            project=project,
                            content_type=chunk.content_type.value,
                            value_type=chunk.value.value,
                            created_at=created_at,
                            conversation_id=conversation_id,
                            source_end_offset=entry.get("_line_end_offset"),
                            sender=metadata.get("sender"),
                            tags=tags,
                            chunk_origin=chunk_origin,
                            content_class=content_class,
                            provenance_class=provenance_class,
                        )
                    except Exception:
                        entry_confirmed = False
                        raise
                    inserted += 1
                else:
                    assert cursor is not None and store is not None
                    deadline_s = _nonnegative_float_env("BRAINLAYER_WATCHER_WRITE_DEADLINE_S", 15.0)
                    deadline = time.monotonic() + deadline_s
                    source_end_offset = entry.get("_line_end_offset")
                    if not isinstance(source_end_offset, int):
                        source_end_offset = None
                    source_position_recorded_at = float(time.time()) if source_end_offset is not None else None
                    attempt = 0
                    while True:
                        transaction_started = False
                        try:
                            ingested_at = int(time.time())
                            cursor.execute("BEGIN IMMEDIATE")
                            transaction_started = True
                            ensure_direct_liveness_schema()
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
                                record_direct_liveness(duplicate.canonical_chunk_id, ingested_at)
                                cursor.execute("COMMIT")
                                mark_direct_liveness_schema_ready()
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
                                record_direct_liveness(chunk_id, ingested_at)
                                cursor.execute("COMMIT")
                                mark_direct_liveness_schema_ready()
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
                                    source_end_offset, source_last_queued_at,
                                    ingested_at, content_class, provenance_class)
                                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                           ?, ?, ?, ?, ?)""",
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
                                    source_end_offset,
                                    source_position_recorded_at,
                                    ingested_at,
                                    content_class,
                                    provenance_class,
                                ),
                            )
                            changed = store.conn.changes() > 0
                            if changed:
                                record_direct_liveness(chunk_id, ingested_at)
                            cursor.execute("COMMIT")
                            mark_direct_liveness_schema_ready()
                            transaction_started = False
                            if changed:
                                inserted += 1
                            else:
                                skipped += 1
                            break
                        except apsw.BusyError:
                            if transaction_started:
                                cursor.execute("ROLLBACK")
                            if time.monotonic() >= deadline:
                                try:
                                    enqueue_chunk(
                                        chunk_id=chunk_id,
                                        clean_content=clean_content,
                                        metadata=metadata,
                                        source_file=source_file,
                                        project=project,
                                        content_type=chunk.content_type.value,
                                        value_type=chunk.value.value,
                                        created_at=created_at,
                                        conversation_id=conversation_id,
                                        source_end_offset=entry.get("_line_end_offset"),
                                        sender=metadata.get("sender"),
                                        tags=tags,
                                        chunk_origin=chunk_origin,
                                        content_class=content_class,
                                        provenance_class=provenance_class,
                                    )
                                except Exception:
                                    logger.exception("spill_failed chunk_id=%s source_file=%s", chunk_id, source_file)
                                    entry_confirmed = False
                                    raise
                                inserted += 1
                                logger.warning("Direct watcher write busy for %s; spilled to queue", chunk_id)
                                break
                            delay = min(0.05 * (2**attempt), 1.0) * random.uniform(0.8, 1.2)
                            attempt += 1
                            time.sleep(delay)
                        except Exception:
                            transaction_started = False
                            try:
                                cursor.execute("ROLLBACK")
                            except Exception:
                                pass
                            entry_confirmed = False
                            raise

            if entry_confirmed:
                confirm_entry(entry, source_file)

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

        return FlushWatermarks(confirmed_offsets, inserted=inserted, skipped=skipped)

    return flush_to_db
