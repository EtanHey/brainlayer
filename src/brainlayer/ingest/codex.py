"""Codex CLI session ingestion adapter for BrainLayer.

Normalizes Codex (GPT-5.4) session transcripts (~/.codex/sessions/YYYY/MM/DD/*.jsonl)
into BrainLayer-compatible chunks and indexes them with source='codex_cli'.

Codex JSONL format per line:
  {"timestamp": ..., "type": "session_meta"|"response_item"|"event_msg"|"turn_context", "payload": {...}}

Key response_item types:
  - type=message, role=user        → user_message (filter system injections)
  - type=message, role=assistant   → assistant_text / ai_code
  - type=function_call             → skip (just metadata)
  - type=function_call_output      → tool result → file_read / stack_trace / build_log
  - type=reasoning                 → skip (encrypted)
  - role=developer                 → skip (system instructions)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

from brainlayer.workspace_identity import canonical_project_from_workspace

logger = logging.getLogger(__name__)

# System injection patterns — skip these user-role messages
_SYSTEM_INJECTION_PATTERNS = [
    re.compile(r"^#\s*AGENTS\.md instructions for ", re.MULTILINE),
    re.compile(r"^<environment_context>", re.MULTILINE),
    re.compile(r"^<permissions instructions>", re.MULTILINE),
    re.compile(r"^<collaboration_mode>", re.MULTILINE),
]

# Minimum content lengths to keep
_MIN_USER_LEN = 15
_MIN_ASSISTANT_LEN = 50
_MIN_TOOL_OUTPUT_LEN = 50

_NATIVE_HEADER_FIELDS = (
    "id",
    "type",
    "role",
    "recipient",
    "channel",
    "call_id",
    "tool",
    "name",
    "turn_id",
    "turnId",
    "message_id",
    "messageId",
    "thread_id",
    "threadId",
    "parent_id",
    "parentId",
    "parentUuid",
    "fork_id",
    "forkId",
    "callId",
    "toolName",
)


def _native_record_context(line: dict, payload: dict, item: dict | None = None) -> dict[str, object]:
    """Keep supplied native identity headers without interpreting authorship."""
    context: dict[str, object] = {}
    scopes = (("/", line, _NATIVE_HEADER_FIELDS + ("timestamp",)), ("/payload", payload, _NATIVE_HEADER_FIELDS))
    if isinstance(item, dict):
        scopes += (("/payload/item", item, _NATIVE_HEADER_FIELDS),)
    for prefix, value, fields in scopes:
        for field in fields:
            candidate = value.get(field)
            if field in value and (candidate is None or isinstance(candidate, (str, int, float, bool))):
                context[f"{prefix}{field}" if prefix == "/" else f"{prefix}/{field}"] = candidate
    return context


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_input_text(blocks: list) -> str:
    """Extract text from input_text blocks (user messages)."""
    parts = []
    for block in blocks or []:
        if isinstance(block, dict) and block.get("type") == "input_text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_output_text(blocks: list) -> str:
    """Extract text from output_text blocks (assistant messages)."""
    parts = []
    for block in blocks or []:
        if isinstance(block, dict) and block.get("type") == "output_text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _extract_tool_output(output_raw) -> str:
    """Decode function_call_output.output field (str or JSON list of text blocks)."""
    if not output_raw:
        return ""
    if isinstance(output_raw, str):
        # May be JSON-encoded list like [{"type":"text","text":"..."}]
        stripped = output_raw.strip()
        if stripped.startswith("["):
            try:
                blocks = json.loads(stripped)
                if isinstance(blocks, list):
                    parts = []
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(b.get("text", ""))
                    result = "\n".join(parts)
                    return result if result else stripped
            except (json.JSONDecodeError, ValueError):
                pass
        return stripped
    return str(output_raw)


def _is_system_injection(text: str) -> bool:
    """Return True if this text is a system/developer injection, not a real user prompt."""
    for pattern in _SYSTEM_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _infer_project_from_cwd(cwd: Optional[str]) -> Optional[str]:
    """Derive canonical repository project from workspace metadata."""
    return canonical_project_from_workspace(cwd)


def _native_event_timestamp_error(line: dict, payload: dict) -> Optional[str]:
    """Return a native completion timestamp validation error, if any.

    Native event messages must retain their provider-recorded event time.  A
    session-start fallback would misrepresent when the completion occurred.
    Operation timing is optional for assistant text, but a supplied completion
    time must be a valid integer and cannot follow the recorded event time.
    """
    timestamp = line.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        return "missing provider timestamp"
    try:
        recorded_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if recorded_at.tzinfo is None:
            return "provider timestamp lacks timezone"
        recorded_seconds = recorded_at.timestamp()
    except (ValueError, OverflowError, OSError):
        return "invalid provider timestamp"

    completed_at_ms = payload.get("completed_at_ms")
    if completed_at_ms is not None:
        if type(completed_at_ms) is not int:
            return "invalid completion timestamp"
        # Keep the comparison integral: an unbounded JSON integer can overflow
        # when divided by a float before it reaches the rejection path.
        recorded_at_ms = int(recorded_seconds * 1000)
        if completed_at_ms > recorded_at_ms:
            return "completion timestamp follows provider timestamp"
    return None


# ---------------------------------------------------------------------------
# Core parsing
# ---------------------------------------------------------------------------


def parse_codex_session(file_path: Path) -> Iterator[dict]:
    """Parse a Codex session JSONL and yield normalized BrainLayer-compatible entries.

    Each yielded dict has:
        content      : str
        content_type : str  (user_message | assistant_text | ai_code | stack_trace |
                              build_log | file_read)
        session_id   : str
        timestamp    : str  (ISO-8601)
        project      : str | None
        source       : "codex_cli"
        metadata     : dict
    """
    session_id: Optional[str] = None
    session_timestamp: Optional[str] = None
    project: Optional[str] = None
    model: Optional[str] = None

    rejected_native_agent_messages = 0

    byte_offset = 0
    with open(file_path, "rb") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            line_start_offset = byte_offset
            byte_offset += len(raw_line)
            raw = raw_line.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                continue
            if not isinstance(line, dict):
                continue

            line_type = line.get("type", "")
            payload_value = line.get("payload")
            payload = payload_value if isinstance(payload_value, dict) else {}
            line_ts = line.get("timestamp", session_timestamp or "")
            record_metadata = {
                "_line_start_offset": line_start_offset,
                "_line_end_offset": line_start_offset + len(raw_line),
                "_raw_record_hash": hashlib.sha256(raw_line).hexdigest(),
                "record_context": _native_record_context(line, payload, payload.get("item")),
            }

            # ------------------------------------------------------------------
            # session_meta — capture session-level metadata
            # ------------------------------------------------------------------
            if line_type == "session_meta":
                session_id = payload.get("id")
                session_timestamp = payload.get("timestamp") or line_ts
                project = _infer_project_from_cwd(payload.get("cwd"))
                model = payload.get("model_provider", "openai")
                continue

            # ------------------------------------------------------------------
            # turn_context — pick up model details if present
            # ------------------------------------------------------------------
            if line_type == "turn_context":
                if not model:
                    model = payload.get("model")
                continue

            # ------------------------------------------------------------------
            # event_msg — accept only completed native agent text. Other lifecycle
            # events remain non-content records.
            # ------------------------------------------------------------------
            if line_type == "event_msg":
                item = payload.get("item")
                if payload.get("type") != "item_completed" or not isinstance(item, dict):
                    continue
                if item.get("type") != "AgentMessage":
                    continue

                timestamp_error = _native_event_timestamp_error(line, payload)
                if timestamp_error:
                    rejected_native_agent_messages += 1
                    logger.warning(
                        "Skipping native Codex AgentMessage in %s line %d: %s "
                        "(rejected_native_agent_messages=%d)",
                        file_path,
                        line_number,
                        timestamp_error,
                        rejected_native_agent_messages,
                    )
                    continue

                blocks = item.get("content")
                if (
                    not isinstance(blocks, list)
                    or not blocks
                    or any(
                        not isinstance(block, dict)
                        or block.get("type") != "Text"
                        or not isinstance(block.get("text"), str)
                        for block in blocks
                    )
                ):
                    rejected_native_agent_messages += 1
                    logger.warning(
                        "Skipping malformed native Codex AgentMessage in %s line %d "
                        "(rejected_native_agent_messages=%d)",
                        file_path,
                        line_number,
                        rejected_native_agent_messages,
                    )
                    continue

                for block_index, block in enumerate(blocks):
                    text = block["text"]
                    if not text.strip() or len(text.strip()) < _MIN_ASSISTANT_LEN:
                        continue
                    ctype = "ai_code" if "```" in text else "assistant_text"
                    yield {
                        "content": text,
                        "content_type": ctype,
                        "session_id": session_id,
                        "timestamp": line_ts,
                        "project": project,
                        "source": "codex_cli",
                        "metadata": {
                            "session_id": session_id,
                            "sender": "assistant",
                            "author_identity": "unverified",
                            "model": model,
                            "source_file": str(file_path),
                            "source_line": line_number,
                            "source_pointer": f"/payload/item/content/{block_index}/text",
                            **record_metadata,
                        },
                    }
                continue

            # ------------------------------------------------------------------
            # response_item — the main content source
            # ------------------------------------------------------------------
            if line_type != "response_item":
                continue

            ptype = payload.get("type", "")
            role = payload.get("role")

            # Skip system/developer injections
            if role == "developer":
                continue

            # Skip encrypted reasoning
            if ptype == "reasoning":
                continue

            # Skip bare function_call entries (metadata only, no readable content)
            if ptype == "function_call":
                continue

            # ------------------------------------------------------------------
            # User messages
            # ------------------------------------------------------------------
            if ptype == "message" and role == "user":
                text = _extract_input_text(payload.get("content") or [])
                if not text.strip():
                    continue
                # Filter system injections
                if _is_system_injection(text):
                    continue
                if len(text.strip()) < _MIN_USER_LEN:
                    continue

                yield {
                    "content": text,
                    "content_type": "user_message",
                    "session_id": session_id,
                    "timestamp": line_ts or session_timestamp,
                    "project": project,
                    "source": "codex_cli",
                    "metadata": {
                        "session_id": session_id,
                        "sender": "user",
                        "model": model,
                        "source_file": str(file_path),
                        "source_line": line_number,
                        "source_pointer": "/payload/content",
                        **record_metadata,
                    },
                }
                continue

            # ------------------------------------------------------------------
            # Assistant messages
            # ------------------------------------------------------------------
            if ptype == "message" and role == "assistant":
                text = _extract_output_text(payload.get("content") or [])
                if not text.strip():
                    continue
                if len(text.strip()) < _MIN_ASSISTANT_LEN:
                    continue

                # Classify: code blocks → ai_code, else assistant_text
                ctype = "ai_code" if "```" in text else "assistant_text"

                yield {
                    "content": text,
                    "content_type": ctype,
                    "session_id": session_id,
                    "timestamp": line_ts or session_timestamp,
                    "project": project,
                    "source": "codex_cli",
                    "metadata": {
                        "session_id": session_id,
                        "sender": "assistant",
                        "model": model,
                        "source_file": str(file_path),
                        "source_line": line_number,
                        "source_pointer": "/payload/content",
                        **record_metadata,
                    },
                }
                continue

            # ------------------------------------------------------------------
            # Tool outputs
            # ------------------------------------------------------------------
            if ptype == "function_call_output":
                raw_output = payload.get("output", "")
                text = _extract_tool_output(raw_output)
                if not text.strip():
                    continue
                if len(text.strip()) < _MIN_TOOL_OUTPUT_LEN:
                    continue

                # Classify tool output content
                ctype = _classify_tool_output(text)

                yield {
                    "content": text,
                    "content_type": ctype,
                    "session_id": session_id,
                    "timestamp": line_ts or session_timestamp,
                    "project": project,
                    "source": "codex_cli",
                    "metadata": {
                        "session_id": session_id,
                        "sender": "tool",
                        "call_id": payload.get("call_id"),
                        "model": model,
                        "source_file": str(file_path),
                        "source_line": line_number,
                        "source_pointer": "/payload/output",
                        **record_metadata,
                    },
                }


_STACK_TRACE_RE = re.compile(
    r"Traceback \(most recent call last\)|" r"at\s+[\w.]+\([\w.]+:\d+\)|" r'File "[^"]+", line \d+',
    re.MULTILINE,
)

_BUILD_LOG_RE = re.compile(
    r"^\s*\d+ (passing|failing)|^npm (ERR!|WARN)|^error\[E\d+\]:|^\[[\d:]+\]",
    re.MULTILINE,
)

_GIT_DIFF_RE = re.compile(r"^diff --git|^@@", re.MULTILINE)

_DIR_LISTING_RE = re.compile(r"^(.*/)?([\w.-]+\.(ts|js|py|json|md|go|rs|sh))\s*$", re.MULTILINE)


def _classify_tool_output(text: str) -> str:
    """Map tool output text to a BrainLayer content_type string."""
    if _STACK_TRACE_RE.search(text):
        return "stack_trace"
    if _BUILD_LOG_RE.search(text):
        return "build_log"
    if _GIT_DIFF_RE.search(text):
        return "git_diff"
    return "file_read"


# ---------------------------------------------------------------------------
# Ingestion entry point
# ---------------------------------------------------------------------------


def ingest_codex_session(
    file_path: Path,
    db_path: Optional[Path] = None,
    project_override: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Ingest a single Codex session JSONL into BrainLayer.

    Returns the number of chunks indexed.
    """
    from ..pipeline.chunk import Chunk, chunk_content
    from ..pipeline.classify import ClassifiedContent, ContentType, ContentValue

    # Map string content_type → pipeline enums
    _type_map = {
        "user_message": (ContentType.USER_MESSAGE, ContentValue.HIGH),
        "assistant_text": (ContentType.ASSISTANT_TEXT, ContentValue.MEDIUM),
        "ai_code": (ContentType.AI_CODE, ContentValue.HIGH),
        "stack_trace": (ContentType.STACK_TRACE, ContentValue.HIGH),
        "build_log": (ContentType.BUILD_LOG, ContentValue.LOW),
        "git_diff": (ContentType.GIT_DIFF, ContentValue.MEDIUM),
        "file_read": (ContentType.FILE_READ, ContentValue.MEDIUM),
    }

    all_chunks: List[Chunk] = []
    project: Optional[str] = project_override
    session_id: Optional[str] = None
    session_ts: Optional[str] = None

    for entry in parse_codex_session(file_path):
        if project is None:
            project = entry.get("project")
        if session_id is None:
            session_id = entry.get("session_id")
        if session_ts is None:
            session_ts = entry.get("timestamp")

        ctype_str = entry.get("content_type", "assistant_text")
        content_type, value = _type_map.get(ctype_str, (ContentType.ASSISTANT_TEXT, ContentValue.MEDIUM))

        classified = ClassifiedContent(
            content=entry["content"],
            content_type=content_type,
            value=value,
            metadata={
                **entry.get("metadata", {}),
                "source": "codex_cli",
            },
        )

        chunks = chunk_content(classified)
        all_chunks.extend(chunks)

        if verbose:
            print(f"  [{ctype_str}] {entry['content'][:80]!r}")

    if not all_chunks:
        logger.info("No indexable content in %s", file_path)
        return 0

    if dry_run:
        print(f"Dry run: {len(all_chunks)} chunks from {file_path.name} (not stored)")
        return len(all_chunks)

    if db_path is None:
        from ..paths import DEFAULT_DB_PATH

        db_path = DEFAULT_DB_PATH

    from ..index_new import index_chunks_to_sqlite

    # Patch source onto each chunk's metadata before indexing
    for chunk in all_chunks:
        chunk.metadata.setdefault("source", "codex_cli")

    indexed = index_chunks_to_sqlite(
        all_chunks,
        source_file=str(file_path),
        project=project,
        db_path=db_path,
        # Pass source through the metadata; index_new reads chunk.metadata["source"]
    )

    # Store session context if we have a session_id
    if session_id:
        try:
            from ..vector_store import VectorStore

            with VectorStore(db_path) as store:
                store.store_session_context(
                    session_id=session_id,
                    project=project or "codex",
                    started_at=session_ts,
                    ended_at=None,
                )
        except Exception as exc:
            logger.debug("Could not store session context for %s: %s", session_id, exc)

    logger.info("Indexed %d chunks from %s (session %s, project %s)", indexed, file_path.name, session_id, project)
    return indexed


def ingest_codex_dir(
    sessions_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    project_override: Optional[str] = None,
    since_days: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, int]:
    """Ingest all Codex session files under sessions_dir.

    Args:
        sessions_dir: Root of Codex sessions (default: ~/.codex/sessions).
        db_path: Override BrainLayer DB path.
        project_override: Force a project name on all sessions.
        since_days: Only ingest sessions from the last N days.
        dry_run: Parse and count but do not write to DB.
        verbose: Print each entry as it's classified.

    Returns:
        (files_processed, total_chunks)
    """
    if sessions_dir is None:
        sessions_dir = Path.home() / ".codex" / "sessions"

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Codex sessions directory not found: {sessions_dir}")

    jsonl_files = sorted(sessions_dir.rglob("*.jsonl"))

    if since_days is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - since_days * 86400
        jsonl_files = [f for f in jsonl_files if f.stat().st_mtime >= cutoff]

    if not jsonl_files:
        logger.info("No Codex session files found in %s", sessions_dir)
        return 0, 0

    # Skip files already indexed (check DB for existing source_file entries)
    if not dry_run and db_path is None:
        from ..paths import DEFAULT_DB_PATH

        db_path = DEFAULT_DB_PATH

    already_indexed: set[str] = set()
    if not dry_run and db_path and db_path.exists():
        try:
            from ..vector_store import VectorStore

            with VectorStore(db_path) as store:
                cursor = store._read_cursor()
                rows = cursor.execute("SELECT DISTINCT source_file FROM chunks WHERE source = 'codex_cli'")
                already_indexed = {row[0] for row in rows}
        except Exception as exc:
            logger.debug("Could not check existing codex_cli chunks: %s", exc)

    files_processed = 0
    total_chunks = 0

    for f in jsonl_files:
        if str(f) in already_indexed:
            logger.debug("Skipping already-indexed %s", f.name)
            continue

        try:
            n = ingest_codex_session(
                f,
                db_path=db_path,
                project_override=project_override,
                dry_run=dry_run,
                verbose=verbose,
            )
            files_processed += 1
            total_chunks += n
        except Exception as exc:
            logger.warning("Failed to ingest %s: %s", f.name, exc)

    return files_processed, total_chunks


# ---------------------------------------------------------------------------
# Module entry point: python -m brainlayer.ingest.codex <session-file>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest Codex session transcripts into BrainLayer.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a Codex session JSONL file or sessions directory (default: ~/.codex/sessions)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Parse but do not write to DB")
    parser.add_argument("--project", default=None, help="Override project name")
    parser.add_argument("--since-days", type=int, default=None, help="Only process last N days")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each entry")
    parser.add_argument("--db", default=None, help="Override BrainLayer DB path")
    args = parser.parse_args()

    db = Path(args.db) if args.db else None
    target = Path(args.path) if args.path else None

    if target and target.is_file():
        n = ingest_codex_session(
            target,
            db_path=db,
            project_override=args.project,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        print(f"Indexed {n} chunks from {target.name}")
    else:
        sessions_root = target if target else None
        files, chunks = ingest_codex_dir(
            sessions_dir=sessions_root,
            db_path=db,
            project_override=args.project,
            since_days=args.since_days,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        print(f"Processed {files} session files, {chunks} chunks indexed")
