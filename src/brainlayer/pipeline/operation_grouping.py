"""Operation Grouping Pipeline — Group chunks into logical operations.

Phase 8a: Detect patterns like read→edit→test, search→read chains,
and user→plan→implement cycles using heuristic rules.

Usage:
    from brainlayer.pipeline.operation_grouping import run_operation_grouping
    run_operation_grouping(vector_store, project="my-project")
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Operation type definitions
OP_EDIT_CYCLE = "edit-cycle"  # read → edit → test
OP_RESEARCH = "research"  # search/grep → read multiple
OP_FEATURE_CYCLE = "feature-cycle"  # user asks → plan → implement
OP_DEBUG = "debug"  # error → read → try fix → test
OP_CONFIG = "config"  # write/edit config files
OP_REVIEW = "review"  # read multiple files, no edits

# Tool action categories (tuples for deterministic order)
SEARCH_TOOLS = ("Grep", "Glob")
READ_TOOLS = ("Read",)
EDIT_TOOLS = ("Edit", "Write")
TEST_TOOLS = ("Bash",)  # detected by content patterns
# Ordered: Edit/Write first so they take priority over Read
ALL_FILE_TOOLS = EDIT_TOOLS + SEARCH_TOOLS + READ_TOOLS

# Max time gap between chunks in same operation (seconds)
MAX_GAP_SECONDS = 300  # 5 minutes


def _parse_timestamp(ts: Optional[str]) -> Optional[float]:
    """Parse ISO timestamp to epoch seconds."""
    if not ts:
        return None
    try:
        from datetime import datetime as dt_cls

        # Handle various ISO formats
        ts_clean = ts.replace("Z", "+00:00")
        dt = dt_cls.fromisoformat(ts_clean)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _extract_tool_info(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Extract tool usage info from a chunk's content."""
    content = chunk.get("content", "")
    content_type = chunk.get("content_type", "")
    metadata = chunk.get("metadata", "{}")

    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

    info = {
        "chunk_id": chunk.get("id", ""),
        "content_type": content_type,
        "tool": None,
        "action": None,
        "file_path": None,
        "is_test": False,
        "is_error": False,
        "is_user_message": content_type == "user_message",
        "is_assistant": content_type == "assistant_text",
        "timestamp": chunk.get("timestamp"),
        "snippet": "",  # brief context for summaries
    }

    # Extract a short snippet for summary context
    # Take first meaningful line (skip empty, very short)
    for line in content.split("\n"):
        line = line.strip()
        if len(line) > 10 and not line.startswith("#"):
            info["snippet"] = line[:80]
            break

    # Extract file paths from content
    import re

    file_match = re.search(
        r'(?:file_path|path)[=:]\s*["\']?'
        r'([^\s"\']+\.\w+)',
        content,
    )
    if file_match:
        info["file_path"] = file_match.group(1)
    else:
        # Also match bare file paths
        bare_match = re.search(
            r"(?:^|\s)(/[^\s]+\.\w{1,5})",
            content,
        )
        if bare_match:
            info["file_path"] = bare_match.group(1)

    # Detect tool calls from content patterns
    # Use word-boundary matching to avoid false positives
    # (e.g., "Read" matching "README")
    has_tool_marker = "tool_use" in content or "Tool:" in content
    if has_tool_marker or content_type == "assistant_text":
        for tool in ALL_FILE_TOOLS:
            # Match tool name as a whole word
            pattern = r"\b" + re.escape(tool) + r"\b"
            if re.search(pattern, content):
                info["tool"] = tool
                if tool in SEARCH_TOOLS:
                    info["action"] = "search"
                elif tool in READ_TOOLS:
                    info["action"] = "read"
                elif tool in EDIT_TOOLS:
                    info["action"] = "edit"
                break

    # Detect from content_type
    if content_type == "ai_code":
        info["action"] = "code"
    elif content_type == "stack_trace":
        info["is_error"] = True
        info["action"] = "error"

    # Detect test runs
    if any(
        kw in content.lower()
        for kw in [
            "bun test",
            "pytest",
            "npm test",
            "jest",
            "vitest",
            "test passed",
            "test failed",
            "tests pass",
        ]
    ):
        info["is_test"] = True
        info["action"] = "test"

    # Detect errors (avoid false positives from test summaries)
    content_lower = content.lower()
    if any(
        kw in content_lower
        for kw in [
            "error:",
            "exception:",
            "traceback",
            "exit code 1",
        ]
    ):
        info["is_error"] = True
    # "failed" only counts as error if not in test summary
    if "failed" in content_lower and "0 fail" not in content_lower and "pass" not in content_lower:
        info["is_error"] = True

    return info


def _classify_operation(
    steps: List[Dict[str, Any]],
) -> str:
    """Classify an operation based on its step patterns."""
    actions = [s["action"] for s in steps if s["action"]]
    has_search = "search" in actions
    has_read = "read" in actions
    has_edit = "edit" in actions
    has_test = any(s["is_test"] for s in steps)
    has_error = any(s["is_error"] for s in steps)
    has_code = "code" in actions
    has_user = any(s["is_user_message"] for s in steps)

    # Debug cycle: error → investigation → fix → test
    if has_error and (has_edit or has_code) and has_test:
        return OP_DEBUG
    if has_error and (has_edit or has_code):
        return OP_DEBUG

    # Edit cycle: read → edit → test
    if has_edit and has_test:
        return OP_EDIT_CYCLE
    if has_edit and has_read:
        return OP_EDIT_CYCLE

    # Feature cycle: user request → implementation
    if has_user and (has_edit or has_code):
        return OP_FEATURE_CYCLE

    # Research: search/read without edits
    if has_search and has_read and not has_edit:
        return OP_RESEARCH
    if has_search and not has_edit:
        return OP_RESEARCH

    # Review: reading multiple files
    read_count = sum(1 for s in steps if s["action"] == "read")
    if read_count >= 3 and not has_edit:
        return OP_REVIEW

    # Config: editing config-like files
    if has_edit:
        return OP_EDIT_CYCLE

    return OP_RESEARCH  # default


def _generate_summary(
    op_type: str,
    steps: List[Dict[str, Any]],
) -> str:
    """Generate a brief summary of an operation."""
    step_count = len(steps)
    actions = [s["action"] for s in steps if s["action"]]
    unique_actions = list(dict.fromkeys(actions))  # ordered

    action_str = " -> ".join(unique_actions[:5])
    if len(unique_actions) > 5:
        action_str += " -> ..."

    type_labels = {
        OP_EDIT_CYCLE: "Edit cycle",
        OP_RESEARCH: "Research",
        OP_FEATURE_CYCLE: "Feature cycle",
        OP_DEBUG: "Debug cycle",
        OP_CONFIG: "Config change",
        OP_REVIEW: "Code review",
    }
    label = type_labels.get(op_type, op_type)

    # Extract file context from steps
    files = []
    for s in steps:
        fp = s.get("file_path")
        if fp:
            # Extract just the filename
            import os

            basename = os.path.basename(fp)
            if basename and basename not in files:
                files.append(basename)
    file_ctx = ""
    if files:
        shown = files[:3]
        file_ctx = " on " + ", ".join(shown)
        if len(files) > 3:
            file_ctx += f" +{len(files) - 3}"

    # Extract topic from first user message or snippet
    topic = ""
    if not file_ctx:
        for s in steps:
            if s.get("snippet"):
                topic = f": {s['snippet'][:50]}"
                break

    parts = [label]
    if action_str:
        parts.append(f"({action_str})")
    if file_ctx:
        parts.append(file_ctx)
    elif topic:
        parts.append(topic)
    parts.append(f"[{step_count} steps]")
    return " ".join(parts)


def group_session_chunks(
    chunks: List[Dict[str, Any]],
    session_id: str,
    max_gap: int = MAX_GAP_SECONDS,
) -> List[Dict[str, Any]]:
    """Group a session's chunks into logical operations.

    Uses temporal proximity and tool-type patterns.

    Args:
        chunks: List of chunk dicts (must have id, content,
            content_type, and optionally timestamp in metadata)
        session_id: The session identifier
        max_gap: Max seconds between chunks in same operation

    Returns:
        List of operation dicts ready for store_operations()
    """
    if not chunks:
        return []

    # Extract tool info for each chunk
    steps = []
    for chunk in chunks:
        info = _extract_tool_info(chunk)
        # Try to get timestamp from chunk metadata
        meta = chunk.get("metadata", "{}")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}
        ts = meta.get("timestamp") or chunk.get("timestamp")
        info["timestamp"] = ts
        info["epoch"] = _parse_timestamp(ts)
        steps.append(info)

    # Group by temporal proximity
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []

    for step in steps:
        if not current_group:
            current_group.append(step)
            continue

        # Check time gap
        prev_epoch = current_group[-1].get("epoch")
        curr_epoch = step.get("epoch")

        if prev_epoch and curr_epoch:
            gap = curr_epoch - prev_epoch
            if gap > max_gap:
                # Time gap too large — start new group
                if len(current_group) >= 2:
                    groups.append(current_group)
                current_group = [step]
                continue

        # Check for natural boundaries
        # A new user message often starts a new operation
        if step["is_user_message"] and len(current_group) >= 3:
            groups.append(current_group)
            current_group = [step]
            continue

        current_group.append(step)

        # Cap group size
        if len(current_group) >= 50:
            groups.append(current_group)
            current_group = []

    # Don't forget the last group
    if len(current_group) >= 2:
        groups.append(current_group)

    # Convert groups to operation dicts
    operations = []
    for group in groups:
        op_type = _classify_operation(group)
        chunk_ids = [s["chunk_id"] for s in group if s["chunk_id"]]

        timestamps = [s["timestamp"] for s in group if s.get("timestamp")]
        started = timestamps[0] if timestamps else None
        ended = timestamps[-1] if timestamps else None

        # Determine outcome
        has_error = any(s["is_error"] for s in group)
        has_test_pass = any(s["is_test"] and not s["is_error"] for s in group)
        if has_test_pass:
            outcome = "success"
        elif has_error:
            outcome = "failure"
        else:
            outcome = "unknown"

        operations.append(
            {
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "operation_type": op_type,
                "chunk_ids": chunk_ids,
                "summary": _generate_summary(op_type, group),
                "outcome": outcome,
                "started_at": started,
                "ended_at": ended,
                "step_count": len(group),
            }
        )

    return operations


def run_operation_grouping(
    vector_store: Any,
    project: Optional[str] = None,
    force: bool = False,
    max_sessions: int = 0,
) -> Dict[str, int]:
    """Run operation grouping on indexed sessions.

    Args:
        vector_store: VectorStore instance
        project: Filter to specific project name
        force: Re-process sessions with existing operations
        max_sessions: Limit number of sessions (0 = all)

    Returns:
        Dict with counts: sessions_processed, operations_added
    """
    stats = {
        "sessions_processed": 0,
        "operations_added": 0,
    }

    # Get all sessions from session_context table
    cursor = vector_store.conn.cursor()
    query = "SELECT session_id, project FROM session_context"
    params: list = []
    if project:
        query += " WHERE project = ?"
        params.append(project)
    query += " ORDER BY started_at"

    sessions = list(cursor.execute(query, params))

    processed = 0
    for session_id, proj in sessions:
        if max_sessions and processed >= max_sessions:
            break

        # Skip if already has operations (unless force)
        if not force:
            existing = vector_store.get_session_operations(session_id)
            if existing:
                continue
        else:
            # Clear existing operations for re-processing
            vector_store.clear_session_operations(session_id)

        # Get chunks for this session
        source_file_pattern = f"%{session_id}%"
        chunk_rows = list(
            cursor.execute(
                """SELECT id, content, content_type, metadata
               FROM chunks
               WHERE source_file LIKE ?
               ORDER BY ROWID""",
                (source_file_pattern,),
            )
        )

        if not chunk_rows:
            continue

        chunks = []
        for row in chunk_rows:
            meta = row[3] or "{}"
            if isinstance(meta, str):
                try:
                    meta_dict = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta_dict = {}
            else:
                meta_dict = meta

            chunks.append(
                {
                    "id": row[0],
                    "content": row[1],
                    "content_type": row[2],
                    "metadata": meta_dict,
                }
            )

        # Group into operations
        operations = group_session_chunks(chunks, session_id)

        if operations:
            count = vector_store.store_operations(operations)
            stats["operations_added"] += count
            logger.info(
                "Session %s: %d operations from %d chunks",
                session_id[:8],
                count,
                len(chunks),
            )

        stats["sessions_processed"] += 1
        processed += 1

    return stats
