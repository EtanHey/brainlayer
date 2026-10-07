"""Local reconstruction of historical conversations; no model producers."""

import json
from datetime import datetime
from typing import Any, Dict, Optional

from ..vector_store import VectorStore

MAX_CONVERSATION_CHARS = 12_000


def reconstruct_session(store: VectorStore, session_id: str) -> Dict[str, Any]:
    """Reassemble ordered chunks from a session into a coherent conversation.

    Chunks are identified by source_file matching the session_id pattern.
    Returns a dict with conversation text, message counts, timing, and metadata.
    """
    cursor = store.conn.cursor()

    # Find chunks belonging to this session, ordered by creation time
    # Session ID maps to source_file (the JSONL filename stem) or conversation_id
    rows = list(
        cursor.execute(
            """SELECT id, content, content_type, source_file, created_at,
                  char_count, source, conversation_id
           FROM chunks
           WHERE (source_file LIKE ? OR conversation_id = ?)
           ORDER BY created_at, rowid""",
            (f"%{session_id}%", session_id),
        )
    )

    if not rows:
        return {"chunks": [], "conversation": "", "message_count": 0}

    chunks = []
    user_count = 0
    assistant_count = 0
    tool_count = 0
    first_time = None
    last_time = None

    for row in rows:
        chunk = {
            "id": row[0],
            "content": row[1],
            "content_type": row[2],
            "source_file": row[3],
            "created_at": row[4],
            "char_count": row[5],
            "source": row[6],
            "conversation_id": row[7],
        }
        chunks.append(chunk)

        # Count message types
        ct = chunk["content_type"] or ""
        if ct == "user_message":
            user_count += 1
        elif ct in ("assistant_text", "ai_code"):
            assistant_count += 1
        elif ct in ("tool_result", "tool_use"):
            tool_count += 1

        # Track timing
        if chunk["created_at"]:
            if first_time is None:
                first_time = chunk["created_at"]
            last_time = chunk["created_at"]

    # Build conversation text for LLM analysis
    conversation_parts = []
    total_chars = 0

    for chunk in chunks:
        ct = chunk["content_type"] or "unknown"
        content = chunk["content"] or ""

        # Skip noise types
        if ct in ("noise", "dir_listing", "build_log", "queue-operation"):
            continue

        # Truncate very long chunks (file reads, large code blocks)
        if len(content) > 2000:
            content = content[:2000] + "\n[... truncated]"

        # Format based on type
        if ct == "user_message":
            conversation_parts.append(f"USER: {content}")
        elif ct == "assistant_text":
            conversation_parts.append(f"ASSISTANT: {content}")
        elif ct == "ai_code":
            conversation_parts.append(f"ASSISTANT [code]: {content}")
        elif ct == "stack_trace":
            conversation_parts.append(f"ERROR: {content}")
        elif ct == "git_diff":
            conversation_parts.append(f"DIFF: {content}")
        elif ct == "file_read":
            # Summarize file reads (often very long)
            lines = content.split("\n")
            conversation_parts.append(f"FILE_READ ({len(lines)} lines): {content[:500]}")
        else:
            conversation_parts.append(f"[{ct}]: {content[:500]}")

        total_chars += len(conversation_parts[-1])

        # Stop if we've exceeded the character limit
        if total_chars > MAX_CONVERSATION_CHARS:
            conversation_parts.append(f"\n[... {len(chunks) - len(conversation_parts)} more chunks truncated]")
            break

    conversation = "\n\n".join(conversation_parts)

    # Calculate duration
    duration_seconds = None
    if first_time and last_time:
        try:
            t1 = datetime.fromisoformat(first_time.replace("Z", "+00:00"))
            t2 = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
            duration_seconds = int((t2 - t1).total_seconds())
        except (ValueError, TypeError):
            pass

    return {
        "chunks": chunks,
        "conversation": conversation,
        "message_count": len(chunks),
        "user_message_count": user_count,
        "assistant_message_count": assistant_count,
        "tool_call_count": tool_count,
        "session_start_time": first_time,
        "session_end_time": last_time,
        "duration_seconds": duration_seconds,
    }


VALID_INTENTS = [
    "debugging",
    "designing",
    "configuring",
    "discussing",
    "deciding",
    "implementing",
    "reviewing",
    "refactoring",
    "deploying",
    "testing",
]


VALID_OUTCOMES = ["success", "partial_success", "failure", "abandoned", "ongoing"]


def parse_session_enrichment(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM's JSON response into session enrichment data."""
    if not text:
        return None
    try:
        # Find JSON in response (handle LLM wrapping in markdown etc.)
        match = None
        for start in range(len(text)):
            if text[start] == "{":
                for end in range(len(text) - 1, start, -1):
                    if text[end] == "}":
                        try:
                            match = json.loads(text[start : end + 1])
                            break
                        except json.JSONDecodeError:
                            continue
                if match:
                    break

        if not match:
            return None

        result: Dict[str, Any] = {}

        # Required: session_summary
        summary = match.get("session_summary", "")
        if isinstance(summary, str) and len(summary) > 10:
            result["session_summary"] = summary[:1000]
        else:
            return None  # Summary is required

        # Intent
        intent = match.get("primary_intent", "")
        if isinstance(intent, str) and intent.lower().strip() in VALID_INTENTS:
            result["primary_intent"] = intent.lower().strip()

        # Outcome
        outcome = match.get("outcome", "")
        if isinstance(outcome, str) and outcome.lower().strip() in VALID_OUTCOMES:
            result["outcome"] = outcome.lower().strip()

        # Scores
        for score_field in ("complexity_score", "session_quality_score"):
            val = match.get(score_field)
            if isinstance(val, (int, float)):
                result[score_field] = max(1, min(10, int(val)))

        # JSON array fields
        for field in ("decisions_made", "corrections", "learnings", "mistakes", "patterns"):
            val = match.get(field, [])
            if isinstance(val, list):
                result[field] = val[:20]  # Cap at 20 items

        # Topic tags
        tags = match.get("topic_tags", [])
        if isinstance(tags, list):
            result["topic_tags"] = [str(t).lower().strip() for t in tags if isinstance(t, str)][:15]

        # Tool usage
        tool_stats = match.get("tool_usage_stats", [])
        if isinstance(tool_stats, list):
            result["tool_usage_stats"] = tool_stats[:20]

        # Narratives
        for field in ("what_worked", "what_failed"):
            val = match.get(field)
            if isinstance(val, str) and val.strip():
                result[field] = val.strip()[:500]

        return result

    except Exception:
        return None
