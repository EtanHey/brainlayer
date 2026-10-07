"""Local reconstruction of historical conversations; no model producers."""

from datetime import datetime
from typing import Any, Dict

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
