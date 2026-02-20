"""Session-level enrichment pipeline — analyze full conversations, not just chunks.

Processes sessions through local LLM (Ollama/MLX) to extract:
- Session summary + primary intent + outcome
- Decisions made (with rationale)
- Corrections (what the user corrected)
- Learnings (new knowledge gained)
- Mistakes (what went wrong)
- Patterns (recurring behaviors)
- Tool usage statistics
- Quality scores

Usage:
    from brainlayer.pipeline.session_enrichment import enrich_session
    result = enrich_session(store, session_id, call_llm_fn)

CLI:
    brainlayer enrich-sessions
    brainlayer enrich-sessions --project golems --since 2026-01-01
"""

import json
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..vector_store import VectorStore

# Valid values for structured fields
VALID_INTENTS = [
    "debugging", "designing", "configuring", "discussing",
    "deciding", "implementing", "reviewing", "refactoring",
    "deploying", "testing",
]
VALID_OUTCOMES = ["success", "partial_success", "failure", "abandoned", "ongoing"]

# Maximum conversation length to send to LLM (in characters)
MAX_CONVERSATION_CHARS = 12_000


def reconstruct_session(store: VectorStore, session_id: str) -> Dict[str, Any]:
    """Reassemble ordered chunks from a session into a coherent conversation.

    Chunks are identified by source_file matching the session_id pattern.
    Returns a dict with conversation text, message counts, timing, and metadata.
    """
    cursor = store.conn.cursor()

    # Find chunks belonging to this session, ordered by creation time
    # Session ID maps to source_file (the JSONL filename stem) or conversation_id
    rows = list(cursor.execute(
        """SELECT id, content, content_type, source_file, created_at,
                  char_count, source, conversation_id
           FROM chunks
           WHERE (source_file LIKE ? OR conversation_id = ?)
           ORDER BY created_at, rowid""",
        (f"%{session_id}%", session_id),
    ))

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
            conversation_parts.append(
                f"FILE_READ ({len(lines)} lines): {content[:500]}"
            )
        else:
            conversation_parts.append(f"[{ct}]: {content[:500]}")

        total_chars += len(conversation_parts[-1])

        # Stop if we've exceeded the character limit
        if total_chars > MAX_CONVERSATION_CHARS:
            conversation_parts.append(
                f"\n[... {len(chunks) - len(conversation_parts)} more chunks truncated]"
            )
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


# Session analysis prompt — single-pass for local LLM efficiency
SESSION_ANALYSIS_PROMPT = """You are a session analysis assistant. Analyze this Claude Code conversation and return ONLY a JSON object.

CONVERSATION (session from project: {project}):
---
{conversation}
---

Return this exact JSON structure:
{{
  "session_summary": "<2-3 sentence summary of what happened in this session>",
  "primary_intent": "<one of: debugging, designing, configuring, discussing, deciding, implementing, reviewing, refactoring, deploying, testing>",
  "outcome": "<one of: success, partial_success, failure, abandoned, ongoing>",
  "complexity_score": <1-10 integer>,
  "session_quality_score": <1-10 integer>,
  "decisions_made": [
    {{"decision": "<what was decided>", "rationale": "<why>"}}
  ],
  "corrections": [
    {{"what_was_wrong": "<what the AI did wrong>", "what_user_wanted": "<correct behavior>"}}
  ],
  "learnings": [
    "<new knowledge or insight gained during this session>"
  ],
  "mistakes": [
    "<what went wrong and how it was resolved>"
  ],
  "patterns": [
    "<recurring behaviors or approaches observed>"
  ],
  "topic_tags": ["<tag1>", "<tag2>"],
  "tool_usage_stats": [
    {{"tool": "<tool name>", "count": <number>}}
  ],
  "what_worked": "<what went well in this session>",
  "what_failed": "<what didn't work or caused problems>"
}}

SCORING RULES:
- complexity_score: 1-3 trivial (quick fix), 4-6 moderate (feature work), 7-9 complex (architecture), 10 critical
- session_quality_score: 1-3 poor (many errors, user frustrated), 4-6 average, 7-9 good (smooth), 10 exceptional

EXTRACTION RULES:
- decisions: Only extract REAL decisions — "we chose X over Y because Z"
- corrections: Only when the user explicitly corrected the AI's approach
- learnings: Concrete knowledge, not vague observations
- mistakes: What actually failed, not hypothetical risks
- topic_tags: lowercase, hyphenated (e.g., "bug-fix", "api-design", "typescript")
- tool_usage_stats: List tools used (Read, Write, Edit, Bash, etc.) with approximate counts
- Empty arrays [] are fine when nothing matches a category

Return ONLY the JSON object, no other text."""


def build_session_prompt(conversation: str, project: str) -> str:
    """Build the session analysis prompt."""
    # Escape braces in conversation to avoid str.format() crash
    safe_conversation = conversation.replace("{", "{{").replace("}", "}}")
    return SESSION_ANALYSIS_PROMPT.format(
        project=project or "unknown",
        conversation=safe_conversation,
    )


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
                            match = json.loads(text[start:end + 1])
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
            result["topic_tags"] = [
                str(t).lower().strip() for t in tags
                if isinstance(t, str)
            ][:15]

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


def list_sessions_for_enrichment(
    store: VectorStore,
    project: Optional[str] = None,
    since: Optional[str] = None,
) -> List[Tuple[str, str, int]]:
    """List session IDs available for enrichment.

    Returns list of (session_id, project, chunk_count) tuples.
    Sessions come from:
    1. session_context table (sessions with git overlay data)
    2. Distinct source_file values in chunks table (all sessions)
    """
    cursor = store.conn.cursor()
    already_enriched = set(store.list_enriched_sessions())

    sessions = []

    # Method 1: session_context table (has richer metadata)
    query = "SELECT session_id, project FROM session_context"
    params: list = []
    if project:
        query += " WHERE project = ?"
        params.append(project)
    for row in cursor.execute(query, params):
        sid, proj = row[0], row[1]
        if sid not in already_enriched:
            # Apply 'since' filter if provided
            if since:
                first_time = list(cursor.execute(
                    "SELECT MIN(created_at) FROM chunks WHERE source_file LIKE ?",
                    (f"%{sid}%",),
                ))[0][0]
                if first_time and first_time < since:
                    continue

            # Count chunks for this session
            count = list(cursor.execute(
                "SELECT COUNT(*) FROM chunks WHERE source_file LIKE ?",
                (f"%{sid}%",),
            ))[0][0]
            if count > 0:
                sessions.append((sid, proj or "", count))
                already_enriched.add(sid)

    # Method 2: Distinct source_files from chunks (catches sessions without git overlay)
    source_query = """
        SELECT DISTINCT source_file, project, COUNT(*) as cnt
        FROM chunks
        WHERE source IS NULL OR source = 'claude_code'
        GROUP BY source_file
        HAVING cnt >= 3
    """
    for row in cursor.execute(source_query):
        source_file = row[0] or ""
        proj = row[1] or ""

        if project and proj != project:
            continue

        # Extract session ID from source_file path
        # Typical format: /path/to/.claude/projects/-Users-janedev-Gits-project/abc123.jsonl
        import os
        sid = os.path.splitext(os.path.basename(source_file))[0] if source_file else ""
        if not sid or sid in already_enriched:
            continue

        # Apply 'since' filter if provided
        if since:
            first_time = list(cursor.execute(
                "SELECT MIN(created_at) FROM chunks WHERE source_file = ?",
                (source_file,),
            ))[0][0]
            if first_time and first_time < since:
                continue

        sessions.append((sid, proj, row[2]))
        already_enriched.add(sid)

    return sessions


def enrich_session(
    store: VectorStore,
    session_id: str,
    call_llm_fn: Callable[[str], Optional[str]],
    project: Optional[str] = None,
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    model_name: str = "",
) -> Optional[Dict[str, Any]]:
    """Enrich a single session with LLM analysis.

    Args:
        store: VectorStore instance.
        session_id: Session to enrich.
        call_llm_fn: Function that takes a prompt string and returns LLM response.
        project: Optional project name override.
        embed_fn: Optional embedding function for session summary.
        model_name: Name of the model used for enrichment tracking.

    Returns:
        Enrichment dict on success, None on failure.
    """
    # Step 1: Reconstruct conversation
    session_data = reconstruct_session(store, session_id)
    if not session_data["chunks"]:
        return None

    conversation = session_data["conversation"]
    if not conversation or len(conversation) < 50:
        return None

    # Determine project from chunks if not provided
    if not project:
        # Try to get project from session_context
        ctx = store.get_session_context(session_id)
        if ctx:
            project = ctx.get("project", "")

    # Step 2: Build prompt and call LLM
    prompt = build_session_prompt(conversation, project or "unknown")
    response = call_llm_fn(prompt)

    # Step 3: Parse response
    enrichment = parse_session_enrichment(response)
    if not enrichment:
        return None

    # Step 4: Add session metadata from reconstruction
    enrichment["session_id"] = session_id
    enrichment["file_path"] = session_data["chunks"][0].get("source_file") if session_data["chunks"] else None
    enrichment["message_count"] = session_data["message_count"]
    enrichment["user_message_count"] = session_data["user_message_count"]
    enrichment["assistant_message_count"] = session_data["assistant_message_count"]
    enrichment["tool_call_count"] = session_data["tool_call_count"]
    enrichment["session_start_time"] = session_data["session_start_time"]
    enrichment["session_end_time"] = session_data["session_end_time"]
    enrichment["duration_seconds"] = session_data["duration_seconds"]
    enrichment["enrichment_model"] = model_name
    enrichment["enrichment_version"] = "1.0"

    # Step 5: Generate summary embedding if embed_fn provided
    if embed_fn and enrichment.get("session_summary"):
        try:
            from ..vector_store import serialize_f32
            embedding = embed_fn(enrichment["session_summary"])
            enrichment["summary_embedding"] = serialize_f32(embedding)
        except Exception:
            pass  # Don't fail enrichment over embedding issues

    # Step 6: Store in database
    store.upsert_session_enrichment(enrichment)

    # Return the stored version (with JSON fields properly deserialized)
    return store.get_session_enrichment(session_id)
