"""Historical session reconstruction and metadata parsing.

Model production and prompt construction are retired. Stored session metadata,
source-class selection and historical conversation reads remain available.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from ..vector_store import VectorStore
from .enrichment_tiers import T1_T2_SOURCES
from .session_history import reconstruct_session as reconstruct_session

# Valid values for structured fields
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

# Maximum conversation length to send to LLM (in characters)
MAX_CONVERSATION_CHARS = 12_000


# Session analysis prompt — single-pass for local LLM efficiency


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
                first_time = list(
                    cursor.execute(
                        "SELECT MIN(created_at) FROM chunks WHERE source_file LIKE ?",
                        (f"%{sid}%",),
                    )
                )[0][0]
                if first_time and first_time < since:
                    continue

            # Count chunks for this session
            count = list(
                cursor.execute(
                    "SELECT COUNT(*) FROM chunks WHERE source_file LIKE ?",
                    (f"%{sid}%",),
                )
            )[0][0]
            if count > 0:
                sessions.append((sid, proj or "", count))
                already_enriched.add(sid)

    # Method 2: Distinct source_files from chunks (catches sessions without git overlay)
    source_placeholders = ",".join("?" for _ in T1_T2_SOURCES)
    source_query = f"""
        SELECT DISTINCT source_file, project, COUNT(*) as cnt
        FROM chunks
        WHERE source IS NULL OR source IN ({source_placeholders})
        GROUP BY source_file
        HAVING cnt >= 3
    """
    for row in cursor.execute(source_query, tuple(T1_T2_SOURCES)):
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
            first_time = list(
                cursor.execute(
                    "SELECT MIN(created_at) FROM chunks WHERE source_file = ?",
                    (source_file,),
                )
            )[0][0]
            if first_time and first_time < since:
                continue

        sessions.append((sid, proj, row[2]))
        already_enriched.add(sid)

    return sessions
