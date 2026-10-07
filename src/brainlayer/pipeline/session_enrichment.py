"""Historical session reconstruction and metadata parsing.

Model production and prompt construction are retired. Stored session metadata,
source-class selection and historical conversation reads remain available.
"""

from typing import List, Optional, Tuple

from ..vector_store import VectorStore
from .enrichment_tiers import T1_T2_SOURCES
from .session_history import parse_session_enrichment as parse_session_enrichment
from .session_history import reconstruct_session as reconstruct_session

# Valid values for structured fields

# Maximum conversation length to send to LLM (in characters)
MAX_CONVERSATION_CHARS = 12_000


# Session analysis prompt — single-pass for local LLM efficiency


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
