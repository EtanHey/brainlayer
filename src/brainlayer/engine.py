"""Think/Recall Engine — Intelligence layer for BrainLayer.

Turns BrainLayer from "search your conversations" into "AI that remembers everything."

Three capabilities:
- think(context) — given current task, retrieve relevant past decisions/patterns
- recall(file_path|topic) — proactive retrieval based on what you're working on
- sessions(project, days) — browse sessions by date/project
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Intent categories for grouping think results
DECISION_INTENTS = {"deciding", "designing"}
DEBUG_INTENTS = {"debugging"}
IMPLEMENT_INTENTS = {"implementing", "configuring"}
REVIEW_INTENTS = {"reviewing", "discussing"}


@dataclass
class ThinkResult:
    """Structured result from think()."""

    decisions: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[dict[str, Any]] = field(default_factory=list)
    bugs: list[dict[str, Any]] = field(default_factory=list)
    context: list[dict[str, Any]] = field(default_factory=list)
    query: str = ""
    total: int = 0

    def format(self) -> str:
        """Format as markdown for MCP response."""
        if self.total == 0:
            return "No relevant memories found."

        parts = [f"## Relevant Memories for: {self.query}\n"]

        if self.decisions:
            parts.append("### Decisions & Design")
            for item in self.decisions:
                parts.append(_format_memory_item(item))

        if self.patterns:
            parts.append("\n### Patterns & Implementations")
            for item in self.patterns:
                parts.append(_format_memory_item(item))

        if self.bugs:
            parts.append("\n### Related Bugs & Fixes")
            for item in self.bugs:
                parts.append(_format_memory_item(item))

        if self.context:
            parts.append("\n### Related Context")
            for item in self.context:
                parts.append(_format_memory_item(item))

        parts.append(f"\n*{self.total} memories retrieved*")
        return "\n".join(parts)


@dataclass
class RecallResult:
    """Structured result from recall()."""

    file_history: list[dict[str, Any]] = field(default_factory=list)
    related_chunks: list[dict[str, Any]] = field(default_factory=list)
    session_summaries: list[dict[str, Any]] = field(default_factory=list)
    target: str = ""

    def format(self) -> str:
        """Format as markdown for MCP response."""
        if not self.file_history and not self.related_chunks:
            return f"No recall data found for '{self.target}'."

        parts = [f"## Recall: {self.target}\n"]

        if self.file_history:
            parts.append("### File History")
            for item in self.file_history:
                ts = (item.get("timestamp") or "?")[:19]
                action = item.get("action", "?")
                session = (item.get("session_id") or "?")[:8]
                parts.append(f"- **{action}** at {ts} (session: {session})")

        if self.session_summaries:
            parts.append("\n### Sessions That Touched This")
            for s in self.session_summaries:
                sid = (s.get("session_id") or "?")[:8]
                branch = s.get("branch") or "?"
                plan = s.get("plan_name") or ""
                ts = (s.get("started_at") or "?")[:19]
                line = f"- {sid} | {branch}"
                if plan:
                    line += f" | plan: {plan}"
                line += f" | {ts}"
                parts.append(line)

        if self.related_chunks:
            parts.append("\n### Related Knowledge")
            for item in self.related_chunks:
                parts.append(_format_memory_item(item))

        return "\n".join(parts)


@dataclass
class SessionInfo:
    """A single session entry."""

    session_id: str = ""
    project: str = ""
    branch: str = ""
    started_at: str = ""
    ended_at: str = ""
    plan_name: str = ""
    plan_phase: str = ""
    files_changed: list[str] = field(default_factory=list)


def _format_memory_item(item: dict[str, Any]) -> str:
    """Format a single memory item as compact markdown."""
    summary = item.get("summary") or ""
    content = item.get("content", "")
    date = (item.get("created_at") or "")[:10]
    project = item.get("project", "")
    importance = item.get("importance")

    # Use summary if available, otherwise truncate content
    display = summary if summary else (content[:200] + "..." if len(content) > 200 else content)

    line = "- "
    if date:
        line += f"[{date}] "
    if project:
        line += f"({project}) "
    if importance is not None and importance >= 7:
        line += "**"
    line += display
    if importance is not None and importance >= 7:
        line += "**"
    return line


def categorize_by_intent(items: list[dict[str, Any]]) -> ThinkResult:
    """Categorize search results by their intent metadata."""
    result = ThinkResult()

    for item in items:
        intent = item.get("intent", "")

        if intent in DECISION_INTENTS:
            result.decisions.append(item)
        elif intent in DEBUG_INTENTS:
            result.bugs.append(item)
        elif intent in IMPLEMENT_INTENTS:
            result.patterns.append(item)
        else:
            result.context.append(item)

    result.total = len(items)
    return result


def think(
    context: str,
    store: VectorStore,
    embed_fn: Any,
    project: str | None = None,
    max_results: int = 10,
) -> ThinkResult:
    """Given current task context, retrieve relevant past knowledge.

    Args:
        context: Free-text description of current task/context
        store: VectorStore instance
        embed_fn: Function that takes text and returns embedding vector
        project: Optional project filter
        max_results: Maximum results to return

    Returns:
        ThinkResult with categorized memories
    """
    if not context or not context.strip():
        return ThinkResult(query=context or "")

    query = context.strip()

    # Generate embedding
    query_embedding = embed_fn(query)

    # Search with importance bias — prefer high-value memories
    results = store.hybrid_search(
        query_embedding=query_embedding,
        query_text=query,
        n_results=max_results,
        project_filter=project,
        importance_min=3.0,  # Skip low-importance noise
    )

    if not results["documents"][0]:
        return ThinkResult(query=query)

    # Build items with metadata
    items = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        items.append(
            {
                "content": doc,
                "summary": meta.get("summary"),
                "intent": meta.get("intent", ""),
                "importance": meta.get("importance"),
                "project": meta.get("project", ""),
                "created_at": meta.get("created_at", ""),
                "content_type": meta.get("content_type", ""),
                "tags": meta.get("tags", []),
            }
        )

    result = categorize_by_intent(items)
    result.query = query
    return result


def recall(
    store: VectorStore,
    embed_fn: Any | None = None,
    file_path: str | None = None,
    topic: str | None = None,
    project: str | None = None,
    max_results: int = 10,
) -> RecallResult:
    """Proactive smart retrieval based on file or topic.

    Args:
        store: VectorStore instance
        embed_fn: Function that takes text and returns embedding vector (needed for topic recall)
        file_path: File path to recall context for
        topic: Topic to recall context for
        project: Optional project filter
        max_results: Maximum results to return

    Returns:
        RecallResult with file history, sessions, and related knowledge
    """
    target = file_path or topic or ""
    result = RecallResult(target=target)

    if file_path:
        # Get file interaction timeline
        timeline = store.get_file_timeline(file_path, project=project, limit=max_results * 2)
        result.file_history = timeline

        # Get sessions that touched this file
        session_ids = list({t.get("session_id") for t in timeline if t.get("session_id")})
        for sid in session_ids[:5]:
            ctx = store.get_session_context(sid)
            if ctx:
                result.session_summaries.append(ctx)

        # Search for related knowledge about this file
        if embed_fn:
            # Use filename as search query
            fname = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
            query_embedding = embed_fn(f"working on {fname}")
            search_results = store.hybrid_search(
                query_embedding=query_embedding,
                query_text=fname,
                n_results=max_results,
                project_filter=project,
            )
            for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
                result.related_chunks.append(
                    {
                        "content": doc,
                        "summary": meta.get("summary"),
                        "intent": meta.get("intent", ""),
                        "importance": meta.get("importance"),
                        "project": meta.get("project", ""),
                        "created_at": meta.get("created_at", ""),
                    }
                )

    elif topic and embed_fn:
        # Topic-based recall — search for related discussions
        query_embedding = embed_fn(topic)
        search_results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text=topic,
            n_results=max_results,
            project_filter=project,
        )
        for doc, meta in zip(search_results["documents"][0], search_results["metadatas"][0]):
            result.related_chunks.append(
                {
                    "content": doc,
                    "summary": meta.get("summary"),
                    "intent": meta.get("intent", ""),
                    "importance": meta.get("importance"),
                    "project": meta.get("project", ""),
                    "created_at": meta.get("created_at", ""),
                }
            )

    return result


def sessions(
    store: VectorStore,
    project: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> list[SessionInfo]:
    """List recent sessions with metadata.

    Args:
        store: VectorStore instance
        project: Optional project filter
        days: How many days back to look
        limit: Maximum sessions to return

    Returns:
        List of SessionInfo objects
    """
    cursor = store.conn.cursor()

    date_from = (datetime.now() - timedelta(days=days)).isoformat()

    where_clauses = ["started_at >= ?"]
    params: list = [date_from]

    if project:
        where_clauses.append("project = ?")
        params.append(project)

    params.append(limit)

    query = f"""
        SELECT session_id, project, branch, started_at, ended_at,
               plan_name, plan_phase, files_changed
        FROM session_context
        WHERE {" AND ".join(where_clauses)}
        ORDER BY started_at DESC
        LIMIT ?
    """

    rows = list(cursor.execute(query, params))

    results = []
    for row in rows:
        files = []
        if row[7]:
            try:
                files = json.loads(row[7])
            except (json.JSONDecodeError, TypeError):
                pass

        results.append(
            SessionInfo(
                session_id=row[0] or "",
                project=row[1] or "",
                branch=row[2] or "",
                started_at=row[3] or "",
                ended_at=row[4] or "",
                plan_name=row[5] or "",
                plan_phase=row[6] or "",
                files_changed=files if isinstance(files, list) else [],
            )
        )

    return results


def format_sessions(session_list: list[SessionInfo], days: int = 7) -> str:
    """Format sessions list as markdown."""
    if not session_list:
        return f"No sessions found in the last {days} days."

    parts = [f"## Recent Sessions (last {days} days)\n"]

    for s in session_list:
        ts = s.started_at[:19] if s.started_at else "?"
        line = f"- **{s.session_id[:8]}** | {s.project or '?'} | {s.branch or '?'}"
        if s.plan_name:
            line += f" | plan: {s.plan_name}"
            if s.plan_phase:
                line += f"/{s.plan_phase}"
        line += f" | {ts}"
        if s.files_changed:
            line += f" | {len(s.files_changed)} files"
        parts.append(line)

    parts.append(f"\n*{len(session_list)} sessions*")
    return "\n".join(parts)
