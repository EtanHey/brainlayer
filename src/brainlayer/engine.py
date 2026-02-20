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


@dataclass
class CurrentContext:
    """Current working context — what the user is working on right now."""

    recent_sessions: list[SessionInfo] = field(default_factory=list)
    active_projects: list[str] = field(default_factory=list)
    recent_files: list[str] = field(default_factory=list)
    active_branches: list[str] = field(default_factory=list)
    active_plan: str = ""

    def format(self) -> str:
        """Format as concise markdown — designed for voice/quick context."""
        if not self.recent_sessions:
            return "No recent session context available."

        parts = ["## Current Context\n"]

        if self.active_projects:
            parts.append(f"**Projects:** {', '.join(self.active_projects)}")
        if self.active_branches:
            parts.append(f"**Branches:** {', '.join(self.active_branches)}")
        if self.active_plan:
            parts.append(f"**Plan:** {self.active_plan}")

        if self.recent_files:
            parts.append(f"\n**Recent files ({len(self.recent_files)}):**")
            for f in self.recent_files[:10]:
                # Show just the filename, not full path
                name = f.rsplit("/", 1)[-1] if "/" in f else f
                parts.append(f"- {name}")

        if self.recent_sessions:
            latest = self.recent_sessions[0]
            parts.append(f"\n**Latest session:** {latest.session_id[:8]}")
            if latest.started_at:
                parts.append(f"**Started:** {latest.started_at[:19]}")
            if latest.project:
                parts.append(f"**Project:** {latest.project}")
            if latest.branch:
                parts.append(f"**Branch:** {latest.branch}")

        return "\n".join(parts)


def current_context(
    store: VectorStore,
    hours: int = 24,
) -> CurrentContext:
    """Get current working context — what the user is doing right now.

    Designed for voice assistants and quick context injection.
    Lightweight — no embedding model needed.

    Uses two data sources:
    1. session_context table (git overlay data — may be sparse)
    2. chunks table (always populated from indexing)

    Args:
        store: VectorStore instance
        hours: How many hours back to look (default: 24)

    Returns:
        CurrentContext with recent sessions, files, projects, branches
    """
    result = CurrentContext()
    cursor = store.conn.cursor()
    date_from = (datetime.now() - timedelta(hours=hours)).isoformat()

    # 1. Try session_context first (richest data)
    # Convert hours to days properly — ceil division, minimum 1
    days = max(1, -(-hours // 24))  # ceiling division trick
    recent = sessions(store, days=days, limit=10)
    result.recent_sessions = recent

    # 2. Also query chunks table directly for recent projects
    # This catches sessions that haven't been through git_overlay yet
    chunk_projects = list(
        cursor.execute(
            """
        SELECT DISTINCT project
        FROM chunks
        WHERE created_at >= ? AND project IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 10
    """,
            (date_from,),
        )
    )

    # Extract active projects and branches from session_context
    projects = []
    branches = []
    plans = []
    for s in recent:
        if s.project and s.project not in projects:
            projects.append(s.project)
        if s.branch and s.branch not in branches:
            branches.append(s.branch)
        if s.plan_name and s.plan_name not in plans:
            plans.append(s.plan_name)

    # Merge in projects from chunks table (may have projects not in session_context)
    for row in chunk_projects:
        if row[0] and row[0] not in projects:
            projects.append(row[0])

    result.active_projects = projects[:5]
    result.active_branches = branches[:5]
    if plans:
        result.active_plan = plans[0]  # Most recent plan

    # 3. Get recent files from file_interactions
    rows = list(
        cursor.execute(
            """
        SELECT DISTINCT file_path
        FROM file_interactions
        WHERE timestamp >= ?
        ORDER BY timestamp DESC
        LIMIT 20
    """,
            (date_from,),
        )
    )
    result.recent_files = [r[0] for r in rows if r[0]]

    # 4. If no files from interactions, try chunks metadata for file references
    if not result.recent_files:
        file_rows = list(
            cursor.execute(
                """
            SELECT DISTINCT source_file
            FROM chunks
            WHERE created_at >= ? AND source_file IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 20
        """,
                (date_from,),
            )
        )
        result.recent_files = [r[0] for r in file_rows if r[0]]

    return result


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
