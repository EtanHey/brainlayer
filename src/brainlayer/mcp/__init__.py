"""BrainLayer MCP Server - Model Context Protocol interface for Claude Code."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ..embeddings import get_embedding_model
from ..paths import DEFAULT_DB_PATH
from ..vector_store import VectorStore

# Create MCP server
server = Server("brainlayer")

# Lazy-loaded globals
_vector_store = None
_embedding_model = None


def normalize_project_name(project: str | None) -> str | None:
    """Normalize project names for consistent filtering.

    Handles:
    - Claude Code encoded paths: "-Users-etanheyman-Gits-golems" → "golems"
    - Worktree paths: "golems-nightshift-1770775282043" → "golems"
    - Path-like names with multiple segments
    - Already-clean names pass through unchanged
    """
    if not project:
        return None

    name = project.strip()
    if not name or name == "-":
        return None

    # Decode Claude Code path encoding
    # "-Users-etanheyman-Gits-golems" → "golems"
    # "-Users-etanheyman-Desktop-Gits-rudy-monorepo" → "rudy-monorepo"
    if name.startswith("-"):
        import os
        # Find the "Gits" segment by splitting on dashes
        segments = name[1:].split("-")  # Remove leading dash, split
        gits_idx = None
        for i, s in enumerate(segments):
            if s == "Gits":
                gits_idx = i
                # Use last occurrence in case of nested "Desktop-Gits"
        if gits_idx is not None and gits_idx + 1 < len(segments):
            # Remaining segments after "Gits" form the project path
            remaining = segments[gits_idx + 1:]
            # Skip secondary "Gits" (e.g., Desktop-Gits)
            while remaining and remaining[0] == "Gits":
                remaining = remaining[1:]
            if not remaining:
                return None
            # Try progressively joining segments with dashes to find a real directory
            gits_dir = "/" + "/".join(segments[:gits_idx]) + "/Gits"
            for length in range(len(remaining), 0, -1):
                candidate_name = "-".join(remaining[:length])
                candidate_path = os.path.join(gits_dir, candidate_name)
                if os.path.isdir(candidate_path):
                    return candidate_name
            # Fallback: return first segment (best guess)
            return remaining[0]
        # No "Gits" found — not a standard project path
        return None

    # Strip worktree suffixes (nightshift-{epoch}, haiku-*, worktree-*)
    import re
    name = re.sub(r'-(?:nightshift|haiku|worktree)-\d+$', '', name)

    return name


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(DEFAULT_DB_PATH)
    return _vector_store


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="brainlayer_search",
            description="""Search through past Claude Code conversations and learnings.

Use this to find:
- How you previously implemented something
- Past solutions to similar problems
- Code patterns and approaches used before
- Error solutions from previous debugging sessions

The knowledge base contains indexed conversations organized by:
- Project (which codebase the conversation was about)
- Content type (ai_code, stack_trace, user_message, etc.)
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'how did I implement authentication' or 'React useEffect cleanup')"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["ai_code", "stack_trace", "user_message", "assistant_text", "file_read", "git_diff"],
                        "description": "Optional: filter by content type"
                    },
                    "num_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return (default: 5)"
                    },
                    "source": {
                        "type": "string",
                        "enum": ["claude_code", "whatsapp", "youtube", "all"],
                        "description": (
                            "Filter by data source (default: claude_code)."
                            " Use 'all' to search everything."
                        )
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag (e.g. 'bug-fix', 'authentication', 'typescript')"
                    },
                    "intent": {
                        "type": "string",
                        "enum": ["debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"],
                        "description": "Filter by intent classification"
                    },
                    "importance_min": {
                        "type": "number",
                        "description": "Minimum importance score (1-10)"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Filter results from this date (ISO 8601, e.g. '2026-02-01')"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Filter results up to this date (ISO 8601, e.g. '2026-02-19')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="brainlayer_stats",
            description="Get statistics about the knowledge base (total chunks, projects, content types).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="brainlayer_list_projects",
            description="List all projects in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="brainlayer_context",
            description="""Get surrounding conversation context for a search result.

Given a chunk ID from a search result, returns the chunks before and after it
from the same conversation. Useful for understanding isolated search results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID from a search result"
                    },
                    "before": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of chunks before the target to include"
                    },
                    "after": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of chunks after the target to include"
                    }
                },
                "required": ["chunk_id"]
            }
        ),
        Tool(
            name="brainlayer_file_timeline",
            description="""Get the interaction timeline for a specific file across sessions.

Shows all Claude Code sessions that read, edited, or wrote to a file,
ordered chronologically. Useful for understanding a file's history.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "File path or partial path to search"
                            " for (e.g., 'telegram-bot.ts')"
                        )
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name"
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of interactions to return"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="brainlayer_operations",
            description=(
                "Get logical operation groups for a session."
                " Operations are patterns like"
                " read→edit→test cycles, research chains,"
                " or debug sequences."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session ID to query"
                    },
                },
                "required": ["session_id"]
            }
        ),
        Tool(
            name="brainlayer_regression",
            description=(
                "Analyze a file for regressions."
                " Shows the last successful operation"
                " and all changes after it."
                " Useful for debugging: 'what changed"
                " since this file last worked?'"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": (
                            "File path or partial path"
                            " to analyze"
                        )
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Optional: filter by project"
                        )
                    },
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="brainlayer_plan_links",
            description=(
                "Query plan-linked sessions."
                " Shows which plan/phase a session belongs"
                " to, or lists all sessions for a plan."
                " Useful for: 'which plan was I working"
                " on in this session?' or 'show all"
                " sessions for my-feature-plan'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_name": {
                        "type": "string",
                        "description": (
                            "Plan name to query"
                            " (e.g. 'local-llm-integration')"
                        )
                    },
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Session ID to look up"
                            " plan info for"
                        )
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Optional: filter by project"
                        )
                    },
                },
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "brainlayer_search":
        return await _search(
            query=arguments["query"],
            project=arguments.get("project"),
            content_type=arguments.get("content_type"),
            num_results=arguments.get("num_results", 5),
            source=arguments.get("source"),
            tag=arguments.get("tag"),
            intent=arguments.get("intent"),
            importance_min=arguments.get("importance_min"),
            date_from=arguments.get("date_from"),
            date_to=arguments.get("date_to"),
        )

    elif name == "brainlayer_stats":
        return await _stats()

    elif name == "brainlayer_list_projects":
        return await _list_projects()

    elif name == "brainlayer_context":
        return await _context(
            chunk_id=arguments["chunk_id"],
            before=min(arguments.get("before", 3), 50),
            after=min(arguments.get("after", 3), 50)
        )

    elif name == "brainlayer_file_timeline":
        return await _file_timeline(
            file_path=arguments["file_path"],
            project=arguments.get("project"),
            limit=arguments.get("limit", 50)
        )

    elif name == "brainlayer_operations":
        return await _operations(
            session_id=arguments["session_id"],
        )

    elif name == "brainlayer_regression":
        return await _regression(
            file_path=arguments["file_path"],
            project=arguments.get("project"),
        )

    elif name == "brainlayer_plan_links":
        return await _plan_links(
            plan_name=arguments.get("plan_name"),
            session_id=arguments.get("session_id"),
            project=arguments.get("project"),
        )

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search(
    query: str,
    project: str | None = None,
    content_type: str | None = None,
    num_results: int = 5,
    source: str | None = None,
    tag: str | None = None,
    intent: str | None = None,
    importance_min: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[TextContent]:
    """Execute a hybrid search query (semantic + keyword via RRF)."""
    try:
        if num_results < 1:
            num_results = 5
        elif num_results > 100:
            num_results = 100

        store = _get_vector_store()

        if store.count() == 0:
            return [TextContent(
                type="text",
                text="Knowledge base is empty. Run 'brainlayer index' to populate it."
            )]

        # Normalize project name for consistent filtering
        normalized_project = normalize_project_name(project)

        # Generate embedding (run in thread to not block)
        loop = asyncio.get_running_loop()
        model = _get_embedding_model()
        query_embedding = await loop.run_in_executor(None, model.embed_query, query)

        # Default to claude_code unless explicitly set to 'all'
        if source == "all":
            source_filter = None
        elif source:
            source_filter = source
        else:
            source_filter = "claude_code"

        # Use hybrid search (semantic + FTS5 keyword via RRF)
        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            n_results=num_results,
            project_filter=normalized_project,
            content_type_filter=content_type,
            source_filter=source_filter,
            tag_filter=tag,
            intent_filter=intent,
            importance_min=importance_min,
            date_from=date_from,
            date_to=date_to,
        )

        if not results["documents"][0]:
            return [TextContent(type="text", text="No results found.")]

        # Format results
        output_parts = [f"## Search Results for: {query}\n"]

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = 1 - dist if dist is not None else 0
            output_parts.append(f"\n### Result {i+1} (score: {score:.3f})")
            # Enrichment header line
            enrichment_parts = []
            if meta.get("intent"):
                enrichment_parts.append(f"Intent: {meta['intent']}")
            if meta.get("importance") is not None:
                enrichment_parts.append(f"Importance: {meta['importance']:.0f}/10")
            if meta.get("tags") and isinstance(meta["tags"], list):
                enrichment_parts.append(f"Tags: {', '.join(str(t) for t in meta['tags'][:5])}")
            project_display = normalize_project_name(meta.get('project')) or meta.get('project', 'unknown')
            header = f"**Project:** {project_display} | **Type:** {meta.get('content_type', 'unknown')}"
            if meta.get("created_at"):
                # Show just the date portion for readability
                date_str = meta["created_at"][:10] if len(meta.get("created_at", "")) >= 10 else meta["created_at"]
                header += f" | **Date:** {date_str}"
            if meta.get("source") and meta["source"] != "claude_code":
                header += f" | **Source:** {meta['source']}"
            output_parts.append(header)
            if enrichment_parts:
                output_parts.append(f"**{' | '.join(enrichment_parts)}**")
            if meta.get("summary"):
                output_parts.append(f"> {meta['summary']}")
            output_parts.append(f"**File:** `{meta.get('source_file', 'unknown')}`\n")
            output_parts.append(doc[:1000] + ("..." if len(doc) > 1000 else ""))
            output_parts.append("\n---")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Search error (query='{query[:50]}...'): {str(e)}")]


async def _stats() -> list[TextContent]:
    """Get knowledge base statistics."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()

        output = f"""## BrainLayer Knowledge Base Stats

- **Total Chunks:** {stats['total_chunks']}
- **Projects:** {', '.join(stats['projects'][:15])}{'...' if len(stats['projects']) > 15 else ''}
- **Content Types:** {', '.join(stats['content_types'])}
"""
        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Stats error: {str(e)}")]


async def _list_projects() -> list[TextContent]:
    """List all projects."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()

        if not stats['projects']:
            return [TextContent(type="text", text="No projects indexed yet.")]

        output = "## Indexed Projects\n\n"
        for proj in sorted(stats['projects']):
            output += f"- {proj}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing projects: {str(e)}")]


async def _context(
    chunk_id: str,
    before: int = 3,
    after: int = 3
) -> list[TextContent]:
    """Get surrounding conversation context for a chunk."""
    try:
        store = _get_vector_store()
        result = store.get_context(chunk_id, before=before, after=after)

        if result.get("error"):
            return [TextContent(type="text", text=f"Context error: {result['error']}")]

        if not result.get("context"):
            return [TextContent(type="text", text="No context available for this chunk.")]

        output_parts = ["## Conversation Context\n"]

        for chunk in result["context"]:
            marker = " **[TARGET]**" if chunk.get("is_target") else ""
            ctype = chunk.get("content_type", "unknown")
            pos = chunk.get("position", "?")
            output_parts.append(f"\n### Position {pos} ({ctype}){marker}\n")
            content = chunk.get("content", "")
            output_parts.append(content[:1500] + ("..." if len(content) > 1500 else ""))
            output_parts.append("\n---")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Context error: {str(e)}")]


async def _file_timeline(
    file_path: str,
    project: str | None = None,
    limit: int = 50,
) -> list[TextContent]:
    """Get interaction timeline for a file."""
    try:
        store = _get_vector_store()
        interactions = store.get_file_timeline(file_path, project=project, limit=limit)

        if not interactions:
            return [TextContent(type="text", text=f"No interactions found for '{file_path}'.")]

        output_parts = [f"## File Timeline: {file_path}\n"]
        output_parts.append(f"Found {len(interactions)} interactions:\n")

        for i, row in enumerate(interactions):
            ts = row.get("timestamp", "?")
            action = row.get("action", "?")
            session = row.get("session_id", "?")[:8]
            proj = row.get("project", "?")
            fp = row.get("file_path", file_path)
            output_parts.append(
                f"{i+1}. **{action}** `{fp}` at {ts}"
                f" (session: {session}, project: {proj})"
            )

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"File timeline error: {str(e)}")]


async def _operations(
    session_id: str,
) -> list[TextContent]:
    """Get operations for a session."""
    try:
        store = _get_vector_store()
        ops = store.get_session_operations(session_id)

        if not ops:
            return [TextContent(
                type="text",
                text=(
                    f"No operations for session"
                    f" '{session_id[:8]}...'."
                ),
            )]

        output_parts = [
            f"## Operations: {session_id[:8]}...\n",
            f"Found {len(ops)} operations:\n",
        ]

        for i, op in enumerate(ops):
            outcome = op.get("outcome", "unknown")
            ts = (op.get("started_at") or "?")[:19]
            output_parts.append(
                f"{i+1}. **{op.get('operation_type', '?')}**"
                f" — {op.get('summary') or '?'}"
                f" [{outcome}] at {ts}"
            )

        return [TextContent(
            type="text",
            text="\n".join(output_parts),
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Operations error: {str(e)}",
        )]


async def _regression(
    file_path: str,
    project: str | None = None,
) -> list[TextContent]:
    """Analyze a file for regressions."""
    try:
        store = _get_vector_store()
        result = store.get_file_regression(
            file_path, project=project
        )

        if not result["timeline"]:
            return [TextContent(
                type="text",
                text=(
                    f"No interactions found"
                    f" for '{file_path}'."
                ),
            )]

        parts = [
            f"## Regression Analysis: {file_path}\n",
            f"Timeline: {len(result['timeline'])}"
            f" interactions\n",
        ]

        if result["last_success"]:
            ls = result["last_success"]
            parts.append(
                f"**Last success:** {ls['timestamp']}"
                f" (session {ls['session_id'][:8]},"
                f" branch {ls.get('branch', '?')})\n"
            )
        else:
            parts.append(
                "**No successful operations found**\n"
            )

        if result["changes_after"]:
            parts.append(
                f"**Changes after last success:**"
                f" {len(result['changes_after'])}\n"
            )
            for i, c in enumerate(
                result["changes_after"][:15]
            ):
                ts = (c["timestamp"] or "?")[:19]
                branch = c.get("branch") or "?"
                parts.append(
                    f"{i+1}. {c['action']}"
                    f" at {ts}"
                    f" (branch: {branch})"
                )

        return [TextContent(
            type="text",
            text="\n".join(parts),
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Regression error: {str(e)}",
        )]


async def _plan_links(
    plan_name: str | None = None,
    session_id: str | None = None,
    project: str | None = None,
) -> list[TextContent]:
    """Query plan-linked sessions."""
    try:
        store = _get_vector_store()

        if session_id:
            ctx = store.get_session_context(session_id)
            if not ctx:
                return [TextContent(
                    type="text",
                    text=f"No context for session '{session_id[:8]}'.",
                )]
            parts = [
                f"## Session {ctx['session_id'][:8]}\n",
                f"- Branch: {ctx.get('branch') or '?'}",
                f"- PR: #{ctx.get('pr_number') or '?'}",
                f"- Plan: {ctx.get('plan_name') or '(none)'}",
                f"- Phase: {ctx.get('plan_phase') or '(none)'}",
                f"- Story: {ctx.get('story_id') or '(none)'}",
            ]
            return [TextContent(
                type="text", text="\n".join(parts),
            )]

        sessions = store.get_sessions_by_plan(
            plan_name=plan_name, project=project
        )
        if not sessions:
            if plan_name:
                msg = f"No sessions linked to plan '{plan_name}'."
            else:
                msg = "No plan-linked sessions found."
            return [TextContent(type="text", text=msg)]

        title = plan_name or "All Plans"
        parts = [f"## Sessions: {title}\n"]
        for s in sessions[:30]:
            sid = (s["session_id"] or "")[:8]
            branch = s.get("branch") or "?"
            pr = f"#{s['pr_number']}" if s.get("pr_number") else ""
            phase = s.get("plan_phase") or ""
            plan = s.get("plan_name") or ""
            started = (s.get("started_at") or "")[:19]
            parts.append(
                f"- {sid} | {plan}/{phase}"
                f" | {branch} {pr}"
                f" | {started}"
            )

        stats = store.get_plan_linking_stats()
        parts.append(
            f"\nTotal: {stats['linked_sessions']}"
            f"/{stats['total_sessions']} linked"
        )

        return [TextContent(
            type="text", text="\n".join(parts),
        )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Plan links error: {str(e)}",
        )]


def serve():
    """Start the MCP server using stdio.

    Note: MCP uses stdin/stdout for communication, not network ports.
    This is designed for integration with Claude Code via mcpServers config.
    """
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    serve()
