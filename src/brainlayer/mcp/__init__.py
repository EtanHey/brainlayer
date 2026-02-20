"""BrainLayer MCP Server - Model Context Protocol interface for Claude Code."""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CompleteResult,
    Completion,
    TextContent,
    Tool,
    ToolAnnotations,
)

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
    - Claude Code encoded paths: "-Users-janedev-Gits-golems" → "golems"
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
    # "-Users-janedev-Gits-golems" → "golems"
    # "-Users-janedev-Desktop-Gits-rudy-monorepo" → "rudy-monorepo"
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
            remaining = segments[gits_idx + 1 :]
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

    name = re.sub(r"-(?:nightshift|haiku|worktree)-\d+$", "", name)

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


# All BrainLayer tools are read-only (search and analyze only)
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)


# --- Output Schemas (MCP spec 2025-06-18+) ---
# Tools with outputSchema MUST return structuredContent alongside text content.

_MEMORY_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "content": {"type": "string"},
        "intent": {"type": "string"},
        "importance": {"type": "number"},
        "project": {"type": "string"},
        "date": {"type": "string"},
        "content_type": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["content"],
}

_SEARCH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "total": {"type": "integer"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "score": {"type": "number"},
                    "project": {"type": "string"},
                    "content_type": {"type": "string"},
                    "content": {"type": "string"},
                    "source_file": {"type": "string"},
                    "date": {"type": "string"},
                    "source": {"type": "string"},
                    "summary": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "intent": {"type": "string"},
                    "importance": {"type": "number"},
                    "chunk_id": {"type": "string"},
                },
                "required": ["content", "project", "content_type", "score"],
            },
        },
    },
    "required": ["query", "total", "results"],
}

_STATS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "total_chunks": {"type": "integer"},
        "projects": {"type": "array", "items": {"type": "string"}},
        "content_types": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["total_chunks", "projects", "content_types"],
}

_THINK_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "total": {"type": "integer"},
        "decisions": {"type": "array", "items": _MEMORY_ITEM_SCHEMA},
        "patterns": {"type": "array", "items": _MEMORY_ITEM_SCHEMA},
        "bugs": {"type": "array", "items": _MEMORY_ITEM_SCHEMA},
        "context": {"type": "array", "items": _MEMORY_ITEM_SCHEMA},
    },
    "required": ["query", "total", "decisions", "patterns", "bugs", "context"],
}

_RECALL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "file_history": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "action": {"type": "string"},
                    "session_id": {"type": "string"},
                    "file_path": {"type": "string"},
                },
            },
        },
        "related_chunks": {"type": "array", "items": _MEMORY_ITEM_SCHEMA},
        "session_summaries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "branch": {"type": "string"},
                    "plan_name": {"type": "string"},
                    "started_at": {"type": "string"},
                },
            },
        },
    },
    "required": ["target", "file_history", "related_chunks", "session_summaries"],
}

_CURRENT_CONTEXT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "active_projects": {"type": "array", "items": {"type": "string"}},
        "active_branches": {"type": "array", "items": {"type": "string"}},
        "active_plan": {"type": "string"},
        "recent_files": {"type": "array", "items": {"type": "string"}},
        "recent_sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "project": {"type": "string"},
                    "branch": {"type": "string"},
                    "started_at": {"type": "string"},
                    "plan_name": {"type": "string"},
                },
            },
        },
    },
    "required": ["active_projects", "active_branches", "active_plan", "recent_files", "recent_sessions"],
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="brainlayer_search",
            title="Search Knowledge Base",
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
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'how did I implement authentication' or 'React useEffect cleanup')",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name",
                    },
                    "content_type": {
                        "type": "string",
                        "enum": [
                            "ai_code",
                            "stack_trace",
                            "user_message",
                            "assistant_text",
                            "file_read",
                            "git_diff",
                        ],
                        "description": "Optional: filter by content type",
                    },
                    "num_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return (default: 5)",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["claude_code", "whatsapp", "youtube", "all"],
                        "description": (
                            "Filter by data source (default: claude_code). Use 'all' to search everything."
                        ),
                    },
                    "tag": {
                        "type": "string",
                        "description": "Filter by tag (e.g. 'bug-fix', 'authentication', 'typescript')",
                    },
                    "intent": {
                        "type": "string",
                        "enum": [
                            "debugging",
                            "designing",
                            "configuring",
                            "discussing",
                            "deciding",
                            "implementing",
                            "reviewing",
                        ],
                        "description": "Filter by intent classification",
                    },
                    "importance_min": {
                        "type": "number",
                        "description": "Minimum importance score (1-10)",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Filter results from this date (ISO 8601, e.g. '2026-02-01')",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Filter results up to this date (ISO 8601, e.g. '2026-02-19')",
                    },
                },
                "required": ["query"],
            },
            outputSchema=_SEARCH_OUTPUT_SCHEMA,
        ),
        Tool(
            name="brainlayer_stats",
            title="Knowledge Base Stats",
            description="Get statistics about the knowledge base (total chunks, projects, content types).",
            annotations=_READ_ONLY,
            inputSchema={"type": "object", "properties": {}},
            outputSchema=_STATS_OUTPUT_SCHEMA,
        ),
        Tool(
            name="brainlayer_list_projects",
            title="List Projects",
            description="List all projects in the knowledge base.",
            annotations=_READ_ONLY,
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="brainlayer_context",
            title="Get Chunk Context",
            description="""Get surrounding conversation context for a search result.

Given a chunk ID from a search result, returns the chunks before and after it
from the same conversation. Useful for understanding isolated search results.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID from a search result",
                    },
                    "before": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of chunks before the target to include",
                    },
                    "after": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of chunks after the target to include",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="brainlayer_file_timeline",
            title="File Interaction Timeline",
            description="""Get the interaction timeline for a specific file across sessions.

Shows all Claude Code sessions that read, edited, or wrote to a file,
ordered chronologically. Useful for understanding a file's history.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": ("File path or partial path to search for (e.g., 'telegram-bot.ts')"),
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of interactions to return",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="brainlayer_operations",
            title="Session Operations",
            description=(
                "Get logical operation groups for a session."
                " Operations are patterns like"
                " read→edit→test cycles, research chains,"
                " or debug sequences."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID to query"},
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="brainlayer_regression",
            title="Regression Analysis",
            description=(
                "Analyze a file for regressions."
                " Shows the last successful operation"
                " and all changes after it."
                " Useful for debugging: 'what changed"
                " since this file last worked?'"
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": ("File path or partial path to analyze"),
                    },
                    "project": {"type": "string", "description": ("Optional: filter by project")},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="brainlayer_plan_links",
            title="Plan-Session Links",
            description=(
                "Query plan-linked sessions."
                " Shows which plan/phase a session belongs"
                " to, or lists all sessions for a plan."
                " Useful for: 'which plan was I working"
                " on in this session?' or 'show all"
                " sessions for my-feature-plan'."
            ),
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "plan_name": {
                        "type": "string",
                        "description": ("Plan name to query (e.g. 'local-llm-integration')"),
                    },
                    "session_id": {
                        "type": "string",
                        "description": ("Session ID to look up plan info for"),
                    },
                    "project": {"type": "string", "description": ("Optional: filter by project")},
                },
            },
        ),
        Tool(
            name="brainlayer_think",
            title="Think — Retrieve Relevant Memories",
            description="""Given your current task context, retrieve relevant past decisions, patterns, and code.

Use this BEFORE starting a task to get informed context instead of cold-starting:
- "I'm implementing JWT authentication for the API"
- "I need to fix the database connection pooling"
- "Working on the deployment pipeline for Railway"

Returns categorized memories: decisions, patterns, bugs/fixes, and related context.
Results are filtered by importance (3+) and grouped by intent.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Describe what you're working on — the engine will find relevant past knowledge",
                    },
                    "project": {"type": "string", "description": "Optional: filter by project"},
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum memories to retrieve (default: 10)",
                    },
                },
                "required": ["context"],
            },
            outputSchema=_THINK_OUTPUT_SCHEMA,
        ),
        Tool(
            name="brainlayer_recall",
            title="Recall — Proactive Context for File or Topic",
            description="""Proactive smart retrieval. Not "search for X" but:
- "What happened with this file before?" (file_path mode)
- "What have I discussed about authentication?" (topic mode)

For files: returns interaction timeline, sessions that touched it, and related knowledge.
For topics: returns related past discussions, decisions, and patterns.

Use when opening a file or starting work on a familiar topic.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path to recall context for (e.g., 'telegram-bot.ts')",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic to recall context for (e.g., 'authentication', 'deployment')",
                    },
                    "project": {"type": "string", "description": "Optional: filter by project"},
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results (default: 10)",
                    },
                },
            },
            outputSchema=_RECALL_OUTPUT_SCHEMA,
        ),
        Tool(
            name="brainlayer_sessions",
            title="Browse Recent Sessions",
            description="""List recent Claude Code sessions with metadata.

Shows session ID, project, branch, plan linkage, and timestamp.
Useful for browsing what you've been working on by date or project.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Optional: filter by project"},
                    "days": {
                        "type": "integer",
                        "default": 7,
                        "description": "How many days back to look (default: 7)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "description": "Maximum sessions to return (default: 20)",
                    },
                },
            },
        ),
        Tool(
            name="brainlayer_current_context",
            title="Current Working Context",
            description="""Get what you're currently working on — recent sessions, projects, files, and active plan.

Lightweight (no embedding needed). Designed for voice assistants and quick context injection.
Use at conversation start to understand current state without asking the user.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "How many hours back to look (default: 24)",
                    },
                },
            },
            outputSchema=_CURRENT_CONTEXT_OUTPUT_SCHEMA,
        ),
    ]


@server.completion()
async def handle_completion(ref, argument) -> CompleteResult:
    """Provide completions for tool arguments."""
    # Only handle tool argument completions
    if not hasattr(ref, "name"):
        return CompleteResult(completion=Completion(values=[]))

    arg_name = argument.name if hasattr(argument, "name") else ""
    arg_value = argument.value if hasattr(argument, "value") else ""

    if arg_name == "project":
        try:
            store = _get_vector_store()
            stats = store.get_stats()
            projects = stats.get("projects", [])
            # Normalize and filter by prefix
            normalized = []
            for p in projects:
                norm = normalize_project_name(p) or p
                if norm not in normalized:
                    normalized.append(norm)
            if arg_value:
                normalized = [p for p in normalized if p.lower().startswith(arg_value.lower())]
            return CompleteResult(completion=Completion(values=sorted(normalized)[:20], hasMore=len(normalized) > 20))
        except Exception:
            return CompleteResult(completion=Completion(values=[]))

    elif arg_name == "content_type":
        types = ["ai_code", "stack_trace", "user_message", "assistant_text", "file_read", "git_diff"]
        if arg_value:
            types = [t for t in types if t.startswith(arg_value)]
        return CompleteResult(completion=Completion(values=types))

    elif arg_name == "source":
        sources = ["claude_code", "whatsapp", "youtube", "all"]
        if arg_value:
            sources = [s for s in sources if s.startswith(arg_value)]
        return CompleteResult(completion=Completion(values=sources))

    elif arg_name == "intent":
        intents = ["debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"]
        if arg_value:
            intents = [i for i in intents if i.startswith(arg_value)]
        return CompleteResult(completion=Completion(values=intents))

    return CompleteResult(completion=Completion(values=[]))


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]):
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
            after=min(arguments.get("after", 3), 50),
        )

    elif name == "brainlayer_file_timeline":
        return await _file_timeline(
            file_path=arguments["file_path"],
            project=arguments.get("project"),
            limit=arguments.get("limit", 50),
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

    elif name == "brainlayer_think":
        return await _think(
            context=arguments["context"],
            project=arguments.get("project"),
            max_results=arguments.get("max_results", 10),
        )

    elif name == "brainlayer_recall":
        return await _recall(
            file_path=arguments.get("file_path"),
            topic=arguments.get("topic"),
            project=arguments.get("project"),
            max_results=arguments.get("max_results", 10),
        )

    elif name == "brainlayer_sessions":
        return await _sessions(
            project=arguments.get("project"),
            days=arguments.get("days", 7),
            limit=arguments.get("limit", 20),
        )

    elif name == "brainlayer_current_context":
        return await _current_context(
            hours=arguments.get("hours", 24),
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
):
    """Execute a hybrid search query (semantic + keyword via RRF)."""
    try:
        if num_results < 1:
            num_results = 5
        elif num_results > 100:
            num_results = 100

        store = _get_vector_store()

        if store.count() == 0:
            empty = {"query": query, "total": 0, "results": []}
            return (
                [TextContent(type="text", text="Knowledge base is empty. Run 'brainlayer index' to populate it.")],
                empty,
            )

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
            empty = {"query": query, "total": 0, "results": []}
            return ([TextContent(type="text", text="No results found.")], empty)

        # Build structured results + formatted text
        output_parts = [f"## Search Results for: {query}\n"]
        structured_results = []

        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            score = 1 - dist if dist is not None else 0

            # Build structured result item
            item = {
                "score": round(score, 4),
                "project": normalize_project_name(meta.get("project")) or meta.get("project", "unknown"),
                "content_type": meta.get("content_type", "unknown"),
                "content": doc[:1000],
                "source_file": meta.get("source_file", "unknown"),
            }
            if meta.get("created_at"):
                item["date"] = meta["created_at"][:10] if len(meta.get("created_at", "")) >= 10 else meta["created_at"]
            if meta.get("source") and meta["source"] != "claude_code":
                item["source"] = meta["source"]
            if meta.get("summary"):
                item["summary"] = meta["summary"]
            if meta.get("tags") and isinstance(meta["tags"], list):
                item["tags"] = [str(t) for t in meta["tags"][:5]]
            if meta.get("intent"):
                item["intent"] = meta["intent"]
            if meta.get("importance") is not None:
                item["importance"] = meta["importance"]
            if meta.get("chunk_id"):
                item["chunk_id"] = meta["chunk_id"]
            structured_results.append(item)

            # Build text output (same as before)
            output_parts.append(f"\n### Result {i + 1} (score: {score:.3f})")
            enrichment_parts = []
            if meta.get("intent"):
                enrichment_parts.append(f"Intent: {meta['intent']}")
            if meta.get("importance") is not None:
                enrichment_parts.append(f"Importance: {meta['importance']:.0f}/10")
            if meta.get("tags") and isinstance(meta["tags"], list):
                enrichment_parts.append(f"Tags: {', '.join(str(t) for t in meta['tags'][:5])}")
            project_display = item["project"]
            if project_display == "unknown" and meta.get("contact_name"):
                project_display = meta["contact_name"]
            header = f"**Project:** {project_display} | **Type:** {meta.get('content_type', 'unknown')}"
            if item.get("date"):
                header += f" | **Date:** {item['date']}"
            if item.get("source"):
                header += f" | **Source:** {item['source']}"
            output_parts.append(header)
            if enrichment_parts:
                output_parts.append(f"**{' | '.join(enrichment_parts)}**")
            if meta.get("summary"):
                output_parts.append(f"> {meta['summary']}")
            output_parts.append(f"**File:** `{meta.get('source_file', 'unknown')}`\n")
            output_parts.append(doc[:1000] + ("..." if len(doc) > 1000 else ""))
            output_parts.append("\n---")

        structured = {
            "query": query,
            "total": len(structured_results),
            "results": structured_results,
        }
        return ([TextContent(type="text", text="\n".join(output_parts))], structured)

    except Exception as e:
        return [TextContent(type="text", text=f"Search error (query='{query[:50]}...'): {str(e)}")]


async def _stats():
    """Get knowledge base statistics."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()

        output = f"""## BrainLayer Knowledge Base Stats

- **Total Chunks:** {stats["total_chunks"]}
- **Projects:** {", ".join(stats["projects"][:15])}{"..." if len(stats["projects"]) > 15 else ""}
- **Content Types:** {", ".join(stats["content_types"])}
"""
        structured = {
            "total_chunks": stats["total_chunks"],
            "projects": stats["projects"],
            "content_types": stats["content_types"],
        }
        return ([TextContent(type="text", text=output)], structured)

    except Exception as e:
        return [TextContent(type="text", text=f"Stats error: {str(e)}")]


async def _list_projects() -> list[TextContent]:
    """List all projects."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()

        if not stats["projects"]:
            return [TextContent(type="text", text="No projects indexed yet.")]

        output = "## Indexed Projects\n\n"
        for proj in sorted(stats["projects"]):
            output += f"- {proj}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing projects: {str(e)}")]


async def _context(chunk_id: str, before: int = 3, after: int = 3) -> list[TextContent]:
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
            output_parts.append(f"{i + 1}. **{action}** `{fp}` at {ts} (session: {session}, project: {proj})")

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
            return [
                TextContent(
                    type="text",
                    text=(f"No operations for session '{session_id[:8]}...'."),
                )
            ]

        output_parts = [
            f"## Operations: {session_id[:8]}...\n",
            f"Found {len(ops)} operations:\n",
        ]

        for i, op in enumerate(ops):
            outcome = op.get("outcome", "unknown")
            ts = (op.get("started_at") or "?")[:19]
            output_parts.append(
                f"{i + 1}. **{op.get('operation_type', '?')}** — {op.get('summary') or '?'} [{outcome}] at {ts}"
            )

        return [
            TextContent(
                type="text",
                text="\n".join(output_parts),
            )
        ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Operations error: {str(e)}",
            )
        ]


async def _regression(
    file_path: str,
    project: str | None = None,
) -> list[TextContent]:
    """Analyze a file for regressions."""
    try:
        store = _get_vector_store()
        result = store.get_file_regression(file_path, project=project)

        if not result["timeline"]:
            return [
                TextContent(
                    type="text",
                    text=(f"No interactions found for '{file_path}'."),
                )
            ]

        parts = [
            f"## Regression Analysis: {file_path}\n",
            f"Timeline: {len(result['timeline'])} interactions\n",
        ]

        if result["last_success"]:
            ls = result["last_success"]
            parts.append(
                f"**Last success:** {ls['timestamp']}"
                f" (session {ls['session_id'][:8]},"
                f" branch {ls.get('branch', '?')})\n"
            )
        else:
            parts.append("**No successful operations found**\n")

        if result["changes_after"]:
            parts.append(f"**Changes after last success:** {len(result['changes_after'])}\n")
            for i, c in enumerate(result["changes_after"][:15]):
                ts = (c["timestamp"] or "?")[:19]
                branch = c.get("branch") or "?"
                parts.append(f"{i + 1}. {c['action']} at {ts} (branch: {branch})")

        return [
            TextContent(
                type="text",
                text="\n".join(parts),
            )
        ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Regression error: {str(e)}",
            )
        ]


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
                return [
                    TextContent(
                        type="text",
                        text=f"No context for session '{session_id[:8]}'.",
                    )
                ]
            parts = [
                f"## Session {ctx['session_id'][:8]}\n",
                f"- Branch: {ctx.get('branch') or '?'}",
                f"- PR: #{ctx.get('pr_number') or '?'}",
                f"- Plan: {ctx.get('plan_name') or '(none)'}",
                f"- Phase: {ctx.get('plan_phase') or '(none)'}",
                f"- Story: {ctx.get('story_id') or '(none)'}",
            ]
            return [
                TextContent(
                    type="text",
                    text="\n".join(parts),
                )
            ]

        sessions = store.get_sessions_by_plan(plan_name=plan_name, project=project)
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
            parts.append(f"- {sid} | {plan}/{phase} | {branch} {pr} | {started}")

        stats = store.get_plan_linking_stats()
        parts.append(f"\nTotal: {stats['linked_sessions']}/{stats['total_sessions']} linked")

        return [
            TextContent(
                type="text",
                text="\n".join(parts),
            )
        ]

    except Exception as e:
        return [
            TextContent(
                type="text",
                text=f"Plan links error: {str(e)}",
            )
        ]


async def _think(
    context: str,
    project: str | None = None,
    max_results: int = 10,
):
    """Execute think — retrieve relevant memories for current task."""
    try:
        from ..engine import think

        store = _get_vector_store()
        model = _get_embedding_model()

        # Run embedding in thread to not block
        loop = asyncio.get_running_loop()

        def _embed(text: str) -> list[float]:
            return model.embed_query(text)

        # Normalize project
        normalized_project = normalize_project_name(project)

        result = await loop.run_in_executor(
            None,
            lambda: think(
                context=context,
                store=store,
                embed_fn=_embed,
                project=normalized_project,
                max_results=max_results,
            ),
        )

        def _memory_to_dict(item: dict) -> dict:
            d: dict = {"content": item.get("content", "")}
            for key in ("summary", "intent", "importance", "project", "content_type"):
                if item.get(key) is not None:
                    d[key] = item[key]
            if item.get("created_at"):
                d["date"] = item["created_at"][:10]
            if item.get("tags") and isinstance(item["tags"], list):
                d["tags"] = [str(t) for t in item["tags"]]
            return d

        structured = {
            "query": result.query,
            "total": result.total,
            "decisions": [_memory_to_dict(i) for i in result.decisions],
            "patterns": [_memory_to_dict(i) for i in result.patterns],
            "bugs": [_memory_to_dict(i) for i in result.bugs],
            "context": [_memory_to_dict(i) for i in result.context],
        }
        return ([TextContent(type="text", text=result.format())], structured)

    except Exception as e:
        return [TextContent(type="text", text=f"Think error: {str(e)}")]


async def _recall(
    file_path: str | None = None,
    topic: str | None = None,
    project: str | None = None,
    max_results: int = 10,
):
    """Execute recall — proactive context retrieval."""
    try:
        from ..engine import recall

        store = _get_vector_store()
        model = _get_embedding_model()
        normalized_project = normalize_project_name(project)

        loop = asyncio.get_running_loop()

        def _embed(text: str) -> list[float]:
            return model.embed_query(text)

        result = await loop.run_in_executor(
            None,
            lambda: recall(
                store=store,
                embed_fn=_embed,
                file_path=file_path,
                topic=topic,
                project=normalized_project,
                max_results=max_results,
            ),
        )

        def _memory_to_dict(item: dict) -> dict:
            d: dict = {"content": item.get("content", "")}
            for key in ("summary", "intent", "importance", "project", "content_type"):
                if item.get(key) is not None:
                    d[key] = item[key]
            if item.get("created_at"):
                d["date"] = item["created_at"][:10]
            if item.get("tags") and isinstance(item["tags"], list):
                d["tags"] = [str(t) for t in item["tags"]]
            return d

        structured = {
            "target": result.target,
            "file_history": [
                {
                    "timestamp": (h.get("timestamp") or "")[:19],
                    "action": h.get("action", ""),
                    "session_id": h.get("session_id", ""),
                    "file_path": h.get("file_path", ""),
                }
                for h in result.file_history
            ],
            "related_chunks": [_memory_to_dict(c) for c in result.related_chunks],
            "session_summaries": [
                {
                    "session_id": s.get("session_id", ""),
                    "branch": s.get("branch", ""),
                    "plan_name": s.get("plan_name", ""),
                    "started_at": (s.get("started_at") or "")[:19],
                }
                for s in result.session_summaries
            ],
        }
        return ([TextContent(type="text", text=result.format())], structured)

    except Exception as e:
        return [TextContent(type="text", text=f"Recall error: {str(e)}")]


async def _sessions(
    project: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> list[TextContent]:
    """Execute sessions — list recent sessions."""
    try:
        from ..engine import format_sessions, sessions

        store = _get_vector_store()
        normalized_project = normalize_project_name(project)

        result = sessions(
            store=store,
            project=normalized_project,
            days=days,
            limit=limit,
        )

        return [TextContent(type="text", text=format_sessions(result, days=days))]

    except Exception as e:
        return [TextContent(type="text", text=f"Sessions error: {str(e)}")]


async def _current_context(
    hours: int = 24,
):
    """Execute current_context — lightweight session awareness."""
    try:
        from ..engine import current_context

        store = _get_vector_store()
        result = current_context(store=store, hours=hours)

        structured = {
            "active_projects": result.active_projects,
            "active_branches": result.active_branches,
            "active_plan": result.active_plan,
            "recent_files": result.recent_files,
            "recent_sessions": [
                {
                    "session_id": s.session_id,
                    "project": s.project,
                    "branch": s.branch,
                    "started_at": s.started_at[:19] if s.started_at else "",
                    "plan_name": s.plan_name,
                }
                for s in result.recent_sessions
            ],
        }
        return ([TextContent(type="text", text=result.format())], structured)

    except Exception as e:
        return [TextContent(type="text", text=f"Current context error: {str(e)}")]


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
