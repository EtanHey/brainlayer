"""BrainLayer MCP Server - Model Context Protocol interface for Claude Code."""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    CompleteResult,
    Completion,
    TextContent,
    Tool,
    ToolAnnotations,
)

from ._shared import (  # noqa: I001
    _auto_importance as _auto_importance,
)
from ._shared import (
    _build_compact_result as _build_compact_result,
)
from ._shared import (
    _detect_memory_type as _detect_memory_type,
)
from ._shared import (
    _error_result,
    _get_vector_store,
    validate_config,
)
from ._shared import (
    _normalize_project_name as _normalize_project_name,
)
from .entity_handler import _brain_entity, _brain_get_person
from .search_handler import (
    _brain_recall,
    _brain_search,
    _context,
    _current_context,
    _file_timeline,
    _list_projects,
    _operations,
    _plan_links,
    _recall,
    _regression,
    _session_summary,
    _sessions,
    _stats,
    _think,
)
from .store_handler import _brain_archive, _brain_digest, _brain_supersede, _brain_update, _store, _store_new
from .tags_handler import _brain_tags_mcp

# MCP query timeout prevents indefinite hangs when DB is locked by enrichment.
MCP_QUERY_TIMEOUT = 15  # seconds — fail fast, return error instead of hanging


async def _with_timeout(coro, timeout: float = MCP_QUERY_TIMEOUT):
    """Wrap an async operation with a timeout. Returns error result on timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=(
                        f"BrainLayer timeout ({timeout}s): DB may be locked by enrichment pipeline. "
                        "Try again in a few minutes, or kill enrichment: "
                        "pkill -f 'brainlayer.pipeline.enrichment'"
                    ),
                )
            ],
            isError=True,
        )


# Create MCP server
server = Server(
    "brainlayer",
    instructions=(
        "Memory layer for Claude Code. 8 tools:\n"
        "- brain_search(query): semantic search across 268K+ indexed conversation chunks. "
        "Returns compact results by default (snippet + chunk_id + score). "
        "Use detail='full' for verbose output. "
        "Filters: project, file_path, chunk_id, content_type, tag, intent, importance_min. "
        "Routing is automatic — pass file_path for file history, no args for current work.\n"
        "- brain_expand(chunk_id): drill into a specific search result. "
        "Returns full content + N surrounding chunks for context. "
        "Use after brain_search to read interesting results in full.\n"
        "- brain_store(content): save decisions, learnings, mistakes, ideas, todos. "
        "type is auto-detected from content. Pass importance (1-10) for critical items.\n"
        "- brain_recall(mode): session/operational context. "
        "mode=context (default, what am I working on), sessions, operations, plan, summary, stats.\n"
        "- brain_digest(content): deeply ingest large content (research, audits, transcripts, docs). "
        "Extracts entities, relations, faceted tags, sentiment, action items, decisions, questions, and sanitizes PII before external enrichment calls. "
        "Creates a new searchable chunk with source='digest'.\n"
        "- brain_entity(query): look up a known entity in the knowledge graph. "
        "Returns entity type, relations, and evidence chunks.\n"
        "- brain_update(action, chunk_id): update, archive, or merge existing memories. "
        "action=update (change content/tags/importance), archive (soft-delete), merge (keep one, archive duplicates).\n"
        "- brain_supersede(old_chunk_id, new_chunk_id): mark old memory as superseded by new one. "
        "Old chunk drops from default search. Safety check for personal data.\n"
        "- brain_archive(chunk_id): archive a memory with timestamp. "
        "Excluded from default search, accessible via direct lookup.\n"
        "Use brain_search at the start of tasks to retrieve past decisions and patterns.\n"
        "Use brain_store when you make a decision, hit a bug, learn something, or finish a phase.\n"
        'project scoping: auto-inferred from cwd. Override with project="all" for cross-project search.\n'
        "All 14 old brainlayer_* tool names still work (backward compat aliases)."
    ),
)

# Tool annotations
_READ_ONLY = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)

# --- Output schemas ---

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

_STORE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "chunk_id": {"type": "string"},
        "related": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "summary": {"type": "string"},
                    "project": {"type": "string"},
                    "type": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    },
    "required": ["chunk_id", "related"],
}


# --- Tool registration ---


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools — 7 consolidated tools."""
    return [
        Tool(
            name="brain_search",
            title="Search Knowledge Base",
            description="""Search through past Claude Code conversations and learnings.

Auto-routes based on input:
- chunk_id → expand surrounding context
- file_path → file timeline + related knowledge (add regression signals like "broke" for regression analysis)
- "what am I working on" → current context + relevant memories
- "how did I implement X" → past decisions, patterns, code
- "history of X" / "discussed about X" → topic recall
- Default → hybrid semantic + keyword search

Returns: Markdown text or structured JSON depending on the route taken.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'how did I implement authentication', 'what happened with auth.ts', 'what am I working on')",
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name. Encoded/worktree names are auto-normalized.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "File path to search for (e.g., 'auth.ts'). Triggers file-aware routing: timeline + related knowledge. Add regression signals in query for regression analysis.",
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "Expand context around a specific chunk from a previous search result.",
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
                        "description": "Filter by content type (search mode only)",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["claude_code", "whatsapp", "youtube", "all"],
                        "description": "Filter by data source (default: claude_code). Use 'all' to search everything.",
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
                    "sentiment": {
                        "type": "string",
                        "enum": ["frustration", "confusion", "positive", "satisfaction", "neutral"],
                        "description": "Filter by sentiment label (Phase 6)",
                    },
                    "num_results": {
                        "type": "integer",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Number of results to return (default: 5, max: 100)",
                    },
                    "before": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 0,
                        "maximum": 50,
                        "description": "Context chunks before target (chunk_id mode only)",
                    },
                    "after": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 0,
                        "maximum": 50,
                        "description": "Context chunks after target (chunk_id mode only)",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 10,
                        "description": "Maximum results for think/recall modes (default: 10)",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Filter results to chunks linked to this entity ID. Used for per-person memory scoping (e.g., get only memories about a specific person). Bypasses routing rules.",
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["compact", "full"],
                        "default": "compact",
                        "description": "Result detail level. 'compact' (default): returns snippet (150 chars), chunk_id, score, date, project, summary — use brain_expand to drill into specific results. 'full': returns full content + all metadata fields.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="brain_store",
            title="Store Memory",
            description="""Persistently store a memory into BrainLayer.

Use this to save ideas, mistakes, decisions, learnings, todos, bookmarks, notes, journal entries, or issues. Stored items are embedded at write time and immediately searchable.

Type is auto-detected from content if omitted (e.g., "Always use bun" → decision, "Bug: overflow" → mistake, "TODO: add X" → todo, "Issue: digest fails" → issue).

Issues support lifecycle tracking (status: open→in_progress→done→archived), severity levels, and code references (file_path, function_name, line_number).

Returns: Structured JSON with `chunk_id` (string) and `related[]` (similar existing memories).""",
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content to store (e.g., a decision, learning, idea)",
                    },
                    "type": {
                        "type": "string",
                        "enum": [
                            "idea",
                            "mistake",
                            "decision",
                            "learning",
                            "todo",
                            "bookmark",
                            "note",
                            "journal",
                            "issue",
                        ],
                        "description": "Memory type. Auto-detected from content if omitted.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: project name to scope the memory",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: tags for categorization (e.g., ['reliability', 'api'])",
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Optional: importance score 1-10. Auto-scored from content if omitted.",
                    },
                    "confidence_score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Decision confidence (0-1). Only for type=decision.",
                    },
                    "outcome": {
                        "type": "string",
                        "enum": ["pending", "validated", "reversed"],
                        "description": "Decision outcome. Only for type=decision.",
                    },
                    "reversibility": {
                        "type": "string",
                        "enum": ["easy", "hard", "destructive"],
                        "description": "How hard to reverse. Only for type=decision.",
                    },
                    "files_changed": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files affected by this decision.",
                    },
                    "entity_id": {
                        "type": "string",
                        "description": "Link this memory to an entity (e.g., a person). The stored chunk will be linked via kg_entity_chunks for per-person memory retrieval.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["open", "in_progress", "done", "archived"],
                        "description": "Issue lifecycle status. Only for type=issue. Defaults to 'open'.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                        "description": "Issue severity. Only for type=issue.",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Code file reference (e.g., 'src/brainlayer/mcp/__init__.py'). Only for type=issue.",
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Function/method reference. Only for type=issue.",
                    },
                    "line_number": {
                        "type": "integer",
                        "description": "Line number reference. Only for type=issue.",
                    },
                    "supersedes": {
                        "type": "string",
                        "description": "Optional chunk_id to supersede. The old chunk is marked as superseded by this new one and removed from default search.",
                    },
                },
                "required": ["content"],
            },
            outputSchema=_STORE_OUTPUT_SCHEMA,
        ),
        Tool(
            name="brain_get_person",
            title="Get Person Context",
            description="""Composite tool: look up a person entity and retrieve their scoped memories in one call.

Returns the person's profile (hard_constraints, preferences, contact_info),
their relations in the knowledge graph, and relevant memory chunks linked to them.

If 'context' is provided, memories are ranked by semantic relevance to the context.
Otherwise, memories are ordered by their entity-chunk relevance score.

Designed for copilot agents that need full person context in a single call.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Person name to look up (e.g., 'Avi Simon'). Searches by FTS + semantic match.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional meeting/conversation context to rank memories by relevance (e.g., 'schedule a meeting next week about product roadmap').",
                    },
                    "num_memories": {
                        "type": "integer",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Number of memory chunks to return (default: 10).",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="brain_recall",
            title="Recall Context",
            description="""Get current working context, browse sessions, or inspect session details.

Modes (auto-inferred from params if omitted):
- context: What am I working on? (default — recent sessions, projects, files, active plan)
- sessions: Browse recent sessions (set days/limit to tune)
- operations: Operation groups for a session (requires session_id)
- plan: Plan-session linkage (requires plan_name or session_id)
- summary: Enriched session summary (requires session_id)
- stats: Knowledge base statistics + project list

Returns: Structured JSON or Markdown depending on mode.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["context", "sessions", "operations", "plan", "summary", "stats"],
                        "description": "Recall mode. Auto-inferred from other params if omitted.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name",
                    },
                    "hours": {
                        "type": "integer",
                        "default": 24,
                        "description": "How many hours back to look (mode=context, default: 24)",
                    },
                    "days": {
                        "type": "integer",
                        "default": 7,
                        "minimum": 1,
                        "maximum": 365,
                        "description": "How many days back (mode=sessions, default: 7)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Max sessions to return (mode=sessions, default: 20)",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID (required for mode=operations/summary, optional for mode=plan)",
                    },
                    "plan_name": {
                        "type": "string",
                        "description": "Plan name (mode=plan, e.g. 'local-llm-integration')",
                    },
                },
            },
        ),
        Tool(
            name="brain_digest",
            title="Digest Content",
            description="""Deeply ingest and enrich large content such as research dumps, audit output, transcripts, and long documents.

When to use:
- after producing large content that should become searchable knowledge
- after brain_store when you want slower, deeper enrichment for a chunk or document
- on schedule for backfill runs over older content

What it does:
- creates a new chunk with source="digest"
- extracts entities and relations into the knowledge graph
- adds faceted tags (topics + dom:* + act:*)
- analyzes sentiment and identifies action items, decisions, and questions
- sanitizes PII before any external API enrichment call

How it differs from brain_store:
- brain_store = fast write, minimal processing, optimized for immediate capture
- brain_digest = slower deep enrichment, may call Gemini, intended for richer indexing

Modes in the digest pipeline:
- realtime: single chunk/document, instant enrichment
- batch: backlog/backfill processing via Gemini Batch API
- local: offline/private enrichment via MLX

Returns: Structured JSON with digest_id, summary, tags, entities, relations,
action_items, decisions, questions, sentiment, enrichment status, and stats.""",
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["digest", "enrich"],
                        "description": "digest (default): ingest provided content. enrich: run realtime enrichment on existing DB chunks.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Raw text content to deeply digest and enrich (research, audit, transcript, article, meeting notes)",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the content",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project name to associate with",
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of known participant names (improves entity extraction)",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 25,
                        "minimum": 1,
                        "maximum": 5000,
                        "description": "For mode=enrich: max number of existing chunks to enrich via realtime mode.",
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="brain_entity",
            title="Entity Lookup",
            description="""Look up a known entity in the knowledge graph.

Searches by name (FTS + semantic), returns structured info including
entity type, relations to other entities, and evidence chunks.

Returns: Structured JSON with name, entity_type, relations[], evidence[], or null if not found.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Entity name or search query (e.g., 'Etan Heyman', 'brainlayer', 'Cantaloupe AI')",
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["person", "company", "project", "golem", "technology", "concept"],
                        "description": "Optional: filter by entity type",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="brain_expand",
            title="Expand Chunk Context",
            description="""Expand a specific chunk from a previous brain_search result.

Returns the full content of the target chunk plus N surrounding chunks for context.
Use this after brain_search returns compact results — pick an interesting chunk_id
and expand it to see full content and conversation flow.

Example workflow:
1. brain_search("auth implementation") → compact results with chunk_ids
2. brain_expand(chunk_id="abc123", context=3) → full content + 3 chunks before/after""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk_id from a brain_search result to expand.",
                    },
                    "context": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 0,
                        "maximum": 20,
                        "description": "Number of surrounding chunks to include (before and after). Default: 3.",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="brain_update",
            title="Update or Archive Memory",
            description="""Update, archive, or merge existing memories in BrainLayer.

Actions:
- **update**: Change content, tags, or importance of an existing memory. If content changes, re-embeds automatically.
- **archive**: Soft-delete a memory (removes from search results, keeps in DB).
- **merge**: Combine multiple duplicate memories into one. Keeps the first chunk_id, archives the rest.

Use brain_search first to find the chunk_id(s) you want to modify.

Returns: Structured JSON with action taken and affected chunk IDs.""",
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["update", "archive", "merge"],
                        "description": "What to do: update fields, archive (soft-delete), or merge duplicates",
                    },
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID to update or archive. For merge, this is the chunk to keep.",
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (update only). Will be re-embedded.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "New tags (update only). Replaces existing tags.",
                    },
                    "importance": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "New importance score (update only).",
                    },
                    "merge_chunk_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For merge: additional chunk IDs to archive (duplicates of chunk_id).",
                    },
                },
                "required": ["action", "chunk_id"],
            },
        ),
        Tool(
            name="brain_tags",
            title="Tag Discovery",
            description="""Discover and explore tags across your knowledge base.

Actions:
- **list**: Return top tags ordered by frequency (most-used first). Optional project filter.
- **search**: Find tags matching a prefix or pattern (case-insensitive). Useful for autocomplete.
- **suggest**: Suggest relevant existing tags for a piece of content. Matches content keywords against tag vocabulary.

Returns: JSON with 'tags' (list of {tag, count}) and 'total' count.""",
            annotations=_READ_ONLY,
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "search", "suggest"],
                        "description": "What to do: list top tags, search by prefix, or suggest tags for content.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Tag prefix or pattern to match (required for action='search').",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to suggest tags for (required for action='suggest').",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name (action='list' only).",
                    },
                    "limit": {
                        "type": "integer",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 200,
                        "description": "Maximum number of tags to return (default: 20).",
                    },
                },
                "required": ["action"],
            },
        ),
        Tool(
            name="brain_supersede",
            title="Supersede Memory",
            description="""Mark an old memory as superseded by a newer one.

The old chunk is removed from default search results and linked to its replacement.
Use this when a fact, decision, or learning has been updated and the old version should
no longer appear in searches (but remains accessible via include_history=true).

Safety: auto-supersede works for technical content. Personal data (journals, notes,
health/finance/relationship content) requires safety_check='confirm' + confirm=true.

Returns: Structured JSON with action taken.""",
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "old_chunk_id": {
                        "type": "string",
                        "description": "The chunk ID to mark as superseded.",
                    },
                    "new_chunk_id": {
                        "type": "string",
                        "description": "The chunk ID that replaces the old one.",
                    },
                    "safety_check": {
                        "type": "string",
                        "enum": ["auto", "confirm"],
                        "default": "auto",
                        "description": "Safety mode: 'auto' for technical facts (auto-supersedes), 'confirm' for personal data (requires confirm=true).",
                    },
                    "confirm": {
                        "type": "boolean",
                        "default": False,
                        "description": "Set to true to confirm superseding personal data (when safety_check='confirm').",
                    },
                },
                "required": ["old_chunk_id", "new_chunk_id"],
            },
        ),
        Tool(
            name="brain_archive",
            title="Archive Memory",
            description="""Archive a memory — soft-delete with timestamp.

Archived chunks are excluded from default search results but remain in the database.
They can be retrieved with include_history=true or by direct chunk_id lookup.

Use this to clean up stale, outdated, or irrelevant memories without permanent deletion.

Returns: Structured JSON with chunk_id and optional reason.""",
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "string",
                        "description": "The chunk ID to archive.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Optional: reason for archiving (stored for audit trail).",
                    },
                },
                "required": ["chunk_id"],
            },
        ),
    ]


# --- Completions ---


@server.completion()
async def handle_completion(ref, argument) -> CompleteResult:
    """Provide completions for tool arguments."""
    if not hasattr(ref, "name"):
        return CompleteResult(completion=Completion(values=[]))

    arg_name = argument.name if hasattr(argument, "name") else ""
    arg_value = argument.value if hasattr(argument, "value") else ""

    if arg_name == "project":
        try:
            store = _get_vector_store()
            stats = store.get_stats()
            projects = stats.get("projects", [])
            normalized = []
            for p in projects:
                norm = _normalize_project_name(p) or p
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

    elif arg_name == "mode":
        modes = ["context", "sessions", "operations", "plan", "summary", "stats"]
        if arg_value:
            modes = [m for m in modes if m.startswith(arg_value)]
        return CompleteResult(completion=Completion(values=modes))

    elif arg_name == "type":
        types = ["idea", "mistake", "decision", "learning", "todo", "bookmark", "note", "journal"]
        if arg_value:
            types = [t for t in types if t.startswith(arg_value)]
        return CompleteResult(completion=Completion(values=types))

    return CompleteResult(completion=Completion(values=[]))


# --- Tool routing ---


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]):
    """Handle tool calls — 7 new tools + 14 backward-compat aliases."""

    # --- New consolidated tools ---

    if name == "brain_search":
        return await _with_timeout(
            _brain_search(
                query=arguments["query"],
                project=arguments.get("project"),
                file_path=arguments.get("file_path"),
                chunk_id=arguments.get("chunk_id"),
                content_type=arguments.get("content_type"),
                source=arguments.get("source"),
                tag=arguments.get("tag"),
                intent=arguments.get("intent"),
                importance_min=arguments.get("importance_min"),
                date_from=arguments.get("date_from"),
                date_to=arguments.get("date_to"),
                sentiment=arguments.get("sentiment"),
                entity_id=arguments.get("entity_id"),
                num_results=arguments.get("num_results", 5),
                before=max(0, min(arguments.get("before", 3), 50)),
                after=max(0, min(arguments.get("after", 3), 50)),
                max_results=arguments.get("max_results", 10),
                detail=arguments.get("detail", "compact"),
            )
        )

    elif name == "brain_store":
        # No timeout wrapper on writes — non-idempotent (UUID-based IDs).
        # If timeout fires but executor thread completes, user retries = duplicate memory.
        imp = arguments.get("importance")
        ln = arguments.get("line_number")
        return await _store_new(
            content=arguments["content"],
            memory_type=arguments.get("type"),
            project=arguments.get("project"),
            tags=arguments.get("tags"),
            importance=max(1, min(imp, 10)) if imp is not None else None,
            confidence_score=arguments.get("confidence_score"),
            outcome=arguments.get("outcome"),
            reversibility=arguments.get("reversibility"),
            files_changed=arguments.get("files_changed"),
            entity_id=arguments.get("entity_id"),
            status=arguments.get("status"),
            severity=arguments.get("severity"),
            file_path=arguments.get("file_path"),
            function_name=arguments.get("function_name"),
            line_number=max(1, ln) if ln is not None else None,
            supersedes=arguments.get("supersedes"),
        )

    elif name == "brain_supersede":
        return await _brain_supersede(
            old_chunk_id=arguments["old_chunk_id"],
            new_chunk_id=arguments["new_chunk_id"],
            safety_check=arguments.get("safety_check", "auto"),
            confirm=arguments.get("confirm", False),
        )

    elif name == "brain_archive":
        return await _brain_archive(
            chunk_id=arguments["chunk_id"],
            reason=arguments.get("reason"),
        )

    elif name == "brain_get_person":
        return await _with_timeout(
            _brain_get_person(
                name=arguments["name"],
                context=arguments.get("context"),
                num_memories=arguments.get("num_memories", 10),
            )
        )

    elif name == "brain_recall":
        return await _with_timeout(
            _brain_recall(
                mode=arguments.get("mode"),
                project=arguments.get("project"),
                hours=arguments.get("hours", 24),
                days=arguments.get("days", 7),
                limit=arguments.get("limit", 20),
                session_id=arguments.get("session_id"),
                plan_name=arguments.get("plan_name"),
            )
        )

    elif name == "brain_digest":
        return await _brain_digest(
            content=arguments.get("content"),
            title=arguments.get("title"),
            project=arguments.get("project"),
            participants=arguments.get("participants"),
            mode=arguments.get("mode", "digest"),
            limit=arguments.get("limit", 25),
        )

    elif name == "brain_expand":
        ctx_n = max(0, min(arguments.get("context", 3), 20))
        return await _with_timeout(
            _context(
                chunk_id=arguments["chunk_id"],
                before=ctx_n,
                after=ctx_n,
            )
        )

    elif name == "brain_entity":
        return await _with_timeout(
            _brain_entity(
                query=arguments["query"],
                entity_type=arguments.get("entity_type"),
            )
        )

    elif name == "brain_update":
        return await _brain_update(
            action=arguments["action"],
            chunk_id=arguments["chunk_id"],
            content=arguments.get("content"),
            tags=arguments.get("tags"),
            importance=arguments.get("importance"),
            merge_chunk_ids=arguments.get("merge_chunk_ids"),
        )

    elif name == "brain_tags":
        return await _with_timeout(
            _brain_tags_mcp(
                action=arguments["action"],
                pattern=arguments.get("pattern"),
                content=arguments.get("content"),
                project=arguments.get("project"),
                limit=arguments.get("limit", 20),
            )
        )

    # --- Backward-compat aliases (old tool names route to same handlers) ---

    elif name == "brainlayer_search":
        return await _with_timeout(
            _brain_search(
                query=arguments["query"],
                project=arguments.get("project"),
                file_path=arguments.get("file_path"),
                chunk_id=arguments.get("chunk_id"),
                content_type=arguments.get("content_type"),
                source=arguments.get("source"),
                tag=arguments.get("tag"),
                intent=arguments.get("intent"),
                importance_min=arguments.get("importance_min"),
                date_from=arguments.get("date_from"),
                date_to=arguments.get("date_to"),
                sentiment=arguments.get("sentiment"),
                entity_id=arguments.get("entity_id"),
                num_results=arguments.get("num_results", 5),
                before=max(0, min(arguments.get("before", 3), 50)),
                after=max(0, min(arguments.get("after", 3), 50)),
                max_results=arguments.get("max_results", 10),
                detail=arguments.get("detail", "compact"),
            )
        )

    elif name == "brainlayer_stats":
        return await _with_timeout(_stats())

    elif name == "brainlayer_list_projects":
        return await _with_timeout(_list_projects())

    elif name == "brainlayer_context":
        return await _with_timeout(
            _context(
                chunk_id=arguments["chunk_id"],
                before=max(0, min(arguments.get("before", 3), 50)),
                after=max(0, min(arguments.get("after", 3), 50)),
            )
        )

    elif name == "brainlayer_file_timeline":
        return await _with_timeout(
            _file_timeline(
                file_path=arguments["file_path"],
                project=arguments.get("project"),
                limit=arguments.get("limit", 50),
            )
        )

    elif name == "brainlayer_operations":
        return await _with_timeout(_operations(session_id=arguments["session_id"]))

    elif name == "brainlayer_regression":
        return await _with_timeout(
            _regression(
                file_path=arguments["file_path"],
                project=arguments.get("project"),
            )
        )

    elif name == "brainlayer_plan_links":
        return await _with_timeout(
            _plan_links(
                plan_name=arguments.get("plan_name"),
                session_id=arguments.get("session_id"),
                project=arguments.get("project"),
            )
        )

    elif name == "brainlayer_think":
        return await _with_timeout(
            _think(
                context=arguments["context"],
                project=arguments.get("project"),
                max_results=arguments.get("max_results", 10),
            )
        )

    elif name == "brainlayer_recall":
        if not arguments.get("file_path") and not arguments.get("topic"):
            return _error_result("Validation error: provide at least one of 'file_path' or 'topic'.")
        return await _with_timeout(
            _recall(
                file_path=arguments.get("file_path"),
                topic=arguments.get("topic"),
                project=arguments.get("project"),
                max_results=arguments.get("max_results", 10),
            )
        )

    elif name == "brainlayer_sessions":
        return await _with_timeout(
            _sessions(
                project=arguments.get("project"),
                days=max(1, min(arguments.get("days", 7), 365)),
                limit=max(1, min(arguments.get("limit", 20), 100)),
            )
        )

    elif name == "brainlayer_current_context":
        return await _with_timeout(_current_context(hours=arguments.get("hours", 24)))

    elif name == "brainlayer_session_summary":
        return await _with_timeout(_session_summary(session_id=arguments["session_id"]))

    elif name == "brainlayer_store":
        # No timeout wrapper on writes — same reason as brain_store above.
        imp = arguments.get("importance")
        return await _store(
            content=arguments["content"],
            memory_type=arguments["type"],
            project=arguments.get("project"),
            tags=arguments.get("tags"),
            importance=max(1, min(imp, 10)) if imp is not None else None,
        )

    else:
        return _error_result(f"Unknown tool: {name}")


# --- Server entry point ---


def serve():
    """Start the MCP server using stdio.

    Note: MCP uses stdin/stdout for communication, not network ports.
    This is designed for integration with Claude Code via mcpServers config.
    """
    # Validate configuration at startup
    config_errors = validate_config()
    fatal = [e for e in config_errors if e["severity"] == "error"]
    if fatal:
        logger.error("BrainLayer MCP: %d config error(s), server may not function correctly", len(fatal))

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    serve()
