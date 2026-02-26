"""BrainLayer MCP Server - Model Context Protocol interface for Claude Code."""

import asyncio
from typing import Any

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

from ..embeddings import get_embedding_model
from ..paths import DEFAULT_DB_PATH
from ..vector_store import VectorStore

# AIDEV-NOTE: MCP query timeout prevents indefinite hangs when DB is locked by enrichment.
# Without this, apsw blocks forever and the entire MCP server freezes, blocking Claude sessions.
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
        "Memory layer for Claude Code. 7 tools:\n"
        "- brain_search(query): semantic search across 268K+ indexed conversation chunks. "
        "Filters: project, file_path, chunk_id, content_type, tag, intent, importance_min. "
        "Routing is automatic — pass file_path for file history, chunk_id to expand context, no args for current work.\n"
        "- brain_store(content): save decisions, learnings, mistakes, ideas, todos. "
        "type is auto-detected from content. Pass importance (1-10) for critical items.\n"
        "- brain_recall(mode): session/operational context. "
        "mode=context (default, what am I working on), sessions, operations, plan, summary, stats.\n"
        "- brain_digest(content): ingest raw content (transcripts, docs, articles). "
        "Extracts entities, relations, sentiment, action items, decisions, questions. "
        "Creates a new searchable chunk with source='digest'.\n"
        "- brain_entity(query): look up a known entity in the knowledge graph. "
        "Returns entity type, relations, and evidence chunks.\n"
        "- brain_update(action, chunk_id): update, archive, or merge existing memories. "
        "action=update (change content/tags/importance), archive (soft-delete), merge (keep one, archive duplicates).\n"
        "Use brain_search at the start of tasks to retrieve past decisions and patterns.\n"
        "Use brain_store when you make a decision, hit a bug, learn something, or finish a phase.\n"
        'project scoping: auto-inferred from cwd. Override with project="all" for cross-project search.\n'
        "All 14 old brainlayer_* tool names still work (backward compat aliases)."
    ),
)

# Lazy-loaded globals
_vector_store = None
_embedding_model = None


def _normalize_project_name(project: str | None) -> str | None:
    """Normalize project names for consistent filtering.

    Handles:
    - Claude Code encoded paths: "-Users-username-Gits-myproject" → "myproject"
    - Worktree paths: "myproject-nightshift-1770775282043" → "myproject"
    - Path-like names with multiple segments
    - Already-clean names pass through unchanged
    """
    if not project:
        return None

    name = project.strip()
    if not name or name == "-":
        return None

    # Decode Claude Code path encoding
    # "-Users-username-Gits-myproject" → "myproject"
    # "-Users-username-Desktop-Gits-my-monorepo" → "my-monorepo"
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

# Write tool annotation — for brainlayer_store
_WRITE = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)


def _error_result(message: str) -> CallToolResult:
    """Return a CallToolResult with isError=True. Bypasses outputSchema validation."""
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


def _memory_to_dict(item: dict) -> dict:
    """Convert a memory item dict to structured output format."""
    d: dict = {"content": item.get("content", "")}
    for key in ("summary", "intent", "importance", "project", "content_type"):
        if item.get(key) is not None:
            d[key] = item[key]
    if item.get("created_at"):
        d["date"] = item["created_at"][:10]
    if item.get("tags") and isinstance(item["tags"], list):
        d["tags"] = [str(t) for t in item["tags"]]
    return d


# --- Routing Heuristics (Phase 4: brain_search auto-routing) ---

import re

FILE_EXTENSIONS = r"\.(ts|tsx|js|jsx|py|go|rs|java|kt|swift|rb|sh|md|json|yaml|yml|toml)$"


def _extract_file_path(query: str) -> str | None:
    """Extract a file path token from a query string."""
    tokens = query.split()
    for token in tokens:
        token = token.strip("'\"(),")
        if re.search(FILE_EXTENSIONS, token, re.IGNORECASE):
            return token
        if "/" in token and len(token) > 3:
            return token
    return None


_CURRENT_CONTEXT_SIGNALS = [
    "what am i working on",
    "what's active",
    "current task",
    "what are you working on",
    "active plan",
    "recent sessions",
    "what have i been doing",
]

_THINK_SIGNALS = [
    "how did i implement",
    "how have i",
    "past decision",
    "my pattern for",
    "how do i usually",
    "what did i decide",
    "how did we handle",
]

_RECALL_SIGNALS = [
    "history of",
    "discussed about",
    "thought about",
    "worked on",
    "context for",
    "what about",
]

_REGRESSION_SIGNALS = [
    "regression",
    "broke",
    "broken",
    "last success",
    "reverted",
    "regressed",
]


def _query_signals_current_context(query: str) -> bool:
    return any(s in query.lower() for s in _CURRENT_CONTEXT_SIGNALS)


def _query_signals_think(query: str) -> bool:
    return any(s in query.lower() for s in _THINK_SIGNALS)


def _query_signals_recall(query: str) -> bool:
    return any(s in query.lower() for s in _RECALL_SIGNALS)


def _query_has_regression_signal(query: str) -> bool:
    return any(s in query.lower() for s in _REGRESSION_SIGNALS)


# --- Auto-type detection for brain_store ---

_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("todo", [r"\bTODO\b", r"\bFIXME\b", r"\bHACK\b", r"^TODO:", r"add\b.*\bsoon\b"]),
    (
        "mistake",
        [
            r"\bBug\b",
            r"\bError:\b",
            r"\bbroke\b",
            r"\bbroken\b",
            r"\boverflow\b",
            r"\bmistake\b",
            r"\bwrong\b",
            r"\bfailed\b",
            r"\bregress",
        ],
    ),
    (
        "decision",
        [
            r"\bAlways\b",
            r"\bNever\b",
            r"\bshould\b.*\binstead\b",
            r"\bdecided\b",
            r"\bprefer\b",
            r"\buse\b.*\bnot\b",
            r"\bconvention\b",
            r"\brule:\b",
        ],
    ),
    (
        "learning",
        [
            r"\blearned\b",
            r"\brealized\b",
            r"\bturns out\b",
            r"\bdiscovered\b",
            r"\bfound out\b",
            r"\bnow I know\b",
        ],
    ),
    ("bookmark", [r"https?://", r"github\.com", r"docs\.", r"\.dev\b"]),
    (
        "idea",
        [
            r"\bidea:\b",
            r"\bwhat if\b",
            r"\bcould\b.*\bbuild\b",
            r"\bmaybe\b.*\badd\b",
            r"\bfeature idea\b",
        ],
    ),
    (
        "journal",
        [
            r"\btoday\b",
            r"\bthis week\b",
            r"\bworked on\b",
            r"\bfinished\b",
            r"\bshipped\b",
        ],
    ),
]


def _detect_memory_type(content: str) -> str:
    """Detect memory type from content using keyword patterns. No LLM call."""
    for memory_type, patterns in _TYPE_RULES:
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return memory_type
    return "note"


# --- Auto-importance scoring for brain_store ---

_ARCH_KEYWORDS = [
    "database",
    "schema",
    "migration",
    "auth",
    "security",
    "api",
    "deploy",
    "infrastructure",
    "architecture",
    "pipeline",
    "config",
]
_PROHIBITION_KEYWORDS = ["never", "always", "must", "critical", "important", "do not", "don't"]


def _auto_importance(content: str) -> int:
    """Keyword-based importance scoring. No LLM call.

    Baseline 3, cap 10. Only used when user doesn't provide explicit importance.
    """
    score = 3
    lower = content.lower()

    # Architectural keywords: +3 (once)
    if any(kw in lower for kw in _ARCH_KEYWORDS):
        score += 3

    # Prohibition/imperative keywords: +2 (once)
    if any(kw in lower for kw in _PROHIBITION_KEYWORDS):
        score += 2

    # Long content (>100 chars): +1
    if len(content) > 100:
        score += 1

    # File path reference: +1
    if re.search(r"[\w/]+\.\w{1,5}", content):
        score += 1

    return min(score, 10)


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


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools — 3 consolidated tools (Phase 4)."""
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
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="brain_store",
            title="Store Memory",
            description="""Persistently store a memory into BrainLayer.

Use this to save ideas, mistakes, decisions, learnings, todos, bookmarks, notes, or journal entries. Stored items are embedded at write time and immediately searchable.

Type is auto-detected from content if omitted (e.g., "Always use bun" → decision, "Bug: overflow" → mistake, "TODO: add X" → todo).

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
                        "enum": ["idea", "mistake", "decision", "learning", "todo", "bookmark", "note", "journal"],
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
            description="""Ingest raw content (transcripts, documents, articles) and extract structured knowledge.

Creates a new chunk with source="digest", extracts entities and relations (KG),
analyzes sentiment, and identifies action items, decisions, and questions.

Returns: Structured JSON with digest_id, summary, entities, relations,
action_items, decisions, questions, sentiment, and stats.""",
            annotations=_WRITE,
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Raw text content to digest (transcript, document, article, meeting notes)",
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
                },
                "required": ["content"],
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
    ]


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


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]):
    """Handle tool calls — 3 new tools + 14 backward-compat aliases."""

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
            )
        )

    elif name == "brain_store":
        # AIDEV-NOTE: No timeout wrapper on writes — non-idempotent (UUID-based IDs).
        # If timeout fires but executor thread completes, user retries = duplicate memory.
        imp = arguments.get("importance")
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
            content=arguments["content"],
            title=arguments.get("title"),
            project=arguments.get("project"),
            participants=arguments.get("participants"),
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

    # --- Backward-compat aliases (old tool names route to same handlers) ---

    elif name == "brainlayer_search":
        return await _with_timeout(
            _search(
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
                sentiment=arguments.get("sentiment"),
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
        # AIDEV-NOTE: No timeout wrapper on writes — same reason as brain_store above.
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


# --- Phase 3: Brain Digest + Brain Entity ---


async def _brain_digest(
    content: str,
    title: str | None = None,
    project: str | None = None,
    participants: list[str] | None = None,
) -> CallToolResult:
    """Handle brain_digest tool call."""
    import json

    from ..pipeline.digest import digest_content

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_event_loop()
    norm_project = _normalize_project_name(project) if project else None

    try:
        result = await loop.run_in_executor(
            None,
            lambda: digest_content(
                content=content,
                store=store,
                embed_fn=model.embed,
                title=title,
                project=norm_project,
                participants=participants,
            ),
        )
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])
    except ValueError as e:
        return _error_result(str(e))
    except Exception as e:
        return _error_result(f"Digest failed: {e}")


async def _brain_entity(
    query: str,
    entity_type: str | None = None,
) -> CallToolResult:
    """Handle brain_entity tool call."""
    import json

    from ..pipeline.digest import entity_lookup

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: entity_lookup(
                query=query,
                store=store,
                embed_fn=model.embed,
                entity_type=entity_type,
            ),
        )
    except Exception as e:
        return _error_result(f"Entity lookup failed: {e}")

    if result is None:
        return CallToolResult(content=[TextContent(type="text", text=f"No entity found matching '{query}'.")])
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])


async def _brain_get_person(
    name: str,
    context: str | None = None,
    num_memories: int = 10,
) -> CallToolResult:
    """Composite tool: look up a person entity + retrieve their scoped memories.

    Returns structured JSON with:
    - profile: entity metadata (constraints, preferences, contact info)
    - relations: entity relations from KG
    - memories: relevant memory chunks linked to this person
    """
    import json

    from ..pipeline.digest import entity_lookup

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_event_loop()

    # Step 1: Look up the person entity
    try:
        entity = await loop.run_in_executor(
            None,
            lambda: entity_lookup(
                query=name,
                store=store,
                embed_fn=model.embed_query,
                entity_type="person",
            ),
        )
    except Exception as e:
        return _error_result(f"Person lookup failed: {e}")

    if entity is None:
        return CallToolResult(content=[TextContent(type="text", text=f"No person entity found matching '{name}'.")])

    entity_id = entity["id"]

    # Step 2: Get per-person scoped memories
    memories = []
    try:
        if context:
            # If context provided, do semantic search scoped to this person's chunks
            query_embedding = await loop.run_in_executor(None, model.embed_query, context)
            results = await loop.run_in_executor(
                None,
                lambda: store.hybrid_search(
                    query_embedding=query_embedding,
                    query_text=context,
                    n_results=num_memories,
                    entity_id=entity_id,
                ),
            )
            if results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    memories.append(
                        {
                            "content": doc[:500],
                            "type": meta.get("content_type", "unknown"),
                            "date": meta.get("created_at", "")[:10] if meta.get("created_at") else None,
                            "summary": meta.get("summary"),
                        }
                    )
        else:
            # No context: return entity's linked chunks ordered by relevance
            entity_chunks = await loop.run_in_executor(
                None,
                lambda: store.get_entity_chunks(entity_id, limit=num_memories),
            )
            for chunk in entity_chunks:
                memories.append(
                    {
                        "content": chunk["content"][:500] if chunk.get("content") else "",
                        "type": chunk.get("content_type", "unknown"),
                        "date": chunk.get("created_at", "")[:10] if chunk.get("created_at") else None,
                        "relevance": chunk.get("relevance"),
                    }
                )
    except Exception as e:
        logger.warning("Memory retrieval for person '%s' failed: %s", name, e)

    # Step 3: Build composite result
    metadata = entity.get("metadata", {})
    result = {
        "entity_id": entity_id,
        "name": entity["name"],
        "profile": metadata,
        "hard_constraints": metadata.get("hard_constraints", {}),
        "preferences": metadata.get("preferences", {}),
        "contact_info": metadata.get("contact_info", {}),
        "relations": entity.get("relations", []),
        "memories": memories,
        "memory_count": len(memories),
    }

    return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])


# --- Consolidated Dispatchers (Phase 4) ---


async def _brain_search(
    query: str,
    project: str | None = None,
    file_path: str | None = None,
    chunk_id: str | None = None,
    content_type: str | None = None,
    source: str | None = None,
    tag: str | None = None,
    intent: str | None = None,
    importance_min: float | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sentiment: str | None = None,
    entity_id: str | None = None,
    num_results: int = 5,
    before: int = 3,
    after: int = 3,
    max_results: int = 10,
):
    """Unified search dispatcher — routes to the right internal handler.

    Args:
        entity_id: If provided, only return chunks linked to this entity.
                   Bypasses most routing rules and goes straight to hybrid search
                   with entity scoping. Used for per-person memory retrieval.
    """

    # Auto-scope project from CWD if not provided — but ONLY for claude_code source.
    # Non-claude_code sources (youtube, whatsapp, etc.) have null/different project values,
    # so auto-scoping filters them out entirely (bug: brain_search(source="youtube") → 0 results).
    # Also skip auto-scope when entity_id is set (entity-scoped search is cross-project).
    if project is None and entity_id is None and source not in ("youtube", "whatsapp", "telegram", "all"):
        try:
            from ..scoping import resolve_project_scope

            project = resolve_project_scope()
        except Exception:
            pass  # Scoping failure should never block search

    # Entity-scoped search: skip routing rules, go straight to hybrid search
    if entity_id is not None:
        return await _search(
            query=query,
            project=project,
            content_type=content_type,
            num_results=num_results,
            source=source,
            tag=tag,
            intent=intent,
            importance_min=importance_min,
            date_from=date_from,
            date_to=date_to,
            sentiment=sentiment,
            entity_id=entity_id,
        )

    # Rule 1: chunk context expand
    if chunk_id is not None:
        return await _context(chunk_id=chunk_id, before=before, after=after)

    # Rule 2: file_path + regression signals
    if file_path is not None and _query_has_regression_signal(query):
        regression_result = await _regression(file_path=file_path, project=project)
        recall_result = await _recall(file_path=file_path, project=project, max_results=max_results)
        # Merge: regression markdown + recall related chunks
        merged_text = []
        if isinstance(regression_result, list):
            merged_text.extend(regression_result)
        if isinstance(recall_result, tuple):
            merged_text.extend(recall_result[0])
        else:
            merged_text.extend(recall_result)
        return merged_text

    # Rule 3: file_path (no regression signal)
    if file_path is not None:
        timeline = await _file_timeline(file_path=file_path, project=project, limit=50)
        recall_result = await _recall(file_path=file_path, project=project, max_results=max_results)
        merged_text = []
        if isinstance(timeline, list):
            merged_text.extend(timeline)
        if isinstance(recall_result, tuple):
            merged_text.extend(recall_result[0])
        else:
            merged_text.extend(recall_result)
        return merged_text

    # Rule 4: file token in query text
    extracted_file = _extract_file_path(query)
    if extracted_file:
        return await _brain_search(
            query=query,
            project=project,
            file_path=extracted_file,
            content_type=content_type,
            source=source,
            tag=tag,
            intent=intent,
            importance_min=importance_min,
            date_from=date_from,
            date_to=date_to,
            sentiment=sentiment,
            num_results=num_results,
            max_results=max_results,
        )

    # Rule 5: current context signals
    if _query_signals_current_context(query):
        ctx = await _current_context(hours=24)
        think_result = await _think(context=query, project=project, max_results=max_results)
        merged_text = []
        if isinstance(ctx, tuple):
            merged_text.extend(ctx[0])
        else:
            merged_text.extend(ctx)
        if isinstance(think_result, tuple):
            merged_text.extend(think_result[0])
        else:
            merged_text.extend(think_result)
        return merged_text

    # Rule 6: think mode signals
    if _query_signals_think(query):
        return await _think(context=query, project=project, max_results=max_results)

    # Rule 7: recall/history signals
    if _query_signals_recall(query):
        return await _recall(topic=query, project=project, max_results=max_results)

    # Rule 8: default — hybrid semantic + FTS5 search
    return await _search(
        query=query,
        project=project,
        content_type=content_type,
        num_results=num_results,
        source=source,
        tag=tag,
        intent=intent,
        importance_min=importance_min,
        date_from=date_from,
        date_to=date_to,
        sentiment=sentiment,
    )


def _infer_recall_mode(arguments: dict) -> str:
    """Auto-infer recall mode from provided arguments."""
    if arguments.get("session_id") and not arguments.get("plan_name"):
        return "summary"
    if arguments.get("plan_name"):
        return "plan"
    if arguments.get("days") or arguments.get("limit"):
        return "sessions"
    return "context"


async def _brain_recall(
    mode: str | None = None,
    project: str | None = None,
    hours: int = 24,
    days: int = 7,
    limit: int = 20,
    session_id: str | None = None,
    plan_name: str | None = None,
):
    """Unified recall dispatcher — routes to session/context handlers."""

    resolved_mode = mode or _infer_recall_mode(
        {
            "session_id": session_id,
            "plan_name": plan_name,
            "days": days if days != 7 else None,
            "limit": limit if limit != 20 else None,
        }
    )

    if resolved_mode == "context":
        return await _current_context(hours=hours)

    elif resolved_mode == "sessions":
        return await _sessions(
            project=project,
            days=max(1, min(days, 365)),
            limit=max(1, min(limit, 100)),
        )

    elif resolved_mode == "operations":
        if not session_id:
            return _error_result("session_id required for mode=operations")
        return await _operations(session_id=session_id)

    elif resolved_mode == "plan":
        return await _plan_links(plan_name=plan_name, session_id=session_id, project=project)

    elif resolved_mode == "summary":
        if not session_id:
            return _error_result("session_id required for mode=summary")
        return await _session_summary(session_id=session_id)

    elif resolved_mode == "stats":
        stats_result = await _stats()
        projects_result = await _list_projects()
        # Merge stats structured + projects markdown
        merged_text = []
        if isinstance(stats_result, tuple):
            merged_text.extend(stats_result[0])
        else:
            merged_text.extend(stats_result)
        if isinstance(projects_result, list):
            merged_text.extend(projects_result)
        return merged_text

    else:
        return _error_result(f"Unknown recall mode: {resolved_mode}")


async def _store_new(
    content: str,
    memory_type: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    confidence_score: float | None = None,
    outcome: str | None = None,
    reversibility: str | None = None,
    files_changed: list[str] | None = None,
    entity_id: str | None = None,
):
    """Wrapper for _store with auto-type detection and auto-importance."""
    resolved_type = memory_type or _detect_memory_type(content)
    resolved_importance = importance if importance is not None else _auto_importance(content)
    return await _store(
        content=content,
        memory_type=resolved_type,
        project=project,
        tags=tags,
        importance=resolved_importance,
        confidence_score=confidence_score,
        outcome=outcome,
        reversibility=reversibility,
        files_changed=files_changed,
        entity_id=entity_id,
    )


async def _brain_update(
    action: str,
    chunk_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    merge_chunk_ids: list[str] | None = None,
):
    """Update, archive, or merge memories."""
    try:
        store = _get_vector_store()

        if action == "archive":
            ok = store.archive_chunk(chunk_id)
            if not ok:
                return _error_result(f"Chunk not found: {chunk_id}")
            return [TextContent(
                type="text",
                text=json.dumps({"action": "archived", "chunk_id": chunk_id}),
            )]

        elif action == "update":
            # Verify chunk exists
            existing = store.get_chunk(chunk_id)
            if not existing:
                return _error_result(f"Chunk not found: {chunk_id}")

            # Re-embed if content changed
            embedding = None
            if content is not None:
                loop = asyncio.get_running_loop()
                model = _get_embedding_model()
                embedding = await loop.run_in_executor(None, model.embed_query, content)

            ok = store.update_chunk(
                chunk_id=chunk_id,
                content=content,
                tags=tags,
                importance=float(importance) if importance is not None else None,
                embedding=embedding,
            )
            if not ok:
                return _error_result(f"Update failed for: {chunk_id}")

            result = {"action": "updated", "chunk_id": chunk_id, "fields": []}
            if content is not None:
                result["fields"].append("content")
            if tags is not None:
                result["fields"].append("tags")
            if importance is not None:
                result["fields"].append("importance")
            return [TextContent(type="text", text=json.dumps(result))]

        elif action == "merge":
            if not merge_chunk_ids:
                return _error_result("merge requires merge_chunk_ids (the duplicates to archive)")

            # Verify the keeper exists
            keeper = store.get_chunk(chunk_id)
            if not keeper:
                return _error_result(f"Keeper chunk not found: {chunk_id}")

            archived = []
            failed = []
            for dup_id in merge_chunk_ids:
                ok = store.archive_chunk(dup_id)
                if ok:
                    archived.append(dup_id)
                else:
                    failed.append(dup_id)

            result = {
                "action": "merged",
                "kept": chunk_id,
                "archived": archived,
                "failed": failed,
            }
            return [TextContent(type="text", text=json.dumps(result))]

        else:
            return _error_result(f"Unknown action: {action}. Use update, archive, or merge.")

    except Exception as e:
        logger.error("brain_update failed: %s", e)
        return _error_result(f"brain_update error: {e}")


# --- Original Handler Functions ---


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
    sentiment: str | None = None,
    entity_id: str | None = None,
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
        normalized_project = _normalize_project_name(project)

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

        # When searching by entity_id, skip source_filter default (entity memories
        # may come from any source: manual, digest, claude_code, etc.)
        if entity_id and not source:
            source_filter = None

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
            sentiment_filter=sentiment,
            entity_id=entity_id,
        )

        if not results["documents"][0]:
            empty = {"query": query, "total": 0, "results": []}
            return ([TextContent(type="text", text="No results found.")], empty)

        # Enrich results with session-level context (Phase 7)
        results = store.enrich_results_with_session_context(results)

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
                "project": _normalize_project_name(meta.get("project")) or meta.get("project", "unknown"),
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
            # Session-level enrichment (Phase 7)
            if meta.get("session_summary"):
                item["session_summary"] = meta["session_summary"]
            if meta.get("session_outcome"):
                item["session_outcome"] = meta["session_outcome"]
            if meta.get("session_quality") is not None:
                item["session_quality"] = meta["session_quality"]
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
            if meta.get("session_summary"):
                output_parts.append(f"**Session:** {meta['session_summary'][:200]}")
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
        return _error_result(f"Search error (query='{query[:50]}...'): {str(e)}")


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
        return _error_result(f"Stats error: {str(e)}")


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
        return _error_result(f"Error listing projects: {str(e)}")


async def _context(chunk_id: str, before: int = 3, after: int = 3) -> list[TextContent]:
    """Get surrounding conversation context for a chunk."""
    try:
        store = _get_vector_store()
        result = store.get_context(chunk_id, before=before, after=after)

        if result.get("error"):
            return _error_result(f"Unknown chunk_id '{chunk_id[:20]}...'. Use chunk_id from brainlayer_search results.")

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
        return _error_result(f"Context error: {str(e)}")


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
        return _error_result(f"File timeline error: {str(e)}")


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
        return _error_result(f"Operations error: {str(e)}")


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
        return _error_result(f"Regression error: {str(e)}")


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
        return _error_result(f"Plan links error: {str(e)}")


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
        normalized_project = _normalize_project_name(project)

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
        return _error_result(f"Think error: {str(e)}")


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
        normalized_project = _normalize_project_name(project)

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
        return _error_result(f"Recall error: {str(e)}")


async def _sessions(
    project: str | None = None,
    days: int = 7,
    limit: int = 20,
) -> list[TextContent]:
    """Execute sessions — list recent sessions."""
    try:
        from ..engine import format_sessions, sessions

        store = _get_vector_store()
        normalized_project = _normalize_project_name(project)

        result = sessions(
            store=store,
            project=normalized_project,
            days=days,
            limit=limit,
        )

        return [TextContent(type="text", text=format_sessions(result, days=days))]

    except Exception as e:
        return _error_result(f"Sessions error: {str(e)}")


async def _session_summary(session_id: str):
    """Get enriched session summary."""
    try:
        store = _get_vector_store()
        enrichment = store.get_session_enrichment(session_id)

        if not enrichment:
            return [
                TextContent(
                    type="text",
                    text=f"No enrichment data for session '{session_id[:8]}...'. Run 'brainlayer enrich-sessions' first.",
                )
            ]

        parts = [f"## Session Summary: {session_id[:8]}...\n"]

        if enrichment.get("session_summary"):
            parts.append(f"**Summary:** {enrichment['session_summary']}\n")
        if enrichment.get("primary_intent"):
            parts.append(f"**Intent:** {enrichment['primary_intent']}")
        if enrichment.get("outcome"):
            parts.append(f"**Outcome:** {enrichment['outcome']}")
        if enrichment.get("session_quality_score"):
            parts.append(f"**Quality:** {enrichment['session_quality_score']}/10")
        if enrichment.get("complexity_score"):
            parts.append(f"**Complexity:** {enrichment['complexity_score']}/10")
        if enrichment.get("duration_seconds"):
            mins = enrichment["duration_seconds"] // 60
            parts.append(f"**Duration:** {mins} min")
        parts.append(
            f"**Messages:** {enrichment.get('message_count', 0)} "
            f"(user: {enrichment.get('user_message_count', 0)}, "
            f"assistant: {enrichment.get('assistant_message_count', 0)})\n"
        )

        if enrichment.get("decisions_made"):
            parts.append("### Decisions")
            for d in enrichment["decisions_made"]:
                if isinstance(d, dict):
                    parts.append(f"- {d.get('decision', '?')} — *{d.get('rationale', '')}*")
                else:
                    parts.append(f"- {d}")

        if enrichment.get("corrections"):
            parts.append("\n### Corrections")
            for c in enrichment["corrections"]:
                if isinstance(c, dict):
                    parts.append(f"- Wrong: {c.get('what_was_wrong', '?')} → Wanted: {c.get('what_user_wanted', '?')}")
                else:
                    parts.append(f"- {c}")

        if enrichment.get("learnings"):
            parts.append("\n### Learnings")
            for l in enrichment["learnings"]:
                parts.append(f"- {l}")

        if enrichment.get("mistakes"):
            parts.append("\n### Mistakes")
            for m in enrichment["mistakes"]:
                parts.append(f"- {m}")

        if enrichment.get("what_worked"):
            parts.append(f"\n**What worked:** {enrichment['what_worked']}")
        if enrichment.get("what_failed"):
            parts.append(f"**What failed:** {enrichment['what_failed']}")

        if enrichment.get("topic_tags"):
            parts.append(f"\n**Tags:** {', '.join(enrichment['topic_tags'][:10])}")

        return [TextContent(type="text", text="\n".join(parts))]

    except Exception as e:
        return _error_result(f"Session summary error: {str(e)}")


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
        return _error_result(f"Current context error: {str(e)}")


def _get_pending_store_path():
    """Path for the store queue buffer file."""
    from ..paths import DEFAULT_DB_PATH

    return DEFAULT_DB_PATH.parent / "pending-stores.jsonl"


def _queue_store(item: dict) -> None:
    """Buffer a store request to JSONL when DB is locked."""
    import json as _json

    path = _get_pending_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(_json.dumps(item) + "\n")


def _flush_pending_stores(store, embed_fn) -> int:
    """Flush pending-stores.jsonl (FIFO). Returns count flushed."""
    import json as _json

    from ..store import store_memory

    path = _get_pending_store_path()
    if not path.exists():
        return 0

    try:
        lines = path.read_text().strip().splitlines()
    except Exception:
        return 0

    if not lines:
        return 0

    flushed = 0
    remaining = []
    for line in lines:
        try:
            item = _json.loads(line)
            store_memory(
                store=store,
                embed_fn=embed_fn,
                content=item["content"],
                memory_type=item["memory_type"],
                project=item.get("project"),
                tags=item.get("tags"),
                importance=item.get("importance"),
                confidence_score=item.get("confidence_score"),
                outcome=item.get("outcome"),
                reversibility=item.get("reversibility"),
                files_changed=item.get("files_changed"),
                entity_id=item.get("entity_id"),
            )
            flushed += 1
        except Exception:
            remaining.append(line)

    # Rewrite file with only failed items
    if remaining:
        path.write_text("\n".join(remaining) + "\n")
    else:
        path.unlink(missing_ok=True)

    return flushed


async def _store(
    content: str,
    memory_type: str,
    project: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    confidence_score: float | None = None,
    outcome: str | None = None,
    reversibility: str | None = None,
    files_changed: list[str] | None = None,
    entity_id: str | None = None,
):
    """Store a memory into BrainLayer. Buffers to JSONL on DB lock."""
    try:
        from ..store import store_memory

        store = _get_vector_store()
        model = _get_embedding_model()
        normalized_project = _normalize_project_name(project)

        loop = asyncio.get_running_loop()

        def _embed(text: str) -> list[float]:
            return model.embed_query(text)

        result = await loop.run_in_executor(
            None,
            lambda: store_memory(
                store=store,
                embed_fn=_embed,
                content=content,
                memory_type=memory_type,
                project=normalized_project,
                tags=tags,
                importance=importance,
                confidence_score=confidence_score,
                outcome=outcome,
                reversibility=reversibility,
                files_changed=files_changed,
                entity_id=entity_id,
            ),
        )

        # Try flushing any pending stores on success
        try:
            flushed = await loop.run_in_executor(None, lambda: _flush_pending_stores(store, _embed))
        except Exception:
            flushed = 0

        # Format text response
        chunk_id = result["id"]
        parts = [f"Stored memory `{chunk_id}`"]
        if flushed > 0:
            parts.append(f"(also flushed {flushed} queued items)")
        if result["related"]:
            parts.append(f"\n**Related memories ({len(result['related'])}):**")
            for r in result["related"]:
                summary = r.get("summary") or r.get("content", "")[:100]
                parts.append(f"- {summary}")

        structured = {
            "chunk_id": chunk_id,
            "related": result["related"],
        }
        return ([TextContent(type="text", text="\n".join(parts))], structured)

    except ValueError as e:
        return _error_result(f"Validation error: {str(e)}")
    except Exception as e:
        # Check if this is a DB lock error — queue instead of failing
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            _queue_store(
                {
                    "content": content,
                    "memory_type": memory_type,
                    "project": _normalize_project_name(project),
                    "tags": tags,
                    "importance": importance,
                    "confidence_score": confidence_score,
                    "outcome": outcome,
                    "reversibility": reversibility,
                    "files_changed": files_changed,
                    "entity_id": entity_id,
                }
            )
            structured = {"chunk_id": "queued", "related": []}
            return (
                [TextContent(type="text", text="Memory queued (DB busy). Will flush on next successful store.")],
                structured,
            )
        return _error_result(f"Store error: {str(e)}")


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
