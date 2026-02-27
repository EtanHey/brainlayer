"""Shared state and utilities for MCP handlers."""

import logging
import os
import re

from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

# Lazy-loaded globals
_vector_store = None
_embedding_model = None


def _get_vector_store():
    """Get or initialize the global VectorStore."""
    global _vector_store
    if _vector_store is None:
        from ..paths import DEFAULT_DB_PATH
        from ..vector_store import VectorStore

        _vector_store = VectorStore(DEFAULT_DB_PATH)
    return _vector_store


def _get_embedding_model():
    """Get or initialize the global embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from ..embeddings import get_embedding_model

        _embedding_model = get_embedding_model()
    return _embedding_model


def validate_config() -> list[dict]:
    """Validate startup configuration. Returns list of {field, message, severity} dicts."""
    errors = []
    from ..paths import DEFAULT_DB_PATH

    # Check DB file exists and is readable
    if not DEFAULT_DB_PATH.exists():
        errors.append(
            {
                "field": "database",
                "message": f"Database not found at {DEFAULT_DB_PATH}. Run 'brainlayer index' first.",
                "severity": "warning",
            }
        )
    elif not os.access(DEFAULT_DB_PATH, os.R_OK):
        errors.append(
            {
                "field": "database",
                "message": f"Database not readable: {DEFAULT_DB_PATH}",
                "severity": "error",
            }
        )

    # Check DB parent directory is writable (for WAL mode)
    db_parent = DEFAULT_DB_PATH.parent
    if db_parent.exists() and not os.access(db_parent, os.W_OK):
        errors.append(
            {
                "field": "database_dir",
                "message": f"Database directory not writable: {db_parent}",
                "severity": "error",
            }
        )

    # Check embedding model availability
    try:
        from ..embeddings import get_embedding_model

        get_embedding_model()
    except Exception as e:
        errors.append(
            {
                "field": "embedding_model",
                "message": f"Embedding model failed to load: {e}",
                "severity": "error",
            }
        )

    for err in errors:
        level = logging.ERROR if err["severity"] == "error" else logging.WARNING
        logger.log(level, "Config: [%s] %s", err["field"], err["message"])

    return errors


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
    name = re.sub(r"-(?:nightshift|haiku|worktree)-\d+$", "", name)

    return name


def _error_result(message: str):
    """Create an error CallToolResult."""
    return CallToolResult(content=[TextContent(type="text", text=message)], isError=True)


def _memory_to_dict(item: dict) -> dict:
    """Convert a memory item dict to structured output format."""
    result = {"content": item.get("content", "")}
    for key in ("summary", "intent", "importance", "project", "content_type", "tags"):
        if item.get(key) is not None:
            result[key] = item[key]
    if item.get("created_at"):
        result["date"] = item["created_at"][:10] if len(item.get("created_at", "")) >= 10 else item["created_at"]
    return result


FILE_EXTENSIONS = re.compile(
    r"\.(py|js|ts|tsx|jsx|rs|go|java|c|cpp|h|hpp|rb|php|swift|kt|scala|sh|bash|zsh"
    r"|yaml|yml|json|toml|ini|cfg|conf|xml|html|css|scss|md|txt|sql|prisma|graphql"
    r"|dockerfile|makefile|gitignore|env|lock)$",
    re.IGNORECASE,
)


def _extract_file_path(query: str) -> str | None:
    """Extract a file path from short query strings."""
    if not query or len(query.split()) > 2:
        return None
    for token in query.split():
        if FILE_EXTENSIONS.search(token):
            return token
    return None


# Query signal lists
_CURRENT_CONTEXT_SIGNALS = [
    "what am i working on",
    "what was i doing",
    "current context",
    "what's happening",
    "what is happening",
    "where was i",
    "catch me up",
    "status update",
]

_THINK_SIGNALS = [
    "how did i",
    "how do i",
    "what approach",
    "best practice",
    "pattern for",
    "similar to",
    "previously",
    "last time",
]

_RECALL_SIGNALS = [
    "history of",
    "discussed about",
    "thought about",
]

_REGRESSION_SIGNALS = [
    "broke",
    "broken",
    "regression",
    "stopped working",
    "was working",
    "used to work",
    "no longer works",
]


def _query_signals_current_context(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _CURRENT_CONTEXT_SIGNALS)


def _query_signals_think(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _THINK_SIGNALS)


def _query_signals_recall(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _RECALL_SIGNALS)


def _query_has_regression_signal(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in _REGRESSION_SIGNALS)


def _build_compact_result(item: dict) -> dict:
    """Build compact search results (~40% token savings)."""
    result = {}
    for key in ("score", "project", "source_file", "date", "importance", "summary"):
        if item.get(key) is not None:
            result[key] = item[key]
    content = item.get("content", "")
    result["content"] = content[:500]
    return result


# Auto-type detection rules (regex-based)
_TYPE_RULES: list[tuple[str, list[str]]] = [
    (
        "issue",
        [
            r"^Issue:",
            r"\bissue\b.*\b(?:with|in|when|on)\b",
            r"\bblocking\b",
            r"\bblocker\b",
            r"\bcrashes?\b",
            r"\bfails?\b.*\bwhen\b",
            r"\bseverity\b",
            r"\bP[0-3]\b",
        ],
    ),
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


def _detect_memory_type(content: str) -> str:
    """Detect memory type from content using regex patterns. No LLM call."""
    for memory_type, patterns in _TYPE_RULES:
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return memory_type
    return "note"


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
