"""Shared state and utilities for MCP handlers."""

import logging
import os
import platform
import queue
import re
import subprocess
import threading
from contextlib import contextmanager

import apsw
from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

# Lazy-loaded globals with thread-safe initialization
_vector_store = None
_search_vector_store = None
_search_vector_store_pool = None
_search_vector_store_pool_handles = []
_embedding_model = None
_store_lock = threading.Lock()
_search_store_lock = threading.Lock()
_model_lock = threading.Lock()

_READ_POOL_RAM_CLAMP_KB = 768 * 1024

_SEARCH_REQUIRED_TABLES = {"chunks", "chunk_vectors"}
_SEARCH_REQUIRED_CHUNK_COLUMNS = {
    "archived",
    "chunk_origin",
    "resolved_queries",
    "status",
    "summary",
}
_SEARCH_REQUIRED_KG_ENTITY_COLUMNS = {"valid_until", "expired_at"}
_SEARCH_REQUIRED_KG_RELATION_COLUMNS = {"valid_from", "valid_until", "expired_at"}


def _search_store_needs_bootstrap(db_path) -> bool:
    """Return true when readonly search would skip required first-run migrations."""
    if not db_path.exists():
        return True

    conn = None
    try:
        conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
        cursor = conn.cursor()
        tables = {row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        if not _SEARCH_REQUIRED_TABLES.issubset(tables):
            return True

        chunk_columns = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
        if not _SEARCH_REQUIRED_CHUNK_COLUMNS.issubset(chunk_columns):
            return True

        kg_tables = {"kg_entities", "kg_relations"}
        if kg_tables & tables:
            if not kg_tables.issubset(tables):
                return True
            entity_columns = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
            relation_columns = {row[1] for row in cursor.execute("PRAGMA table_info(kg_relations)")}
            if not _SEARCH_REQUIRED_KG_ENTITY_COLUMNS.issubset(entity_columns):
                return True
            if not _SEARCH_REQUIRED_KG_RELATION_COLUMNS.issubset(relation_columns):
                return True

        return False
    finally:
        if conn is not None:
            conn.close()


def _detected_default_read_pool_size() -> int:
    """Return the platform default for the readonly WAL pool."""
    if platform.system() == "Darwin":
        try:
            brand = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                check=False,
                text=True,
                timeout=0.5,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            brand = ""
        if "Apple M1" in brand:
            return 4
    return 8


def _read_pool_size() -> int:
    raw = os.environ.get("BRAINLAYER_READ_POOL_SIZE")
    if raw is None:
        return _detected_default_read_pool_size()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _detected_default_read_pool_size()
    return value if value > 0 else _detected_default_read_pool_size()


def _assert_read_pool_ram_clamp(pool_size: int) -> None:
    from ..vector_store import _read_cache_size_kb

    read_cache_kb = abs(_read_cache_size_kb())
    total_kb = pool_size * read_cache_kb
    if total_kb > _READ_POOL_RAM_CLAMP_KB:
        raise ValueError(
            "read pool RAM clamp exceeded: "
            f"pool_size={pool_size} * read_cache_kb={read_cache_kb} = {total_kb}KB "
            f"> {_READ_POOL_RAM_CLAMP_KB}KB. Lower BRAINLAYER_READ_POOL_SIZE or BRAINLAYER_READ_CACHE_KB."
        )


def _initialize_search_vector_store_pool() -> None:
    """Pre-open the fixed readonly VectorStore pool."""
    global _search_vector_store, _search_vector_store_pool, _search_vector_store_pool_handles

    from ..paths import get_db_path
    from ..runtime_store import ReadonlyStore

    db_path = get_db_path()
    pool_size = _read_pool_size()
    _assert_read_pool_ram_clamp(pool_size)
    handles = []
    try:
        for _ in range(pool_size):
            handles.append(ReadonlyStore(db_path))
    except Exception:
        for store in handles:
            store.close()
        raise

    pool: queue.Queue = queue.Queue(maxsize=pool_size)
    for store in handles:
        pool.put(store)

    _search_vector_store_pool_handles = handles
    _search_vector_store = handles[0]
    _search_vector_store_pool = pool


def _get_vector_store(timeout: float | None = None):
    """Get or initialize the global VectorStore (thread-safe)."""
    global _vector_store
    if _vector_store is None:
        acquired = _store_lock.acquire() if timeout is None else _store_lock.acquire(timeout=max(0.0, timeout))
        if not acquired:
            raise apsw.BusyError("timed out waiting for vector store initialization")
        try:
            if _vector_store is None:
                from ..paths import get_db_path
                from ..runtime_store import open_writer_store

                _vector_store = open_writer_store(get_db_path())
        finally:
            _store_lock.release()
    return _vector_store


def _get_search_vector_store():
    """Get or initialize one read-only VectorStore for compatibility callers."""
    if _search_vector_store_pool is None:
        with _search_store_lock:
            if _search_vector_store_pool is None:
                _initialize_search_vector_store_pool()
    return _search_vector_store


@contextmanager
def _search_store_checkout():
    """Checkout a bounded readonly VectorStore handle and return it to the pool."""
    if _search_vector_store_pool is None:
        with _search_store_lock:
            if _search_vector_store_pool is None:
                _initialize_search_vector_store_pool()

    from ..vector_store import _read_busy_timeout_ms

    try:
        store = _search_vector_store_pool.get(timeout=_read_busy_timeout_ms() / 1000.0)
    except queue.Empty as exc:
        raise apsw.BusyError("timed out waiting for read pool checkout") from exc

    try:
        yield store
    finally:
        _search_vector_store_pool.put(store)


def _close_search_vector_store() -> None:
    """Close and reset the readonly search pool."""
    global _search_vector_store, _search_vector_store_pool, _search_vector_store_pool_handles
    with _search_store_lock:
        handles = list(_search_vector_store_pool_handles)
        _search_vector_store = None
        _search_vector_store_pool = None
        _search_vector_store_pool_handles = []
    for store in handles:
        store.close()


def _get_embedding_model():
    """Get or initialize the global embedding model (thread-safe)."""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:
                from ..embeddings import get_embedding_model

                _embedding_model = get_embedding_model()
    return _embedding_model


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
    return CallToolResult(content=[TextContent(type="text", text=message)], is_error=True)


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
    """Build compact search results — pointers for drill-down, not full content.

    Returns: chunk_id, score, project, date, source_file, snippet (150 chars), summary, importance.
    """
    result = {}
    for key in (
        "chunk_id",
        "score",
        "project",
        "date",
        "source_file",
        "summary",
        "importance",
        "tags",
        "provenance_class",
        "superseded_by",
    ):
        if item.get(key) is not None:
            result[key] = item[key]
    content = item.get("content", "")
    result["snippet"] = content[:150]
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
