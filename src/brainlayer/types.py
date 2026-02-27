"""Shared type definitions for BrainLayer.

Replaces Dict[str, Any] returns with proper TypedDict models across
the codebase: vector_store, MCP handlers, daemon, and engine.
"""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict

# --- Chunk types ---


class ChunkDict(TypedDict):
    """A single chunk as returned by VectorStore.get_chunk()."""

    id: str
    content: str
    metadata: str  # JSON string
    source_file: Optional[str]
    project: Optional[str]
    content_type: Optional[str]
    value_type: Optional[str]
    tags: Optional[str]  # JSON string
    importance: Optional[float]
    created_at: Optional[str]
    summary: Optional[str]


class ChunkMetadata(TypedDict, total=False):
    """Metadata dict attached to search result chunks."""

    source_file: str
    project: str
    content_type: str
    value_type: str
    char_count: int
    summary: str
    tags: list[str]
    importance: float
    intent: str
    created_at: str
    source: str
    session_summary: str
    session_outcome: str
    session_quality_score: float


# --- Search result types ---


class SearchResults(TypedDict):
    """Return type of VectorStore.search() and hybrid_search()."""

    ids: list[list[str]]
    documents: list[list[str]]
    metadatas: list[list[dict[str, Any]]]
    distances: list[list[Optional[float]]]


class SearchResultItem(TypedDict, total=False):
    """A single search result as returned by MCP brain_search."""

    score: float
    project: str
    content_type: str
    content: str
    source_file: str
    date: str
    source: str
    summary: str
    tags: list[str]
    intent: str
    importance: float
    chunk_id: str


class SearchResponse(TypedDict):
    """Structured response from MCP brain_search."""

    query: str
    total: int
    results: list[SearchResultItem]


# --- Stats types ---


class StatsResponse(TypedDict):
    """Return type of VectorStore.get_stats() and MCP brain_search(query='stats')."""

    total_chunks: int
    projects: list[str]
    content_types: list[str]


# --- Store types ---


class StoreRelated(TypedDict, total=False):
    """A related memory returned alongside a newly stored memory."""

    content: str
    summary: str
    project: str
    type: str
    date: str


class StoreResponse(TypedDict):
    """Structured response from MCP brain_store."""

    chunk_id: str
    related: list[StoreRelated]


# --- Recall types ---


class RecallFileHistory(TypedDict, total=False):
    """A file history entry from recall mode."""

    timestamp: str
    action: str
    session_id: str
    file_path: str


class RecallSessionSummary(TypedDict, total=False):
    """A session summary from recall mode."""

    session_id: str
    branch: str
    plan_name: str
    started_at: str


class RecallResponse(TypedDict):
    """Structured response from MCP brain_recall."""

    target: str
    file_history: list[RecallFileHistory]
    related_chunks: list[dict[str, Any]]
    session_summaries: list[RecallSessionSummary]


# --- Think types ---


class ThinkResponse(TypedDict):
    """Structured response from MCP brain_search think mode."""

    query: str
    total: int
    decisions: list[dict[str, Any]]
    patterns: list[dict[str, Any]]
    bugs: list[dict[str, Any]]
    context: list[dict[str, Any]]


# --- Current context types ---


class CurrentContextSession(TypedDict, total=False):
    """A session in current context response."""

    session_id: str
    project: str
    branch: str
    started_at: str
    plan_name: str


class CurrentContextResponse(TypedDict):
    """Structured response from current context mode."""

    active_projects: list[str]
    active_branches: list[str]
    active_plan: str
    recent_files: list[str]
    recent_sessions: list[CurrentContextSession]


# --- Entity types ---


class EntityDict(TypedDict, total=False):
    """An entity from the knowledge graph."""

    id: str
    entity_type: str
    name: str
    metadata: str  # JSON string
    source: str
    confidence: float
    first_seen: str
    last_seen: str
    mention_count: int


class EntityRelation(TypedDict, total=False):
    """A relation between entities."""

    relation_id: str
    relation_type: str
    target_id: str
    target_name: str
    target_type: str
    confidence: float
    source: str
    valid_from: str
    valid_to: Optional[str]
    direction: str


# --- Session types ---


class SessionContext(TypedDict, total=False):
    """Git overlay / session context."""

    session_id: str
    branch: str
    pr_number: int
    commit_count: int
    files_changed: str  # JSON string
    started_at: str
    plan_name: str
    plan_phase: str
    story_id: str


class SessionEnrichment(TypedDict, total=False):
    """Session-level enrichment data."""

    session_id: str
    summary: str
    outcome: str
    quality_score: float
    key_decisions: list[str]
    mistakes: list[str]
    tools_used: list[str]
    files_touched: list[str]
    intent: str
    enriched_at: str


# --- KG hybrid search types ---


class KGFact(TypedDict, total=False):
    """A fact from the knowledge graph used in hybrid retrieval."""

    entity: str
    entity_type: str
    relation: str
    target: str
    target_type: str
    confidence: float


class KGHybridResult(TypedDict):
    """Result from kg_hybrid_search combining chunks and KG facts."""

    chunks: list[dict[str, Any]]
    facts: list[KGFact]
    query: str


# --- Config validation ---


class ConfigError(TypedDict):
    """A startup configuration error."""

    field: str
    message: str
    severity: str  # "error" | "warning"
