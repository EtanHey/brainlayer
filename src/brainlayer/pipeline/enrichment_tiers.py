"""Tiered enrichment selectivity for BrainLayer.

Assigns each chunk a processing tier based on source, content type, and age:

  T0 (IMMEDIATE): manual/digest — always enrich, highest priority
  T1 (HOURLY):    recent agent-session chunks — hourly local enrichment
  T2 (LAZY):      old agent-session backlog — lazy remote batch
  T3 (EXPLICIT):  youtube transcripts — only when explicitly triggered

Design note: tiers are IntEnum so T0 < T1 < T2 < T3 comparisons work naturally
and callers can filter "up to tier N" with `tier <= max_tier`.
"""

from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import List, Optional, Set

# ── Tier definition ──────────────────────────────────────────────────────


class EnrichmentTier(IntEnum):
    T0_IMMEDIATE = 0  # manual / digest — enrich immediately
    T1_HOURLY = 1  # recent agent-session chunks — hourly local run
    T2_LAZY = 2  # old agent-session backlog — lazy / remote batch
    T3_EXPLICIT = 3  # youtube — explicit request only


# ── Constants ────────────────────────────────────────────────────────────

# Sources that are always T0 regardless of age or content type.
T0_SOURCES: frozenset = frozenset({"manual", "digest"})

# Sources that are always T3 regardless of age or content type.
T3_SOURCES: frozenset = frozenset({"youtube"})

# Default recency window: chunks within this many days are T1, older are T2.
DEFAULT_RECENCY_DAYS: int = 7

# High-value content types enriched at T1 (and T2 for old backlog).
T1_CONTENT_TYPES: List[str] = ["ai_code", "stack_trace", "user_message", "assistant_text"]

# Content types that should never be enriched (low-signal noise).
SKIP_CONTENT_TYPES: frozenset = frozenset({"noise"})

# Only these sources participate in the T1/T2 recency gate.
# Unrecognised sources fall through to T2 (lazy backlog) rather than T1.
T1_T2_SOURCES: frozenset = frozenset({"claude_code", "codex_cli", "cursor_cli", "gemini_cli"})


# ── Classifier ───────────────────────────────────────────────────────────


def classify_chunk_tier(
    source: str,
    content_type: str,
    created_at: Optional[str],
    recency_days: int = DEFAULT_RECENCY_DAYS,
) -> EnrichmentTier:
    """Return the enrichment tier for a chunk.

    Args:
        source:        The chunk source field (e.g. "claude_code", "codex_cli", "youtube", "manual").
        content_type:  The chunk content_type field (e.g. "ai_code", "assistant_text").
        created_at:    ISO timestamp string when the chunk was created, or None.
        recency_days:  Window (days) for T1 vs T2 split (default 7).

    Returns:
        EnrichmentTier for the chunk.
    """
    # Noise is never worth enriching, regardless of source.
    if content_type in SKIP_CONTENT_TYPES:
        return EnrichmentTier.T3_EXPLICIT

    # T0: always-on sources (manual brain_store, digested documents)
    if source in T0_SOURCES:
        return EnrichmentTier.T0_IMMEDIATE

    # T3: archival sources never touched by the default pipeline
    if source in T3_SOURCES:
        return EnrichmentTier.T3_EXPLICIT

    # Only recognised T1/T2 sources participate in the recency gate.
    # Unknown sources default to lazy backlog (T2) rather than crowding T1.
    if source not in T1_T2_SOURCES:
        return EnrichmentTier.T2_LAZY

    # Agent-session sources: age determines tier.
    if _is_recent(created_at, recency_days):
        return EnrichmentTier.T1_HOURLY
    return EnrichmentTier.T2_LAZY


def _is_recent(created_at: Optional[str], recency_days: int) -> bool:
    """Return True if created_at falls within the recency window."""
    if created_at is None:
        return False
    try:
        dt = datetime.fromisoformat(created_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=recency_days)
        return dt >= cutoff
    except (ValueError, TypeError):
        return False


# ── Selectors ────────────────────────────────────────────────────────────


def get_tier_content_types(tier: EnrichmentTier) -> List[str]:
    """Return the content types relevant for a given tier.

    All tiers currently share the same high-value content type list.
    This function exists so callers can parametrize queries without
    hard-coding content types.
    """
    return list(T1_CONTENT_TYPES)


def get_tier_source_filter(tier: EnrichmentTier) -> Set[str]:
    """Return the set of *allowed* sources for a given tier.

    Useful for building SQL IN clauses or filtering chunk lists.

    T0 → {manual, digest}
    T1 → {claude_code, codex_cli, cursor_cli, gemini_cli}   (explicitly excludes youtube)
    T2 → {claude_code, codex_cli, cursor_cli, gemini_cli}   (old backlog only)
    T3 → {youtube}
    """
    if tier == EnrichmentTier.T0_IMMEDIATE:
        return set(T0_SOURCES)
    if tier == EnrichmentTier.T1_HOURLY:
        return set(T1_T2_SOURCES)
    if tier == EnrichmentTier.T2_LAZY:
        return set(T1_T2_SOURCES)
    if tier == EnrichmentTier.T3_EXPLICIT:
        return set(T3_SOURCES)
    return set()  # pragma: no cover
