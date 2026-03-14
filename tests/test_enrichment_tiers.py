"""TDD tests for tiered enrichment selectivity.

Tiers:
  T0 (IMMEDIATE): manual/digest sources and high-signal types (decision, mistake)
  T1 (HOURLY):    recent claude_code chunks (ai_code, stack_trace, user_message, assistant_text)
  T2 (LAZY):      old claude_code backlog (older than recency_days)
  T3 (EXPLICIT):  youtube transcripts — only when explicitly requested
"""

from datetime import datetime, timedelta, timezone

import pytest

from brainlayer.pipeline.enrichment_tiers import (
    EnrichmentTier,
    classify_chunk_tier,
    get_tier_content_types,
    get_tier_source_filter,
)

# ── Helpers ─────────────────────────────────────────────────────────────

def _dt(days_ago: int) -> str:
    """Return ISO timestamp N days ago in UTC."""
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


RECENT = _dt(3)
OLD = _dt(30)
VERY_OLD = _dt(365)

# ── T0: IMMEDIATE — manual / digest / high-signal memory types ───────────

def test_manual_source_is_tier0():
    """Manually stored chunks (brain_store) must always be tier 0."""
    tier = classify_chunk_tier(source="manual", content_type="assistant_text", created_at=RECENT)
    assert tier == EnrichmentTier.T0_IMMEDIATE


def test_digest_source_is_tier0():
    """Digested documents must always be tier 0."""
    tier = classify_chunk_tier(source="digest", content_type="assistant_text", created_at=RECENT)
    assert tier == EnrichmentTier.T0_IMMEDIATE


def test_manual_source_is_tier0_regardless_of_age():
    """Manual chunks should be T0 even if they are old."""
    tier = classify_chunk_tier(source="manual", content_type="assistant_text", created_at=VERY_OLD)
    assert tier == EnrichmentTier.T0_IMMEDIATE


def test_digest_source_is_tier0_regardless_of_age():
    """Digest chunks should be T0 even if they are old."""
    tier = classify_chunk_tier(source="digest", content_type="ai_code", created_at=VERY_OLD)
    assert tier == EnrichmentTier.T0_IMMEDIATE


# ── T1: HOURLY — recent claude_code ──────────────────────────────────────

def test_recent_claude_code_ai_code_is_tier1():
    """Recent ai_code from claude_code sessions is tier 1."""
    tier = classify_chunk_tier(source="claude_code", content_type="ai_code", created_at=RECENT)
    assert tier == EnrichmentTier.T1_HOURLY


def test_recent_claude_code_stack_trace_is_tier1():
    """Recent stack_trace from claude_code sessions is tier 1."""
    tier = classify_chunk_tier(source="claude_code", content_type="stack_trace", created_at=RECENT)
    assert tier == EnrichmentTier.T1_HOURLY


def test_recent_claude_code_user_message_is_tier1():
    """Recent user_message from claude_code sessions is tier 1."""
    tier = classify_chunk_tier(source="claude_code", content_type="user_message", created_at=RECENT)
    assert tier == EnrichmentTier.T1_HOURLY


def test_recent_claude_code_assistant_text_is_tier1():
    """Recent assistant_text from claude_code sessions is tier 1."""
    tier = classify_chunk_tier(source="claude_code", content_type="assistant_text", created_at=RECENT)
    assert tier == EnrichmentTier.T1_HOURLY


def test_today_claude_code_is_tier1():
    """Chunks from today (0 days ago) are recent."""
    today = _dt(0)
    tier = classify_chunk_tier(source="claude_code", content_type="ai_code", created_at=today)
    assert tier == EnrichmentTier.T1_HOURLY


def test_within_recency_window_is_tier1():
    """Chunks well within the recency window (6 days) are T1.

    The boundary (7 days) is not tested at microsecond precision here because
    classify_chunk_tier evaluates 'now' slightly after the timestamp is captured.
    test_just_past_recency_window_is_tier2 covers the other side at 8 days.
    """
    at_boundary = _dt(6)
    tier = classify_chunk_tier(source="claude_code", content_type="ai_code", created_at=at_boundary)
    assert tier == EnrichmentTier.T1_HOURLY


# ── T2: LAZY — old claude_code backlog ───────────────────────────────────

def test_old_claude_code_assistant_text_is_tier2():
    """Old assistant_text backlog (>7 days) from claude_code is tier 2."""
    tier = classify_chunk_tier(source="claude_code", content_type="assistant_text", created_at=OLD)
    assert tier == EnrichmentTier.T2_LAZY


def test_old_claude_code_user_message_is_tier2():
    """Old user_message backlog from claude_code is tier 2."""
    tier = classify_chunk_tier(source="claude_code", content_type="user_message", created_at=OLD)
    assert tier == EnrichmentTier.T2_LAZY


def test_very_old_claude_code_is_tier2():
    """Very old claude_code chunks (>1 year) are still only T2, not T3."""
    tier = classify_chunk_tier(source="claude_code", content_type="assistant_text", created_at=VERY_OLD)
    assert tier == EnrichmentTier.T2_LAZY


def test_just_past_recency_window_is_tier2():
    """Chunks just past the recency window (8 days) are T2."""
    just_past = _dt(8)
    tier = classify_chunk_tier(source="claude_code", content_type="assistant_text", created_at=just_past)
    assert tier == EnrichmentTier.T2_LAZY


# ── T3: EXPLICIT — youtube transcripts ──────────────────────────────────

def test_youtube_source_is_tier3():
    """YouTube transcript chunks are always tier 3 regardless of content type or age."""
    tier = classify_chunk_tier(source="youtube", content_type="assistant_text", created_at=RECENT)
    assert tier == EnrichmentTier.T3_EXPLICIT


def test_youtube_source_old_is_still_tier3():
    """Old YouTube chunks stay T3."""
    tier = classify_chunk_tier(source="youtube", content_type="assistant_text", created_at=VERY_OLD)
    assert tier == EnrichmentTier.T3_EXPLICIT


def test_unknown_source_old_is_tier2():
    """Unknown/unrecognized source that is old defaults to T2 (lazy backlog)."""
    tier = classify_chunk_tier(source="unknown", content_type="assistant_text", created_at=OLD)
    assert tier == EnrichmentTier.T2_LAZY


def test_unknown_source_recent_is_tier1():
    """Unknown source that is recent defaults to T1."""
    tier = classify_chunk_tier(source="unknown", content_type="assistant_text", created_at=RECENT)
    assert tier == EnrichmentTier.T1_HOURLY


# ── None / missing created_at ────────────────────────────────────────────

def test_none_created_at_defaults_to_tier2_for_claude_code():
    """Chunks with no created_at timestamp are treated as old (T2 for claude_code)."""
    tier = classify_chunk_tier(source="claude_code", content_type="assistant_text", created_at=None)
    assert tier == EnrichmentTier.T2_LAZY


# ── get_tier_content_types ───────────────────────────────────────────────

def test_get_tier_content_types_returns_high_value_for_t1():
    """T1 content types should include the core high-value types."""
    types = get_tier_content_types(EnrichmentTier.T1_HOURLY)
    assert "ai_code" in types
    assert "stack_trace" in types
    assert "user_message" in types
    assert "assistant_text" in types


# ── get_tier_source_filter ───────────────────────────────────────────────

def test_get_tier_source_filter_t1_excludes_youtube():
    """T1 source filter must exclude youtube so it's not processed hourly."""
    sources = get_tier_source_filter(EnrichmentTier.T1_HOURLY)
    assert "youtube" not in sources


def test_get_tier_source_filter_t0_includes_manual_and_digest():
    """T0 source filter must include manual and digest sources."""
    sources = get_tier_source_filter(EnrichmentTier.T0_IMMEDIATE)
    assert "manual" in sources
    assert "digest" in sources


def test_get_tier_source_filter_t3_only_youtube():
    """T3 source filter should only include youtube."""
    sources = get_tier_source_filter(EnrichmentTier.T3_EXPLICIT)
    assert sources == {"youtube"}


def test_tier_ordering():
    """Lower tier numbers should be higher priority (T0 < T1 < T2 < T3)."""
    assert EnrichmentTier.T0_IMMEDIATE < EnrichmentTier.T1_HOURLY
    assert EnrichmentTier.T1_HOURLY < EnrichmentTier.T2_LAZY
    assert EnrichmentTier.T2_LAZY < EnrichmentTier.T3_EXPLICIT
