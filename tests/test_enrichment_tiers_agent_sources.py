from datetime import datetime, timedelta, timezone

from brainlayer.pipeline.enrichment_tiers import EnrichmentTier, classify_chunk_tier, get_tier_source_filter


def _recent(days_ago: int = 2) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def _old(days_ago: int = 30) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()


def test_recent_agent_cli_sources_are_hourly_candidates():
    recent = _recent()

    assert classify_chunk_tier("codex_cli", "assistant_text", recent) == EnrichmentTier.T1_HOURLY
    assert classify_chunk_tier("cursor_cli", "assistant_text", recent) == EnrichmentTier.T1_HOURLY
    assert classify_chunk_tier("gemini_cli", "assistant_text", recent) == EnrichmentTier.T1_HOURLY


def test_old_agent_cli_sources_fall_back_to_lazy_backlog():
    old = _old()

    assert classify_chunk_tier("codex_cli", "assistant_text", old) == EnrichmentTier.T2_LAZY
    assert classify_chunk_tier("cursor_cli", "assistant_text", old) == EnrichmentTier.T2_LAZY
    assert classify_chunk_tier("gemini_cli", "assistant_text", old) == EnrichmentTier.T2_LAZY


def test_hourly_source_filter_includes_all_agent_cli_families():
    sources = get_tier_source_filter(EnrichmentTier.T1_HOURLY)

    assert {"claude_code", "codex_cli", "cursor_cli", "gemini_cli"} <= sources
