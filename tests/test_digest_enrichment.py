"""Tests for digest-time faceted enrichment."""

from brainlayer.pipeline.digest import _build_faceted_gemini_config


def test_faceted_gemini_config_disables_thinking():
    """Flash models must always force thinkingBudget=0."""
    config = _build_faceted_gemini_config()

    assert config["response_mime_type"] == "application/json"
    assert config["thinking_config"]["thinking_budget"] == 0
