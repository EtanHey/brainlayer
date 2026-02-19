"""Tests for source-aware enrichment thresholds."""

from brainlayer.vector_store import source_aware_min_chars


class TestSourceAwareMinChars:
    """Test source-aware minimum character thresholds for enrichment."""

    def test_whatsapp_threshold_is_15(self):
        assert source_aware_min_chars("whatsapp") == 15

    def test_claude_code_threshold_is_50(self):
        assert source_aware_min_chars("claude_code") == 50

    def test_youtube_threshold_is_50(self):
        assert source_aware_min_chars("youtube") == 50

    def test_unknown_source_defaults_to_50(self):
        assert source_aware_min_chars("unknown") == 50

    def test_none_source_defaults_to_50(self):
        assert source_aware_min_chars(None) == 50

    def test_telegram_threshold_is_15(self):
        """Telegram messages are also short-form like WhatsApp."""
        assert source_aware_min_chars("telegram") == 15
