"""Tests for mid-run backend fallback in enrichment pipeline."""

from unittest.mock import patch

from brainlayer.pipeline import enrichment


def _reset_fallback_state():
    """Reset module-level fallback state between tests."""
    enrichment._consecutive_failures = 0
    enrichment._fallback_active = False
    enrichment._fallback_available = None


class TestCallLlmFallback:
    """Test call_llm mid-run fallback behavior."""

    def setup_method(self):
        _reset_fallback_state()

    def test_success_resets_consecutive_failures(self):
        """Successful call resets the failure counter."""
        enrichment._consecutive_failures = 2
        with patch.object(enrichment, "call_mlx", return_value='{"summary":"ok"}'):
            result = enrichment.call_llm("test prompt", backend="mlx")
        assert result == '{"summary":"ok"}'
        assert enrichment._consecutive_failures == 0

    def test_failure_increments_counter(self):
        """Failed call increments consecutive failure counter."""
        with patch.object(enrichment, "call_mlx", return_value=None), \
             patch.object(enrichment, "_check_fallback_available", return_value=False):
            result = enrichment.call_llm("test prompt", backend="mlx")
        assert result is None
        assert enrichment._consecutive_failures == 1

    def test_fallback_triggers_after_threshold(self):
        """After N consecutive failures, switches to fallback backend."""
        enrichment._consecutive_failures = 2  # One more failure will trigger threshold (3)
        with patch.object(enrichment, "call_mlx", return_value=None), \
             patch.object(enrichment, "call_glm", return_value='{"summary":"fallback"}'), \
             patch.object(enrichment, "_check_fallback_available", return_value=True):
            result = enrichment.call_llm("test prompt", backend="mlx")
        assert result == '{"summary":"fallback"}'
        assert enrichment._fallback_active is True

    def test_fallback_stays_active_once_triggered(self):
        """Once fallback is active, subsequent calls use fallback directly."""
        enrichment._fallback_active = True
        with patch.object(enrichment, "call_glm", return_value='{"summary":"via fallback"}') as mock_glm, \
             patch.object(enrichment, "call_mlx") as mock_mlx:
            result = enrichment.call_llm("test prompt", backend="mlx")
        assert result == '{"summary":"via fallback"}'
        mock_glm.assert_called_once()
        mock_mlx.assert_not_called()

    def test_no_fallback_when_unavailable(self):
        """If fallback backend isn't running, don't switch."""
        enrichment._consecutive_failures = 2
        with patch.object(enrichment, "call_mlx", return_value=None), \
             patch.object(enrichment, "_check_fallback_available", return_value=False):
            result = enrichment.call_llm("test prompt", backend="mlx")
        assert result is None
        assert enrichment._fallback_active is False

    def test_ollama_to_mlx_fallback(self):
        """Fallback works in reverse: ollama primary → mlx fallback."""
        enrichment._consecutive_failures = 2
        with patch.object(enrichment, "call_glm", return_value=None), \
             patch.object(enrichment, "call_mlx", return_value='{"summary":"mlx ok"}'), \
             patch.object(enrichment, "_check_fallback_available", return_value=True):
            result = enrichment.call_llm("test prompt", backend="ollama")
        assert result == '{"summary":"mlx ok"}'
        assert enrichment._fallback_active is True


class TestRunEnrichmentResetsState:
    """Test that run_enrichment resets fallback state."""

    def test_resets_fallback_state(self):
        """run_enrichment resets all fallback state at start."""
        enrichment._consecutive_failures = 5
        enrichment._fallback_active = True
        enrichment._fallback_available = True

        with patch.object(enrichment, "VectorStore") as mock_vs:
            mock_store = mock_vs.return_value
            mock_store.get_enrichment_stats.return_value = {
                "enriched": 0, "enrichable": 0, "remaining": 0,
                "skipped": 0, "percent": "0", "total_chunks": 0,
                "by_intent": {},
            }
            try:
                enrichment.run_enrichment(max_chunks=0, batch_size=1)
            except Exception:
                pass  # Will fail on backend check, that's fine

        assert enrichment._consecutive_failures == 0
        assert enrichment._fallback_active is False
        assert enrichment._fallback_available is None


class TestCheckFallbackAvailable:
    """Test fallback availability detection."""

    def setup_method(self):
        _reset_fallback_state()

    def test_caches_result(self):
        """Availability check is cached after first call."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status = lambda: None
            result1 = enrichment._check_fallback_available("mlx")
            result2 = enrichment._check_fallback_available("mlx")
        assert result1 is True
        assert result2 is True
        mock_get.assert_called_once()  # Only checked once

    def test_returns_false_on_connection_error(self):
        """Returns False if fallback backend is unreachable."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = enrichment._check_fallback_available("mlx")
        assert result is False
