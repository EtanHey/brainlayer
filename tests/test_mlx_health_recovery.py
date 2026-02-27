"""Tests for MLX health checking and backend recovery in enrichment pipeline."""

from unittest.mock import MagicMock, patch

import requests

from brainlayer.pipeline import enrichment


class TestCheckBackendHealth:
    """Backend health check function."""

    def test_mlx_healthy(self):
        with patch("brainlayer.pipeline.enrichment.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert enrichment.check_backend_health("mlx") is True

    def test_mlx_dead(self):
        with patch("brainlayer.pipeline.enrichment.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("refused")
            assert enrichment.check_backend_health("mlx") is False

    def test_ollama_healthy(self):
        with patch("brainlayer.pipeline.enrichment.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert enrichment.check_backend_health("ollama") is True

    def test_groq_with_key(self):
        with patch.object(enrichment, "GROQ_API_KEY", "sk-test"):
            assert enrichment.check_backend_health("groq") is True

    def test_groq_without_key(self):
        with patch.object(enrichment, "GROQ_API_KEY", ""):
            assert enrichment.check_backend_health("groq") is False


class TestCallMlxErrorCategorization:
    """call_mlx should distinguish connection errors from timeouts."""

    def test_connection_refused_logged_distinctly(self, capsys):
        with patch("brainlayer.pipeline.enrichment.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
            result = enrichment.call_mlx("test prompt")

        assert result is None
        captured = capsys.readouterr()
        assert "connection error" in captured.err.lower()

    def test_timeout_logged_distinctly(self, capsys):
        with patch("brainlayer.pipeline.enrichment.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("timed out")
            result = enrichment.call_mlx("test prompt")

        assert result is None
        captured = capsys.readouterr()
        assert "timeout" in captured.err.lower()


class TestHighFailRatioBehavior:
    """run_enrichment should detect high fail ratios and check health."""

    def test_high_fail_ratio_triggers_health_check(self):
        """When >80% of a batch fails, health is checked before continuing."""
        store = MagicMock()
        store.get_enrichment_stats.return_value = {
            "enriched": 100,
            "enrichable": 1000,
            "remaining": 900,
            "skipped": 0,
            "total_chunks": 1000,
            "percent": "10.0",
            "by_intent": {},
        }

        batch_results = [
            # Batch 1: 90% fail rate — triggers health check
            {"processed": 50, "success": 5, "failed": 45, "circuit_broken": False},
            # Batch 2: empty — ends loop
            {"processed": 0, "success": 0, "failed": 0},
        ]
        batch_iter = iter(batch_results)

        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.object(enrichment, "enrich_batch", side_effect=lambda *a, **kw: next(batch_iter)),
            patch.object(enrichment, "mark_unenrichable", return_value=0),
            patch.object(enrichment, "check_backend_health", return_value=True) as mock_health,
            patch.object(enrichment, "_sync_stats_to_supabase"),
            patch.object(enrichment, "HEALTH_CHECK_PAUSE", 0),  # No sleep in tests
            patch.object(enrichment, "BATCH_FAIL_RATIO_THRESHOLD", 0.8),
            patch("brainlayer.pipeline.enrichment.requests.get", return_value=mock_resp),
            patch("brainlayer.pipeline.enrichment.VectorStore", return_value=store),
        ):
            enrichment.run_enrichment(
                db_path=None,
                batch_size=50,
                max_chunks=0,
            )

        # Health check should have been called after the high-fail batch
        mock_health.assert_called()

    def test_circuit_breaker_triggers_recovery(self):
        """Circuit breaker should attempt backend recovery before stopping."""
        store = MagicMock()
        store.get_enrichment_stats.return_value = {
            "enriched": 100,
            "enrichable": 1000,
            "remaining": 900,
            "skipped": 0,
            "total_chunks": 1000,
            "percent": "10.0",
            "by_intent": {},
        }

        batch_results = [
            # Batch 1: circuit breaker tripped
            {"processed": 10, "success": 0, "failed": 10, "circuit_broken": True},
        ]
        batch_iter = iter(batch_results)

        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()

        with (
            patch.object(enrichment, "enrich_batch", side_effect=lambda *a, **kw: next(batch_iter)),
            patch.object(enrichment, "mark_unenrichable", return_value=0),
            patch.object(enrichment, "check_backend_health", return_value=False) as mock_health,
            patch.object(enrichment, "_recover_backend", return_value=False) as mock_recover,
            patch.object(enrichment, "_sync_stats_to_supabase"),
            patch("brainlayer.pipeline.enrichment.requests.get", return_value=mock_resp),
            patch("brainlayer.pipeline.enrichment.VectorStore", return_value=store),
        ):
            enrichment.run_enrichment(
                db_path=None,
                batch_size=50,
                max_chunks=0,
            )

        mock_health.assert_called()
        mock_recover.assert_called()
