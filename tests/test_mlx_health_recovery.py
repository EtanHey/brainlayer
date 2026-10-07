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
