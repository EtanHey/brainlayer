"""Tests for Groq backend in enrichment pipeline.

Tests call_groq(), backend selection, privacy enforcement (sanitization),
and CLI --backend flag.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from brainlayer.pipeline import enrichment
from brainlayer.pipeline.groq import GroqServiceUnavailableError, validate_groq_model

# ── call_groq unit tests ──────────────────────────────────────────────


# ── Backend selection ──────────────────────────────────────────────────


class TestGroqBackendSelection:
    """Test that BRAINLAYER_ENRICH_BACKEND=groq routes to call_groq."""

    def setup_method(self):
        enrichment._consecutive_failures = 0
        enrichment._fallback_active = False
        enrichment._fallback_available = None

    def test_detect_backend_groq_from_env(self):
        """BRAINLAYER_ENRICH_BACKEND=groq is recognized."""
        with patch.dict(os.environ, {"BRAINLAYER_ENRICH_BACKEND": "groq"}):
            backend = enrichment._detect_default_backend()
        assert backend == "groq"

    def test_model_validation_distinguishes_unreachable_service(self):
        """A timeout must not be reported as proof that the model is dead."""
        with (
            patch("requests.get", side_effect=requests.Timeout("timed out")),
            pytest.raises(
                GroqServiceUnavailableError,
                match=r"retired/model.*could not be checked.*service is unavailable",
            ),
        ):
            validate_groq_model(
                "gsk_test123",
                "retired/model",
                "https://api.groq.com/openai/v1/chat/completions",
            )

    @pytest.mark.parametrize(
        "catalog",
        [ValueError("bad json"), [], {"data": [None]}, {"data": [{"id": None}]}],
    )
    def test_model_validation_reports_malformed_catalog_as_service_failure(self, catalog):
        """A 2xx response with an unusable catalog does not prove model retirement."""
        response = MagicMock()
        if isinstance(catalog, Exception):
            response.json.side_effect = catalog
        else:
            response.json.return_value = catalog

        with (
            patch("requests.get", return_value=response),
            pytest.raises(GroqServiceUnavailableError, match=r"could not be checked.*service is unavailable"),
        ):
            validate_groq_model(
                "gsk_test123",
                "live/model",
                "https://api.groq.com/openai/v1/chat/completions",
            )


# ── Privacy enforcement ────────────────────────────────────────────────


# ── Config constants ──────────────────────────────────────────────────
