"""Tests for Groq backend in enrichment pipeline.

Tests call_groq(), backend selection, privacy enforcement (sanitization),
and CLI --backend flag.
"""

import os
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
import requests

from brainlayer.pipeline import enrichment
from brainlayer.pipeline.groq import GroqModelUnavailableError, GroqServiceUnavailableError, validate_groq_model

# ── call_groq unit tests ──────────────────────────────────────────────


class TestCallGroq:
    """Test call_groq() function — OpenAI-compatible API call to Groq."""

    def test_call_groq_success(self):
        """Successful Groq API call returns response content."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"summary":"test","tags":["groq"]}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("requests.post", return_value=mock_response) as mock_post,
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
        ):
            result = enrichment.call_groq("test prompt")

        assert result == '{"summary":"test","tags":["groq"]}'
        # Verify it called Groq API URL
        call_args = mock_post.call_args
        assert "groq.com" in call_args[0][0] or "groq.com" in str(call_args)

    def test_call_groq_returns_none_on_error(self):
        """Groq API error returns None (not raises)."""
        with (
            patch("requests.post", side_effect=Exception("Connection refused")),
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
        ):
            result = enrichment.call_groq("test prompt")
        assert result is None

    def test_call_groq_raises_named_error_when_model_is_unavailable(self):
        """A dead configured model must stop the run instead of entering retries."""
        mock_response = MagicMock(status_code=404)
        mock_response.json.return_value = {"error": {"code": "model_not_found", "message": "model does not exist"}}

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
            patch.object(enrichment, "GROQ_MODEL", "retired/model"),
            pytest.raises(GroqModelUnavailableError, match=r"retired/model.*unavailable"),
        ):
            enrichment.call_groq("test prompt")

    def test_call_groq_does_not_call_generic_404_a_dead_model(self):
        """A bad proxy/path is an endpoint error unless Groq names model_not_found."""
        mock_response = MagicMock(status_code=404)
        mock_response.json.return_value = {"error": {"code": "route_not_found"}}
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
            patch.object(enrichment, "GROQ_MODEL", "live/model"),
        ):
            assert enrichment.call_groq("test prompt") is None

    def test_call_groq_requires_api_key(self):
        """call_groq returns None when GROQ_API_KEY is not set."""
        with patch.object(enrichment, "GROQ_API_KEY", ""):
            result = enrichment.call_groq("test prompt")
        assert result is None

    def test_call_groq_sends_auth_header(self):
        """Groq API call includes Authorization: Bearer header."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("requests.post", return_value=mock_response) as mock_post,
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
        ):
            enrichment.call_groq("test prompt")

        call_kwargs = mock_post.call_args
        headers = call_kwargs[1].get("headers") if call_kwargs[1] else call_kwargs.kwargs.get("headers")
        assert headers is not None
        assert "Bearer gsk_test123" in headers.get("Authorization", "")

    def test_call_groq_uses_correct_model(self):
        """Groq API call uses the confirmed live default model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("requests.post", return_value=mock_response) as mock_post,
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
        ):
            enrichment.call_groq("test prompt")

        call_args = mock_post.call_args
        json_body = call_args[1].get("json") if call_args[1] else call_args.kwargs.get("json")
        assert json_body["model"] == "openai/gpt-oss-120b"

    def test_call_groq_logs_usage(self):
        """Groq API call logs token usage to Supabase."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }
        mock_response.raise_for_status = MagicMock()

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
            patch.object(enrichment, "_log_glm_usage") as mock_log,
        ):
            enrichment.call_groq("test prompt")

        mock_log.assert_called_once()
        # _log_glm_usage(prompt_tokens, completion_tokens, duration_ms, model=...)
        args, kwargs = mock_log.call_args
        assert args[0] == 100  # prompt_tokens
        assert args[1] == 50  # completion_tokens
        assert "groq" in kwargs.get("model", "")


# ── Backend selection ──────────────────────────────────────────────────


class TestGroqBackendSelection:
    """Test that BRAINLAYER_ENRICH_BACKEND=groq routes to call_groq."""

    def setup_method(self):
        enrichment._consecutive_failures = 0
        enrichment._fallback_active = False
        enrichment._fallback_available = None

    def test_call_llm_routes_to_groq(self):
        """call_llm with backend='groq' calls call_groq."""
        with patch.object(enrichment, "call_groq", return_value='{"summary":"ok"}') as mock_groq:
            result = enrichment.call_llm("test prompt", backend="groq")
        assert result == '{"summary":"ok"}'
        mock_groq.assert_called_once()

    def test_detect_backend_groq_from_env(self):
        """BRAINLAYER_ENRICH_BACKEND=groq is recognized."""
        with patch.dict(os.environ, {"BRAINLAYER_ENRICH_BACKEND": "groq"}):
            backend = enrichment._detect_default_backend()
        assert backend == "groq"

    def test_run_enrichment_refuses_unavailable_groq_model_before_queue_work(self):
        """Startup validates the model catalog before touching queued chunks."""
        store = MagicMock()
        unavailable = GroqModelUnavailableError("Groq model 'retired/model' is unavailable")

        with (
            patch.object(enrichment, "VectorStore", return_value=store),
            patch.object(enrichment, "ENRICH_BACKEND", "groq"),
            patch.object(enrichment, "GROQ_API_KEY", "gsk_test123"),
            patch.object(enrichment, "validate_groq_model", side_effect=unavailable) as mock_validate,
            patch.object(enrichment, "mark_unenrichable") as mock_mark,
            pytest.raises(GroqModelUnavailableError, match=r"retired/model.*unavailable"),
        ):
            enrichment.run_enrichment()

        mock_validate.assert_called_once()
        mock_mark.assert_not_called()
        store.close.assert_called_once()

    def test_parallel_batch_does_not_wait_on_fatal_model_error(self):
        """A fatal model result cancels queued futures without blocking shutdown."""
        store = MagicMock()
        store.get_unenriched_chunks.return_value = [{"id": "chunk-1"}]
        future = Future()
        future.set_exception(GroqModelUnavailableError("model unavailable"))
        pool = MagicMock()
        pool.submit.return_value = future

        with (
            patch.object(enrichment, "ThreadPoolExecutor", return_value=pool),
            patch.object(enrichment, "as_completed", return_value=[future]),
            pytest.raises(GroqModelUnavailableError),
        ):
            enrichment.enrich_batch(store, parallel=2, backend="groq")

        pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

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

    @pytest.mark.parametrize("catalog", [ValueError("bad json"), [], {"data": [None]}])
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


class TestGroqPrivacy:
    """Test that Groq backend enforces sanitization (no raw PII to cloud)."""

    def test_enrich_one_uses_external_prompt_for_groq(self):
        """When backend=groq, _enrich_one uses build_external_prompt with Sanitizer."""
        store = MagicMock()
        store.get_context.return_value = {"context": []}
        chunk = {
            "id": "test-chunk-groq-1",
            "content": "Etan fixed the bug in auth.py",
            "content_type": "user_message",
            "project": "test",
        }

        with (
            patch.object(
                enrichment,
                "build_external_prompt",
                return_value=("sanitized prompt", MagicMock()),
            ) as mock_ext_prompt,
            patch.object(enrichment, "call_llm", return_value='{"summary":"ok","tags":["test"]}'),
            patch.object(enrichment, "parse_enrichment", return_value={"summary": "ok", "tags": ["test"]}),
        ):
            result = enrichment._enrich_one(store, chunk, with_context=False, backend="groq")

        assert result is True
        mock_ext_prompt.assert_called_once()

    def test_enrich_one_uses_local_prompt_for_mlx(self):
        """When backend=mlx, _enrich_one uses build_prompt (no sanitization)."""
        store = MagicMock()
        store.get_context.return_value = {"context": []}
        chunk = {
            "id": "test-chunk-mlx-1",
            "content": "test content",
            "content_type": "user_message",
            "project": "test",
        }

        with (
            patch.object(enrichment, "build_prompt", return_value="local prompt") as mock_local_prompt,
            patch.object(enrichment, "call_llm", return_value='{"summary":"ok","tags":["test"]}'),
            patch.object(enrichment, "parse_enrichment", return_value={"summary": "ok", "tags": ["test"]}),
        ):
            result = enrichment._enrich_one(store, chunk, with_context=False, backend="mlx")

        assert result is True
        mock_local_prompt.assert_called_once()


# ── Config constants ──────────────────────────────────────────────────


class TestGroqConfig:
    """Test Groq-specific configuration constants."""

    @pytest.mark.parametrize(
        "attr,expected",
        [
            ("GROQ_URL", "groq.com"),
            ("GROQ_MODEL", "openai/gpt-oss-120b"),
        ],
    )
    def test_groq_config_defaults(self, attr, expected):
        """Groq config constants have expected default values."""
        value = getattr(enrichment, attr)
        assert expected in value or value == expected
