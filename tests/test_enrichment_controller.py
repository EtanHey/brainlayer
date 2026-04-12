"""Tests for the unified enrichment controller.

Covers: 3 backends (realtime/batch/local), content-hash dedup, retry logic,
rate limiting, telemetry, MCP handler, stats, error handling, idempotency,
LaunchAgent plist, and CLI integration.

Target: 35+ tests per A-R2 acceptance criteria.
"""

import json
import sqlite3
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _candidate(chunk_id: str = "c1", content: str = "x" * 120) -> dict:
    return {
        "id": chunk_id,
        "content": content,
        "project": "brainlayer",
        "content_type": "assistant_text",
        "source": "claude_code",
    }


def _fake_gemini_client(response_text='{"summary":"sum","tags":["python"]}'):
    """Create a fake Gemini client that returns the given response text."""

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(text=response_text)

        def __init__(self):
            self.models = self._Models()

    return FakeClient()


def _patch_realtime_deps(monkeypatch, controller, store, response_text=None):
    """Common monkeypatching for realtime enrichment tests."""
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        controller,
        "_get_gemini_client",
        lambda: _fake_gemini_client(response_text or '{"summary":"sum","tags":["python"]}'),
    )


# ── Existing realtime tests ──────────────────────────────────────────────────


def test_enrich_realtime_calls_get_enrichment_candidates(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    result = controller.enrich_realtime(store, limit=7, since_hours=12)

    store.get_enrichment_candidates.assert_called_once_with(limit=7, since_hours=12, chunk_ids=None)
    assert result.mode == "realtime"


def test_enrich_realtime_calls_build_external_prompt_for_every_chunk(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate("c1"), _candidate("c2")]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "_retry_with_backoff", lambda fn, **kwargs: '{"summary":"sum","tags":["python"]}')
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: MagicMock())
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    controller.enrich_realtime(store, limit=2, since_hours=24)

    assert controller.build_external_prompt.call_count == 2


def test_enrich_realtime_sets_thinking_budget_zero_in_gemini_config(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    captured = {}

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(text='{"summary":"sum","tags":["python"]}')

        def __init__(self):
            self.models = self._Models()

    monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())

    controller.enrich_realtime(store)

    assert captured["config"]["thinking_config"]["thinking_budget"] == 0


def test_enrich_realtime_passes_flex_service_tier_in_gemini_config(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.delenv("BRAINLAYER_GEMINI_SERVICE_TIER", raising=False)
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    captured = {}

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(text='{"summary":"sum","tags":["python"]}')

        def __init__(self):
            self.models = self._Models()

    monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())

    controller.enrich_realtime(store)

    assert captured["config"]["http_options"]["extra_body"]["serviceTier"] == "flex"


def test_enrich_realtime_calls_parse_enrichment_on_response(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    parse_mock = MagicMock(return_value={"summary": "sum", "tags": ["python"]})
    monkeypatch.setattr(controller, "parse_enrichment", parse_mock)
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):  # noqa: ARG002
                return SimpleNamespace(text='{"summary":"sum","tags":["python"]}')

        def __init__(self):
            self.models = self._Models()

    monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())

    controller.enrich_realtime(store)

    parse_mock.assert_called_once_with('{"summary":"sum","tags":["python"]}')


def test_enrich_realtime_writes_via_update_enrichment_only(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):  # noqa: ARG002
                return SimpleNamespace(text='{"summary":"sum","tags":["python"]}')

        def __init__(self):
            self.models = self._Models()

    monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())

    controller.enrich_realtime(store)

    store.update_enrichment.assert_called_once()


def test_enrich_realtime_is_idempotent_on_rerun(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.side_effect = [[_candidate()], []]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)

    class FakeClient:
        class _Models:
            def generate_content(self, **kwargs):  # noqa: ARG002
                return SimpleNamespace(text='{"summary":"sum","tags":["python"]}')

        def __init__(self):
            self.models = self._Models()

    monkeypatch.setattr(controller, "_get_gemini_client", lambda: FakeClient())

    first = controller.enrich_realtime(store)
    second = controller.enrich_realtime(store)

    assert first.enriched == 1
    assert second.enriched == 0
    assert store.update_enrichment.call_count == 1


def test_retry_with_backoff_retries_12_times_on_429(monkeypatch):
    from brainlayer import enrichment_controller as controller

    sleeps = []
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    monkeypatch.setattr(controller.random, "uniform", lambda a, b: 0.0)

    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError):
        controller._retry_with_backoff(flaky, max_retries=12)

    assert attempts["count"] == 13
    assert len(sleeps) == 12


def test_retry_with_backoff_respects_max_delay_cap(monkeypatch):
    from brainlayer import enrichment_controller as controller

    sleeps = []
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)
    monkeypatch.setattr(controller.random, "uniform", lambda a, b: b)

    attempts = {"count": 0}

    def flaky_then_success():
        attempts["count"] += 1
        if attempts["count"] < 4:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    result = controller._retry_with_backoff(flaky_then_success, max_retries=5, base_delay=10, max_delay=15)

    assert result == "ok"
    assert max(sleeps) <= 15


def test_enrich_local_does_not_call_build_external_prompt(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    external_prompt = MagicMock(side_effect=AssertionError("should not be called"))
    monkeypatch.setattr(controller, "build_external_prompt", external_prompt)
    monkeypatch.setattr(controller, "build_prompt", MagicMock(return_value="local-prompt"))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "sum", "tags": ["python"]}))
    monkeypatch.setattr(controller, "_retry_with_backoff", lambda fn, **kwargs: '{"summary":"sum","tags":["python"]}')
    monkeypatch.setattr(
        controller, "_call_local_backend", lambda *args, **kwargs: '{"summary":"sum","tags":["python"]}'
    )

    result = controller.enrich_local(store, limit=1)

    assert result.mode == "local"
    external_prompt.assert_not_called()


def test_enrich_batch_returns_early_for_no_candidates(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    # _get_gemini_client should never be reached when there are no candidates
    monkeypatch.setattr(
        controller, "_get_gemini_client", lambda: (_ for _ in ()).throw(AssertionError("should not be called"))
    )

    result = controller.enrich_batch(store, limit=100)

    store.get_enrichment_candidates.assert_called_once_with(limit=100, chunk_ids=None)
    assert result.mode == "batch"
    assert result.enriched == 0


# ── Content-hash dedup tests ─────────────────────────────────────────────────


def test_content_hash_deterministic():
    from brainlayer.enrichment_controller import _content_hash

    h1 = _content_hash("hello world")
    h2 = _content_hash("hello world")
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hex


def test_content_hash_strips_whitespace():
    from brainlayer.enrichment_controller import _content_hash

    h1 = _content_hash("  hello world  ")
    h2 = _content_hash("hello world")
    assert h1 == h2


def test_content_hash_differs_for_different_content():
    from brainlayer.enrichment_controller import _content_hash

    h1 = _content_hash("hello")
    h2 = _content_hash("world")
    assert h1 != h2


def test_is_duplicate_returns_false_when_column_missing():
    from brainlayer.enrichment_controller import _is_duplicate_content

    store = MagicMock()
    store._read_cursor.side_effect = Exception("no such column: content_hash")
    assert _is_duplicate_content(store, "content") is False


def test_is_duplicate_returns_true_when_hash_exists_enriched():
    from brainlayer.enrichment_controller import _is_duplicate_content

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (1,)
    store._read_cursor.return_value = cursor
    assert _is_duplicate_content(store, "content") is True


def test_is_duplicate_returns_false_when_summary_cleared():
    from brainlayer.enrichment_controller import _is_duplicate_content

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (0,)
    store._read_cursor.return_value = cursor

    assert _is_duplicate_content(store, "content") is False
    query, params = cursor.execute.call_args[0]
    assert "summary IS NOT NULL" in query
    assert params == (_is_duplicate_content.__globals__["_content_hash"]("content"),)


def test_is_duplicate_returns_false_when_hash_not_found():
    from brainlayer.enrichment_controller import _is_duplicate_content

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (0,)
    store._read_cursor.return_value = cursor
    assert _is_duplicate_content(store, "content") is False


def test_ensure_content_hash_column_creates_if_missing():
    from brainlayer.enrichment_controller import _ensure_content_hash_column

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.side_effect = [Exception("no such column"), None]
    store.conn.cursor.return_value = cursor
    assert _ensure_content_hash_column(store) is True


def test_ensure_content_hash_column_noop_if_exists():
    from brainlayer.enrichment_controller import _ensure_content_hash_column

    store = MagicMock()
    cursor = MagicMock()
    store.conn.cursor.return_value = cursor
    assert _ensure_content_hash_column(store) is True


def test_backfill_content_hashes_processes_null_rows():
    from brainlayer.enrichment_controller import _backfill_content_hashes

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = [("id1", "content1"), ("id2", "content2")]
    store.conn.cursor.return_value = cursor

    count = _backfill_content_hashes(store, limit=10)
    assert count == 2


def test_backfill_content_hashes_skips_empty_content():
    from brainlayer.enrichment_controller import _backfill_content_hashes

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = [("id1", ""), ("id2", None)]
    store.conn.cursor.return_value = cursor

    count = _backfill_content_hashes(store, limit=10)
    assert count == 0


def test_realtime_skips_duplicate_content(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate("c1", "dup"), _candidate("c2", "unique")]
    _patch_realtime_deps(monkeypatch, controller, store)
    monkeypatch.setattr(controller, "_is_duplicate_content", lambda s, c: c == "dup")

    result = controller.enrich_realtime(store, limit=2)
    assert result.skipped == 1
    assert result.enriched == 1


def test_local_skips_duplicate_content(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate("c1", "dup"), _candidate("c2", "unique")]
    monkeypatch.setattr(controller, "build_prompt", MagicMock(return_value="prompt"))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "s", "tags": ["t"]}))
    monkeypatch.setattr(controller, "_retry_with_backoff", lambda fn, **kw: '{"summary":"s"}')
    monkeypatch.setattr(controller, "_call_local_backend", lambda *a, **kw: '{"summary":"s"}')
    monkeypatch.setattr(controller, "_is_duplicate_content", lambda s, c: c == "dup")

    result = controller.enrich_local(store, limit=2)
    assert result.skipped == 1


# ── EnrichmentResult dataclass tests ─────────────────────────────────────────


def test_enrichment_result_defaults():
    from brainlayer.enrichment_controller import EnrichmentResult

    r = EnrichmentResult(mode="test", attempted=0, enriched=0, skipped=0, failed=0)
    assert r.errors == []
    assert r.mode == "test"


def test_enrichment_result_with_errors():
    from brainlayer.enrichment_controller import EnrichmentResult

    r = EnrichmentResult(mode="realtime", attempted=3, enriched=1, skipped=0, failed=2, errors=["e1", "e2"])
    assert len(r.errors) == 2
    assert r.failed == 2


# ── Gemini config tests ──────────────────────────────────────────────────────


def test_build_gemini_config_has_json_mime_type():
    from brainlayer.enrichment_controller import _build_gemini_config

    config = _build_gemini_config()
    assert config["response_mime_type"] == "application/json"


def test_build_gemini_config_disables_thinking():
    from brainlayer.enrichment_controller import _build_gemini_config

    config = _build_gemini_config()
    assert config["thinking_config"]["thinking_budget"] == 0


def test_build_gemini_config_allows_service_tier_override(monkeypatch):
    from brainlayer.enrichment_controller import _build_gemini_config

    monkeypatch.setenv("BRAINLAYER_GEMINI_SERVICE_TIER", "standard")

    config = _build_gemini_config()

    assert config["http_options"]["extra_body"]["serviceTier"] == "standard"


def test_build_gemini_config_validates_against_sdk(monkeypatch):
    genai = pytest.importorskip("google.genai")

    from brainlayer.enrichment_controller import _build_gemini_config

    monkeypatch.setenv("BRAINLAYER_GEMINI_SERVICE_TIER", "flex")

    config = _build_gemini_config()
    validated = genai.types.GenerateContentConfig.model_validate(config)

    assert validated.http_options.extra_body["serviceTier"] == "flex"


def test_gemini_client_requires_api_key(monkeypatch):
    from brainlayer.enrichment_controller import _get_gemini_client

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not set|not installed"):
        _get_gemini_client()


def test_gemini_sdk_retries_disabled(monkeypatch):
    from brainlayer.enrichment_controller import _get_gemini_client

    client_ctor = MagicMock(return_value=object())
    fake_google = types.ModuleType("google")
    fake_google.genai = SimpleNamespace(Client=client_ctor)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_REGION", raising=False)

    _get_gemini_client()

    _, kwargs = client_ctor.call_args
    assert kwargs["http_options"]["retry_options"] == {"attempts": 1}


def test_store_lifecycle_waits_for_cleanup_before_new_operations():
    from brainlayer import enrichment_controller as controller

    store = SimpleNamespace(db_path="/tmp/pr-a3-lifecycle.db")
    key = controller._store_queue_key(store)
    stop_started = threading.Event()
    release_stop = threading.Event()
    begin_finished = threading.Event()

    old_queue = MagicMock()

    def blocking_stop(timeout=1.0):  # noqa: ARG001
        stop_started.set()
        release_stop.wait(timeout=2)

    old_queue.stop.side_effect = blocking_stop

    with controller._WRITE_QUEUE_LOCK:
        controller._WRITE_QUEUE_REGISTRY[key] = old_queue
    with controller._STORE_OPERATION_CONDITION:
        controller._STORE_OPERATION_COUNTS[key] = 1
        controller._STORE_CLOSING.discard(key)

    def cleanup():
        controller._end_store_operation(store)

    def begin_new_operation():
        controller._begin_store_operation(store)
        begin_finished.set()

    cleanup_thread = threading.Thread(target=cleanup)
    cleanup_thread.start()
    assert stop_started.wait(timeout=1)

    begin_thread = threading.Thread(target=begin_new_operation)
    begin_thread.start()
    assert not begin_finished.wait(timeout=0.05)

    release_stop.set()
    cleanup_thread.join(timeout=1)
    begin_thread.join(timeout=1)

    assert begin_finished.is_set()
    with controller._STORE_OPERATION_CONDITION:
        assert controller._STORE_OPERATION_COUNTS[key] == 1
        assert key not in controller._STORE_CLOSING

    controller._end_store_operation(store)


def test_rate_limiter_survives_store_cleanup():
    from brainlayer import enrichment_controller as controller

    store = SimpleNamespace(db_path="/tmp/pr-a3-rate-limiter.db")
    registry_key = controller._store_queue_key(store)
    limiter_key = (registry_key, 5.0, 10)

    with controller._RATE_LIMITER_LOCK:
        controller._RATE_LIMITER_REGISTRY.pop(limiter_key, None)

    try:
        controller._begin_store_operation(store)
        limiter = controller._get_store_rate_limiter(store, rate_per_second=5.0, burst=10)
        controller._end_store_operation(store)

        controller._begin_store_operation(store)
        same_limiter = controller._get_store_rate_limiter(store, rate_per_second=5.0, burst=10)
        controller._end_store_operation(store)

        assert same_limiter is limiter
    finally:
        with controller._RATE_LIMITER_LOCK:
            controller._RATE_LIMITER_REGISTRY.pop(limiter_key, None)


# ── Rate limiting tests ──────────────────────────────────────────────────────


def test_rate_limits_defaults():
    from brainlayer.enrichment_controller import RATE_LIMITS

    assert RATE_LIMITS["realtime"] > 0
    assert RATE_LIMITS["local"] == 0  # No limit for local
    assert RATE_LIMITS["batch"] == 0  # No limit for batch (async)


def test_realtime_rate_limit_acquires_token_per_chunk(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate("c1"), _candidate("c2"), _candidate("c3")]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value={"summary": "s", "tags": []}))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: _fake_gemini_client())

    acquires = []

    class FakeLimiter:
        def acquire(self, n=1):
            acquires.append(n)

    monkeypatch.setattr(controller, "_get_store_rate_limiter", lambda *args, **kwargs: FakeLimiter())

    controller.enrich_realtime(store, limit=3, rate_per_second=2.0)

    assert acquires == [1, 1, 1]


def test_realtime_no_sleep_when_rate_zero(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate("c1"), _candidate("c2")]
    _patch_realtime_deps(monkeypatch, controller, store)

    sleeps = []
    monkeypatch.setattr(controller.time, "sleep", sleeps.append)

    controller.enrich_realtime(store, limit=2, rate_per_second=0)
    assert len(sleeps) == 0


# ── Retry logic tests ────────────────────────────────────────────────────────


def test_retry_succeeds_on_first_try(monkeypatch):
    from brainlayer.enrichment_controller import _retry_with_backoff

    monkeypatch.setattr("brainlayer.enrichment_controller.time.sleep", lambda _: None)
    monkeypatch.setattr("brainlayer.enrichment_controller.random.uniform", lambda a, b: 0.0)

    result = _retry_with_backoff(lambda: "ok", max_retries=3)
    assert result == "ok"


def test_retry_succeeds_after_transient_failure(monkeypatch):
    from brainlayer.enrichment_controller import _retry_with_backoff

    monkeypatch.setattr("brainlayer.enrichment_controller.time.sleep", lambda _: None)
    monkeypatch.setattr("brainlayer.enrichment_controller.random.uniform", lambda a, b: 0.0)

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "recovered"

    result = _retry_with_backoff(flaky, max_retries=5)
    assert result == "recovered"
    assert attempts["n"] == 3


def test_retry_only_catches_specified_errors(monkeypatch):
    from brainlayer.enrichment_controller import _retry_with_backoff

    monkeypatch.setattr("brainlayer.enrichment_controller.time.sleep", lambda _: None)
    monkeypatch.setattr("brainlayer.enrichment_controller.random.uniform", lambda a, b: 0.0)

    with pytest.raises(ValueError):
        _retry_with_backoff(
            lambda: (_ for _ in ()).throw(ValueError("not retryable")),
            max_retries=3,
            retryable_errors=(RuntimeError,),
        )


# ── Error handling tests ──────────────────────────────────────────────────────


def test_realtime_counts_failed_on_invalid_enrichment(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(return_value=("prompt", SimpleNamespace())))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value=None))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: _fake_gemini_client("garbage"))

    result = controller.enrich_realtime(store, limit=1)
    assert result.failed == 1
    assert result.enriched == 0
    assert "invalid_enrichment" in result.errors[0]


def test_realtime_counts_failed_on_exception(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_external_prompt", MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(controller, "Sanitizer", SimpleNamespace(from_env=lambda: SimpleNamespace()))
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    monkeypatch.setattr(controller, "_get_gemini_client", lambda: _fake_gemini_client())

    result = controller.enrich_realtime(store, limit=1)
    assert result.failed == 1
    assert "boom" in result.errors[0]


def test_local_counts_failed_on_exception(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_prompt", MagicMock(side_effect=RuntimeError("local fail")))

    result = controller.enrich_local(store, limit=1)
    assert result.failed == 1
    assert "local fail" in result.errors[0]


def test_local_counts_failed_on_invalid_parse(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]
    monkeypatch.setattr(controller, "build_prompt", MagicMock(return_value="prompt"))
    monkeypatch.setattr(controller, "parse_enrichment", MagicMock(return_value=None))
    monkeypatch.setattr(controller, "_retry_with_backoff", lambda fn, **kw: "garbage")
    monkeypatch.setattr(controller, "_call_local_backend", lambda *a, **kw: "garbage")

    result = controller.enrich_local(store, limit=1)
    assert result.failed == 1
    assert result.enriched == 0


# ── Meta-research filter tests ───────────────────────────────────────────────


def test_meta_research_filter_detects_common_patterns():
    from brainlayer.enrichment_controller import is_meta_research

    samples = [
        "brain_search(query='crypto trading bot')",
        'brain_search query="crypto trading bot"',
        "Search results for 'crypto trading bot Ofir strategy'",
        "Query 3 for 'crypto trading bot Ofir strategy' degraded from 2.4/5 to 2.2/5",
        "Eval score: 2.6/5 after ingestion",
        "Grade: 3/5",
        "[BrainLayer auto] Memories matching: 5",
        '{"hookEventName":"search","additionalContext":"tool payload"}',
    ]

    assert all(is_meta_research(sample) for sample in samples)


def test_meta_research_filter_preserves_real_content():
    from brainlayer.enrichment_controller import is_meta_research

    samples = [
        "We decided to keep the enrichment controller in a single file until the batch path is stabilized.",
        "def build_index(query: str) -> list[str]:\n    return [query.strip()]",
        "Ofir said the strategy should defer position sizing until volatility normalizes.",
        "Conversation note: Etan wants the daemon restart deferred until after the migration lands.",
    ]

    assert all(not is_meta_research(sample) for sample in samples)


# ── Apply enrichment tests ───────────────────────────────────────────────────


def test_apply_enrichment_calls_update_enrichment_with_all_fields():
    from brainlayer.enrichment_controller import _apply_enrichment

    store = MagicMock()
    chunk = _candidate("c1")
    enrichment = {
        "summary": "test summary",
        "tags": ["python", "test"],
        "importance": 7,
        "intent": "implementation",
        "primary_symbols": ["func_a"],
        "resolved_query": None,
        "key_facts": ["PR #1722"],
        "resolved_queries": [
            "What changed in enrichment v2?",
            "enrichment v2 key_facts resolved_queries",
            "Enrichment v2 added key_facts and resolved_queries.",
        ],
        "epistemic_level": "certain",
        "version_scope": "v1.0",
        "debt_impact": "low",
        "external_deps": ["pytest"],
        "sentiment_label": "frustration",
        "sentiment_score": -0.6,
        "sentiment_signals": ["damn", "broken"],
    }

    _apply_enrichment(store, chunk, enrichment)

    store.update_enrichment.assert_called_once_with(
        chunk_id="c1",
        summary="test summary",
        tags=["python", "test"],
        importance=7,
        intent="implementation",
        primary_symbols=["func_a"],
        resolved_query="What changed in enrichment v2?",
        key_facts=["PR #1722"],
        resolved_queries=[
            "What changed in enrichment v2?",
            "enrichment v2 key_facts resolved_queries",
            "Enrichment v2 added key_facts and resolved_queries.",
        ],
        epistemic_level="certain",
        version_scope="v1.0",
        debt_impact="low",
        external_deps=["pytest"],
        sentiment_label="frustration",
        sentiment_score=-0.6,
        sentiment_signals=["damn", "broken"],
    )


def test_apply_enrichment_sets_content_hash():
    from brainlayer.enrichment_controller import _apply_enrichment, _content_hash

    store = MagicMock()
    cursor = MagicMock()
    store.conn.cursor.return_value = cursor
    chunk = _candidate("c1", "test content")

    _apply_enrichment(store, chunk, {"summary": "s"})

    expected_hash = _content_hash("test content")
    args = cursor.execute.call_args_list[-1][0]
    assert args[1] == (expected_hash, "c1")


def test_apply_enrichment_persists_raw_entities():
    from brainlayer.enrichment_controller import _apply_enrichment

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY, raw_entities_json TEXT, content_hash TEXT)")
    conn.execute("INSERT INTO chunks (id, raw_entities_json, content_hash) VALUES (?, NULL, NULL)", ("c1",))

    store = MagicMock()
    store.conn = conn
    chunk = _candidate("c1", "test content")
    entities = [
        {"name": "Ofir", "type": "person", "relation": "described the strategy"},
        {"name": "BrainLayer", "type": "project", "relation": "stores the chunk"},
    ]

    _apply_enrichment(store, chunk, {"summary": "s", "entities": entities})

    row = conn.execute("SELECT raw_entities_json FROM chunks WHERE id = ?", ("c1",)).fetchone()
    assert row == (json.dumps(entities),)


# ── Telemetry tests ──────────────────────────────────────────────────────────


def test_emit_enrichment_start_fires(monkeypatch):
    from brainlayer import enrichment_controller as controller

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    controller._emit_enrichment_start("realtime", 25)
    assert len(events) == 1
    assert events[0]["_type"] == "start"
    assert events[0]["mode"] == "realtime"


def test_emit_enrichment_start_swallows_oserror_and_logs_debug(monkeypatch):
    from brainlayer import enrichment_controller as controller

    events = []
    debug_logs = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)
    monkeypatch.setattr(controller.os, "write", lambda *_args: (_ for _ in ()).throw(OSError("pipe closed")))
    monkeypatch.setattr(controller.logger, "debug", lambda msg, *args: debug_logs.append(msg % args if args else msg))

    controller._emit_enrichment_start("realtime", 25)

    assert len(events) == 1
    assert events[0]["_type"] == "start"
    assert any("ENRICHMENT_RUNTIME_LOADED" in entry for entry in debug_logs)


def test_emit_enrichment_complete_fires(monkeypatch):
    from brainlayer import enrichment_controller as controller
    from brainlayer.enrichment_controller import EnrichmentResult

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    result = EnrichmentResult(mode="local", attempted=10, enriched=8, skipped=1, failed=1)
    controller._emit_enrichment_complete(result, 1500.0)

    assert len(events) == 1
    assert events[0]["_type"] == "complete"
    assert events[0]["enriched"] == 8
    assert events[0]["duration_ms"] == 1500.0


def test_emit_enrichment_error_truncates_long_errors(monkeypatch):
    from brainlayer import enrichment_controller as controller

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    long_error = "x" * 500
    controller._emit_enrichment_error("realtime", "chunk123", long_error)

    assert len(events[0]["error"]) == 300


def test_realtime_emits_start_and_complete_events(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    events = []
    monkeypatch.setattr(controller, "_emit_enrichment_event", lambda e: events.append(e) or True)

    controller.enrich_realtime(store, limit=5)

    types = [e["_type"] for e in events]
    assert "start" in types
    assert "complete" in types


# ── Telemetry module tests ───────────────────────────────────────────────────


def test_telemetry_emit_returns_false_without_axiom_token(monkeypatch):
    import brainlayer.telemetry as telemetry

    monkeypatch.delenv("AXIOM_TOKEN", raising=False)
    telemetry._client = None
    telemetry._client_failed = False

    result = telemetry.emit("test-dataset", {"key": "value"})
    assert result is False


def test_telemetry_emit_many_returns_true_for_empty_list():
    import brainlayer.telemetry as telemetry

    result = telemetry.emit_many("test-dataset", [])
    assert result is True


def test_telemetry_enrichment_helpers_exist():
    from brainlayer.telemetry import (
        emit_enrichment_complete,
        emit_enrichment_error,
        emit_enrichment_start,
    )

    assert callable(emit_enrichment_start)
    assert callable(emit_enrichment_complete)
    assert callable(emit_enrichment_error)


# ── MCP handler tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_brain_enrich_handler_returns_error_for_unknown_mode(monkeypatch):
    from brainlayer.mcp.enrich_handler import _brain_enrich

    result = await _brain_enrich(mode="unknown")
    assert result.isError is True
    assert "Unknown mode" in result.content[0].text


@pytest.mark.asyncio
async def test_brain_enrich_handler_stats_mode(monkeypatch):
    from brainlayer.mcp.enrich_handler import _brain_enrich

    store = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value.fetchone.return_value = (100,)
    store._read_cursor.return_value = cursor

    monkeypatch.setattr("brainlayer.mcp.enrich_handler._get_vector_store", lambda: store)

    result = await _brain_enrich(stats=True)
    assert result.isError is not True
    text = result.content[0].text
    # _enrich_stats returns formatted text with box-drawing chars, not JSON
    assert "Total:" in text
    assert "Enriched:" in text


@pytest.mark.asyncio
async def test_enrich_stats_returns_correct_structure():
    from brainlayer.mcp.enrich_handler import _enrich_stats

    store = MagicMock()
    cursor = MagicMock()
    # Simulate: total=1000, enriched=600, unenriched=350, skipped=50, recent=20
    cursor.execute.return_value.fetchone.side_effect = [(1000,), (600,), (350,), (50,), (20,)]
    store._read_cursor.return_value = cursor

    result = await _enrich_stats(store)
    text = result.content[0].text

    # _enrich_stats returns formatted text lines, not JSON
    assert "Total: 1,000" in text
    assert "Enriched: 600" in text
    assert "(60.0%)" in text
    assert "Remaining: 350" in text
    assert "Skipped: 50" in text
    assert "Last 24h: 20" in text


# ── Batch mode tests ─────────────────────────────────────────────────────────


def test_enrich_batch_processes_candidates_with_gemini(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate("c1"), _candidate("c2")]
    _patch_realtime_deps(monkeypatch, controller, store)

    result = controller.enrich_batch(store, limit=10)

    assert result.mode == "batch"
    assert result.attempted == 2
    assert result.enriched == 2


def test_enrich_batch_graceful_when_no_gemini_key(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = [_candidate()]

    monkeypatch.setattr(controller, "_get_gemini_client", lambda: (_ for _ in ()).throw(RuntimeError("no key")))

    result = controller.enrich_batch(store, limit=5)

    assert result.mode == "batch"
    assert result.enriched == 0
    assert any("No Gemini client" in e for e in result.errors)


# ── Realtime chunk_ids filter test ────────────────────────────────────────────


def test_realtime_passes_chunk_ids_to_candidates(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    controller.enrich_realtime(store, chunk_ids=["a", "b"])

    store.get_enrichment_candidates.assert_called_once_with(limit=500, since_hours=8760, chunk_ids=["a", "b"])


# ── LaunchAgent plist validation ──────────────────────────────────────────────


def test_enrichment_plist_has_correct_label():
    import xml.etree.ElementTree as ET
    from pathlib import Path

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    tree = ET.parse(plist_path)
    root = tree.getroot()
    d = root.find("dict")

    # Find the string element right after the Label key
    elements = list(d)
    for i, el in enumerate(elements):
        if el.tag == "key" and el.text == "Label":
            assert elements[i + 1].text == "com.brainlayer.enrichment"
            break


def test_enrichment_plist_uses_realtime_mode():
    from pathlib import Path

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    content = plist_path.read_text()
    assert "realtime" in content


def test_enrichment_plist_has_start_interval():
    from pathlib import Path

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    content = plist_path.read_text()
    assert "StartInterval" in content
    assert "3600" in content


def test_enrichment_plist_invokes_python_enrich_realtime_entrypoint():
    from pathlib import Path

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    content = plist_path.read_text()
    assert "__PYTHON3__" in content
    assert "brainlayer.enrichment_controller" in content
    assert "enrich_realtime" in content


def test_enrichment_plist_uses_low_priority_and_library_logs():
    from pathlib import Path

    plist_path = Path(__file__).parent.parent / "scripts" / "launchd" / "com.brainlayer.enrichment.plist"
    content = plist_path.read_text()
    assert "<key>Nice</key>" in content
    assert "<integer>10</integer>" in content
    assert "__HOME__/Library/Logs/brainlayer-enrichment.log" in content


def test_launchd_installer_supports_enrichment_load_and_unload():
    from pathlib import Path

    install_script = (Path(__file__).parent.parent / "scripts" / "launchd" / "install.sh").read_text()
    assert 'LAUNCH_DIR="$HOME/Library/LaunchAgents"' in install_script
    assert "load)" in install_script
    assert "unload)" in install_script
    assert "com.brainlayer.enrichment" in install_script


def test_launchd_installer_reads_google_api_key_from_zshrc():
    from pathlib import Path

    install_script = (Path(__file__).parent.parent / "scripts" / "launchd" / "install.sh").read_text()
    assert ".zshrc" in install_script
    assert "GOOGLE_API_KEY" in install_script


# ── Gemini model constant test ────────────────────────────────────────────────


def test_gemini_realtime_model_default():
    from brainlayer.enrichment_controller import GEMINI_REALTIME_MODEL

    assert "flash-lite" in GEMINI_REALTIME_MODEL
    assert "2.5" in GEMINI_REALTIME_MODEL


# ── Empty candidates handling ─────────────────────────────────────────────────


def test_realtime_returns_zero_counts_for_no_candidates(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    result = controller.enrich_realtime(store)

    assert result.attempted == 0
    assert result.enriched == 0
    assert result.skipped == 0
    assert result.failed == 0


def test_local_returns_zero_counts_for_no_candidates():
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    store.get_enrichment_candidates.return_value = []

    result = controller.enrich_local(store)

    assert result.attempted == 0
    assert result.enriched == 0
