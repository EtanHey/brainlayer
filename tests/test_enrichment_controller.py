"""Tests for the unified enrichment controller."""

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


def test_enrich_batch_uses_checkpoint_db_for_resume(monkeypatch):
    from brainlayer import enrichment_controller as controller

    store = MagicMock()
    ensure_mock = MagicMock()
    monkeypatch.setattr(controller, "ensure_checkpoint_table", ensure_mock)
    monkeypatch.setattr(controller, "get_pending_jobs", MagicMock(return_value=[]))
    monkeypatch.setattr(controller, "get_unsubmitted_export_files", MagicMock(return_value=[]))

    result = controller.enrich_batch(store, phase="run", limit=100)

    ensure_mock.assert_called_once_with(store)
    assert result.mode == "batch"
