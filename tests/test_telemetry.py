"""Tests for Axiom telemetry module.

Covers:
- Graceful degradation when AXIOM_TOKEN is unset
- Event emission with mocked client
- Watcher-specific helper functions
- Error telemetry includes traceback snippet
"""

from unittest.mock import MagicMock

import pytest

# Reset module-level state before each test
import brainlayer.telemetry as telemetry_mod


@pytest.fixture(autouse=True)
def reset_telemetry_state():
    """Reset the lazy-init client state between tests."""
    telemetry_mod._client = None
    telemetry_mod._client_failed = False
    yield
    telemetry_mod._client = None
    telemetry_mod._client_failed = False


# ── Graceful Degradation ─────────────────────────────────────────────────────


class TestGracefulDegradation:
    def test_emit_returns_false_without_token(self, monkeypatch):
        monkeypatch.delenv("AXIOM_TOKEN", raising=False)
        from brainlayer.telemetry import emit

        result = emit("test-dataset", {"_type": "test"})
        assert result is False

    def test_emit_many_returns_false_without_token(self, monkeypatch):
        monkeypatch.delenv("AXIOM_TOKEN", raising=False)
        from brainlayer.telemetry import emit_many

        result = emit_many("test-dataset", [{"_type": "test"}])
        assert result is False

    def test_emit_many_empty_list_returns_true(self, monkeypatch):
        monkeypatch.delenv("AXIOM_TOKEN", raising=False)
        from brainlayer.telemetry import emit_many

        result = emit_many("test-dataset", [])
        assert result is True

    def test_startup_returns_false_without_token(self, monkeypatch):
        monkeypatch.delenv("AXIOM_TOKEN", raising=False)
        from brainlayer.telemetry import emit_watcher_startup

        result = emit_watcher_startup(sessions_watched=5, watcher_pid=1234)
        assert result is False


# ── Mocked Client Emission ───────────────────────────────────────────────────


class TestMockedEmission:
    @pytest.fixture
    def mock_client(self, monkeypatch):
        """Inject a mock Axiom client."""
        monkeypatch.setenv("AXIOM_TOKEN", "test-token")
        client = MagicMock()
        telemetry_mod._client = client
        return client

    def test_emit_calls_ingest(self, mock_client):
        from brainlayer.telemetry import emit

        result = emit("brainlayer-watcher", {"_type": "test", "value": 42})
        assert result is True
        mock_client.ingest_events.assert_called_once()
        args = mock_client.ingest_events.call_args
        assert args.kwargs["dataset"] == "brainlayer-watcher"
        assert args.kwargs["events"][0]["_type"] == "test"

    def test_emit_many_calls_ingest(self, mock_client):
        from brainlayer.telemetry import emit_many

        events = [{"_type": "a"}, {"_type": "b"}]
        result = emit_many("brainlayer-watcher", events)
        assert result is True
        args = mock_client.ingest_events.call_args
        assert len(args.kwargs["events"]) == 2

    def test_emit_handles_client_error(self, mock_client):
        mock_client.ingest_events.side_effect = RuntimeError("network error")
        from brainlayer.telemetry import emit

        result = emit("brainlayer-watcher", {"_type": "test"})
        assert result is False  # Doesn't crash


# ── Watcher Helpers ──────────────────────────────────────────────────────────


class TestWatcherHelpers:
    @pytest.fixture
    def mock_client(self, monkeypatch):
        monkeypatch.setenv("AXIOM_TOKEN", "test-token")
        client = MagicMock()
        telemetry_mod._client = client
        return client

    def test_startup_event_shape(self, mock_client):
        from brainlayer.telemetry import emit_watcher_startup

        emit_watcher_startup(sessions_watched=10, watcher_pid=5678)
        event = mock_client.ingest_events.call_args.kwargs["events"][0]
        assert event["_type"] == "startup"
        assert event["sessions_watched"] == 10
        assert event["watcher_pid"] == 5678
        assert "hostname" in event

    def test_flush_event_shape(self, mock_client):
        from brainlayer.telemetry import emit_watcher_flush

        emit_watcher_flush(
            chunks_indexed=5,
            chunks_skipped=2,
            latency_ms=45.678,
            source_files=["/a.jsonl", "/b.jsonl"],
        )
        event = mock_client.ingest_events.call_args.kwargs["events"][0]
        assert event["_type"] == "flush"
        assert event["chunks_indexed"] == 5
        assert event["chunks_skipped"] == 2
        assert event["latency_ms"] == 45.68
        assert len(event["source_files"]) == 2

    def test_error_event_shape(self, mock_client):
        from brainlayer.telemetry import emit_watcher_error

        try:
            raise ValueError("test error")
        except ValueError:
            emit_watcher_error("classify", "test error", file_path="/test.jsonl")

        event = mock_client.ingest_events.call_args.kwargs["events"][0]
        assert event["_type"] == "error"
        assert event["error_type"] == "classify"
        assert event["message"] == "test error"
        assert event["file_path"] == "/test.jsonl"
        assert "ValueError" in event["traceback_snippet"]

    def test_heartbeat_event_shape(self, mock_client):
        from brainlayer.telemetry import emit_watcher_heartbeat

        emit_watcher_heartbeat(
            sessions_tracked=8,
            chunks_indexed_total=250,
            uptime_seconds=3600.5,
        )
        event = mock_client.ingest_events.call_args.kwargs["events"][0]
        assert event["_type"] == "heartbeat"
        assert event["sessions_tracked"] == 8
        assert event["chunks_indexed_total"] == 250
        assert event["uptime_seconds"] == 3600.5
        assert "watcher_pid" in event

    def test_flush_caps_source_files(self, mock_client):
        from brainlayer.telemetry import emit_watcher_flush

        emit_watcher_flush(
            chunks_indexed=1,
            chunks_skipped=0,
            latency_ms=1.0,
            source_files=[f"/file{i}.jsonl" for i in range(20)],
        )
        event = mock_client.ingest_events.call_args.kwargs["events"][0]
        assert len(event["source_files"]) == 5  # Capped at 5
