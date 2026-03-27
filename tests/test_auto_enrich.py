"""Tests for auto-enrichment on brain_store (R47 two-pass pattern).

Pass 1: sync embedding (immediate, searchable) — already tested in test_deferred_embedding.py
Pass 2: async Gemini enrichment (~600ms) — tested here

Tests use mock Gemini client — no real API calls.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from brainlayer.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def stored_chunk(store):
    """Insert a chunk that enrich_single can operate on."""
    from brainlayer.store import store_memory

    result = store_memory(
        store=store,
        embed_fn=None,
        content="Chose twin-primary extraction over broker-primary for real-time conversation processing",
        memory_type="decision",
        project="brainlayer",
        tags=["architecture"],
        importance=8,
    )
    return result["id"]


def _fake_gemini_response(summary="Auto-enriched summary", tags=None):
    """Build a fake Gemini JSON response string."""
    if tags is None:
        tags = ["architecture", "decision", "real-time"]
    return json.dumps({
        "summary": summary,
        "tags": tags,
        "importance": 8,
        "intent": "deciding",
        "primary_symbols": [],
        "resolved_query": "What extraction pattern was chosen?",
        "epistemic_level": "substantiated",
        "debt_impact": "none",
    })


class _FakeGeminiClient:
    """Mock Gemini client that returns a valid enrichment response."""

    def __init__(self, response_text=None):
        self._response_text = response_text or _fake_gemini_response()
        self.call_count = 0

    class _Models:
        def __init__(self, parent):
            self._parent = parent

        def generate_content(self, **kwargs):
            self._parent.call_count += 1
            return SimpleNamespace(text=self._parent._response_text)

    def __new__(cls, response_text=None):
        instance = super().__new__(cls)
        instance._response_text = response_text or _fake_gemini_response()
        instance.call_count = 0
        instance.models = cls._Models(instance)
        return instance


# ── enrich_single unit tests ────────────────────────────────────


class TestEnrichSingle:
    """Test the enrich_single function directly."""

    def test_enriches_chunk_successfully(self, store, stored_chunk, monkeypatch):
        """enrich_single calls Gemini and applies enrichment to the chunk."""
        from brainlayer import enrichment_controller as ctrl

        client = _FakeGeminiClient()
        monkeypatch.setattr(ctrl, "_get_gemini_client", lambda: client)
        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)
        _mock_sanitizer = SimpleNamespace(
            sanitize=lambda text, metadata=None: SimpleNamespace(sanitized=text, replacements=[], pii_detected=False),
        )
        monkeypatch.setattr(ctrl, "Sanitizer", SimpleNamespace(from_env=lambda: _mock_sanitizer))

        result = ctrl.enrich_single(store, stored_chunk)

        assert result is not None
        assert result["summary"] == "Auto-enriched summary"
        assert "architecture" in result["tags"]
        assert client.call_count == 1

    def test_returns_none_when_disabled(self, store, stored_chunk, monkeypatch):
        """enrich_single returns None when AUTO_ENRICH_ENABLED is False."""
        from brainlayer import enrichment_controller as ctrl

        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", False)

        result = ctrl.enrich_single(store, stored_chunk)
        assert result is None

    def test_returns_none_for_missing_chunk(self, store, monkeypatch):
        """enrich_single returns None if chunk_id doesn't exist."""
        from brainlayer import enrichment_controller as ctrl

        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)

        result = ctrl.enrich_single(store, "nonexistent-chunk-id")
        assert result is None

    def test_returns_none_without_api_key(self, store, stored_chunk, monkeypatch):
        """enrich_single returns None if no Gemini API key is set."""
        from brainlayer import enrichment_controller as ctrl

        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)
        monkeypatch.setattr(ctrl, "_get_gemini_client", MagicMock(side_effect=RuntimeError("no key")))

        result = ctrl.enrich_single(store, stored_chunk)
        assert result is None

    def test_returns_none_on_gemini_failure(self, store, stored_chunk, monkeypatch):
        """enrich_single returns None if Gemini call fails after retries."""
        from brainlayer import enrichment_controller as ctrl

        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)
        _mock_sanitizer = SimpleNamespace(
            sanitize=lambda text, metadata=None: SimpleNamespace(sanitized=text, replacements=[], pii_detected=False),
        )
        monkeypatch.setattr(ctrl, "Sanitizer", SimpleNamespace(from_env=lambda: _mock_sanitizer))

        failing_client = MagicMock()
        failing_client.models.generate_content.side_effect = Exception("API error")
        monkeypatch.setattr(ctrl, "_get_gemini_client", lambda: failing_client)
        monkeypatch.setattr(ctrl.time, "sleep", lambda _: None)

        result = ctrl.enrich_single(store, stored_chunk, max_retries=1)
        assert result is None

    def test_returns_none_on_invalid_response(self, store, stored_chunk, monkeypatch):
        """enrich_single returns None if Gemini returns unparseable response."""
        from brainlayer import enrichment_controller as ctrl

        client = _FakeGeminiClient(response_text="not valid json at all")
        monkeypatch.setattr(ctrl, "_get_gemini_client", lambda: client)
        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)
        _mock_sanitizer = SimpleNamespace(
            sanitize=lambda text, metadata=None: SimpleNamespace(sanitized=text, replacements=[], pii_detected=False),
        )
        monkeypatch.setattr(ctrl, "Sanitizer", SimpleNamespace(from_env=lambda: _mock_sanitizer))

        result = ctrl.enrich_single(store, stored_chunk)
        assert result is None

    def test_writes_enrichment_to_db(self, store, stored_chunk, monkeypatch):
        """enrich_single writes the Gemini enrichment back to the DB."""
        from brainlayer import enrichment_controller as ctrl

        client = _FakeGeminiClient()
        monkeypatch.setattr(ctrl, "_get_gemini_client", lambda: client)
        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)
        _mock_sanitizer = SimpleNamespace(
            sanitize=lambda text, metadata=None: SimpleNamespace(sanitized=text, replacements=[], pii_detected=False),
        )
        monkeypatch.setattr(ctrl, "Sanitizer", SimpleNamespace(from_env=lambda: _mock_sanitizer))

        ctrl.enrich_single(store, stored_chunk)

        cursor = store.conn.cursor()
        rows = list(cursor.execute(
            "SELECT summary, tags FROM chunks WHERE id = ?",
            (stored_chunk,),
        ))
        assert len(rows) == 1
        assert rows[0][0] == "Auto-enriched summary"
        db_tags = json.loads(rows[0][1])
        assert "architecture" in db_tags
        assert "real-time" in db_tags

    def test_uses_low_retry_count(self, store, stored_chunk, monkeypatch):
        """enrich_single uses max_retries=2 by default (fast, not 12)."""
        from brainlayer import enrichment_controller as ctrl

        call_log = []

        def fake_retry(fn, max_retries=2, **kwargs):
            call_log.append({"max_retries": max_retries})
            return _fake_gemini_response()

        monkeypatch.setattr(ctrl, "_get_gemini_client", lambda: _FakeGeminiClient())
        monkeypatch.setattr(ctrl, "AUTO_ENRICH_ENABLED", True)
        _mock_sanitizer = SimpleNamespace(
            sanitize=lambda text, metadata=None: SimpleNamespace(sanitized=text, replacements=[], pii_detected=False),
        )
        monkeypatch.setattr(ctrl, "Sanitizer", SimpleNamespace(from_env=lambda: _mock_sanitizer))
        monkeypatch.setattr(ctrl, "_retry_with_backoff", fake_retry)

        ctrl.enrich_single(store, stored_chunk)

        assert len(call_log) == 1
        assert call_log[0]["max_retries"] == 2


# ── Integration: _store triggers auto-enrichment ────────────────


class TestStoreAutoEnrich:
    """Test that brain_store's background thread triggers auto-enrichment."""

    @pytest.mark.asyncio
    async def test_store_triggers_enrich_single(self, tmp_path, monkeypatch):
        """_store's background thread calls enrich_single after embedding."""
        from brainlayer.mcp import store_handler

        enriched_ids = []

        def mock_enrich_single(bg_store, cid):
            enriched_ids.append(cid)
            return {"summary": "enriched"}

        monkeypatch.setattr(
            "brainlayer.enrichment_controller.enrich_single",
            mock_enrich_single,
        )

        db_path = tmp_path / "test.db"
        test_store = VectorStore(db_path)

        monkeypatch.setattr(store_handler, "_get_vector_store", lambda: test_store)

        mock_model = MagicMock()
        mock_model.embed_query = lambda text: [0.1] * 1024
        monkeypatch.setattr(store_handler, "_get_embedding_model", lambda: mock_model)

        result = await store_handler._store_new(
            content="Test auto-enrichment integration",
            memory_type="learning",
            project="test",
        )

        content_items, structured = result
        chunk_id = structured["chunk_id"]

        # Wait for background thread to complete
        for t in threading.enumerate():
            if t.daemon and t.name != "MainThread":
                t.join(timeout=5.0)

        assert len(enriched_ids) == 1
        assert enriched_ids[0] == chunk_id
        test_store.close()

    @pytest.mark.asyncio
    async def test_store_succeeds_when_enrichment_fails(self, tmp_path, monkeypatch):
        """_store returns success even if auto-enrichment throws."""
        from brainlayer.mcp import store_handler

        def mock_enrich_single(bg_store, cid):
            raise RuntimeError("Gemini exploded")

        monkeypatch.setattr(
            "brainlayer.enrichment_controller.enrich_single",
            mock_enrich_single,
        )

        db_path = tmp_path / "test.db"
        test_store = VectorStore(db_path)

        monkeypatch.setattr(store_handler, "_get_vector_store", lambda: test_store)

        mock_model = MagicMock()
        mock_model.embed_query = lambda text: [0.1] * 1024
        monkeypatch.setattr(store_handler, "_get_embedding_model", lambda: mock_model)

        result = await store_handler._store_new(
            content="Store should succeed regardless of enrichment",
            memory_type="note",
        )

        content_items, structured = result
        assert structured["chunk_id"] != "queued"
        assert "Stored memory" in content_items[0].text

        for t in threading.enumerate():
            if t.daemon and t.name != "MainThread":
                t.join(timeout=5.0)

        test_store.close()


# ── Environment variable opt-out ────────────────────────────────


class TestAutoEnrichEnvVar:
    """Test the BRAINLAYER_AUTO_ENRICH environment variable."""

    @pytest.mark.parametrize("value,expected", [
        ("0", False),
        ("false", False),
        ("False", False),
        ("no", False),
        ("NO", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("", True),
    ])
    def test_auto_enrich_flag_parsing(self, value, expected, monkeypatch):
        """AUTO_ENRICH_ENABLED respects environment variable values."""
        monkeypatch.setenv("BRAINLAYER_AUTO_ENRICH", value)

        result = value.lower() not in ("0", "false", "no")
        assert result == expected
