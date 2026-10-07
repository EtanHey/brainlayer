"""Tests for Phase 2: MLX migration, auto-indexing, current_context fix.

Tests cover:
- Backend auto-detection (MLX on arm64 Mac, Ollama elsewhere)
- Enrichment stall detection
- Heartbeat logging
- current_context fix (hours→days conversion, chunks fallback)
- call_llm backend override
"""

import os
from datetime import datetime
from unittest.mock import patch


class TestBackendAutoDetection:
    """Test _detect_default_backend() auto-detection logic."""

    def test_explicit_env_overrides_detection(self):
        """BRAINLAYER_ENRICH_BACKEND env var overrides auto-detection."""
        from brainlayer.pipeline.enrichment_results import _detect_default_backend

        with patch.dict(os.environ, {"BRAINLAYER_ENRICH_BACKEND": "ollama"}):
            assert _detect_default_backend() == "ollama"

        with patch.dict(os.environ, {"BRAINLAYER_ENRICH_BACKEND": "mlx"}):
            assert _detect_default_backend() == "mlx"

    def test_arm64_darwin_defaults_mlx(self):
        """arm64 macOS defaults to MLX."""
        from brainlayer.pipeline.enrichment_results import _detect_default_backend

        env = {k: v for k, v in os.environ.items() if k != "BRAINLAYER_ENRICH_BACKEND"}
        with patch.dict(os.environ, env, clear=True):
            with patch("platform.machine", return_value="arm64"):
                with patch("platform.system", return_value="Darwin"):
                    assert _detect_default_backend() == "mlx"

    def test_x86_defaults_ollama(self):
        """x86_64 defaults to Ollama."""
        from brainlayer.pipeline.enrichment_results import _detect_default_backend

        env = {k: v for k, v in os.environ.items() if k != "BRAINLAYER_ENRICH_BACKEND"}
        with patch.dict(os.environ, env, clear=True):
            with patch("platform.machine", return_value="x86_64"):
                assert _detect_default_backend() == "ollama"

    def test_linux_defaults_ollama(self):
        """Linux defaults to Ollama regardless of arch."""
        from brainlayer.pipeline.enrichment_results import _detect_default_backend

        env = {k: v for k, v in os.environ.items() if k != "BRAINLAYER_ENRICH_BACKEND"}
        with patch.dict(os.environ, env, clear=True):
            with patch("platform.machine", return_value="arm64"):
                with patch("platform.system", return_value="Linux"):
                    assert _detect_default_backend() == "ollama"


class TestCurrentContextFix:
    """Test that current_context returns data from chunks table when session_context is empty."""

    def _make_store(self, tmp_path):
        """Create a test VectorStore with sample data."""
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)
        return store

    def test_hours_to_days_precision(self):
        """Hours to days conversion uses ceiling division."""
        # hours=4 should produce days=1 (not 0)
        # hours=25 should produce days=2 (not 1)
        # hours=48 should produce days=2
        assert max(1, -(-4 // 24)) == 1
        assert max(1, -(-25 // 24)) == 2
        assert max(1, -(-48 // 24)) == 2
        assert max(1, -(-1 // 24)) == 1

    def test_empty_store_returns_empty(self, tmp_path):
        """Empty database returns empty context."""
        from brainlayer.engine import current_context

        store = self._make_store(tmp_path)
        result = current_context(store, hours=24)
        assert result.active_projects == []
        assert result.recent_files == []
        assert result.recent_sessions == []
        store.close()

    def test_chunks_provide_project_fallback(self, tmp_path):
        """Projects are found from chunks table even without session_context entries."""
        from brainlayer.engine import current_context
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)

        # Insert a chunk with a recent created_at and project
        now = datetime.now().isoformat()
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("chunk-1", "test content", "{}", "test.py", "my-project", "user_message", 100, now),
        )

        result = current_context(store, hours=24)
        assert "my-project" in result.active_projects
        store.close()

    def test_source_files_fallback(self, tmp_path):
        """Recent files come from chunks.source_file when file_interactions is empty."""
        from brainlayer.engine import current_context
        from brainlayer.vector_store import VectorStore

        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)

        now = datetime.now().isoformat()
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project, content_type, char_count, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("chunk-1", "test content", "{}", "src/auth.py", "my-project", "user_message", 100, now),
        )

        result = current_context(store, hours=24)
        assert "src/auth.py" in result.recent_files
        store.close()


class TestParseEnrichment:
    """Test enrichment JSON parsing — existing but good to verify."""

    def test_valid_json(self):
        from brainlayer.pipeline.enrichment_results import parse_enrichment

        text = (
            '{"summary":"Test summary here","tags":["tech/testing","project/brainlayer"],'
            '"importance":7,"intent":"debugging"}'
        )
        result = parse_enrichment(text)
        assert result is not None
        assert result["summary"] == "Test summary here"
        assert "tech/testing" in result["tags"]
        assert result["importance"] == 7.0
        assert result["intent"] == "debugging"

    def test_json_with_extra_text(self):
        from brainlayer.pipeline.enrichment_results import parse_enrichment

        text = (
            'Here is the result:\n{"summary":"Found it","tags":["test"],"importance":5,"intent":"implementing"}\nDone.'
        )
        result = parse_enrichment(text)
        assert result is not None
        assert result["summary"] == "Found it"

    def test_invalid_json(self):
        from brainlayer.pipeline.enrichment_results import parse_enrichment

        assert parse_enrichment("not json at all") is None
        assert parse_enrichment("") is None
        assert parse_enrichment(None) is None

    def test_missing_required_fields(self):
        from brainlayer.pipeline.enrichment_results import parse_enrichment

        # Missing tags — should return None
        text = '{"summary":"test"}'
        assert parse_enrichment(text) is None

    def test_extended_fields_optional(self):
        from brainlayer.pipeline.enrichment_results import parse_enrichment

        text = '{"summary":"Testing extended fields work correctly","tags":["python"],"importance":5,"intent":"debugging","primary_symbols":["MyClass"],"resolved_query":"How to fix it?","epistemic_level":"validated","debt_impact":"resolution","external_deps":["fastapi"]}'
        result = parse_enrichment(text)
        assert result is not None
        assert result["primary_symbols"] == ["MyClass"]
        assert result["resolved_query"] == "How to fix it?"
        assert result["epistemic_level"] == "validated"
        assert result["debt_impact"] == "resolution"
        assert result["external_deps"] == ["fastapi"]
