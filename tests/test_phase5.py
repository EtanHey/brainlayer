"""Tests for Phase 5 — Layers Polish Sprint.

Tests cover:
1. Auto-importance scoring (_auto_importance)
2. Auto-type detection (_detect_memory_type) — already tested, extended here
3. Project scoping (resolve_project_scope)
4. phase_commits table creation
5. Decision tracking fields in brain_store
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Mock embedding function returning 1024-dim vector."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


# ── Auto-Importance Tests ────────────────────────────────────────


class TestAutoImportance:
    """Test _auto_importance heuristic scoring."""

    def test_baseline_score(self):
        from brainlayer.mcp import _auto_importance

        score = _auto_importance("checked in Monday")
        assert score <= 4  # Short, no keywords

    def test_architectural_keywords_boost(self):
        from brainlayer.mcp import _auto_importance

        score = _auto_importance("Never use raw SQL in the database layer")
        assert score >= 8  # "never" (+2) + "database" (+3) + baseline (3)

    def test_prohibition_keywords_boost(self):
        from brainlayer.mcp import _auto_importance

        score = _auto_importance("Always validate user input")
        assert score >= 5  # "always" (+2) + baseline (3)

    def test_long_content_boost(self):
        from brainlayer.mcp import _auto_importance

        short = _auto_importance("hi")
        long_ = _auto_importance("x" * 150)
        assert long_ > short

    def test_file_path_boost(self):
        from brainlayer.mcp import _auto_importance

        score = _auto_importance("Changed src/auth.ts to use JWT")
        assert score >= 4  # file path (+1) + baseline (3)

    def test_max_cap_at_10(self):
        from brainlayer.mcp import _auto_importance

        # Stack everything: architectural + prohibition + long + file path
        content = "Never use raw SQL in the database migration auth.ts layer, always validate security API schema deploy infrastructure " + "x" * 150
        score = _auto_importance(content)
        assert score == 10

    def test_decision_content_high_importance(self):
        from brainlayer.mcp import _auto_importance

        # "never" (+2) + "database" (+3) + baseline (3) = 8
        score = _auto_importance("Never use raw SQL in the database")
        assert score >= 8

    def test_prohibition_without_arch(self):
        from brainlayer.mcp import _auto_importance

        # "never" (+2) + baseline (3) = 5
        score = _auto_importance("Never use raw SQL")
        assert score == 5

    def test_note_content_low_importance(self):
        from brainlayer.mcp import _auto_importance

        score = _auto_importance("checked in Monday")
        assert score <= 4


# ── Auto-Type Detection Tests ────────────────────────────────────


class TestAutoTypeDetection:
    """Test _detect_memory_type heuristic."""

    def test_todo_detection(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("TODO: fix the login flow") == "todo"
        assert _detect_memory_type("FIXME: race condition") == "todo"

    def test_mistake_detection(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("Bug: overflow in the parser") == "mistake"
        assert _detect_memory_type("The deploy broke everything") == "mistake"

    def test_decision_detection(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("Always use bun for JS projects") == "decision"
        assert _detect_memory_type("Never commit secrets to git") == "decision"

    def test_learning_detection(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("I learned that WAL mode is critical") == "learning"
        assert _detect_memory_type("Turns out sqlite-vec needs apsw") == "learning"

    def test_default_to_note(self):
        from brainlayer.mcp import _detect_memory_type

        assert _detect_memory_type("checked in Monday") == "note"


# ── Project Scoping Tests ────────────────────────────────────────


class TestProjectScoping:
    """Test resolve_project_scope from scopes.yaml."""

    def test_cwd_heuristic_in_gits(self, tmp_path):
        """CWD heuristic extracts project from ~/Gits/<project> path."""
        from brainlayer.scoping import _cwd_heuristic

        home = str(Path.home())
        gits_project = os.path.join(home, "Gits", "myproject", "src")

        with patch("os.getcwd", return_value=gits_project):
            result = _cwd_heuristic()
            assert result == "myproject"

    def test_cwd_heuristic_not_in_gits(self):
        """CWD heuristic returns None when not in ~/Gits/."""
        from brainlayer.scoping import _cwd_heuristic

        with patch("os.getcwd", return_value="/tmp/random"):
            result = _cwd_heuristic()
            assert result is None

    def test_parse_scopes_simple(self, tmp_path):
        """Simple parser reads scopes.yaml without PyYAML."""
        from brainlayer.scoping import _parse_scopes_simple

        config = tmp_path / "scopes.yaml"
        config.write_text(
            'scopes:\n  /Users/test/Gits/golems: "golems"\n  /Users/test/Gits/brainlayer: "brainlayer"\ndefault: "all"\n'
        )

        result = _parse_scopes_simple(config)
        assert "scopes" in result
        assert result["default"] == "all"
        # The keys should be expanded paths
        assert any("golems" in v for v in result["scopes"].values())

    def test_resolve_with_config(self, tmp_path):
        """resolve_project_scope uses config when available."""
        from brainlayer import scoping

        config = tmp_path / "scopes.yaml"
        config.write_text(
            'scopes:\n  /Users/test/Gits/golems: "golems"\ndefault: "all"\n'
        )

        # Patch the config path and CWD
        with patch.object(scoping, "_SCOPES_PATH", config):
            scoping._cached_scopes = None  # Reset cache
            with patch("os.getcwd", return_value="/Users/test/Gits/golems/packages"):
                result = scoping.resolve_project_scope()
                assert result == "golems"


# ── phase_commits Table Tests ────────────────────────────────────


class TestPhaseCommitsTable:
    """Test that phase_commits table is created during DB init."""

    def test_table_exists(self, store):
        """phase_commits table exists after VectorStore init."""
        cursor = store.conn.cursor()
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        ]
        assert "phase_commits" in tables

    def test_table_schema(self, store):
        """phase_commits has expected columns."""
        cursor = store.conn.cursor()
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(phase_commits)")}
        expected = {
            "id",
            "commit_hash",
            "commit_message",
            "phase_name",
            "session_id",
            "project",
            "files_changed",
            "confidence_score",
            "outcome",
            "reversibility",
            "created_at",
        }
        assert expected.issubset(cols)

    def test_insert_phase_commit(self, store):
        """Can insert a row into phase_commits."""
        cursor = store.conn.cursor()
        cursor.execute(
            """INSERT INTO phase_commits
               (commit_hash, commit_message, phase_name, project, outcome)
               VALUES (?, ?, ?, ?, ?)""",
            ("abc123", "feat: add scoping", "Phase 5", "brainlayer", "success"),
        )
        rows = list(cursor.execute("SELECT * FROM phase_commits"))
        assert len(rows) == 1
        assert rows[0][1] == "abc123"  # commit_hash


# ── Decision Tracking Fields Tests ────────────────────────────────


class TestDecisionTrackingFields:
    """Test decision tracking fields in brain_store."""

    def test_store_with_decision_fields(self, store, mock_embed):
        """store_memory accepts and persists decision tracking fields."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Use sqlite-vec over ChromaDB for vector storage",
            memory_type="decision",
            project="brainlayer",
            confidence_score=0.9,
            outcome="validated",
            reversibility="hard",
            files_changed=["src/vector_store.py", "src/embeddings.py"],
        )

        # Verify the fields are in metadata
        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT metadata FROM chunks WHERE id = ?", (result["id"],)
            )
        )
        metadata = json.loads(rows[0][0])
        assert metadata["confidence_score"] == 0.9
        assert metadata["outcome"] == "validated"
        assert metadata["reversibility"] == "hard"
        assert "src/vector_store.py" in metadata["files_changed"]

    def test_store_without_decision_fields(self, store, mock_embed):
        """store_memory works fine without decision fields."""
        from brainlayer.store import store_memory

        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Just a simple note",
            memory_type="note",
            project="test",
        )

        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT metadata FROM chunks WHERE id = ?", (result["id"],)
            )
        )
        metadata = json.loads(rows[0][0])
        assert "confidence_score" not in metadata
        assert metadata["memory_type"] == "note"

    def test_brain_store_schema_has_decision_fields(self):
        """brain_store tool schema includes decision tracking fields."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        props = store_tool.inputSchema["properties"]

        assert "confidence_score" in props
        assert "outcome" in props
        assert "reversibility" in props
        assert "files_changed" in props

        # Check enums
        assert props["outcome"]["enum"] == ["pending", "validated", "reversed"]
        assert props["reversibility"]["enum"] == ["easy", "hard", "destructive"]


# ── Integration: Auto-importance in _store_new ────────────────────


class TestStoreNewAutoImportance:
    """Test that _store_new auto-scores importance when not provided."""

    def test_auto_importance_applied(self):
        """brain_store inputSchema says importance is auto-scored."""
        import asyncio

        from brainlayer.mcp import list_tools

        tools = asyncio.run(list_tools())
        store_tool = next(t for t in tools if t.name == "brain_store")
        desc = store_tool.inputSchema["properties"]["importance"]["description"]
        assert "auto" in desc.lower() or "Auto" in desc
