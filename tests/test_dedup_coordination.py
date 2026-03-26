"""Tests for session-scoped dedup coordination between BrainLayer hooks.

Covers:
- Coordination file read/write/atomic semantics
- Chunk registration and dedup
- Handoff prompt detection
- Graceful degradation on missing/corrupt files
"""

import json
import os
import sys

import pytest

# Add hooks dir to path so we can import dedup_coordination
HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "hooks")
sys.path.insert(0, HOOKS_DIR)

from dedup_coordination import (
    coord_path,
    get_injected_ids,
    is_handoff_prompt,
    mark_handoff_session,
    read_coord,
    register_chunks,
    write_coord,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def session_id():
    return "test-session-abc123"


@pytest.fixture(autouse=True)
def cleanup_coord_file(session_id):
    """Remove coordination file after each test."""
    yield
    path = coord_path(session_id)
    try:
        os.unlink(path)
    except OSError:
        pass


# ── Coordination File Read/Write ─────────────────────────────────────────────


class TestCoordFileIO:
    def test_write_and_read(self, session_id):
        data = {
            "session_id": session_id,
            "injected_chunks": [],
            "injected_ids_set": [],
            "total_tokens_injected": 0,
        }
        ok = write_coord(session_id, data)
        assert ok is True

        result = read_coord(session_id)
        assert result is not None
        assert result["session_id"] == session_id
        assert result["schema_version"] == 1

    def test_read_nonexistent_returns_none(self):
        result = read_coord("nonexistent-session-xyz")
        assert result is None

    def test_read_corrupt_returns_none(self, session_id):
        path = coord_path(session_id)
        with open(path, "w") as f:
            f.write("not valid json{{{")
        result = read_coord(session_id)
        assert result is None

    def test_read_wrong_schema_version_returns_none(self, session_id):
        path = coord_path(session_id)
        with open(path, "w") as f:
            json.dump({"schema_version": 999, "session_id": session_id}, f)
        result = read_coord(session_id)
        assert result is None

    def test_atomic_write(self, session_id):
        """Verify no .tmp files are left behind."""
        data = {"injected_chunks": [], "injected_ids_set": [], "total_tokens_injected": 0}
        write_coord(session_id, data)

        # Check no .tmp files in /tmp matching our pattern
        coord = coord_path(session_id)
        dir_name = os.path.dirname(coord)
        tmp_files = [f for f in os.listdir(dir_name) if f.endswith(".tmp") and "brainlayer" in f]
        # Should be empty (atomic rename cleans up)
        assert len(tmp_files) == 0


# ── Chunk Registration ───────────────────────────────────────────────────────


class TestRegisterChunks:
    def test_register_new_chunks(self, session_id):
        data = register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-1", "chunk-2"],
            source_hook="SessionStart",
            briefs=["Decision about auth", "Milestone: shipped v2"],
            token_estimate=150,
        )
        assert "chunk-1" in data["injected_ids_set"]
        assert "chunk-2" in data["injected_ids_set"]
        assert data["total_tokens_injected"] == 150
        assert len(data["injected_chunks"]) == 2

    def test_register_dedup_existing(self, session_id):
        # First registration
        register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-1", "chunk-2"],
            source_hook="SessionStart",
            token_estimate=100,
        )

        # Second registration with overlap
        data = register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-2", "chunk-3"],
            source_hook="UserPromptSubmit",
            token_estimate=50,
        )
        # chunk-2 should NOT be duplicated
        assert data["injected_ids_set"].count("chunk-2") == 1
        assert "chunk-3" in data["injected_ids_set"]
        # Only chunk-3 should be added as new entry
        assert len(data["injected_chunks"]) == 3  # 2 from first + 1 new
        assert data["total_tokens_injected"] == 150  # 100 + 50

    def test_register_preserves_source_hook(self, session_id):
        register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-1"],
            source_hook="SessionStart",
        )
        data = register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-2"],
            source_hook="UserPromptSubmit",
        )
        sources = [c["source_hook"] for c in data["injected_chunks"]]
        assert "SessionStart" in sources
        assert "UserPromptSubmit" in sources


# ── Get Injected IDs ─────────────────────────────────────────────────────────


class TestGetInjectedIds:
    def test_returns_empty_for_new_session(self, session_id):
        ids = get_injected_ids(session_id)
        assert ids == set()

    def test_returns_registered_ids(self, session_id):
        register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-a", "chunk-b"],
            source_hook="SessionStart",
        )
        ids = get_injected_ids(session_id)
        assert ids == {"chunk-a", "chunk-b"}


# ── Handoff Detection ────────────────────────────────────────────────────────


class TestHandoffDetection:
    def test_handoff_keyword(self):
        assert is_handoff_prompt("Please handoff to coachClaude")

    def test_session_handoff(self):
        assert is_handoff_prompt("This is a session-handoff from brainClaude")

    def test_pick_up_where(self):
        assert is_handoff_prompt("Pick up where we left off with the auth refactor")

    def test_resume_session(self):
        assert is_handoff_prompt("Resume session from yesterday")

    def test_normal_prompt_not_handoff(self):
        assert not is_handoff_prompt("Fix the bug in the authentication module")

    def test_short_prompt_not_handoff(self):
        assert not is_handoff_prompt("hello")

    def test_code_prompt_not_handoff(self):
        assert not is_handoff_prompt("Add a new test for the vector store search function")

    def test_case_insensitive(self):
        assert is_handoff_prompt("HANDOFF to the next agent")


# ── Mark Handoff Session ─────────────────────────────────────────────────────


class TestMarkHandoffSession:
    def test_mark_creates_file(self, session_id):
        mark_handoff_session(session_id)
        data = read_coord(session_id)
        assert data is not None
        assert data["is_handoff_session"] is True

    def test_mark_preserves_existing_data(self, session_id):
        register_chunks(
            session_id=session_id,
            chunk_ids=["chunk-1"],
            source_hook="SessionStart",
        )
        mark_handoff_session(session_id)
        data = read_coord(session_id)
        assert data["is_handoff_session"] is True
        assert "chunk-1" in data["injected_ids_set"]
