"""Tests for chunk lifecycle management — supersede, archive, search filtering.

Covers:
- Schema migration (new columns exist)
- supersede_chunk() and archive_chunk() on VectorStore
- Search filtering (superseded/archived excluded by default, included with include_archived)
- MCP handler: brain_supersede with safety_check
- MCP handler: brain_archive
- brain_store with supersedes param
"""

import json
from datetime import datetime

import pytest

from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore  # noqa: F401

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore for testing."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Mock embedding function that returns a deterministic 1024-dim vector."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:50]) % 100
        return [float(seed + i) / 1000.0 for i in range(1024)]

    return _embed


def _store_chunk(store, mock_embed, content, **kwargs):
    """Helper: store a chunk and return its ID."""
    return store_memory(
        store=store,
        embed_fn=mock_embed,
        content=content,
        memory_type=kwargs.pop("memory_type", "learning"),
        **kwargs,
    )


# ── Schema Migration Tests ───────────────────────────────────────────────────


class TestSchemaLifecycleColumns:
    """Verify lifecycle columns are created on init."""

    def test_superseded_by_column_exists(self, store):
        cols = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        assert "superseded_by" in cols

    def test_aggregated_into_column_exists(self, store):
        cols = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        assert "aggregated_into" in cols

    def test_archived_at_column_exists(self, store):
        cols = {row[1] for row in store.conn.cursor().execute("PRAGMA table_info(chunks)")}
        assert "archived_at" in cols

    def test_columns_default_null(self, store, mock_embed):
        """New chunks have NULL lifecycle columns."""
        result = _store_chunk(store, mock_embed, "test content")
        chunk = store.get_chunk(result["id"])
        assert chunk["superseded_by"] is None
        assert chunk["aggregated_into"] is None
        assert chunk["archived_at"] is None


# ── VectorStore.supersede_chunk Tests ────────────────────────────────────────


class TestSupersedeChunk:
    def test_supersede_marks_old_chunk(self, store, mock_embed):
        old = _store_chunk(store, mock_embed, "Old fact: X=1")
        new = _store_chunk(store, mock_embed, "Updated fact: X=2")

        ok = store.supersede_chunk(old["id"], new["id"])
        assert ok is True

        chunk = store.get_chunk(old["id"])
        assert chunk["superseded_by"] == new["id"]

    def test_supersede_removes_from_vector_index(self, store, mock_embed):
        old = _store_chunk(store, mock_embed, "Old fact for vector")
        new = _store_chunk(store, mock_embed, "New fact for vector")

        store.supersede_chunk(old["id"], new["id"])

        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT chunk_id FROM chunk_vectors WHERE chunk_id = ?",
                (old["id"],),
            )
        )
        assert len(rows) == 0

    def test_supersede_nonexistent_old_returns_false(self, store, mock_embed):
        new = _store_chunk(store, mock_embed, "New fact")
        ok = store.supersede_chunk("nonexistent-id", new["id"])
        assert ok is False

    def test_supersede_nonexistent_new_returns_false(self, store, mock_embed):
        old = _store_chunk(store, mock_embed, "Old fact")
        ok = store.supersede_chunk(old["id"], "nonexistent-id")
        assert ok is False

    def test_new_chunk_unaffected(self, store, mock_embed):
        old = _store_chunk(store, mock_embed, "Old")
        new = _store_chunk(store, mock_embed, "New")
        store.supersede_chunk(old["id"], new["id"])

        chunk = store.get_chunk(new["id"])
        assert chunk["superseded_by"] is None


# ── VectorStore.archive_chunk Tests (updated behavior) ──────────────────────


class TestArchiveChunkLifecycle:
    def test_archive_sets_archived_at(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "To be archived")
        store.archive_chunk(result["id"])

        chunk = store.get_chunk(result["id"])
        assert chunk["archived_at"] is not None
        assert chunk["value_type"] == "ARCHIVED"

    def test_archive_timestamp_is_iso(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "Archive me")
        store.archive_chunk(result["id"])

        chunk = store.get_chunk(result["id"])
        # Should parse as ISO 8601
        dt = datetime.fromisoformat(chunk["archived_at"])
        assert dt.year >= 2026


# ── Search Filtering Tests ───────────────────────────────────────────────────


class TestSearchLifecycleFiltering:
    """Default search excludes superseded/archived; include_archived=True shows all."""

    def test_superseded_excluded_from_text_search(self, store, mock_embed):
        old = _store_chunk(store, mock_embed, "UniqueTestToken superseded search")
        new = _store_chunk(store, mock_embed, "UniqueTestToken replacement search")
        store.supersede_chunk(old["id"], new["id"])

        results = store.search(query_text="UniqueTestToken")
        ids = results["ids"][0] if results["ids"] else []
        assert old["id"] not in ids
        assert new["id"] in ids

    def test_archived_excluded_from_text_search(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "UniqueArchToken archived search")
        store.archive_chunk(result["id"])

        results = store.search(query_text="UniqueArchToken")
        ids = results["ids"][0] if results["ids"] else []
        assert result["id"] not in ids

    def test_include_archived_shows_superseded(self, store, mock_embed):
        old = _store_chunk(store, mock_embed, "UniqueHistToken history old")
        new = _store_chunk(store, mock_embed, "UniqueHistToken history new")
        store.supersede_chunk(old["id"], new["id"])

        results = store.search(query_text="UniqueHistToken", include_archived=True)
        ids = results["ids"][0] if results["ids"] else []
        assert old["id"] in ids
        assert new["id"] in ids

    def test_include_archived_shows_archived(self, store, mock_embed):
        result = _store_chunk(store, mock_embed, "UniqueHistArchToken include archived")
        store.archive_chunk(result["id"])

        results = store.search(query_text="UniqueHistArchToken", include_archived=True)
        ids = results["ids"][0] if results["ids"] else []
        assert result["id"] in ids


# ── MCP Handler: brain_supersede Tests ───────────────────────────────────────


class TestBrainSupersedeHandler:
    @pytest.fixture(autouse=True)
    def _patch_store(self, store, monkeypatch):
        """Patch _get_vector_store to return our test store."""
        self.store = store
        monkeypatch.setattr(
            "brainlayer.mcp.store_handler._get_vector_store",
            lambda: store,
        )

    @pytest.mark.asyncio
    async def test_supersede_auto_technical(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_supersede

        old = _store_chunk(self.store, mock_embed, "Test count: 100 passing")
        new = _store_chunk(self.store, mock_embed, "Test count: 200 passing")

        result = await _brain_supersede(old["id"], new["id"], safety_check="auto")
        data = json.loads(result[0].text)
        assert data["action"] == "superseded"
        assert data["old_chunk_id"] == old["id"]

    @pytest.mark.asyncio
    async def test_supersede_auto_personal_requires_confirm(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_supersede

        old = _store_chunk(self.store, mock_embed, "My health update: feeling better", memory_type="journal")
        new = _store_chunk(self.store, mock_embed, "Health update v2: fully recovered", memory_type="journal")

        result = await _brain_supersede(old["id"], new["id"], safety_check="auto")
        data = json.loads(result[0].text)
        assert data["action"] == "confirm_required"

    @pytest.mark.asyncio
    async def test_supersede_confirm_without_flag(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_supersede

        old = _store_chunk(self.store, mock_embed, "Some content")
        new = _store_chunk(self.store, mock_embed, "Replacement content")

        result = await _brain_supersede(old["id"], new["id"], safety_check="confirm", confirm=False)
        data = json.loads(result[0].text)
        assert data["action"] == "confirm_required"
        assert "confirm=true" in data["instruction"]

    @pytest.mark.asyncio
    async def test_supersede_confirm_with_flag(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_supersede

        old = _store_chunk(self.store, mock_embed, "My journal: personal thoughts")
        new = _store_chunk(self.store, mock_embed, "Updated journal: new thoughts")

        result = await _brain_supersede(old["id"], new["id"], safety_check="confirm", confirm=True)
        data = json.loads(result[0].text)
        assert data["action"] == "superseded"

    @pytest.mark.asyncio
    async def test_supersede_nonexistent_old(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_supersede

        new = _store_chunk(self.store, mock_embed, "New thing")
        result = await _brain_supersede("nonexistent", new["id"])
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_supersede_nonexistent_new(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_supersede

        old = _store_chunk(self.store, mock_embed, "Old thing")
        result = await _brain_supersede(old["id"], "nonexistent")
        assert result.isError is True


# ── MCP Handler: brain_archive Tests ─────────────────────────────────────────


class TestBrainArchiveHandler:
    @pytest.fixture(autouse=True)
    def _patch_store(self, store, monkeypatch):
        self.store = store
        monkeypatch.setattr(
            "brainlayer.mcp.store_handler._get_vector_store",
            lambda: store,
        )

    @pytest.mark.asyncio
    async def test_archive_success(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_archive

        result = _store_chunk(self.store, mock_embed, "To be archived")
        resp = await _brain_archive(result["id"])
        data = json.loads(resp[0].text)
        assert data["action"] == "archived"
        assert data["chunk_id"] == result["id"]

    @pytest.mark.asyncio
    async def test_archive_with_reason(self, mock_embed):
        from brainlayer.mcp.store_handler import _brain_archive

        result = _store_chunk(self.store, mock_embed, "Old info")
        resp = await _brain_archive(result["id"], reason="outdated")
        data = json.loads(resp[0].text)
        assert data["reason"] == "outdated"

    @pytest.mark.asyncio
    async def test_archive_nonexistent(self):
        from brainlayer.mcp.store_handler import _brain_archive

        resp = await _brain_archive("nonexistent-id")
        assert resp.isError is True


# ── brain_store with supersedes Tests ────────────────────────────────────────


class TestStoreWithSupersedes:
    @pytest.fixture(autouse=True)
    def _patch(self, store, mock_embed, monkeypatch):
        self.store = store
        self.mock_embed = mock_embed
        monkeypatch.setattr(
            "brainlayer.mcp.store_handler._get_vector_store",
            lambda: store,
        )
        # Patch embedding model to return our mock
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.embed_query = mock_embed
        monkeypatch.setattr(
            "brainlayer.mcp.store_handler._get_embedding_model",
            lambda: mock_model,
        )

    @pytest.mark.asyncio
    async def test_store_with_supersedes_marks_old(self):
        from brainlayer.mcp.store_handler import _store_new

        old = _store_chunk(self.store, self.mock_embed, "Old test count: 100")

        resp = await _store_new(
            content="New test count: 200",
            supersedes=old["id"],
        )
        # _store_new returns (text_parts, structured)
        text_parts, structured = resp
        assert structured["superseded"] == old["id"]

        # Old chunk should be superseded
        old_chunk = self.store.get_chunk(old["id"])
        assert old_chunk["superseded_by"] is not None

    @pytest.mark.asyncio
    async def test_store_with_invalid_supersedes(self):
        from brainlayer.mcp.store_handler import _store_new

        resp = await _store_new(
            content="New content",
            supersedes="nonexistent-id",
        )
        text_parts, structured = resp
        # Should warn but still succeed at storing new chunk
        assert structured["chunk_id"] is not None
        assert structured["superseded"] is None


# ── Personal Content Detection Tests ─────────────────────────────────────────


class TestPersonalContentDetection:
    def test_journal_type_is_personal(self):
        from brainlayer.mcp.store_handler import _is_personal_content

        assert _is_personal_content({"content_type": "journal", "content": "today was good"})

    def test_note_type_is_personal(self):
        from brainlayer.mcp.store_handler import _is_personal_content

        assert _is_personal_content({"content_type": "note", "content": "remember this"})

    def test_health_keyword_is_personal(self):
        from brainlayer.mcp.store_handler import _is_personal_content

        assert _is_personal_content({"content_type": "learning", "content": "health checkup results"})

    def test_finance_keyword_is_personal(self):
        from brainlayer.mcp.store_handler import _is_personal_content

        assert _is_personal_content({"content_type": "learning", "content": "finance portfolio update"})

    def test_technical_content_not_personal(self):
        from brainlayer.mcp.store_handler import _is_personal_content

        assert not _is_personal_content({"content_type": "learning", "content": "pytest runs 200 tests"})

    def test_decision_not_personal(self):
        from brainlayer.mcp.store_handler import _is_personal_content

        assert not _is_personal_content({"content_type": "decision", "content": "Use WAL mode for SQLite"})
