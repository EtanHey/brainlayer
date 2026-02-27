"""Phase 6 critical tests — 7 missing high-value tests from test quality audit.

Covers:
1. brain_store → brain_search roundtrip (via public surface)
2. brain_recall(mode="context") returns structured data
3. Search routing dispatcher (each rule)
4. brain_update re-embedding correctness
5. Concurrent DB access (two sessions)
6. DB corruption recovery
7. format="compact" output size
"""

import asyncio
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import apsw
import pytest

from brainlayer.mcp._shared import _build_compact_result
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore

# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    """Create a fresh VectorStore with isolated tmp_path DB."""
    db_path = tmp_path / "test.db"
    s = VectorStore(db_path)
    yield s
    s.close()


@pytest.fixture
def mock_embed():
    """Deterministic mock embedding: different text → different vector."""

    def _embed(text: str) -> list[float]:
        seed = sum(ord(c) for c in text[:100]) % 1000
        return [float(seed + i) / 10000.0 for i in range(1024)]

    return _embed


# ── 1. brain_store → brain_search roundtrip ────────────────────


class TestStoreSearchRoundtrip:
    """Store a memory via store_memory, search via VectorStore, verify match."""

    def test_store_search_roundtrip(self, store, mock_embed):
        """Stored chunk is findable via vector search with matching content."""
        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Authentication uses JWT tokens with RS256 signing",
            memory_type="decision",
            project="test-project",
            tags=["auth", "jwt"],
            importance=8,
        )
        chunk_id = result["id"]
        assert chunk_id.startswith("manual-")

        # Search with the same embedding
        query_vec = mock_embed("JWT authentication tokens")
        search_results = store.search(query_embedding=query_vec, n_results=5)

        docs = search_results["documents"][0]
        metas = search_results["metadatas"][0]

        assert len(docs) >= 1
        # Check the content is findable
        found = any("JWT tokens" in doc for doc in docs)
        assert found, f"Stored chunk not found in search results. Docs: {[d[:50] for d in docs]}"


# ── 2. brain_recall(mode="context") structured data ───────────


class TestRecallContext:
    """brain_recall(mode="context") returns expected structured fields."""

    def test_recall_context_returns_structured_data(self):
        """brain_recall mode=context returns active_projects, active_branches, etc."""
        mock_ctx = MagicMock()
        mock_ctx.active_projects = ["brainlayer"]
        mock_ctx.active_branches = ["main"]
        mock_ctx.active_plan = ""
        mock_ctx.recent_files = ["src/mcp/__init__.py"]
        mock_session = MagicMock()
        mock_session.session_id = "abc123"
        mock_session.project = "brainlayer"
        mock_session.branch = "main"
        mock_session.started_at = "2026-02-27T10:00:00"
        mock_session.plan_name = ""
        mock_ctx.recent_sessions = [mock_session]
        mock_ctx.format.return_value = "## Current Context\n..."

        with (
            patch("brainlayer.engine.current_context", return_value=mock_ctx),
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=MagicMock()),
        ):
            from brainlayer.mcp.search_handler import _current_context

            result = asyncio.run(_current_context(hours=24))

        # Result is a tuple of (text_content_list, structured_dict)
        assert isinstance(result, tuple)
        text_list, structured = result
        assert "active_projects" in structured
        assert "active_branches" in structured
        assert "active_plan" in structured
        assert "recent_files" in structured
        assert "recent_sessions" in structured
        assert structured["active_projects"] == ["brainlayer"]
        assert len(structured["recent_sessions"]) == 1
        assert structured["recent_sessions"][0]["session_id"] == "abc123"


# ── 3. Search routing dispatcher ──────────────────────────────


class TestSearchRouting:
    """Verify _brain_search dispatcher routes to correct handler."""

    def _make_mock_store(self):
        """Create a mock store with all methods needed by routing."""
        mock = MagicMock()
        mock.count.return_value = 100
        mock.hybrid_search.return_value = {
            "documents": [["test doc"]],
            "metadatas": [[{"project": "test", "content_type": "note", "created_at": "2026-01-01"}]],
            "distances": [[0.1]],
        }
        mock.enrich_results_with_session_context.return_value = mock.hybrid_search.return_value
        mock.get_context.return_value = {"context": [{"content": "test", "content_type": "note", "position": 1}]}
        mock.get_file_timeline.return_value = []
        return mock

    def test_search_routing_chunk_id(self):
        """chunk_id parameter → routes to context expansion."""
        mock_store = self._make_mock_store()

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._context", new_callable=AsyncMock) as mock_context,
        ):
            mock_context.return_value = [MagicMock()]
            from brainlayer.mcp.search_handler import _brain_search

            asyncio.run(_brain_search(query="expand this", chunk_id="test-chunk-001"))
        mock_context.assert_called_once_with(chunk_id="test-chunk-001", before=3, after=3)

    def test_search_routing_file_path(self):
        """file_path parameter → routes to file timeline + recall."""
        mock_store = self._make_mock_store()

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._file_timeline", new_callable=AsyncMock) as mock_timeline,
            patch("brainlayer.mcp.search_handler._recall", new_callable=AsyncMock) as mock_recall,
        ):
            mock_timeline.return_value = [MagicMock()]
            mock_recall.return_value = [MagicMock()]
            from brainlayer.mcp.search_handler import _brain_search

            asyncio.run(_brain_search(query="show history", file_path="src/auth.ts"))
        mock_timeline.assert_called_once()
        mock_recall.assert_called_once()

    def test_search_routing_current_context_signal(self):
        """'what am I working on' → routes to current_context + think."""
        with (
            patch("brainlayer.mcp.search_handler._current_context", new_callable=AsyncMock) as mock_ctx,
            patch("brainlayer.mcp.search_handler._think", new_callable=AsyncMock) as mock_think,
        ):
            mock_ctx.return_value = [MagicMock()]
            mock_think.return_value = [MagicMock()]
            from brainlayer.mcp.search_handler import _brain_search

            asyncio.run(_brain_search(query="what am I working on", project="test"))
        mock_ctx.assert_called_once()
        mock_think.assert_called_once()

    def test_search_routing_think_signal(self):
        """'how did I implement X' → routes to think handler."""
        with (
            patch("brainlayer.mcp.search_handler._think", new_callable=AsyncMock) as mock_think,
        ):
            mock_think.return_value = [MagicMock()]
            from brainlayer.mcp.search_handler import _brain_search

            asyncio.run(_brain_search(query="how did I implement authentication", project="test"))
        mock_think.assert_called_once()

    def test_search_routing_recall_signal(self):
        """'history of X' → routes to recall handler."""
        with (
            patch("brainlayer.mcp.search_handler._recall", new_callable=AsyncMock) as mock_recall,
        ):
            mock_recall.return_value = [MagicMock()]
            from brainlayer.mcp.search_handler import _brain_search

            asyncio.run(_brain_search(query="history of authentication changes", project="test"))
        mock_recall.assert_called_once()

    def test_search_routing_default_semantic(self):
        """No signal keywords → routes to default hybrid search."""
        mock_store = self._make_mock_store()
        mock_model = MagicMock()
        mock_model.embed_query.return_value = [0.1] * 1024

        with (
            patch("brainlayer.mcp.search_handler._get_vector_store", return_value=mock_store),
            patch("brainlayer.mcp.search_handler._get_embedding_model", return_value=mock_model),
        ):
            from brainlayer.mcp.search_handler import _brain_search

            result = asyncio.run(_brain_search(query="sqlite performance tuning tips", project="test"))
        # Default search calls hybrid_search on the store
        mock_store.hybrid_search.assert_called_once()


# ── 4. brain_update re-embedding ───────────────────────────────


class TestUpdateReembedding:
    """brain_update(action="update") with new content re-embeds the vector."""

    def test_update_reembedding(self, store, mock_embed):
        """Updating content changes the vector in chunk_vectors."""
        # Store initial chunk
        result = store_memory(
            store=store,
            embed_fn=mock_embed,
            content="Original authentication approach using sessions",
            memory_type="decision",
            project="test",
        )
        chunk_id = result["id"]

        # Read original embedding bytes
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT embedding FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)))
        assert len(rows) == 1
        original_embedding = rows[0][0]

        # Update content with new text (different embedding)
        new_content = "Switched to JWT tokens with RS256 for stateless auth"
        new_embedding = mock_embed(new_content)
        store.update_chunk(
            chunk_id=chunk_id,
            content=new_content,
            embedding=new_embedding,
        )

        # Read updated embedding bytes
        rows = list(cursor.execute("SELECT embedding FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)))
        assert len(rows) == 1
        updated_embedding = rows[0][0]

        # Embeddings should differ (different text → different vector)
        assert original_embedding != updated_embedding

        # Verify content was also updated in chunks table
        chunk = store.get_chunk(chunk_id)
        assert chunk["content"] == new_content


# ── 5. Concurrent DB access ───────────────────────────────────


class TestConcurrentDBAccess:
    """Two VectorStore instances on the same DB — no lost writes."""

    def test_concurrent_db_access(self, tmp_path, mock_embed):
        """Concurrent writes from two connections produce no lost chunks."""
        db_path = tmp_path / "concurrent.db"
        store1 = VectorStore(db_path)
        store2 = VectorStore(db_path)

        errors = []
        ids_per_store = {1: [], 2: []}

        def write_chunks(store_instance, store_num, count):
            try:
                for i in range(count):
                    result = store_memory(
                        store=store_instance,
                        embed_fn=mock_embed,
                        content=f"Chunk from store {store_num} number {i}: unique content here",
                        memory_type="note",
                        project=f"concurrent-test-{store_num}",
                    )
                    ids_per_store[store_num].append(result["id"])
            except Exception as e:
                errors.append(f"store{store_num}: {e}")

        # Write 5 chunks from each store concurrently
        t1 = threading.Thread(target=write_chunks, args=(store1, 1, 5))
        t2 = threading.Thread(target=write_chunks, args=(store2, 2, 5))
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)
        assert not t1.is_alive(), "Thread 1 did not finish within 30s"
        assert not t2.is_alive(), "Thread 2 did not finish within 30s"

        assert not errors, f"Concurrent write errors: {errors}"

        # Verify all chunks exist — no lost writes
        all_ids = ids_per_store[1] + ids_per_store[2]
        assert len(all_ids) == 10, f"Expected 10 total IDs, got {len(all_ids)}"

        # Verify each chunk is actually in the DB
        cursor = store1.conn.cursor()
        for chunk_id in all_ids:
            rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
            assert len(rows) == 1, f"Lost chunk: {chunk_id}"

        store1.close()
        store2.close()


# ── 6. DB corruption recovery ─────────────────────────────────


class TestDBCorruptionRecovery:
    """Malformed DB file → predictable error, not silent data loss."""

    def test_corrupt_db_recovery(self, tmp_path):
        """Opening a corrupt DB file raises a clear error."""
        corrupt_path = tmp_path / "corrupt.db"
        corrupt_path.write_bytes(b"THIS IS NOT A VALID SQLITE DATABASE FILE AT ALL")

        with pytest.raises((apsw.NotADBError, apsw.CorruptError)):
            VectorStore(corrupt_path)

    def test_empty_file_handled(self, tmp_path):
        """Opening an empty file (0 bytes) initializes a fresh DB."""
        empty_path = tmp_path / "empty.db"
        empty_path.touch()

        # This should succeed — SQLite treats empty files as new databases
        store = VectorStore(empty_path)
        assert store.count() == 0
        store.close()


# ── 7. format="compact" output size ───────────────────────────


class TestCompactFormatSize:
    """format='compact' produces fewer keys and smaller content than full."""

    def test_compact_format_output_size(self):
        """Compact output has fewer keys and content truncated to 500 chars."""
        full_item = {
            "score": 0.92,
            "project": "brainlayer",
            "content_type": "ai_code",
            "content": "x" * 1000,
            "source_file": "src/mcp/__init__.py",
            "date": "2026-02-15",
            "source": "claude_code",
            "summary": "MCP server implementation",
            "tags": ["mcp", "python"],
            "intent": "implementing",
            "importance": 8,
            "chunk_id": "abc-123-def",
            "session_summary": "Worked on MCP tools",
            "session_outcome": "completed",
            "session_quality": 0.9,
        }

        compact = _build_compact_result(full_item)

        # Compact should have fewer keys
        assert len(compact) < len(full_item)

        # Content truncated to 500
        assert len(compact["content"]) <= 500

        # Core fields present
        assert "score" in compact
        assert "project" in compact
        assert "source_file" in compact

        # Verbose fields dropped
        for dropped in (
            "content_type",
            "tags",
            "intent",
            "chunk_id",
            "session_summary",
            "session_outcome",
            "session_quality",
        ):
            assert dropped not in compact, f"'{dropped}' should be dropped in compact format"

    def test_compact_total_chars_less_than_full(self):
        """Total serialized size of compact is strictly less than full."""
        full_item = {
            "score": 0.85,
            "project": "golems",
            "content_type": "user_message",
            "content": "A detailed discussion about authentication patterns " * 20,
            "source_file": "packages/auth/src/index.ts",
            "date": "2026-02-20",
            "summary": "Auth patterns discussion",
            "tags": ["auth", "patterns", "security"],
            "intent": "discussing",
            "importance": 7,
            "chunk_id": "xyz-789",
        }

        compact = _build_compact_result(full_item)

        full_size = len(json.dumps(full_item))
        compact_size = len(json.dumps(compact))

        assert compact_size < full_size, (
            f"Compact ({compact_size} chars) should be smaller than full ({full_size} chars)"
        )
