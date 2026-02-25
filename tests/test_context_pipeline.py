"""Tests for Gate 1: Fix dead context pipeline.

conversation_id and position must be populated in chunks table
so that get_context() can reconstruct conversation flow and
entity extraction can use surrounding context for disambiguation.
"""

import json
from pathlib import Path

from brainlayer.pipeline.chunk import Chunk, chunk_content
from brainlayer.pipeline.classify import ClassifiedContent, ContentType, classify_content
from brainlayer.vector_store import VectorStore

# ── classify_content should extract session_id ──


class TestClassifyExtractsSessionId:
    """classify_content must thread sessionId from JSONL entries into metadata."""

    def test_user_entry_gets_session_id(self):
        """User entries should carry sessionId in metadata."""
        entry = {
            "type": "user",
            "sessionId": "abc-123-def",
            "message": {"role": "user", "content": "How do I implement auth?"},
            "timestamp": "2026-02-25T14:00:00Z",
        }
        result = classify_content(entry)
        assert result is not None
        assert result.metadata.get("session_id") == "abc-123-def"

    def test_assistant_entry_gets_session_id(self):
        """Assistant entries should carry sessionId in metadata."""
        entry = {
            "type": "assistant",
            "sessionId": "abc-123-def",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Here is a detailed explanation of authentication patterns."}
                ],
            },
        }
        result = classify_content(entry)
        assert result is not None
        assert result.metadata.get("session_id") == "abc-123-def"

    def test_missing_session_id_is_none(self):
        """Entries without sessionId should have session_id=None in metadata."""
        entry = {
            "type": "user",
            "message": {"role": "user", "content": "How do I implement auth?"},
        }
        result = classify_content(entry)
        assert result is not None
        # session_id should be None or absent, not crash
        assert result.metadata.get("session_id") is None

    def test_entry_timestamp_in_metadata(self):
        """Entry timestamp should be carried in metadata for ordering."""
        entry = {
            "type": "user",
            "sessionId": "abc-123",
            "timestamp": "2026-02-25T14:00:00Z",
            "message": {"role": "user", "content": "How do I implement auth?"},
        }
        result = classify_content(entry)
        assert result is not None
        assert result.metadata.get("timestamp") == "2026-02-25T14:00:00Z"

    def test_entry_type_in_metadata(self):
        """Entry type (user/assistant) should be carried as sender role."""
        entry = {
            "type": "user",
            "sessionId": "abc-123",
            "message": {"role": "user", "content": "How do I implement auth?"},
        }
        result = classify_content(entry)
        assert result is not None
        assert result.metadata.get("sender") == "user"


# ── chunk_content should preserve session metadata ──


class TestChunkPreservesSessionMetadata:
    """Chunking must not lose session_id from metadata."""

    def test_single_chunk_preserves_session_id(self):
        """A small text that becomes one chunk should keep session_id."""
        classified = ClassifiedContent(
            content="Short user question about auth",
            content_type=ContentType.USER_MESSAGE,
            value=ContentType.USER_MESSAGE,  # doesn't matter for this test
            metadata={"session_id": "sess-001", "timestamp": "2026-02-25T14:00:00Z"},
        )
        # chunk_content expects value to be ContentValue, fix:
        from brainlayer.pipeline.classify import ContentValue

        classified.value = ContentValue.HIGH
        chunks = chunk_content(classified)
        assert len(chunks) >= 1
        assert chunks[0].metadata.get("session_id") == "sess-001"

    def test_split_chunks_preserve_session_id(self):
        """When text is split into multiple chunks, all should keep session_id."""
        # Create content large enough to be split
        long_content = "This is a detailed paragraph about authentication. " * 100
        classified = ClassifiedContent(
            content=long_content,
            content_type=ContentType.ASSISTANT_TEXT,
            value=None,
            metadata={"session_id": "sess-002"},
        )
        from brainlayer.pipeline.classify import ContentValue

        classified.value = ContentValue.MEDIUM
        chunks = chunk_content(classified)
        assert len(chunks) > 1, "Content should be split into multiple chunks"
        for chunk in chunks:
            assert chunk.metadata.get("session_id") == "sess-002"


# ── index_chunks_to_sqlite should populate conversation_id and position ──


class TestIndexPopulatesContext:
    """The indexer must set conversation_id and position on stored chunks."""

    def test_conversation_id_stored(self, tmp_path):
        """Chunks should have conversation_id set from session_id metadata."""
        from unittest.mock import patch

        db_path = tmp_path / "test.db"

        # Create chunks with session_id in metadata
        chunks = [
            Chunk(
                content="First message about auth",
                content_type=ContentType.USER_MESSAGE,
                value=ContentType.USER_MESSAGE,  # will be overridden
                metadata={"session_id": "sess-abc-123"},
                char_count=25,
            ),
            Chunk(
                content="Here is how to implement authentication in your app with JWT tokens and refresh logic.",
                content_type=ContentType.ASSISTANT_TEXT,
                value=ContentType.ASSISTANT_TEXT,
                metadata={"session_id": "sess-abc-123"},
                char_count=80,
            ),
        ]
        # Fix value types
        from brainlayer.pipeline.classify import ContentValue

        chunks[0].value = ContentValue.HIGH
        chunks[1].value = ContentValue.MEDIUM

        # Mock embeddings to avoid loading the model
        fake_embeddings = [[0.1] * 1024, [0.2] * 1024]

        with patch("brainlayer.index_new.embed_chunks") as mock_embed:
            from brainlayer.embeddings import EmbeddedChunk

            mock_embed.return_value = [
                EmbeddedChunk(chunk=chunks[0], embedding=fake_embeddings[0]),
                EmbeddedChunk(chunk=chunks[1], embedding=fake_embeddings[1]),
            ]

            from brainlayer.index_new import index_chunks_to_sqlite

            count = index_chunks_to_sqlite(
                chunks,
                source_file="test.jsonl",
                project="test-project",
                db_path=db_path,
            )
            assert count == 2

        # Verify conversation_id and position are stored
        store = VectorStore(db_path)
        cursor = store.conn.cursor()
        rows = list(cursor.execute(
            "SELECT id, conversation_id, position FROM chunks ORDER BY position"
        ))
        assert len(rows) == 2

        # conversation_id should be set
        assert rows[0][1] == "sess-abc-123"
        assert rows[1][1] == "sess-abc-123"

        # position should be sequential
        assert rows[0][2] == 0
        assert rows[1][2] == 1

        store.close()

    def test_position_is_sequential(self, tmp_path):
        """Positions should be 0-indexed and sequential within a file."""
        from unittest.mock import patch

        db_path = tmp_path / "test.db"

        chunks = []
        for i in range(5):
            chunks.append(
                Chunk(
                    content=f"Message number {i} with enough text to pass filtering thresholds easily",
                    content_type=ContentType.USER_MESSAGE,
                    value=ContentType.USER_MESSAGE,
                    metadata={"session_id": "sess-xyz"},
                    char_count=60,
                )
            )
            from brainlayer.pipeline.classify import ContentValue

            chunks[-1].value = ContentValue.HIGH

        fake_embeddings = [[0.1] * 1024] * 5

        with patch("brainlayer.index_new.embed_chunks") as mock_embed:
            from brainlayer.embeddings import EmbeddedChunk

            mock_embed.return_value = [
                EmbeddedChunk(chunk=c, embedding=e) for c, e in zip(chunks, fake_embeddings)
            ]

            from brainlayer.index_new import index_chunks_to_sqlite

            index_chunks_to_sqlite(
                chunks, source_file="test.jsonl", project="test", db_path=db_path
            )

        store = VectorStore(db_path)
        cursor = store.conn.cursor()
        rows = list(cursor.execute("SELECT position FROM chunks ORDER BY position"))
        positions = [r[0] for r in rows]
        assert positions == [0, 1, 2, 3, 4]
        store.close()

    def test_missing_session_id_uses_file_stem(self, tmp_path):
        """Chunks without session_id in metadata should use source file stem as fallback."""
        from unittest.mock import patch

        db_path = tmp_path / "test.db"

        chunk = Chunk(
            content="A message without session ID metadata for some reason or another here",
            content_type=ContentType.USER_MESSAGE,
            value=ContentType.USER_MESSAGE,
            metadata={},  # No session_id
            char_count=60,
        )
        from brainlayer.pipeline.classify import ContentValue

        chunk.value = ContentValue.HIGH

        with patch("brainlayer.index_new.embed_chunks") as mock_embed:
            from brainlayer.embeddings import EmbeddedChunk

            mock_embed.return_value = [
                EmbeddedChunk(chunk=chunk, embedding=[0.1] * 1024)
            ]

            from brainlayer.index_new import index_chunks_to_sqlite

            # Source file path contains session UUID
            index_chunks_to_sqlite(
                [chunk],
                source_file="/path/to/abc-def-123.jsonl",
                project="test",
                db_path=db_path,
            )

        store = VectorStore(db_path)
        cursor = store.conn.cursor()
        row = list(cursor.execute("SELECT conversation_id FROM chunks"))[0]
        # Should fall back to file stem
        assert row[0] == "abc-def-123"
        store.close()


# ── get_context should work with populated fields ──


class TestGetContextWorks:
    """get_context() should return surrounding chunks when conversation_id/position are set."""

    def _make_store_with_conversation(self, tmp_path):
        """Create a store with a 5-chunk conversation."""
        db_path = tmp_path / "test.db"
        store = VectorStore(db_path)
        cursor = store.conn.cursor()

        for i in range(5):
            cursor.execute(
                """INSERT INTO chunks
                   (id, content, metadata, source_file, project, content_type,
                    char_count, conversation_id, position)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"chunk-{i}",
                    f"Content of message {i} in the conversation",
                    "{}",
                    "test.jsonl",
                    "test",
                    "user_message" if i % 2 == 0 else "assistant_text",
                    50,
                    "conv-001",
                    i,
                ),
            )

        return store

    def test_get_context_returns_surrounding_chunks(self, tmp_path):
        """get_context for chunk-2 should return chunks 0-4."""
        store = self._make_store_with_conversation(tmp_path)
        result = store.get_context("chunk-2", before=2, after=2)

        assert "error" not in result, f"get_context returned error: {result.get('error')}"
        assert "context" in result
        context = result["context"]
        assert len(context) == 5  # chunks 0,1,2,3,4
        store.close()

    def test_get_context_marks_target(self, tmp_path):
        """The target chunk should be marked in the result."""
        store = self._make_store_with_conversation(tmp_path)
        result = store.get_context("chunk-2", before=1, after=1)

        context = result["context"]
        targets = [c for c in context if c.get("is_target")]
        assert len(targets) == 1
        assert targets[0]["id"] == "chunk-2"
        store.close()

    def test_get_context_respects_before_after(self, tmp_path):
        """Before/after params should limit the window."""
        store = self._make_store_with_conversation(tmp_path)
        result = store.get_context("chunk-2", before=1, after=1)

        context = result["context"]
        ids = [c["id"] for c in context]
        assert ids == ["chunk-1", "chunk-2", "chunk-3"]
        store.close()


# ── Full pipeline integration: JSONL → chunks with context ──


class TestFullPipelineIntegration:
    """End-to-end: JSONL file → indexed chunks with conversation_id + position."""

    def _write_jsonl(self, path: Path, entries: list[dict]):
        """Write entries as JSONL."""
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_index_fast_populates_context(self, tmp_path):
        """Full pipeline from JSONL to stored chunks should set conversation_id and position."""
        from unittest.mock import patch

        # Create a fake JSONL file
        jsonl_path = tmp_path / "sess-001.jsonl"
        entries = [
            {
                "type": "user",
                "sessionId": "sess-001",
                "timestamp": "2026-02-25T14:00:00Z",
                "message": {"role": "user", "content": "How do I implement authentication in my app?"},
            },
            {
                "type": "assistant",
                "sessionId": "sess-001",
                "timestamp": "2026-02-25T14:00:05Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is how to implement authentication. You need to set up JWT tokens with a refresh token rotation strategy for security.",
                        }
                    ],
                },
            },
            {
                "type": "user",
                "sessionId": "sess-001",
                "timestamp": "2026-02-25T14:00:30Z",
                "message": {"role": "user", "content": "Can you show me a code example for the JWT part?"},
            },
        ]
        self._write_jsonl(jsonl_path, entries)

        db_path = tmp_path / "test.db"

        # Mock embeddings to avoid loading model
        with patch("brainlayer.index_new.embed_chunks") as mock_embed:
            from brainlayer.embeddings import EmbeddedChunk
            from brainlayer.index_new import index_chunks_to_sqlite
            from brainlayer.pipeline.chunk import chunk_content
            from brainlayer.pipeline.classify import classify_content
            from brainlayer.pipeline.extract import parse_jsonl

            # Run the same pipeline as CLI index_fast
            all_chunks = []
            for entry in parse_jsonl(jsonl_path):
                classified = classify_content(entry)
                if classified is not None:
                    chunks = chunk_content(classified)
                    all_chunks.extend(chunks)

            assert len(all_chunks) >= 2, f"Expected at least 2 chunks, got {len(all_chunks)}"

            # All chunks should have session_id in metadata
            for chunk in all_chunks:
                assert chunk.metadata.get("session_id") == "sess-001", (
                    f"Chunk missing session_id: {chunk.metadata}"
                )

            # Mock the embeddings
            mock_embed.return_value = [
                EmbeddedChunk(chunk=c, embedding=[0.1] * 1024) for c in all_chunks
            ]

            count = index_chunks_to_sqlite(
                all_chunks,
                source_file=str(jsonl_path),
                project="test",
                db_path=db_path,
            )
            assert count >= 2

        # Verify DB state
        store = VectorStore(db_path)
        cursor = store.conn.cursor()
        rows = list(cursor.execute(
            "SELECT conversation_id, position FROM chunks ORDER BY position"
        ))

        # All chunks should have conversation_id set
        for row in rows:
            assert row[0] is not None, "conversation_id should not be NULL"
            assert row[1] is not None, "position should not be NULL"

        # conversation_id should be the sessionId
        assert rows[0][0] == "sess-001"

        # Positions should be sequential
        positions = [r[1] for r in rows]
        assert positions == list(range(len(positions)))

        store.close()
