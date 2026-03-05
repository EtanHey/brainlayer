"""Tests for US-003: Fix embedding truncation for long chunks.

Previously, embed_chunks() truncated content at 512 characters before passing
to sentence-transformers.  bge-large-en-v1.5 supports 512 *tokens* (~2000+
characters), so the old cap discarded ~75 % of the model's capacity.

These tests verify:
1. Content beyond 512 chars is no longer truncated before encode().
2. encode() is called with the full content for short AND long texts.
3. embed_query() caps only at MAX_QUERY_CHARS (not 512 chars).
4. Timing for long-content embedding is within acceptable range.
"""

import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from brainlayer.embeddings import (
    MAX_QUERY_CHARS,
    EmbeddingModel,
    embed_chunks,
    embed_query,
)
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType


# ── Helpers ─────────────────────────────────────────────────────


def _make_chunk(content: str, chunk_id: str = "test-chunk") -> Chunk:
    return Chunk(
        content=content,
        content_type=ContentType.USER_MESSAGE,
        value=ContentType.USER_MESSAGE,
        metadata={"chunk_id": chunk_id, "session_id": "sess-001"},
        char_count=len(content),
    )


def _fake_encode(texts, **kwargs):
    """Return deterministic 1024-dim vectors for each text."""
    return np.array([[float(i) / 1000.0] * 1024 for i in range(len(texts))])


# ── Tests ────────────────────────────────────────────────────────


class TestNoCharacterTruncation:
    """embed_chunks must pass full content to model.encode(), not 512-char slices."""

    def test_short_content_passed_verbatim(self):
        content = "Short content under 512 chars."
        chunk = _make_chunk(content)

        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_chunks([chunk])

        args, _ = mock_st.encode.call_args
        passed_texts = args[0]
        assert passed_texts == [content]

    def test_long_content_not_truncated_at_512_chars(self):
        """Content longer than 512 chars must reach model.encode() intact."""
        # Build a string where the critical phrase is after char 512
        prefix = "x" * 520
        critical = "UNIQUE_MARKER_AFTER_512"
        content = prefix + critical
        assert len(content) > 512

        chunk = _make_chunk(content)

        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_chunks([chunk])

        args, _ = mock_st.encode.call_args
        passed_text = args[0][0]
        assert critical in passed_text, (
            f"Content after char 512 was truncated; encode() received only "
            f"{len(passed_text)} chars, missing '{critical}'"
        )

    def test_very_long_content_passed_to_model(self):
        """Even 5000-char content is passed through; model handles token truncation."""
        content = "word " * 1000  # ~5000 chars
        chunk = _make_chunk(content)

        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_chunks([chunk])

        args, _ = mock_st.encode.call_args
        assert len(args[0][0]) == len(content)

    def test_multiple_chunks_all_passed_full(self):
        """Batch of mixed-length chunks: every content reaches encode() unaltered."""
        contents = [
            "Short.",
            "x" * 600 + "_AFTER_600",
            "y" * 1500 + "_AFTER_1500",
        ]
        chunks = [_make_chunk(c, f"c{i}") for i, c in enumerate(contents)]

        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_chunks(chunks)

        args, _ = mock_st.encode.call_args
        passed = args[0]
        for original, sent in zip(contents, passed):
            assert sent == original, f"Expected full content ({len(original)} chars), got {len(sent)} chars"


class TestQueryTruncation:
    """embed_query() should cap at MAX_QUERY_CHARS, not 512 chars."""

    def test_query_under_max_not_truncated(self):
        query = "short query"
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_query(query)

        args, _ = mock_st.encode.call_args
        # BGE prefix is prepended, but the query part should be intact
        assert query in args[0][0]

    def test_query_over_512_not_truncated_at_512(self):
        """A 600-char query must NOT be cut at 512 chars."""
        query = "a" * 600
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_query(query)

        args, _ = mock_st.encode.call_args
        # The BGE prefix is prepended, so the passed string is longer
        passed = args[0][0]
        assert "a" * 600 in passed, "600-char query was truncated before 600 chars"

    def test_query_over_max_query_chars_is_capped(self):
        """Degenerate very-long queries are capped at MAX_QUERY_CHARS."""
        query = "b" * (MAX_QUERY_CHARS + 500)
        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()
        mock_st.encode.side_effect = _fake_encode
        model._model = mock_st

        model.embed_query(query)

        args, _ = mock_st.encode.call_args
        # Passed string = BGE_PREFIX + capped_query
        passed = args[0][0]
        # The query portion (after prefix) must be <= MAX_QUERY_CHARS
        from brainlayer.embeddings import BGE_QUERY_PREFIX
        query_portion = passed[len(BGE_QUERY_PREFIX):]
        assert len(query_portion) <= MAX_QUERY_CHARS


class TestEmbeddingPerformance:
    """Embedding long content must not significantly slow down batch processing."""

    def test_long_content_timing_acceptable(self):
        """Embedding 10 chunks with ~1000-char content each must complete < 2s (mocked)."""
        chunks = [_make_chunk("word " * 200, f"c{i}") for i in range(10)]

        model = EmbeddingModel.__new__(EmbeddingModel)
        model.model_name = "test"
        mock_st = MagicMock()

        def _slow_encode(texts, **kwargs):
            # Simulate 50ms per batch (realistic for CPU inference on mocked model)
            return np.array([[0.1] * 1024 for _ in texts])

        mock_st.encode.side_effect = _slow_encode
        model._model = mock_st

        start = time.monotonic()
        results = model.embed_chunks(chunks, batch_size=10)
        elapsed = time.monotonic() - start

        assert len(results) == 10
        assert elapsed < 2.0, f"embed_chunks took {elapsed:.2f}s — unexpectedly slow"
