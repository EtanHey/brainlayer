"""Tests for pipeline/chunk.py — sentence-aware splitting."""

from brainlayer.pipeline.chunk import (
    TARGET_CHUNK_SIZE,
    _split_at_sentences,
    chunk_content,
)
from brainlayer.pipeline.classify import ClassifiedContent, ContentType, ContentValue


class TestSplitAtSentences:
    """Test sentence boundary splitting."""

    def test_short_text_single_chunk(self):
        """Short text returns single chunk."""
        result = _split_at_sentences("Hello world. This is a test.")
        assert len(result) == 1

    def test_long_text_splits_at_periods(self):
        """Long text splits at sentence boundaries."""
        # Create text longer than TARGET_CHUNK_SIZE
        sentences = ["This is sentence number %d." % i for i in range(200)]
        text = " ".join(sentences)
        assert len(text) > TARGET_CHUNK_SIZE

        result = _split_at_sentences(text)
        assert len(result) > 1
        # Each chunk should end at a sentence boundary (period)
        for chunk in result[:-1]:  # Last chunk might not end with period
            assert chunk.rstrip().endswith(".")

    def test_no_sentence_boundaries_falls_back(self):
        """Text without sentence boundaries falls back to line-based."""
        text = "word " * 500  # Long text with no periods
        result = _split_at_sentences(text)
        assert len(result) >= 1


class TestChunkContent:
    """Test content-type-aware chunking."""

    def test_short_user_message_preserved(self):
        """Short user messages (>15 chars) are preserved."""
        classified = ClassifiedContent(
            content="How do I fix this bug?",
            content_type=ContentType.USER_MESSAGE,
            value=ContentValue.HIGH,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) == 1
        assert chunks[0].content == "How do I fix this bug?"

    def test_very_short_message_filtered(self):
        """Very short messages (<15 chars) are filtered out."""
        classified = ClassifiedContent(
            content="ok",
            content_type=ContentType.USER_MESSAGE,
            value=ContentValue.LOW,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) == 0

    def test_stack_trace_never_split(self):
        """Stack traces are never split, regardless of size."""
        long_trace = "Traceback (most recent call last):\n" + "\n".join(
            f"  File 'module_{i}.py', line {i}, in func_{i}" for i in range(100)
        )
        classified = ClassifiedContent(
            content=long_trace,
            content_type=ContentType.STACK_TRACE,
            value=ContentValue.HIGH,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) == 1
        assert chunks[0].content == long_trace

    def test_long_text_chunked_at_paragraphs(self):
        """Long text is chunked at paragraph boundaries."""
        paragraphs = ["This is paragraph %d with some text." % i for i in range(100)]
        long_text = "\n\n".join(paragraphs)
        classified = ClassifiedContent(
            content=long_text,
            content_type=ContentType.ASSISTANT_TEXT,
            value=ContentValue.MEDIUM,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) > 1
