"""Tests for content chunking."""

from brainlayer.pipeline.chunk import (
    TARGET_CHUNK_SIZE,
    _extract_code_blocks,
    chunk_content,
)
from brainlayer.pipeline.classify import (
    ClassifiedContent,
    ContentType,
    ContentValue,
)


class TestChunkContent:
    """Test content chunking logic."""

    def test_stack_traces_never_split(self):
        """Stack traces should never be split, regardless of size."""
        large_trace = "Traceback (most recent call last):\n" + "  at foo()\n" * 100
        classified = ClassifiedContent(
            content=large_trace,
            content_type=ContentType.STACK_TRACE,
            value=ContentValue.HIGH,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) == 1
        assert chunks[0].content == large_trace

    def test_small_content_single_chunk(self):
        """Content smaller than target size stays as single chunk."""
        small_content = (
            "This is a small piece of content that should be kept as a single chunk for the assistant text type."
        )
        classified = ClassifiedContent(
            content=small_content,
            content_type=ContentType.ASSISTANT_TEXT,
            value=ContentValue.MEDIUM,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) == 1
        assert chunks[0].content == small_content

    def test_large_content_split(self):
        """Content larger than target size should be split."""
        large_content = "This is a paragraph.\n\n" * 200  # >4000 chars
        classified = ClassifiedContent(
            content=large_content,
            content_type=ContentType.ASSISTANT_TEXT,
            value=ContentValue.MEDIUM,
            metadata={},
        )
        chunks = chunk_content(classified)
        assert len(chunks) > 1
        # Each chunk should be reasonable size
        for chunk in chunks:
            assert chunk.char_count <= TARGET_CHUNK_SIZE * 1.5  # Allow some overflow

    def test_observation_masking_large_output(self):
        """Large tool outputs should be masked."""
        large_output = "line of output\n" * 500  # Very large
        classified = ClassifiedContent(
            content=large_output,
            content_type=ContentType.FILE_READ,
            value=ContentValue.MEDIUM,
            metadata={},
        )
        chunks = chunk_content(classified)
        # Should be masked to single chunk
        assert len(chunks) == 1
        assert "[... " in chunks[0].content
        assert "lines elided" in chunks[0].content


class TestCodeBlockExtraction:
    """Test code block extraction from markdown."""

    def test_extracts_python_block(self):
        """Extract Python code blocks."""
        text = "Here's code:\n```python\ndef foo():\n    pass\n```\nDone."
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] == "python"
        assert "def foo():" in blocks[0][1]

    def test_extracts_multiple_blocks(self):
        """Extract multiple code blocks."""
        text = """First:
```typescript
const x = 1;
```

Second:
```python
y = 2
```
"""
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 2
        assert blocks[0][0] == "typescript"
        assert blocks[1][0] == "python"

    def test_handles_no_language(self):
        """Handle code blocks without language specifier."""
        text = "```\nsome code\n```"
        blocks = _extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0][0] is None
