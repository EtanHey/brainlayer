"""Tests for content classification."""

from brainlayer.pipeline.classify import (
    ContentType,
    ContentValue,
    classify_content,
)


class TestClassifyContent:
    """Test content classification logic."""

    def test_filters_progress_type(self):
        """Progress entries should be filtered (return None)."""
        entry = {"type": "progress", "data": {"type": "hook_progress"}}
        result = classify_content(entry)
        assert result is None

    def test_filters_queue_operation(self):
        """Queue operation entries should be filtered."""
        entry = {"type": "queue-operation", "operation": "dequeue"}
        result = classify_content(entry)
        assert result is None

    def test_classifies_user_message(self):
        """User messages should be classified as HIGH value."""
        entry = {"type": "user", "message": {"role": "user", "content": "How do I implement auth?"}}
        result = classify_content(entry)
        assert result is not None
        assert result.content_type == ContentType.USER_MESSAGE
        assert result.value == ContentValue.HIGH

    def test_detects_system_prompt(self):
        """Long first messages are likely system prompts."""
        long_content = "CLAUDE.md instructions " * 500  # >2000 chars
        entry = {"type": "user", "message": {"role": "user", "content": long_content}}
        result = classify_content(entry)
        assert result is not None
        assert result.metadata.get("is_system_prompt") is True

    def test_classifies_code_blocks(self):
        """Code blocks should be classified as AI_CODE."""
        entry = {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Here's the code:\n```python\ndef foo():\n    pass\n```",
                    }
                ],
            },
        }
        result = classify_content(entry)
        assert result is not None
        assert result.content_type == ContentType.AI_CODE
        assert result.value == ContentValue.HIGH

    def test_detects_python_stack_trace(self):
        """Python stack traces should be detected."""
        stack_trace = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    foo()
  File "test.py", line 5, in foo
    raise ValueError("oops")
ValueError: oops"""
        entry = {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": stack_trace}]},
        }
        result = classify_content(entry)
        assert result is not None
        assert result.content_type == ContentType.STACK_TRACE
        assert result.value == ContentValue.HIGH

    def test_detects_javascript_stack_trace(self):
        """JavaScript stack traces should be detected."""
        stack_trace = """Error: Something went wrong
    at foo (/app/src/index.js:10:15)
    at bar (/app/src/utils.js:25:3)
    at Object.<anonymous> (/app/src/main.js:5:1)"""
        entry = {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": stack_trace}]},
        }
        result = classify_content(entry)
        assert result is not None
        assert result.content_type == ContentType.STACK_TRACE


class TestContentValue:
    """Test value assignment logic."""

    def test_high_value_preserved(self):
        """HIGH value content should be preserved verbatim."""
        # This is a design principle - HIGH value = never summarize
        assert ContentValue.HIGH.value == "high"

    def test_low_value_can_be_masked(self):
        """LOW value content can be summarized or masked."""
        assert ContentValue.LOW.value == "low"
