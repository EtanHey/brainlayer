"""Stage 2: Classify content blocks by type and value."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContentType(Enum):
    """Types of content in Claude Code conversations."""
    AI_CODE = "ai_code"           # Code written by Claude (HIGH VALUE)
    STACK_TRACE = "stack_trace"   # Error traces (HIGH VALUE - preserve exact)
    USER_MESSAGE = "user_message" # Human questions (HIGH VALUE)
    ASSISTANT_TEXT = "assistant_text"  # Claude's explanations
    FILE_READ = "file_read"       # Code from tool reads (MEDIUM)
    GIT_DIFF = "git_diff"         # Git changes (MEDIUM)
    BUILD_LOG = "build_log"       # Build/test output (LOW - summarize)
    DIRECTORY_LISTING = "dir_listing"  # ls output (LOW - structure only)
    CONFIG = "config"             # Config files (MEDIUM)
    NOISE = "noise"               # progress, queue-operation (SKIP)


class ContentValue(Enum):
    """Value level for preservation decisions."""
    HIGH = "high"       # Preserve verbatim
    MEDIUM = "medium"   # Context-dependent
    LOW = "low"         # Summarize or mask


@dataclass
class ClassifiedContent:
    """A classified piece of content."""
    content: str
    content_type: ContentType
    value: ContentValue
    metadata: dict[str, Any]


# Stack trace detection patterns
STACK_TRACE_PATTERNS = [
    r"^Traceback \(most recent call last\):",  # Python
    r"^\s+at\s+[\w.]+\([\w.]+:\d+\)",          # Java
    r"at\s+.*\(.*:\d+:\d+\)",                   # JavaScript/Node
    r"^\s+File \".*\", line \d+",              # Python file reference
]

# Build log patterns
BUILD_LOG_PATTERNS = [
    r"^\s*\d+ passing",           # Test results
    r"^\s*\d+ failing",
    r"^npm (ERR!|WARN)",          # npm output
    r"^error\[E\d+\]:",           # Rust errors
    r"^\[[\d:]+\]",               # Timestamped logs
]


def classify_content(entry: dict) -> ClassifiedContent | None:
    """
    Classify a JSONL entry by content type and value.

    Returns None for entries that should be skipped entirely (noise).
    """
    entry_type = entry.get("type", "")

    # Skip noise types entirely
    if entry_type in ("progress", "queue-operation"):
        return None

    if entry_type == "user":
        content = entry.get("message", {}).get("content", "")
        # Check if it's a system prompt (first message, very long)
        if len(content) > 2000:
            return ClassifiedContent(
                content=content,
                content_type=ContentType.USER_MESSAGE,
                value=ContentValue.MEDIUM,  # System prompts are deduplicated elsewhere
                metadata={"is_system_prompt": True}
            )
        return ClassifiedContent(
            content=content,
            content_type=ContentType.USER_MESSAGE,
            value=ContentValue.HIGH,
            metadata={}
        )

    if entry_type == "assistant":
        message = entry.get("message", {})
        content_blocks = message.get("content", [])

        # Process each content block
        results = []
        for block in content_blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "")

                if block_type == "text":
                    text = block.get("text", "")
                    classified = _classify_text(text)
                    results.append(classified)

                elif block_type == "tool_use":
                    # Tool calls themselves are metadata
                    tool_name = block.get("name", "")
                    tool_input = block.get("input", {})
                    results.append(ClassifiedContent(
                        content=str(tool_input),
                        content_type=ContentType.ASSISTANT_TEXT,
                        value=ContentValue.LOW,
                        metadata={"tool_name": tool_name}
                    ))

                elif block_type == "tool_result":
                    # Tool results need careful classification
                    result_content = block.get("content", "")
                    classified = _classify_tool_result(result_content, block)
                    results.append(classified)

        # Return the highest-value content from this entry
        if results:
            return max(results, key=lambda x: list(ContentValue).index(x.value))

    return None


def _classify_text(text: str) -> ClassifiedContent:
    """Classify assistant text content."""
    # Check for stack traces
    for pattern in STACK_TRACE_PATTERNS:
        if re.search(pattern, text, re.MULTILINE):
            return ClassifiedContent(
                content=text,
                content_type=ContentType.STACK_TRACE,
                value=ContentValue.HIGH,
                metadata={}
            )

    # Check for code blocks (AI-generated code)
    if "```" in text:
        return ClassifiedContent(
            content=text,
            content_type=ContentType.AI_CODE,
            value=ContentValue.HIGH,
            metadata={}
        )

    return ClassifiedContent(
        content=text,
        content_type=ContentType.ASSISTANT_TEXT,
        value=ContentValue.MEDIUM,
        metadata={}
    )


def _classify_tool_result(content: str, block: dict) -> ClassifiedContent:
    """Classify tool result content."""
    # Check for stack traces in tool output
    for pattern in STACK_TRACE_PATTERNS:
        if re.search(pattern, content, re.MULTILINE):
            return ClassifiedContent(
                content=content,
                content_type=ContentType.STACK_TRACE,
                value=ContentValue.HIGH,
                metadata={}
            )

    # Check for build logs
    for pattern in BUILD_LOG_PATTERNS:
        if re.search(pattern, content, re.MULTILINE):
            return ClassifiedContent(
                content=content,
                content_type=ContentType.BUILD_LOG,
                value=ContentValue.LOW,
                metadata={"action": "summarize"}
            )

    # Git diff detection
    if content.startswith("diff --git") or "@@" in content[:500]:
        return ClassifiedContent(
            content=content,
            content_type=ContentType.GIT_DIFF,
            value=ContentValue.MEDIUM,
            metadata={}
        )

    # Directory listing detection
    if content.count("\n") > 10 and all(
        line.strip().endswith(("/", ".ts", ".js", ".py", ".json", ".md"))
        for line in content.split("\n")[:10]
        if line.strip()
    ):
        return ClassifiedContent(
            content=content,
            content_type=ContentType.DIRECTORY_LISTING,
            value=ContentValue.LOW,
            metadata={"action": "structure_only"}
        )

    # Default: file read or general output
    return ClassifiedContent(
        content=content,
        content_type=ContentType.FILE_READ,
        value=ContentValue.MEDIUM,
        metadata={}
    )
