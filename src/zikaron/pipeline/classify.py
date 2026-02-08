"""Stage 2: Classify content blocks by type and value.

Smart filtering to maximize search quality:
- Detects and skips tool JSON garbage (e.g., {'file_path': ...})
- Preserves questions regardless of length (ends with ?)
- Filters pure acknowledgments ("yes", "ok", "do it")
- Content-type-aware thresholds
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


# =============================================================================
# SMART FILTERING CONFIGURATION
# =============================================================================

# Minimum length thresholds by content type
MIN_LENGTH_THRESHOLDS = {
    "user_message": 15,      # Short questions can be valuable
    "assistant_text": 50,    # Technical explanations can be concise
    "ai_code": 30,           # Code can be short but meaningful
    "tool_result": 50,       # Tool outputs need context
    "stack_trace": 0,        # Always keep stack traces
}
DEFAULT_MIN_LENGTH = 50

# Pure acknowledgments to skip (case-insensitive, exact match after strip)
ACKNOWLEDGMENTS = {
    "yes", "no", "ok", "okay", "sure", "done", "got it", "thanks", "thank you",
    "do it", "go ahead", "proceed", "continue", "next", "skip", "stop",
    "warmup", "y", "n", "k", "yep", "nope", "yup", "ah", "oh", "hmm",
    "ah gotcha", "gotcha", "i see", "makes sense", "perfect", "great",
}

# Patterns that indicate tool JSON garbage (stringified tool_use inputs)
TOOL_JSON_PATTERNS = [
    r"^\s*\{\s*['\"](?:command|file_path|pattern|query|url|old_string|new_string|content)['\"]:",
    r"^\s*\{\s*['\"](?:tool_name|function|input|arguments)['\"]:",
    r"^\s*\{\}$",  # Empty dict
    r"^\s*\{['\"]todos['\"]\s*:\s*\[\s*\]\s*\}$",  # Empty todos
]

# Transitional phrases to skip (assistant text that's ONLY filler, no substance)
# Only match if the ENTIRE message is transitional (short and starts with filler)
TRANSITIONAL_PATTERNS = [
    r"^(?:Let me|Now I'll|I'll now|I will now|Checking|Looking at|Running)[^.]*[:.]?$",
    r"^(?:Here's|Here is) (?:the|what)[^.]*[:.]?$",
    r"^(?:Let me check|I'll check|checking now)[:.]?$",
]
# Maximum length for transitional text (longer messages likely have substance)
MAX_TRANSITIONAL_LENGTH = 60

# High-value patterns to keep regardless of length
HIGH_VALUE_PATTERNS = [
    r"\?$",  # Questions
    r"(?:error|failed|issue|bug|broken|fix|problem|crash)",  # Debugging
    r"(?:because|decided|chose|recommend|should|must)",  # Decisions
    r"(?:todo|fixme|hack|workaround)",  # Code notes
]


class ContentType(Enum):
    """Types of content in Claude Code conversations."""
    # Conversation content
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

    # Markdown content types
    LEARNING = "learning"         # Curated learnings (HIGH VALUE)
    SKILL = "skill"               # Skill definitions (HIGH VALUE)
    PROJECT_CONFIG = "project_config"  # CLAUDE.md files (HIGH VALUE)
    RESEARCH = "research"         # Research documents (HIGH VALUE)
    PRD_ARCHIVE = "prd_archive"   # PRD archives (MEDIUM)
    VERIFICATION = "verification" # Verification rounds (LOW)
    DOCUMENTATION = "documentation"  # General markdown docs (MEDIUM)


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


# Stack trace detection patterns (ReDoS-safe: avoid unbounded .*)
STACK_TRACE_PATTERNS = [
    r"^Traceback \(most recent call last\):",  # Python
    r"^\s+at\s+[\w.]+\([\w.]+:\d+\)",          # Java
    r"at\s+[^\(]+\([^\)]+:\d+:\d+\)",          # JavaScript/Node (ReDoS-safe)
    r"^\s+File \"[^\"]+\", line \d+",          # Python file reference (ReDoS-safe)
]

# Build log patterns
BUILD_LOG_PATTERNS = [
    r"^\s*\d+ passing",           # Test results
    r"^\s*\d+ failing",
    r"^npm (ERR!|WARN)",          # npm output
    r"^error\[E\d+\]:",           # Rust errors
    r"^\[[\d:]+\]",               # Timestamped logs
]


def _extract_text_content(content: Any) -> str:
    """
    Extract text from various content formats.

    Content can be:
    - A plain string
    - A list of content blocks: [{"type": "text", "text": "..."}, ...]
    - A dict with a "text" key
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    if isinstance(content, dict) and "text" in content:
        return content.get("text", "")
    return str(content) if content else ""


# =============================================================================
# SMART FILTERING HELPERS
# =============================================================================

def _is_tool_json(content: str) -> bool:
    """Detect stringified tool_use input JSON that has no semantic value."""
    stripped = content.strip()
    for pattern in TOOL_JSON_PATTERNS:
        if re.match(pattern, stripped):
            return True
    return False


def _is_acknowledgment(content: str) -> bool:
    """Check if content is a pure acknowledgment with no semantic value."""
    # Normalize: lowercase, strip, remove trailing punctuation
    normalized = content.strip().lower().rstrip(".,!?")
    return normalized in ACKNOWLEDGMENTS


def _is_transitional(content: str) -> bool:
    """Check if content is transitional filler text.

    Only considers content transitional if it's short AND matches a pattern.
    Longer messages that start with transitional phrases but contain substance are kept.
    """
    stripped = content.strip()

    # Long messages are not transitional even if they start with filler
    if len(stripped) > MAX_TRANSITIONAL_LENGTH:
        return False

    for pattern in TRANSITIONAL_PATTERNS:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    return False


def _has_high_value_signal(content: str) -> bool:
    """Check if content has high-value patterns worth keeping regardless of length."""
    for pattern in HIGH_VALUE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False


def _should_keep_user_message(content: str) -> bool:
    """
    Smart filtering for user messages.

    Keep if:
    - Is a question (ends with ?)
    - Has high-value signals (error, bug, fix, etc.)
    - Meets minimum length threshold

    Skip if:
    - Empty
    - Pure acknowledgment
    - Too short without high-value signals
    """
    stripped = content.strip()

    # Always skip empty
    if not stripped:
        return False

    # Always skip pure acknowledgments
    if _is_acknowledgment(stripped):
        return False

    # Always keep questions
    if stripped.endswith("?"):
        return True

    # Keep if has high-value signals
    if _has_high_value_signal(stripped):
        return True

    # Otherwise, apply length threshold
    min_len = MIN_LENGTH_THRESHOLDS.get("user_message", DEFAULT_MIN_LENGTH)
    return len(stripped) >= min_len


def _should_keep_assistant_text(content: str) -> bool:
    """
    Smart filtering for assistant text.

    Skip if:
    - Tool JSON garbage
    - Transitional filler
    - Too short

    Keep if:
    - Has code blocks
    - Has high-value signals
    - Meets minimum length threshold
    """
    stripped = content.strip()

    # Always skip empty
    if not stripped:
        return False

    # Skip tool JSON garbage
    if _is_tool_json(stripped):
        return False

    # Keep if has code blocks (valuable) — check before transitional filter
    # because "Here's the code: ```python..." starts transitional but has real code
    if "```" in stripped:
        return True

    # Keep if has high-value signals
    if _has_high_value_signal(stripped):
        return True

    # Skip transitional filler (after code/signal checks)
    if _is_transitional(stripped):
        return False

    # Otherwise, apply length threshold
    min_len = MIN_LENGTH_THRESHOLDS.get("assistant_text", DEFAULT_MIN_LENGTH)
    return len(stripped) >= min_len


def classify_content(entry: dict) -> ClassifiedContent | None:
    """
    Classify a JSONL entry by content type and value.

    Uses smart filtering to maximize search quality:
    - Preserves questions regardless of length
    - Filters tool JSON garbage
    - Filters pure acknowledgments
    - Content-type-aware thresholds

    Returns None for entries that should be skipped entirely.
    """
    entry_type = entry.get("type", "")

    # Skip noise types entirely
    if entry_type in ("progress", "queue-operation"):
        return None

    if entry_type == "user":
        raw_content = entry.get("message", {}).get("content", "")
        content = _extract_text_content(raw_content)

        # Smart filtering for user messages
        if not _should_keep_user_message(content):
            return None

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
                    # Smart filtering for assistant text
                    if not _should_keep_assistant_text(text):
                        continue
                    classified = _classify_text(text)
                    results.append(classified)

                elif block_type == "tool_use":
                    # Skip tool_use - these are just API calls, not useful content
                    # The tool_result contains the actual content
                    continue

                elif block_type == "tool_result":
                    # Tool results need careful classification
                    result_content = _extract_text_content(block.get("content", ""))
                    # Skip tool JSON and short results
                    if _is_tool_json(result_content):
                        continue
                    min_len = MIN_LENGTH_THRESHOLDS.get("tool_result", DEFAULT_MIN_LENGTH)
                    if len(result_content.strip()) < min_len:
                        continue
                    classified = _classify_tool_result(result_content, block)
                    results.append(classified)

        # Return the highest-value content from this entry
        # Priority: HIGH > MEDIUM > LOW
        if results:
            return min(results, key=lambda x: list(ContentValue).index(x.value))

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
