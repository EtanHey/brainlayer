"""Chunk-origin classification helpers."""

from __future__ import annotations

CHUNK_ORIGIN_USER_EXPLICIT = "user_explicit"
CHUNK_ORIGIN_AGENT_EXPLICIT = "agent_explicit"
CHUNK_ORIGIN_AUTO_SUMMARY = "auto_summary"
CHUNK_ORIGIN_MANUAL = "manual"
CHUNK_ORIGIN_GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
CHUNK_ORIGIN_GROQ = "groq"
CHUNK_ORIGIN_OLLAMA = "ollama"
CHUNK_ORIGIN_MLX = "mlx"
CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT = "precompact_checkpoint"
CHUNK_ORIGIN_UNKNOWN = "unknown"

VALID_CHUNK_ORIGINS = frozenset(
    {
        CHUNK_ORIGIN_USER_EXPLICIT,
        CHUNK_ORIGIN_AGENT_EXPLICIT,
        CHUNK_ORIGIN_AUTO_SUMMARY,
        CHUNK_ORIGIN_MANUAL,
        CHUNK_ORIGIN_GEMINI_FLASH_LITE,
        CHUNK_ORIGIN_GROQ,
        CHUNK_ORIGIN_OLLAMA,
        CHUNK_ORIGIN_MLX,
        CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
        CHUNK_ORIGIN_UNKNOWN,
    }
)

_DIRECT_PRECOMPACT_PREFIXES = (
    "[precompact checkpoint]",
    "# precompact checkpoint",
    "session-restore",
    "# session-restore",
)
_WRAPPED_PRECOMPACT_MARKERS = (
    'content="[precompact checkpoint]',
    "content='[precompact checkpoint]",
    '"content": "[precompact checkpoint]',
    "'content': '[precompact checkpoint]",
)


def is_precompact_checkpoint_content(content: str | None) -> bool:
    """Return True when content is a stored PreCompact checkpoint or its write prompt."""
    if not content:
        return False

    stripped = content.lstrip().casefold()
    if stripped.startswith(_DIRECT_PRECOMPACT_PREFIXES):
        return True

    prefix = stripped[:1024]
    return any(marker in prefix for marker in _WRAPPED_PRECOMPACT_MARKERS)


def detect_chunk_origin(content: str | None, explicit_origin: str | None = None) -> str:
    """Classify the origin enum for a chunk."""
    if explicit_origin in VALID_CHUNK_ORIGINS and explicit_origin != CHUNK_ORIGIN_UNKNOWN:
        return explicit_origin
    if is_precompact_checkpoint_content(content):
        return CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT
    return CHUNK_ORIGIN_UNKNOWN
