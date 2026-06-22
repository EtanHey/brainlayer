"""Lightweight system-prompt detection for write-path validation."""

import re

SYSTEM_PROMPT_MARKERS = {
    "# base context",
    "## iron rules",
    "> this context contains universal rules",
    "claude.md instructions",
    "agents.md instructions for",
    "global agent instructions",
}


def looks_like_system_prompt(content: str) -> bool:
    """Detect agent/base-context prompt scaffolding that should not be indexed."""
    stripped = content.strip()
    if not stripped:
        return False

    lowered = stripped.lower()
    score = sum(1 for marker in SYSTEM_PROMPT_MARKERS if marker in lowered)

    if re.search(r"(?im)^(?:>\s*)?you are (?:codex|claude|[\w-]*(?:codex|claude)|a coding agent)\b", stripped):
        score += 2

    if re.search(r"(?im)^##\s*first:\s*load context\b", stripped):
        score += 1

    return score >= 2 or (score >= 1 and len(stripped) > 2000)
