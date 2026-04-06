"""Rule-based detection for user corrections in prompts and realtime chunks."""

from __future__ import annotations

import re

CORRECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"(?:^|\b)(?:no|nope|wrong|incorrect|that'?s?\s+(?:not|wrong|incorrect))",
            re.IGNORECASE,
        ),
        "factual",
    ),
    (
        re.compile(
            r"(?:^|\b)(?:"
            r"don'?t|do\s+not|stop|never|please\s+don'?t|"
            r"i\s+don'?t\s+(?:want|like)|"
            r"i\s+prefer\b.+\b(?:instead|rather\s+than)\b|"
            r"i\s+want\b.+\binstead\b|"
            r"i\s+like\b.+\bbetter\b"
            r")",
            re.IGNORECASE,
        ),
        "preference",
    ),
    (
        re.compile(
            r"(?:^|\b)(?:actually|i\s+meant|i\s+said|not\s+[\w]+[,]\s+(?:it'?s|but))",
            re.IGNORECASE,
        ),
        "factual",
    ),
    (
        re.compile(
            r"(?:^|\b)(?:"
            r"too\s+(?:long|short|verbose|brief)|"
            r"(?:change|fix|adjust)\s+(?:the\s+)?(?:format|style|tone)|"
            r"(?:the\s+)?(?:format|style|tone)\s+(?:is\s+)?(?:wrong|off)"
            r")",
            re.IGNORECASE,
        ),
        "style",
    ),
    (
        re.compile(r"(?:לא\s+נכון|טעות|לא\s+ככה|תתקן|שגוי)"),
        "factual",
    ),
]


def detect_correction(prompt: str) -> str | None:
    """Detect whether text looks like a user correction and return its category."""
    stripped = prompt.strip()
    if len(stripped) < 3:
        return None
    for pattern, category in CORRECTION_PATTERNS:
        if pattern.search(stripped):
            return category
    return None


def build_correction_tags(content: str) -> list[str]:
    """Return canonical correction tags for auto-tagging persisted chunks."""
    category = detect_correction(content)
    if not category:
        return []
    return ["correction", f"correction:{category}", "auto-detected"]
