"""Rule-based detection for user corrections in prompts and realtime chunks."""

from __future__ import annotations

import os
import re

_RELAY_HEADER = re.compile(
    r"^\s*\[(?=[^\]\n]{0,80}\b(?:agent|brainlayer|claude|codex|cursor|golem|lead|orc|worker)\b)"
    r"[^\]\n]{0,80}?(?:→|->)[^\]\n]{0,80}?\]",
    re.IGNORECASE,
)
_FLEET_TICK = re.compile(r"^\s*FLEET\s+TICK\b", re.IGNORECASE)
_STANDING_RULE_MARKER = re.compile(r"\bstanding\b", re.IGNORECASE)
_STANDING_RULE_LIVE_CORRECTION = re.compile(
    r"\bstanding\b.*\b(?:wrong|incorrect)\b|\b(?:wrong|incorrect)\b.*\bstanding\b",
    re.IGNORECASE,
)

_HARNESS_MARKERS = (
    "<system-reminder>",
    "userpromptsubmit hook additional context",
    "important: after completing your current task",
    "# autonomous loop",
    "autonomous loop tick",
    "<<autonomous-loop",
    "[frustration signal detected",
    "[brainlayer auto]",
    "[brainlayer deep]",
    "this is ambient context",
    "the user sent a new message while you were working",
    "silent orchestrator-monitor",
    "stay silent unless a real event",
    "# autonomous loop tick",
    "run the autonomous check",
    "<task-notification",
    "</task-notification>",
    '"commandmode":"task-notification"',
    '"commandmode": "task-notification"',
    "queued_command",
    '"type":"queue-operation"',
    '"type": "queue-operation"',
    '"operation":"enqueue"',
    '"operation": "enqueue"',
)
_ENTITY_CONTEXT_MARKERS = ("[entity:",)

_DISPATCH_MARKERS = (
    "dispatch brief",
    "worker brief",
    "dispatched by:",
    "quoted in dispatch",
    "brief quoting",
    "you are the new brainlayer",
    "outgoing lead",
    "grill answer",
)
_QUOTED_DISPATCH_MARKERS = (
    "quoted in dispatch",
    "brief quoting",
)

_STANDING_RULE_TERMS = (
    "no --print",
    "no -p",
    "--print/-p",
    "no source ~/.zshrc",
    "do not use source ~/.zshrc",
)

_LIVE_CORRECTION_CUES = re.compile(
    r"^\s*(no\b|stop\b|wait\b|why\b|what\b|are you\b)|"
    r"\b(i told you|as i said|we spoke about|this is live)\b",
    re.IGNORECASE | re.MULTILINE,
)
_STRONG_LIVE_CORRECTION_CUES = re.compile(
    r"\b(i told you|as i said|we spoke about|this is live)\b",
    re.IGNORECASE,
)
_DIRECT_LIVE_CORRECTION_LINE = re.compile(
    r"^\s*(?:no\b\s*(?:[,.:;]|\s-\s)|nope\b\s*(?:[,.:;]|\s-\s)|"
    r"that'?s?\s+wrong\b|stop\b\s*[.!?:;-]|לא\s+נכון\b)",
    re.IGNORECASE,
)

CORRECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:no|nope|wrong|incorrect)\b|\bthat'?s?\s+(?:not|wrong|incorrect)\b",
            re.IGNORECASE,
        ),
        "factual",
    ),
    (
        re.compile(
            r"\b(?:don'?t|do\s+not|stop|never)\b|"
            r"\bplease\s+don'?t\b|"
            r"\bi\s+(?:don'?t\s+)?(?:want|like|prefer)\b",
            re.IGNORECASE,
        ),
        "preference",
    ),
    (
        re.compile(
            r"\b(?:actually|i\s+meant|i\s+said)\b|\bnot\s+[\w]+,\s+(?:it'?s|but)\b",
            re.IGNORECASE,
        ),
        "factual",
    ),
    (
        re.compile(
            r"\btoo\s+(?:long|short|verbose|brief)\b|\b(?:format(?:ting)?|styl(?:e|ing)|tone)\b",
            re.IGNORECASE,
        ),
        "style",
    ),
    (
        re.compile(r"(?:לא\s+נכון|טעות|לא\s+ככה|תתקן|שגוי)"),
        "factual",
    ),
]


def looks_like_live_correction(prompt: str) -> bool:
    """Return True when a prompt has explicit live user-correction cues."""
    return bool(_LIVE_CORRECTION_CUES.search(prompt))


def has_strong_live_correction_cue(prompt: str) -> bool:
    """Return True for live cues strong enough to override relay/dispatch markers."""
    return bool(_STRONG_LIVE_CORRECTION_CUES.search(prompt))


def has_direct_live_correction_line(prompt: str) -> bool:
    """Return True when any line starts like a direct user correction."""
    return any(_DIRECT_LIVE_CORRECTION_LINE.search(line) for line in prompt.splitlines())


def has_standing_rule_live_correction(prompt: str) -> bool:
    """Return True when standing-rule text is itself being corrected."""
    return bool(_STANDING_RULE_LIVE_CORRECTION.search(prompt))


def should_suppress_correction_detection(prompt: str) -> tuple[bool, str]:
    """Return (suppress, reason) for prompts that are confidently non-user payloads.

    The gate is intentionally conservative: relay, harness, cron, and dispatch
    markers suppress false fires, but ambiguous prompts still pass to regexes.
    """
    if os.environ.get("BRAINLAYER_CORRECTION_STAGE_A_DISABLED") == "1":
        return (False, "")
    if not prompt:
        return (False, "")
    if _RELAY_HEADER.search(prompt):
        return (True, "agent-relay header")
    if _FLEET_TICK.search(prompt):
        return (True, "fleet tick")

    lowered = prompt.lower()
    for marker in _HARNESS_MARKERS:
        if marker in lowered:
            return (True, f"harness marker: {marker!r}")

    if any(marker in lowered for marker in _ENTITY_CONTEXT_MARKERS) and not (
        looks_like_live_correction(prompt) or has_direct_live_correction_line(prompt)
    ):
        return (True, "entity context marker")

    if any(marker in lowered for marker in _QUOTED_DISPATCH_MARKERS):
        return (True, "quoted dispatch marker")

    if any(marker in lowered for marker in _DISPATCH_MARKERS) and not (
        has_strong_live_correction_cue(prompt) or has_direct_live_correction_line(prompt)
    ):
        return (True, "dispatch marker")

    if _STANDING_RULE_MARKER.search(prompt) and any(term in lowered for term in _STANDING_RULE_TERMS):
        if not (
            has_strong_live_correction_cue(prompt)
            or has_direct_live_correction_line(prompt)
            or has_standing_rule_live_correction(prompt)
        ):
            return (True, "relayed standing-rules block")

    return (False, "")


def detect_correction(prompt: str) -> str | None:
    """Detect whether text looks like a user correction and return its category."""
    stripped = prompt.strip()
    if len(stripped) < 5:
        return None
    suppress, _reason = should_suppress_correction_detection(stripped)
    if suppress:
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
