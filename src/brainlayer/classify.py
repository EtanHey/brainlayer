"""Lightweight prompt classification for search routing."""

from __future__ import annotations

import re

COMMAND_PATTERNS = [
    r"^/",
    r"^(git|npm|pip|python|pytest|ruff|cd|ls|cat|grep)\s",
    r"^(run|execute|install|build|deploy|test|lint|format)\s",
]

CASUAL_PATTERNS = [
    r"^(hey|hi|hello|thanks|thank you|ok|okay|yes|no|sure|sounds good|got it|perfect|great|nice)[\s!.?]*$",
    r"^(good morning|good night|gm|gn)[\s!.?]*$",
]

FOLLOW_UP_PATTERNS = [
    r"^(tell me more|what about|and the|how about|what else|go on|continue|elaborate)",
    r"^(that|this|it|those|these)\s",
    r"^(why|how)\??\s*$",
]

HEBREW_RE = re.compile(r"[\u0590-\u05FF]{2,}")
CASUAL_TOKENS = {
    "hey",
    "hi",
    "hello",
    "thanks",
    "thank",
    "you",
    "ok",
    "okay",
    "yes",
    "no",
    "sure",
    "sounds",
    "good",
    "got",
    "it",
    "perfect",
    "great",
    "nice",
}


def classify_prompt(prompt: str, detected_entities: list | None = None) -> str:
    """Classify prompt into one of 6 route categories."""
    stripped = prompt.strip()
    lower = stripped.lower()

    for pattern in COMMAND_PATTERNS:
        if re.match(pattern, lower):
            return "command"

    if len(stripped) < 40:
        for pattern in CASUAL_PATTERNS:
            if re.match(pattern, lower):
                return "casual_chat"
        words = re.findall(r"[a-z]+", lower)
        if words and all(word in CASUAL_TOKENS for word in words):
            return "casual_chat"

    if HEBREW_RE.search(stripped):
        return "hebrew_query"

    if detected_entities:
        return "entity_lookup"

    if len(stripped) < 60:
        for pattern in FOLLOW_UP_PATTERNS:
            if re.match(pattern, lower):
                return "follow_up"

    return "knowledge_question"
