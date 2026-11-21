"""Which source classes default search hides: the one list every reader shares.

Dependency-free on purpose: the prompt and session hooks import it on every fire,
and pulling in ``agent_provenance`` there would cost about 0.1 s per prompt.

Agents' thinking (reasoning) blocks are stored, hidden from default search, opt-in
only (Etan, 2026-11-17, choice A): the same visibility the letter gives desktop apps.
"""

from __future__ import annotations

THINKING_SOURCE_CLASS = "agent-thinking"
# The unadvertised opt-in reveals these...
OPT_IN_HIDDEN_SOURCE_CLASSES: tuple[str, ...] = ("desktop", THINKING_SOURCE_CLASS)
# ...and never these.
ALWAYS_HIDDEN_SOURCE_CLASSES: tuple[str, ...] = ("brain-worker",)
DEFAULT_HIDDEN_SOURCE_CLASSES: tuple[str, ...] = OPT_IN_HIDDEN_SOURCE_CLASSES + ALWAYS_HIDDEN_SOURCE_CLASSES


def hidden_source_classes(*, include_hidden_source_classes: bool = False) -> tuple[str, ...]:
    return ALWAYS_HIDDEN_SOURCE_CLASSES if include_hidden_source_classes else DEFAULT_HIDDEN_SOURCE_CLASSES


def is_hidden_source_class(value: object) -> bool:
    """True when a row of this class is kept out of default search (NULL and unknown are visible)."""
    return str(value or "").strip() in DEFAULT_HIDDEN_SOURCE_CLASSES


def hidden_source_class_sql(column: str, *, include_hidden_source_classes: bool = False) -> str:
    """SQL keeping a row visible: legacy NULL rows stay visible, the hidden list is excluded.

    ``column`` is a trusted identifier (for example ``c.source_class``), never user input.
    """
    hidden = hidden_source_classes(include_hidden_source_classes=include_hidden_source_classes)
    values = ", ".join(f"'{value}'" for value in hidden)
    return f"({column} IS NULL OR {column} NOT IN ({values}))"
