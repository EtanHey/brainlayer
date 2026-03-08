"""Shared helper functions for vector store modules."""

import json
import struct
from typing import Any, List, Optional

_SOURCE_MIN_CHARS = {
    "whatsapp": 15,
    "telegram": 15,
}
_DEFAULT_MIN_CHARS = 50


def source_aware_min_chars(source: Optional[str]) -> int:
    """Return minimum character count for enrichment based on message source.

    Short-form messaging sources (WhatsApp, Telegram) use a lower threshold
    since meaningful messages are often 15-50 chars.
    """
    if source is None:
        return _DEFAULT_MIN_CHARS
    return _SOURCE_MIN_CHARS.get(source, _DEFAULT_MIN_CHARS)


def _safe_json_loads(value: Any) -> list:
    """Safely parse a JSON string, returning [] on None or invalid JSON."""
    if not value:
        return []
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def _escape_fts5_query(query: str, *, match_mode: str = "auto") -> str:
    """Escape a query string for FTS5 MATCH.

    FTS5 treats certain characters as syntax: ., *, ^, ", (, ), +, -, NOT, AND, OR, NEAR.
    We wrap each word in double quotes so they're treated as literal terms.

    match_mode:
        "auto" — AND for ≤3 terms (precision), OR for 4+ (recall via RRF).
        "or"   — always OR (for entity search where any term should match).
    Empty/whitespace-only queries return a wildcard match-all.
    """
    if not query or not query.strip():
        return "*"
    # Split into words, wrap each in double quotes (escaping any internal quotes)
    terms = []
    for word in query.split():
        # Remove internal double quotes to prevent FTS5 injection
        clean = word.replace('"', "")
        if clean:
            terms.append(f'"{clean}"')
    if not terms:
        return "*"
    if match_mode == "or":
        joiner = " OR "
    else:
        # Short queries (≤3 terms): AND for precision; 4+: OR for recall
        joiner = " " if len(terms) <= 3 else " OR "
    return joiner.join(terms)


def serialize_f32(vector: List[float]) -> bytes:
    """Serialize a float32 vector to bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)
