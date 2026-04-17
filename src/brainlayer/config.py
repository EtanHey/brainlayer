"""Small configuration helpers shared across BrainLayer entrypoints."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back cleanly on malformed values."""
    raw = os.environ.get(name)
    if raw is None:
        return default

    value = raw.strip()
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


DEFAULT_REALTIME_ENRICH_SINCE_HOURS = get_int_env(
    "BRAINLAYER_DEFAULT_ENRICH_SINCE_HOURS",
    8760,
)
