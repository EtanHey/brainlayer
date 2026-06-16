"""Small configuration helpers shared across BrainLayer entrypoints."""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

logger = logging.getLogger(__name__)


def get_user_env_path() -> Path:
    """Return the per-user BrainLayer env-file path."""
    return Path.home() / ".config" / "brainlayer" / "brainlayer.env"


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").lstrip()

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None

    value_text = raw_value.strip()
    if "$(" in value_text or "`" in value_text:
        return None

    try:
        parsed = shlex.split(value_text, comments=False, posix=True)
    except ValueError:
        return None
    value = parsed[0] if parsed else ""
    return key, value


def load_brainlayer_env(
    env_path: Path | None = None,
    *,
    repo_env_path: Path | None = None,
) -> dict[str, str]:
    """Load simple assignments from ~/.config/brainlayer/brainlayer.env.

    Precedence is process environment first, then the user env file. Repo-root
    .env files are deliberately ignored; pass repo_env_path only to document
    that it is not part of the loader contract.
    """
    del repo_env_path

    target = env_path or get_user_env_path()
    if not target.exists():
        return {}

    loaded: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        assignment = _parse_env_assignment(line)
        if assignment is None:
            continue
        key, value = assignment
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


load_brainlayer_env()


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
