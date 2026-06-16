"""Small configuration helpers shared across BrainLayer entrypoints."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
_OP_READ_PREFIX = "$(op read "


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
        value = _resolve_op_read_value(value_text)
        return (key, value) if value is not None else None

    try:
        parsed = shlex.split(value_text, comments=False, posix=True)
    except ValueError:
        return None
    value = parsed[0] if parsed else ""
    return key, value


def _resolve_op_read_value(value_text: str) -> str | None:
    """Resolve exactly quoted $(op read 'op://...') values without a shell."""
    try:
        parsed = shlex.split(value_text, comments=False, posix=True)
    except ValueError:
        return None
    if len(parsed) != 1:
        return None

    command = parsed[0].strip()
    if not command.startswith(_OP_READ_PREFIX) or not command.endswith(")"):
        return None

    inner = command[2:-1]
    try:
        args = shlex.split(inner, comments=False, posix=True)
    except ValueError:
        return None
    if len(args) != 3 or args[:2] != ["op", "read"] or not args[2].startswith("op://"):
        return None

    try:
        result = subprocess.run(
            ["op", "read", args[2]],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Could not resolve 1Password env reference %s: %s", args[2], exc)
        return None

    if result.returncode != 0:
        logger.warning("Could not resolve 1Password env reference %s: op read exited %s", args[2], result.returncode)
        return None
    return result.stdout.rstrip("\n")


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

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read BrainLayer env file %s: %s", target, exc)
        return {}
    except UnicodeDecodeError as exc:
        logger.warning("Could not decode BrainLayer env file %s: %s", target, exc)
        return {}

    loaded: dict[str, str] = {}
    for line in lines:
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
