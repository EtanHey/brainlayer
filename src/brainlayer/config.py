"""Small configuration helpers shared across BrainLayer entrypoints."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)
_OP_READ_PREFIX = "$(op read "
_brainlayer_db_config_error: str | None = None


def get_user_env_path() -> Path:
    """Return the per-user BrainLayer env-file path."""
    return Path.home() / ".config" / "brainlayer" / "brainlayer.env"


def _split_env_assignment(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ").lstrip()

    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    return key, raw_value


def _parse_env_value(raw_value: str) -> str | None:
    value_text = raw_value.strip()
    if "$(" in value_text or "`" in value_text:
        return _resolve_op_read_value(value_text)

    try:
        parsed = shlex.split(value_text, comments=False, posix=True)
    except ValueError:
        return None
    return parsed[0] if parsed else ""


def _parse_env_assignment(line: str) -> tuple[str, str] | None:
    assignment = _split_env_assignment(line)
    if assignment is None:
        return None
    key, raw_value = assignment
    value = _parse_env_value(raw_value)
    return (key, value) if value is not None else None


def _parse_launchd_env_value(raw_value: str) -> str:
    """Match brainlayer-env-run.sh: trim and remove one pair of matching quotes."""
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _split_launchd_env_assignment(line: str) -> tuple[str, str] | None:
    """Match brainlayer-env-run.sh's deliberately simple assignment grammar."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    if stripped.startswith("export "):
        stripped = stripped.removeprefix("export ")

    key, raw_value = stripped.split("=", 1)
    key = key.rstrip()
    if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
        return None
    return key, raw_value.lstrip()


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


def configured_brainlayer_env_value(name: str, env_path: Path | None = None) -> str | None:
    """Return one unambiguous launchd-compatible assignment from the selected env file."""
    target = env_path or get_user_env_path()
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning("Could not read BrainLayer env file %s: %s", target, exc)
        raise RuntimeError(f"Could not read BrainLayer env file {target}: {exc}") from exc

    selected: str | None = None
    for line in lines:
        assignment = _split_launchd_env_assignment(line)
        if assignment is None or assignment[0] != name:
            continue
        if _OP_READ_PREFIX in assignment[1]:
            raise RuntimeError(f"{name} must not use an op command substitution in {target}")
        if "$(" in assignment[1] or "`" in assignment[1]:
            continue
        value = _parse_launchd_env_value(assignment[1])
        if _parse_env_value(assignment[1]) != value:
            raise RuntimeError(f"{name} must parse identically for launchd and direct CLI use in {target}")
        if selected is not None:
            raise RuntimeError(f"duplicate valid {name} assignments in {target}")
        selected = value
    return selected


def get_brainlayer_db_config_error() -> str | None:
    """Return the runtime DB env validation error that triggered canonical fallback."""
    return _brainlayer_db_config_error


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
    global _brainlayer_db_config_error

    del repo_env_path

    target = env_path or get_user_env_path()
    record_db_error = env_path is None or target == get_user_env_path()
    if not target.exists():
        if record_db_error:
            _brainlayer_db_config_error = None
        return {}

    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read BrainLayer env file %s: %s", target, exc)
        if record_db_error:
            _brainlayer_db_config_error = f"Could not read BrainLayer env file {target}: {exc}"
        return {}
    except UnicodeDecodeError as exc:
        logger.warning("Could not decode BrainLayer env file %s: %s", target, exc)
        if record_db_error:
            _brainlayer_db_config_error = f"Could not decode BrainLayer env file {target}: {exc}"
        return {}

    loaded: dict[str, str] = {}
    if "BRAINLAYER_DB" not in os.environ:
        try:
            configured_db = configured_brainlayer_env_value("BRAINLAYER_DB", target)
            direct_db_values = []
            for line in lines:
                assignment = _split_env_assignment(line)
                if assignment is None or assignment[0] != "BRAINLAYER_DB":
                    continue
                value = _parse_env_value(assignment[1])
                if value is not None:
                    direct_db_values.append(value)
            if any(value != configured_db for value in direct_db_values):
                raise RuntimeError(f"BRAINLAYER_DB must parse identically for launchd and direct CLI use in {target}")
        except RuntimeError as exc:
            if record_db_error:
                _brainlayer_db_config_error = str(exc)
            logger.error("Invalid BRAINLAYER_DB configuration; falling back to the canonical database: %s", exc)
        else:
            if record_db_error:
                _brainlayer_db_config_error = None
            if configured_db is not None:
                os.environ["BRAINLAYER_DB"] = configured_db
                loaded["BRAINLAYER_DB"] = configured_db
    else:
        if record_db_error:
            _brainlayer_db_config_error = None

    for line in lines:
        assignment = _split_env_assignment(line)
        if assignment is None:
            continue
        key, raw_value = assignment
        if key == "BRAINLAYER_DB":
            continue
        if key in os.environ:
            continue
        value = _parse_env_value(raw_value)
        if value is None:
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
