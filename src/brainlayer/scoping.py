"""Project scoping — auto-detect project from CWD using scopes.yaml config.

Reads ~/.config/brainlayer/scopes.yaml to map directory prefixes to project names.
Falls back to parsing CWD if no config exists.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCOPES_PATH = Path.home() / ".config" / "brainlayer" / "scopes.yaml"

# Cache parsed config
_cached_scopes: Optional[dict] = None
_cached_mtime: float = 0.0


def _load_scopes() -> dict:
    """Load scopes.yaml, returning parsed dict. Caches by mtime."""
    global _cached_scopes, _cached_mtime

    if not _SCOPES_PATH.exists():
        return {}

    mtime = _SCOPES_PATH.stat().st_mtime
    if _cached_scopes is not None and mtime == _cached_mtime:
        return _cached_scopes

    try:
        import yaml

        with open(_SCOPES_PATH) as f:
            data = yaml.safe_load(f) or {}
        _cached_scopes = data
        _cached_mtime = mtime
        return data
    except ImportError:
        # Fall back to simple line parsing if PyYAML not installed
        return _parse_scopes_simple(_SCOPES_PATH)
    except Exception as e:
        logger.debug("Failed to load scopes config from %s: %s", _SCOPES_PATH, e)
        return {}


def _parse_scopes_simple(path: Path) -> dict:
    """Minimal scopes.yaml parser without PyYAML dependency."""
    global _cached_scopes, _cached_mtime
    scopes: dict = {"scopes": {}, "default": "all"}

    try:
        lines = path.read_text().splitlines()
        in_scopes = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped == "scopes:":
                in_scopes = True
                continue
            if stripped.startswith("default:"):
                scopes["default"] = stripped.split(":", 1)[1].strip().strip("\"'")
                in_scopes = False
                continue
            if in_scopes and ":" in stripped:
                # Parse "  ~/Gits/golems: golems" or "  ~/Gits/golems: \"golems\""
                key, _, val = stripped.partition(":")
                key = key.strip().strip("\"'")
                val = val.strip().strip("\"'")
                # Expand ~ in the key
                key = str(Path(key).expanduser())
                scopes["scopes"][key] = val

        _cached_scopes = scopes
        _cached_mtime = path.stat().st_mtime
        return scopes
    except Exception:
        return {"scopes": {}, "default": "all"}


def resolve_project_scope() -> Optional[str]:
    """Resolve the current project from CWD using scopes.yaml.

    Returns:
        Project name if matched, None if default is "all" or no match.
    """
    config = _load_scopes()
    scope_map = config.get("scopes", {})
    default = config.get("default", "all")

    if not scope_map:
        # No config — try CWD heuristic
        return _cwd_heuristic()

    cwd = os.getcwd()

    # Check each scope prefix (longest match first)
    matches = []
    for prefix, project in scope_map.items():
        expanded = str(Path(prefix).expanduser())
        if cwd == expanded or cwd.startswith(expanded + os.sep):
            matches.append((len(expanded), project))

    if matches:
        # Return longest (most specific) match
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    return None if default == "all" else default


def _cwd_heuristic() -> Optional[str]:
    """Fallback: extract project name from CWD if it looks like ~/Gits/<project>."""
    cwd = os.getcwd()
    home = str(Path.home())

    # Check ~/Gits/<project> pattern
    gits_dir = os.path.join(home, "Gits")
    if cwd.startswith(gits_dir + os.sep):
        relative = cwd[len(gits_dir) + 1 :]
        # First segment is the project
        project = relative.split(os.sep)[0]
        if project:
            return project

    return None
