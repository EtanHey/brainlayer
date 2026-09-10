"""Resolve a workspace path to its canonical repository project name."""

from __future__ import annotations

import os
import subprocess
from collections import OrderedDict
from pathlib import Path

_CACHE: OrderedDict[str, str | None] = OrderedDict()
_CACHE_LIMIT = 256
_GIT_OVERRIDE_VARS = {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}


def canonical_project_from_workspace(workspace: str | None) -> str | None:
    """Return repository root name for a Git workspace, with leaf fallback."""
    if not isinstance(workspace, str) or not workspace.strip():
        return None
    raw = workspace.strip()
    if raw in _CACHE:
        value = _CACHE.pop(raw)
        _CACHE[raw] = value
        return value
    path = Path(raw).expanduser()
    fallback = path.name if path.name not in {"", ".", "/"} else None
    clean_env = {key: value for key, value in os.environ.items() if key not in _GIT_OVERRIDE_VARS}
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "rev-parse",
                "--path-format=absolute",
                "--show-toplevel",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
            env=clean_env,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) != 2:
            raise ValueError("unexpected git identity output")
        top_level, common_dir = (Path(line).expanduser().resolve() for line in lines)
        if not top_level.is_dir() or common_dir.name != ".git" or not common_dir.is_dir():
            raise ValueError("ambiguous git identity")
        value = top_level.name if top_level == common_dir.parent else common_dir.parent.name
        value = value or fallback
    except (OSError, subprocess.SubprocessError, ValueError):
        value = fallback
    _CACHE[raw] = value
    if len(_CACHE) > _CACHE_LIMIT:
        _CACHE.popitem(last=False)
    return value
