"""Source-path denylist for transcript roots that must not be ingested."""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

BRAINLAYER_INGEST_DENYLIST_ENV = "BRAINLAYER_INGEST_DENYLIST"

DEFAULT_INGEST_DENYLIST = ("~/.claude/projects/**/wf_*/**",)

_SUBAGENT_ATTRIBUTION_CACHE: dict[str, tuple[int, int, str | None]] = {}


def _configured_patterns() -> tuple[str, ...]:
    override = os.environ.get(BRAINLAYER_INGEST_DENYLIST_ENV)
    if override is None:
        return DEFAULT_INGEST_DENYLIST
    return tuple(pattern.strip() for pattern in override.split(",") if pattern.strip())


def _inferred_homes(path: Path) -> tuple[Path, ...]:
    homes: list[Path] = [Path.home()]
    for provider_dir in (".claude", ".codex", ".cursor", ".gemini"):
        if provider_dir not in path.parts:
            continue
        provider_index = path.parts.index(provider_dir)
        if provider_index > 0:
            homes.append(Path(*path.parts[:provider_index]))
    return tuple(dict.fromkeys(homes))


def _expand_globs(pattern: str, homes: tuple[Path, ...]) -> tuple[Path, ...]:
    if pattern.startswith("~/"):
        return tuple(Path(os.path.abspath(str(home / pattern[2:]))) for home in homes)
    return (Path(os.path.abspath(os.path.expanduser(pattern))),)


def _match_parts(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts
    if pattern_parts[0] == "**":
        return _match_parts(path_parts, pattern_parts[1:]) or (
            bool(path_parts) and _match_parts(path_parts[1:], pattern_parts)
        )
    if not path_parts:
        return False
    return fnmatch.fnmatchcase(path_parts[0], pattern_parts[0]) and _match_parts(path_parts[1:], pattern_parts[1:])


def _is_claude_subagent(path: Path) -> bool:
    parts = path.parts
    return ".claude" in parts and "projects" in parts and "subagents" in parts


def _claude_subagent_attribution(path: Path) -> str | None:
    """Read the first stable worker attribution without rescanning unchanged JSONLs."""
    try:
        stat = path.stat()
    except OSError:
        return None

    cache_key = str(path)
    cached = _SUBAGENT_ATTRIBUTION_CACHE.get(cache_key)
    if cached is not None and cached[:2] == (stat.st_size, stat.st_mtime_ns):
        return cached[2]

    attribution = None
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                raw_attribution = entry.get("attributionAgent")
                if isinstance(raw_attribution, str) and raw_attribution.strip():
                    attribution = raw_attribution.strip()
                    break
    except OSError:
        return None

    _SUBAGENT_ATTRIBUTION_CACHE[cache_key] = (stat.st_size, stat.st_mtime_ns, attribution)
    return attribution


def is_denylisted(path: str | Path, *, unknown_subagent_is_denylisted: bool = True) -> bool:
    """Return True when a source path is under an ingest-denylisted transcript root."""
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    homes = _inferred_homes(candidate)
    for pattern in _configured_patterns():
        for expanded_pattern in _expand_globs(pattern, homes):
            if _match_parts(candidate.parts, expanded_pattern.parts):
                return True
    if BRAINLAYER_INGEST_DENYLIST_ENV not in os.environ and _is_claude_subagent(candidate):
        attribution = _claude_subagent_attribution(candidate)
        return (attribution is None and unknown_subagent_is_denylisted) or attribution == "brain-worker"
    return False
