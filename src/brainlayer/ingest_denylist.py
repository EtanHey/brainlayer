"""Source-path denylist for transcript roots that must not be ingested."""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path

BRAINLAYER_INGEST_DENYLIST_ENV = "BRAINLAYER_INGEST_DENYLIST"

DEFAULT_INGEST_DENYLIST = (
    "~/.claude/projects/*/**/subagents/**",
    "~/.claude/projects/**/wf_*/**",
    "~/.codex/sessions/**",
    "~/.cursor/**/agent-transcripts/**",
    "~/.gemini/sessions/**",
)


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


def is_denylisted(path: str | Path) -> bool:
    """Return True when a source path is under an ingest-denylisted transcript root."""
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    homes = _inferred_homes(candidate)
    for pattern in _configured_patterns():
        for expanded_pattern in _expand_globs(pattern, homes):
            if _match_parts(candidate.parts, expanded_pattern.parts):
                return True
    return False
