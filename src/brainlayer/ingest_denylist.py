"""Source-path denylist for transcript roots that must not be ingested."""

from __future__ import annotations

import fnmatch
import json
import os
from dataclasses import dataclass
from pathlib import Path

BRAINLAYER_INGEST_DENYLIST_ENV = "BRAINLAYER_INGEST_DENYLIST"

DEFAULT_INGEST_DENYLIST = ("~/.claude/projects/**/wf_*/**",)
RETIRED_BLANKET_INGEST_DENYLIST = (
    "~/.claude/projects/*/**/subagents/**",
    "~/.codex/sessions/**",
    "~/.cursor/**/agent-transcripts/**",
    "~/.gemini/sessions/**",
)


@dataclass(frozen=True)
class _AttributionCacheEntry:
    device: int
    inode: int
    size: int
    mtime_ns: int
    scanned_offset: int
    attribution: str | None


_SUBAGENT_ATTRIBUTION_CACHE: dict[str, _AttributionCacheEntry] = {}


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


def _matches_patterns(candidate: Path, patterns: tuple[str, ...]) -> bool:
    homes = _inferred_homes(candidate)
    return any(
        _match_parts(candidate.parts, expanded_pattern.parts)
        for pattern in patterns
        for expanded_pattern in _expand_globs(pattern, homes)
    )


def _is_claude_subagent(path: Path) -> bool:
    parts = path.parts
    return ".claude" in parts and "projects" in parts and "subagents" in parts


def _claude_subagent_attribution(path: Path) -> str | None:
    """Incrementally read the first stable worker attribution from an appending JSONL."""
    try:
        stat = path.stat()
    except OSError:
        return None

    cache_key = str(path)
    cached = _SUBAGENT_ATTRIBUTION_CACHE.get(cache_key)
    same_file = cached is not None and (cached.device, cached.inode) == (stat.st_dev, stat.st_ino)
    if same_file and (cached.size, cached.mtime_ns) == (stat.st_size, stat.st_mtime_ns):
        return cached.attribution
    if same_file and cached.attribution is not None and stat.st_size > cached.size:
        _SUBAGENT_ATTRIBUTION_CACHE[cache_key] = _AttributionCacheEntry(
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            cached.scanned_offset,
            cached.attribution,
        )
        return cached.attribution

    attribution = None
    scan_offset = (
        cached.scanned_offset if same_file and cached.attribution is None and stat.st_size > cached.size else 0
    )
    scanned_offset = scan_offset
    final_stat = stat
    try:
        with path.open("rb") as handle:
            handle.seek(scan_offset)
            for raw_line in handle:
                line_end = handle.tell()
                if raw_line.endswith(b"\n"):
                    scanned_offset = line_end
                try:
                    entry = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                raw_attribution = entry.get("attributionAgent")
                if isinstance(raw_attribution, str) and raw_attribution.strip():
                    attribution = raw_attribution.strip()
                    scanned_offset = line_end
                    break
            final_stat = os.fstat(handle.fileno())
    except OSError:
        return None

    _SUBAGENT_ATTRIBUTION_CACHE[cache_key] = _AttributionCacheEntry(
        final_stat.st_dev,
        final_stat.st_ino,
        final_stat.st_size,
        final_stat.st_mtime_ns,
        scanned_offset,
        attribution,
    )
    return attribution


def is_denylisted(path: str | Path, *, unknown_subagent_is_denylisted: bool = True) -> bool:
    """Return True when a source path is under an ingest-denylisted transcript root."""
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    if _matches_patterns(candidate, _configured_patterns()):
        return True
    if BRAINLAYER_INGEST_DENYLIST_ENV not in os.environ and _is_claude_subagent(candidate):
        attribution = _claude_subagent_attribution(candidate)
        return (attribution is None and unknown_subagent_is_denylisted) or attribution == "brain-worker"
    return False


def is_legacy_backfill_denylisted(path: str | Path) -> bool:
    """Apply current safety exclusions while retiring only the old blanket roots."""
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    configured = tuple(pattern for pattern in _configured_patterns() if pattern not in RETIRED_BLANKET_INGEST_DENYLIST)
    active_patterns = tuple(dict.fromkeys((*DEFAULT_INGEST_DENYLIST, *configured)))
    if _matches_patterns(candidate, active_patterns):
        return True
    if _is_claude_subagent(candidate):
        return _claude_subagent_attribution(candidate) == "brain-worker"
    return False
