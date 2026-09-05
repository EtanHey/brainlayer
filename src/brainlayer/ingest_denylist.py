"""Source-path denylist for transcript roots that must not be ingested."""

from __future__ import annotations

import fnmatch
import functools
import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path

BRAINLAYER_INGEST_DENYLIST_ENV = "BRAINLAYER_INGEST_DENYLIST"

# The class rule is deliberately narrower than the historical wf_* blanket:
# only memory-reading workers are out of the index. Explicit deployment globs
# remain available through BRAINLAYER_INGEST_DENYLIST.
DEFAULT_INGEST_DENYLIST: tuple[str, ...] = ()
MEMORY_READER_ATTRIBUTIONS = frozenset({"brain-worker", "session-miner", "weave"})


@dataclass(frozen=True)
class _AttributionCacheEntry:
    device: int
    inode: int
    size: int
    mtime_ns: int
    scanned_offset: int
    prefix_sha256: str
    attribution: str | None


_SUBAGENT_ATTRIBUTION_CACHE: dict[str, _AttributionCacheEntry] = {}


def _configured_patterns() -> tuple[str, ...]:
    override = os.environ.get(BRAINLAYER_INGEST_DENYLIST_ENV)
    if override is None:
        return DEFAULT_INGEST_DENYLIST
    return _split_patterns(override)


@functools.lru_cache(maxsize=16)
def _split_patterns(raw: str) -> tuple[str, ...]:
    return tuple(pattern.strip() for pattern in raw.split(",") if pattern.strip())


@functools.lru_cache(maxsize=16)
def _subtree_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """The patterns that, when they match a directory, also match every path below it."""
    return tuple(pattern for pattern in patterns if pattern.rstrip("/").rsplit("/", 1)[-1] == "**")


def _inferred_homes(path: Path, home: Path | None = None) -> tuple[Path, ...]:
    """Homes a `~/` pattern expands against: the process home plus any home implied by the path.

    `home` is explicit so a memoised caller can key its cache on the exact value it matched
    with; left None it reads the live Path.home().
    """
    homes: list[Path] = [home if home is not None else Path.home()]
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


def memory_reader_attribution(path: str | Path) -> str | None:
    """Return an exact memory-reader role from path/JSONL attribution, never task text."""
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    if not _is_claude_subagent(candidate):
        return None
    if "subagents" in candidate.parts:
        subagents_index = candidate.parts.index("subagents")
        for part in candidate.parts[subagents_index + 1 :]:
            normalized = part.strip().casefold().replace("_", "-")
            if normalized in MEMORY_READER_ATTRIBUTIONS:
                return normalized
    attribution = _claude_subagent_attribution(candidate)
    normalized = str(attribution or "").strip().casefold().replace("_", "-")
    return normalized if normalized in MEMORY_READER_ATTRIBUTIONS else None


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

    attribution: str | None = None
    scanned_offset = 0
    final_stat = stat
    try:
        with path.open("rb") as handle:
            prefix_hasher = hashlib.sha256()
            prefix_matches = False
            if same_file and cached.scanned_offset <= stat.st_size:
                remaining = cached.scanned_offset
                while remaining:
                    chunk = handle.read(min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    prefix_hasher.update(chunk)
                    remaining -= len(chunk)
                prefix_matches = remaining == 0 and prefix_hasher.hexdigest() == cached.prefix_sha256

            if prefix_matches:
                scanned_offset = cached.scanned_offset
                if cached.attribution is not None:
                    final_stat = os.fstat(handle.fileno())
                    _SUBAGENT_ATTRIBUTION_CACHE[cache_key] = _AttributionCacheEntry(
                        final_stat.st_dev,
                        final_stat.st_ino,
                        final_stat.st_size,
                        final_stat.st_mtime_ns,
                        scanned_offset,
                        cached.prefix_sha256,
                        cached.attribution,
                    )
                    return cached.attribution
            else:
                prefix_hasher = hashlib.sha256()
                handle.seek(0)

            scan_offset = scanned_offset
            handle.seek(scan_offset)
            for raw_line in handle:
                line_end = handle.tell()
                if raw_line.endswith(b"\n"):
                    scanned_offset = line_end
                    prefix_hasher.update(raw_line)
                try:
                    entry = json.loads(raw_line.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                raw_attribution = entry.get("attributionAgent")
                if isinstance(raw_attribution, str) and raw_attribution.strip():
                    attribution = raw_attribution.strip()
                    if not raw_line.endswith(b"\n"):
                        prefix_hasher.update(raw_line)
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
        prefix_hasher.hexdigest(),
        attribution,
    )
    return attribution


@functools.lru_cache(maxsize=65536)
def _matches_configured_pattern(path: str, patterns: tuple[str, ...], home: str) -> bool:
    """Glob-match one absolute path against the configured denylist, memoised.

    The verdict is a pure function of the path, the pattern tuple, and the home the
    patterns expand against -- so it is cached on exactly those three. The watcher asks
    this for every file on every poll: with the deployed 5-pattern denylist that was
    60,625 glob expansions and ~258,000 recursive `_match_parts` calls per poll over the
    12,125-file corpus (0.35s CPU on a performance core, ~1.8s on the efficiency cores
    launchd schedules the watcher onto), all producing the same answers as the poll
    before. A changed BRAINLAYER_INGEST_DENYLIST is still picked up on the very next call,
    because the patterns are part of the key. `home` is in the key for the same reason, and it
    is the home the match is computed with -- not a live Path.home() read that merely happens
    to agree with it (round-1 review of #781: a key that does not drive the computation is a
    latent flip between a cached verdict and the expansion it was cached under).
    """
    candidate = Path(path)
    homes = _inferred_homes(candidate, Path(home))
    for pattern in patterns:
        for expanded_pattern in _expand_globs(pattern, homes):
            if _match_parts(candidate.parts, expanded_pattern.parts):
                return True
    return False


def clear_pattern_match_cache() -> None:
    """Drop memoised glob verdicts (tests; a process that rewrites its own environment)."""
    _matches_configured_pattern.cache_clear()


def is_directory_denylisted(path: str | Path) -> bool:
    """Return True when every file that could exist under this directory is denylisted.

    Only a configured pattern ending in `**` can say that: it matches the directory itself
    and, because the trailing `**` absorbs any suffix, everything beneath it. A pattern with
    any other tail (`.../*.jsonl`) matching a directory's name says nothing about the files
    inside, and the default subagent policy judges files one at a time by their attribution,
    so both answer False here. The watcher uses this to skip whole subtrees during discovery:
    on the M4 the deployed `~/.cursor/**/agent-transcripts/**` covers 4,315 of the 5,086
    directories under `~/.cursor/projects`, all of which were being read on every poll to
    find files that were then discarded.
    """
    patterns = _subtree_patterns(_configured_patterns())
    if not patterns:
        return False
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    return _matches_configured_pattern(str(candidate), patterns, str(Path.home()))


def is_denylisted(path: str | Path, *, unknown_subagent_is_denylisted: bool = True) -> bool:
    """Return True when a source path is under an ingest-denylisted transcript root."""
    candidate = Path(os.path.abspath(os.path.expanduser(str(path))))
    patterns = _configured_patterns()
    if patterns and _matches_configured_pattern(str(candidate), patterns, str(Path.home())):
        return True
    if BRAINLAYER_INGEST_DENYLIST_ENV not in os.environ and _is_claude_subagent(candidate):
        if memory_reader_attribution(candidate) is not None:
            return True
        attribution = _claude_subagent_attribution(candidate)
        normalized = str(attribution or "").strip().casefold().replace("_", "-")
        return (attribution is None and unknown_subagent_is_denylisted) or normalized in MEMORY_READER_ATTRIBUTIONS
    return False


@dataclass(frozen=True)
class EnvDenylistOverreach:
    """An explicit env pattern that excludes a path the class rule would keep."""

    pattern: str
    kept_example: str


# Neutral probes standing for the transcript classes the class rule KEEPS: a plain
# provider session, an ordinary Claude subagent, an ordinary workflow agent, and the
# other CLI coding agents. The project token is deliberately generic so a
# deployment-scoped pattern (one naming a single project or repo) is not mistaken
# for class overreach.
_CLASS_KEPT_PROBES: tuple[tuple[str, ...], ...] = (
    (".claude", "projects", "probe-project", "probe-session.jsonl"),
    (".claude", "projects", "probe-project", "probe-session", "subagents", "agent-aprobe.jsonl"),
    (
        ".claude",
        "projects",
        "probe-project",
        "probe-session",
        "subagents",
        "workflows",
        "wf_probe",
        "agent-aprobe.jsonl",
    ),
    (".cursor", "projects", "probe-project", "agent-transcripts", "probe-session", "agent-aprobe.jsonl"),
    (".codex", "sessions", "probe-session.jsonl"),
    (".gemini", "sessions", "probe-session.jsonl"),
)


def split_denylist_patterns(raw: str) -> tuple[str, ...]:
    """Public split for a raw BRAINLAYER_INGEST_DENYLIST value (comma- or newline-separated)."""
    return _split_patterns(raw)


def env_file_denylist_patterns(env_file: str | Path) -> tuple[str, ...] | None:
    """Read BRAINLAYER_INGEST_DENYLIST out of a resolved env file.

    `brainlayer setup` writes this file and launchd is pointed at it, so it -- not the process
    environment of whatever shell ran setup -- is where a deployment blanket actually lives.
    Returns None when the key is absent, which is a different thing from an empty override.

    This deliberately does not reuse `config._parse_env_value`: importing `config` runs
    `load_brainlayer_env()` at import time, and that parser resolves `$(op read ...)` values.
    Neither belongs in a read-only audit of a denylist line, so the quoting is matched with shlex
    and a command-substitution value is refused rather than executed.
    """
    path = Path(os.path.abspath(os.path.expanduser(str(env_file))))
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    value: str | None = None
    for raw_line in raw_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        key, _, candidate = stripped.partition("=")
        if key.strip() != BRAINLAYER_INGEST_DENYLIST_ENV:
            continue
        candidate = candidate.strip()
        if "$(" in candidate or "`" in candidate:
            continue
        try:
            parsed = shlex.split(candidate, comments=False, posix=True)
        except ValueError:
            continue
        value = parsed[0] if parsed else ""
    if value is None:
        return None
    return _split_patterns(value)


def _probe_roots(expanded_pattern: Path) -> tuple[Path, ...]:
    """Home, plus any provider root the pattern itself is anchored at.

    A blanket does not have to live under $HOME -- an absolute
    /var/.../.claude/projects/**/subagents/** overrides ingest just as wholesale -- so probes are
    rebuilt at the pattern's own anchor as well. Mirrors _inferred_homes, which does this for paths.
    """
    roots: list[Path] = [Path.home()]
    parts = expanded_pattern.parts
    for provider_dir in (".claude", ".codex", ".cursor", ".gemini"):
        if provider_dir not in parts:
            continue
        provider_index = parts.index(provider_dir)
        if provider_index > 0:
            roots.append(Path(*parts[:provider_index]))
    return tuple(dict.fromkeys(roots))


def env_denylist_overreach(patterns: tuple[str, ...] | None = None) -> tuple[EnvDenylistOverreach, ...]:
    """Return the configured patterns that exclude transcript classes the class rule keeps.

    Pass `patterns` to audit a source other than the process environment -- notably the env file
    `brainlayer setup` just resolved, which is where a deployment blanket actually lives.

    The class rule excludes only memory-reading workers (see MEMORY_READER_ATTRIBUTIONS),
    read from each transcript's own attribution. A pattern that also swallows an ordinary
    subagent, an ordinary workflow agent, or a plain provider session is broader than that
    rule, and callers must say so rather than degrade silently.
    """
    findings: list[EnvDenylistOverreach] = []
    home = Path.home()
    for pattern in _configured_patterns() if patterns is None else patterns:
        matched: str | None = None
        for expanded_pattern in _expand_globs(pattern, (home,)):
            for root in _probe_roots(expanded_pattern):
                probes = (Path(os.path.abspath(str(root.joinpath(*parts)))) for parts in _CLASS_KEPT_PROBES)
                match = next((probe for probe in probes if _match_parts(probe.parts, expanded_pattern.parts)), None)
                if match is not None:
                    matched = str(match)
                    break
            if matched is not None:
                break
        if matched is not None:
            findings.append(EnvDenylistOverreach(pattern=pattern, kept_example=matched))
    return tuple(findings)
