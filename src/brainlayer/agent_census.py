"""Known-agent census: a local agent from a fixed list is running and writes where BrainLayer does not watch.

The ingest guarantee (formal/ingest-coverage) holds only for wired roots, and new sources are wired on
purpose, never auto-detected. This is the honest add-on: it names a running agent from the list below
whose transcripts land outside every watched root. It cannot see an agent that is not on the list.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

RECENT_WRITE_S = 24 * 3600
MAX_ENTRIES = 2000
MAX_DEPTH = 4


@dataclass(frozen=True)
class KnownAgent:
    name: str
    process_names: tuple[str, ...]  # argv[0] basename, or argv[1] for an interpreter-run CLI
    roots: tuple[str, ...]  # home-relative directories the agent writes its transcripts under


KNOWN_AGENTS: tuple[KnownAgent, ...] = (
    KnownAgent("claude", ("claude",), (".claude/projects",)),
    KnownAgent("codex", ("codex",), (".codex/sessions",)),
    KnownAgent("cursor-agent", ("cursor-agent",), (".cursor/projects",)),
    # Gemini CLI saves chats as tmp/<project-hash>/chats/session-*.json, not under sessions/.
    KnownAgent("gemini", ("gemini",), (".gemini/tmp", ".gemini/sessions")),
    KnownAgent("opencode", ("opencode",), (".local/share/opencode/storage",)),
)


def running_agents(ps_output: str, agents: Sequence[KnownAgent] = KNOWN_AGENTS) -> list[KnownAgent]:
    """Agents whose process name is argv[0] or argv[1] of some ``ps axo pid=,command=`` line."""
    seen: set[str] = set()
    for line in ps_output.splitlines():
        parts = line.split()
        names = {os.path.basename(token) for token in parts[1:3]}
        seen.update(agent.name for agent in agents if names.intersection(agent.process_names))
    return [agent for agent in agents if agent.name in seen]


def _watched(path: Path, watch_roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in watch_roots)


def _recent_write(root: Path, now: float, max_age_s: float) -> bool:
    """A file under ``root`` (depth <= MAX_DEPTH, at most MAX_ENTRIES looked at) modified within ``max_age_s``."""
    stack, looked = [(root, 0)], 0
    while stack and looked < MAX_ENTRIES:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    looked += 1
                    if looked > MAX_ENTRIES:
                        break
                    if entry.is_dir(follow_symlinks=False):
                        if depth < MAX_DEPTH:
                            stack.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False) and now - entry.stat().st_mtime <= max_age_s:
                        return True
        except OSError:
            continue
    return False


def unindexed_agents(
    ps_output: str,
    *,
    home: Path,
    watch_roots: Iterable[Path],
    now: float | None = None,
    agents: Sequence[KnownAgent] = KNOWN_AGENTS,
    max_age_s: float = RECENT_WRITE_S,
) -> list[str]:
    """``"<agent>: <root>"`` for each running known agent writing recently under a root nobody watches."""
    now = time.time() if now is None else now
    watched = [Path(root).expanduser() for root in watch_roots]
    found = []
    for agent in running_agents(ps_output, agents):
        for relative in agent.roots:
            root = home / relative
            if root.is_dir() and not _watched(root, watched) and _recent_write(root, now, max_age_s):
                found.append(f"{agent.name}: {root}")
    return found
