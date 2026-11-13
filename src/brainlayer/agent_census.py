"""Known-agent census: a local agent from a fixed list is running and writes where BrainLayer does not watch.

The ingest guarantee (formal/ingest-coverage) holds only for wired roots, and new sources are wired on
purpose, never auto-detected. This is the honest add-on: it names a running agent from the list below
that wrote one of its transcripts, during its current run, outside every watched root. It cannot see an
agent that is not on the list.

Result, per agent: ``found`` lists ``"<agent>: <root>"`` for each agent measured positive this check;
``incomplete`` names each agent that could not be measured (a running instance without a start time, or a
scan cut short by a cap or a read error, with nothing found). Every other listed agent is a measured
negative, including one that is not running. The verdict merges per agent, so an incomplete agent keeps
whatever was last measured for it.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

MAX_ENTRIES = 5000
MAX_DEPTH = 6
START_SLACK_S = 1.0  # ps reports elapsed time in whole seconds
_INTERPRETER = re.compile(r"^(node|nodejs|bun|deno|python[0-9.]*)$")

ProcessStarts = Callable[[Sequence[int]], "dict[int, float] | None"]


@dataclass(frozen=True)
class Census:
    found: list[str]  # "<agent>: <root>", measured positive this check
    incomplete: list[str]  # agent names that could not be measured this check

    @property
    def complete(self) -> bool:
        return not self.incomplete


@dataclass(frozen=True)
class KnownAgent:
    name: str
    executables: tuple[str, ...]  # argv[0] basename of the agent's own binary
    entry_scripts: tuple[str, ...]  # script an interpreter runs: a basename, or a path fragment containing "/"
    transcripts: tuple[tuple[str, str], ...]  # (home-relative root, fnmatch pattern for the path under it)


KNOWN_AGENTS: tuple[KnownAgent, ...] = (
    KnownAgent("claude", ("claude",), ("claude", "@anthropic-ai/claude-code/"), ((".claude/projects", "*.jsonl"),)),
    KnownAgent("codex", ("codex",), ("codex", "@openai/codex/"), ((".codex/sessions", "*rollout-*.jsonl"),)),
    KnownAgent(
        "cursor-agent", ("cursor-agent",), ("cursor-agent",), ((".cursor/projects", "*/agent-transcripts/*.jsonl"),)
    ),
    # Gemini CLI saves chats as tmp/<project-hash>/chats/session-*.json, not under sessions/.
    KnownAgent(
        "gemini",
        ("gemini",),
        ("gemini", "@google/gemini-cli/"),
        ((".gemini/tmp", "*/chats/session-*.json"), (".gemini/sessions", "*.jsonl")),
    ),
    KnownAgent("opencode", ("opencode",), ("opencode", "opencode-ai/"), ((".local/share/opencode/storage", "*.json"),)),
)


# Options that take the next token as their value (it is never the script).
_VALUE_OPTIONS = {
    "python": {"-W", "-X", "-Q", "--check-hash-based-pycs"},
    "node": {"-r", "--require", "--import", "--loader", "--experimental-loader", "-C", "--conditions"},
    "bun": {"-r", "--preload", "--cwd", "--config", "-c"},
    "deno": {"-c", "--config", "-L", "--log-level", "--import-map"},
}
# Options whose value is inline code or a module: the process runs no script at all.
_CODE_OPTIONS = {
    "python": ("-c", "-m"),
    "node": ("-e", "--eval", "-p", "--print"),
    "bun": ("-e", "--eval", "-p", "--print"),
    "deno": (),
}
_SUBCOMMANDS = {"bun": {"run"}, "deno": {"run"}}  # a subcommand that is followed by a script
_NO_SCRIPT = {"bun": {"x", "exec", "eval", "repl", "install", "add", "test"}, "deno": {"eval", "repl", "test", "task"}}


def _family(exe: str) -> str:
    return "python" if exe.startswith("python") else "node" if exe in ("node", "nodejs") else exe


def _entry_script(tokens: Sequence[str]) -> str | None:
    """The script an interpreter runs, reading its options; None for ``-c``/``-e``/``-m``/``eval`` and the like."""
    family = _family(os.path.basename(tokens[0]))
    rest, i = list(tokens[1:]), 0
    while i < len(rest):
        token = rest[i]
        if token == "--":
            return rest[i + 1] if i + 1 < len(rest) else None
        if any(
            token == code
            or (code.startswith("--") and token.startswith(code + "="))
            or (not code.startswith("--") and token.startswith(code))
            for code in _CODE_OPTIONS.get(family, ())
        ):
            return None  # inline code, e.g. ``python3 -c gemini`` or ``node -egemini``: not a script
        if family == "python" and token.startswith("-") and not token.startswith("--") and len(token) > 1:
            # A cluster such as -uc: -c or -m anywhere ends the options with code; -W/-X take a value.
            for j, char in enumerate(token[1:], 1):
                if char in "cm":
                    return None
                if char in "WXQ":
                    i += 1 if j < len(token) - 1 else 2
                    break
            else:
                i += 1
            continue
        if token.startswith("-"):
            i += 2 if token in _VALUE_OPTIONS.get(family, ()) else 1
            continue
        if token in _NO_SCRIPT.get(family, ()):
            return None
        if token in _SUBCOMMANDS.get(family, ()):
            i += 1
            continue
        return token
    return None


def _is_agent(tokens: Sequence[str], agent: KnownAgent) -> bool:
    exe = os.path.basename(tokens[0])
    if exe in agent.executables:
        return True
    if not _INTERPRETER.match(exe):
        return False
    script = _entry_script(tokens)
    if script is None:
        return False
    return any(
        (fragment in script) if "/" in fragment else os.path.basename(script) == fragment
        for fragment in agent.entry_scripts
    )


def running_agents(ps_output: str, agents: Sequence[KnownAgent] = KNOWN_AGENTS) -> dict[str, list[int]]:
    """PIDs per agent from a ``ps axo pid=,command=`` snapshot, matched on the executable or entry script only."""
    found: dict[str, list[int]] = {}
    for line in ps_output.splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        for agent in agents:
            if _is_agent(parts[1:], agent):
                found.setdefault(agent.name, []).append(int(parts[0]))
    return found


def parse_etime(value: str) -> int | None:
    """``ps`` elapsed time ``[[dd-]hh:]mm:ss`` in seconds."""
    days, _, clock = value.strip().rpartition("-")
    fields = clock.split(":")
    if not 2 <= len(fields) <= 3 or not all(f.isdigit() for f in fields) or (days and not days.isdigit()):
        return None
    seconds = 0
    for field in fields:
        seconds = seconds * 60 + int(field)
    return seconds + int(days or 0) * 86400


def _watched(path: Path, watch_roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in watch_roots)


def _scan(root: Path, pattern: str, since: float) -> tuple[bool, bool]:
    """(found, complete): a file under ``root`` matching ``pattern`` modified at or after ``since``."""
    stack, looked, complete = [(root, 0)], 0, True
    while stack:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    looked += 1
                    if looked > MAX_ENTRIES:
                        return False, False
                    if entry.is_dir(follow_symlinks=False):
                        if depth + 1 >= MAX_DEPTH:
                            complete = False
                        else:
                            stack.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        relative = Path(entry.path).relative_to(root).as_posix()
                        if fnmatch(relative, pattern) and entry.stat().st_mtime >= since:
                            return True, True
        except OSError:
            complete = False
    return False, complete


def unindexed_agents(
    ps_output: str,
    *,
    home: Path,
    watch_roots: Iterable[Path],
    process_starts: ProcessStarts | None = None,
    agents: Sequence[KnownAgent] = KNOWN_AGENTS,
) -> Census:
    """Per agent: wrote a transcript during its current run under a root nobody watches, did not, or unknown."""
    running = running_agents(ps_output, agents)
    if not running:
        return Census([], [])
    watched = [Path(root).expanduser() for root in watch_roots]
    pids = sorted(pid for found in running.values() for pid in found)
    starts = process_starts(pids) if process_starts is not None else None
    found, incomplete = [], []
    for agent in agents:
        if agent.name not in running:
            continue
        known = [starts[pid] for pid in running[agent.name] if starts and pid in starts]
        unwatched = [
            (home / rel, pattern)
            for rel, pattern in agent.transcripts
            if (home / rel).is_dir() and not _watched(home / rel, watched)
        ]
        if not unwatched:
            continue
        if not known:
            incomplete.append(agent.name)  # running, but no start time to tie a write to this run
            continue
        since = min(known) - START_SLACK_S  # any write newer than a live instance's start
        complete = True
        for root, pattern in unwatched:
            hit, scanned = _scan(root, pattern, since)
            if hit:
                found.append(f"{agent.name}: {root}")
                break
            complete = complete and scanned
        else:
            if not complete:
                incomplete.append(agent.name)
    return Census(found, incomplete)


def ps_process_starts(run: Callable[[list[str]], object], now: Callable[[], float] = time.time) -> ProcessStarts:
    """Start times (epoch seconds) from ``ps -o pid=,etime= -p <pids>``; None when ps cannot answer."""

    def starts(pids: Sequence[int]) -> dict[int, float] | None:
        result = run(["ps", "-o", "pid=,etime=", "-p", ",".join(str(pid) for pid in pids)])
        if getattr(result, "returncode", 1) not in (0, 1):  # 1: some PIDs already gone
            return None
        at, out = now(), {}
        for line in str(getattr(result, "stdout", "") or "").splitlines():
            parts = line.split()
            elapsed = parse_etime(parts[1]) if len(parts) == 2 and parts[0].isdigit() else None
            if elapsed is not None:
                out[int(parts[0])] = at - elapsed
        return out

    return starts
