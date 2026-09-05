"""Resolve the interpreter that BrainLayer's Claude Code hooks must run under.

Hooks are wired into `~/.claude/settings.json` as `<python> <script>`. A bare
`python3` there hands the choice of interpreter — and therefore the choice of
`brainlayer` library — to whatever PATH the Claude Code process inherited.

That is not hypothetical. On the M4, `python3` fronts the framework python at
`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`, whose
`site-packages/_brainlayer.pth` injects `/Users/etanheyman/Gits/brainlayer/src` — a
live, agent-editable checkout carrying no `__build_sha__`. When that checkout held a
stale snapshot, every hook fire executed weeks-old library code under a current CLI,
silently: no import error, no version mismatch, no log line.

So the interpreter is named outright, and a silent PATH fallback is refused — the
same fail-closed stance `scripts/launchd/install.sh` takes when a keg is present.
`BRAINLAYER_HOOK_PYTHON` remains the explicit, loud escape hatch.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

__all__ = [
    "BRAINLAYER_HOOK_SCRIPTS",
    "DEFAULT_KEG_PYTHON",
    "HOOK_PYTHON_ENV",
    "HookPythonUnresolved",
    "UnpinnedHookCommand",
    "find_unpinned_hook_commands",
    "is_bare_python3",
    "main",
    "render_hook_command",
    "resolve_hook_python",
    "shebang_of",
]

#: Explicit override. Honoured first, and it must point at something that exists —
#: an override that silently misses is the same failure in a new costume.
HOOK_PYTHON_ENV = "BRAINLAYER_HOOK_PYTHON"

#: The `opt/` symlink, not a `Cellar/<version>` path: a hook command rendered today
#: outlives the keg version it was rendered against, exactly as a launchd plist does.
DEFAULT_KEG_PYTHON = "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python"

#: Intel-prefix homebrew and the Cellar path a `brew --prefix` answer resolves to.
_FALLBACK_CANDIDATES: tuple[str, ...] = (
    DEFAULT_KEG_PYTHON,
    "/usr/local/opt/brainlayer/libexec/venv/bin/python",
)

#: Hook scripts this repo owns. The settings.json lint matches on these basenames so
#: it never touches a hook belonging to another repo. `tests/test_hook_python.py`
#: asserts every shebang-bearing file under `hooks/` appears here.
BRAINLAYER_HOOK_SCRIPTS = frozenset(
    {
        "brainbar-postcompact.py",
        "brainbar-prompt-capture.py",
        "brainbar-stop-index.py",
        "brainlayer-prompt-search.py",
        "brainlayer-session-start.py",
        "post-commit.py",
        "session-cleanup.py",
    }
)

# `python`, `python3`, `python3.13`, with or without a directory — anything whose
# answer depends on PATH or on a system/framework install rather than on the keg.
_BARE_NAME = re.compile(r"^python(\d+(\.\d+)*)?$")


class HookPythonUnresolved(RuntimeError):
    """No keg python could be found, and PATH is not an acceptable answer."""


@dataclass(frozen=True)
class UnpinnedHookCommand:
    """A configured hook command whose interpreter is decided by PATH."""

    event: str
    script: str
    command: str
    interpreter: str


def is_bare_python3(value: str | None) -> bool:
    """True when `value` names an interpreter PATH or the system decides.

    Accepts a raw token, a full shebang line, or an `/usr/bin/env python3` form.
    A path under a brainlayer keg (`libexec/venv`) is pinned and returns False.
    """
    if value is None:
        return False
    token = value.strip()
    if not token:
        return False
    if token.startswith("#!"):
        token = token[2:].strip()
    parts = token.split()
    if not parts:
        return False
    head = parts[0]
    if os.path.basename(head) == "env":
        # `/usr/bin/env python3` — env's whole job is to ask PATH.
        return len(parts) > 1 and bool(_BARE_NAME.match(os.path.basename(parts[1])))
    if "libexec/venv" in head:
        return False
    return bool(_BARE_NAME.match(os.path.basename(head)))


def shebang_of(path) -> str | None:
    """Return the file's shebang line (stripped), or None when it has none."""
    with open(path, "rb") as handle:
        first = handle.readline()
    if not first.startswith(b"#!"):
        return None
    return first.decode("utf-8", "replace").strip()


def resolve_hook_python(
    *,
    env: Mapping[str, str] | None = None,
    candidates: Sequence[str] | None = None,
) -> str:
    """Return the interpreter BrainLayer hooks must run under.

    Order: `BRAINLAYER_HOOK_PYTHON`, then the keg pythons. Never PATH — raises
    `HookPythonUnresolved` instead, naming everything it looked for.
    """
    env = os.environ if env is None else env
    candidates = _FALLBACK_CANDIDATES if candidates is None else tuple(candidates)

    looked_at: list[str] = []
    override = (env.get(HOOK_PYTHON_ENV) or "").strip()
    if override:
        if os.path.exists(override):
            return override
        looked_at.append(f"{override} (from {HOOK_PYTHON_ENV})")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        looked_at.append(candidate)

    raise HookPythonUnresolved(
        "no brainlayer keg python found; refusing to fall back to PATH because PATH is "
        "what let hooks import a stale checkout silently. Looked at: "
        + ", ".join(looked_at)
        + f". Reinstall the formula (`brew install brainlayer`) or set {HOOK_PYTHON_ENV} explicitly."
    )


def render_hook_command(
    script_path: str,
    *,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Render the `settings.json` command string for one hook script."""
    interpreter = python or resolve_hook_python(env=env)
    return f"{shlex.quote(interpreter) if ' ' in interpreter else interpreter} {script_path}"


def _iter_hook_entries(settings: Mapping) -> Iterator[tuple[str, str]]:
    """Yield `(event, command)` for every command hook configured in `settings`."""
    hooks = settings.get("hooks") if isinstance(settings, Mapping) else None
    if not isinstance(hooks, Mapping):
        return
    for event, matchers in hooks.items():
        if not isinstance(matchers, Iterable) or isinstance(matchers, (str, bytes)):
            continue
        for matcher in matchers:
            if not isinstance(matcher, Mapping):
                continue
            entries = matcher.get("hooks")
            if not isinstance(entries, Iterable) or isinstance(entries, (str, bytes)):
                continue
            for entry in entries:
                if not isinstance(entry, Mapping):
                    continue
                command = entry.get("command")
                if isinstance(command, str) and command.strip():
                    yield str(event), command


def _brainlayer_script_in(command: str) -> tuple[str, str] | None:
    """Find `(script_basename, interpreter_token)` for a BrainLayer hook in `command`.

    Handles wrapper forms — the Stop hook runs through skill-creator's
    `stop-telemetry.mjs <name> -- <python> <script>` shim, so the interpreter is not
    necessarily the first token.
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if os.path.basename(token) in BRAINLAYER_HOOK_SCRIPTS:
            interpreter = tokens[index - 1] if index else ""
            # `/usr/bin/env python3 script.py` puts `env` two tokens back.
            if index >= 2 and os.path.basename(tokens[index - 2]) == "env":
                interpreter = f"{tokens[index - 2]} {tokens[index - 1]}"
            return os.path.basename(token), interpreter
    return None


def find_unpinned_hook_commands(settings: Mapping) -> list[UnpinnedHookCommand]:
    """Return every BrainLayer hook command whose interpreter comes from PATH.

    Hooks owned by other repos are deliberately ignored — they are not ours to repin.
    """
    findings: list[UnpinnedHookCommand] = []
    for event, command in _iter_hook_entries(settings):
        found = _brainlayer_script_in(command)
        if found is None:
            continue
        script, interpreter = found
        if is_bare_python3(interpreter):
            findings.append(UnpinnedHookCommand(event=event, script=script, command=command, interpreter=interpreter))
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m brainlayer.hook_python [settings.json]` — lint a settings file.

    Exits 0 when every BrainLayer hook names its interpreter, 1 when any is
    PATH-resolved, 2 when the file cannot be read. Hooks owned by other repos are
    never reported.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="brainlayer.hook_python")
    parser.add_argument(
        "settings",
        nargs="?",
        default=os.path.expanduser("~/.claude/settings.json"),
        help="path to a Claude Code settings.json (default: ~/.claude/settings.json)",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.settings, encoding="utf-8") as handle:
            settings = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.settings}: {exc}", flush=True)
        return 2

    findings = find_unpinned_hook_commands(settings)
    if not findings:
        print(f"OK: every BrainLayer hook in {args.settings} names its interpreter")
        return 0
    print(f"UNPINNED BrainLayer hooks in {args.settings}:")
    for finding in findings:
        print(f"  {finding.event}: {finding.script} runs under {finding.interpreter!r}")
        print(f"    {finding.command}")
    try:
        print(f"  pin them to: {resolve_hook_python()}")
    except HookPythonUnresolved as exc:
        print(f"  (and the keg python is missing: {exc})")
    return 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
