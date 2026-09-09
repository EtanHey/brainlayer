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
from xml.sax.saxutils import escape

__all__ = [
    "BRAINLAYER_HOOK_SCRIPTS",
    "DEFAULT_KEG_PYTHON",
    "HOOK_PYTHON_ENV",
    "HookPythonUnresolved",
    "UnpinnedHookCommand",
    "find_unpinned_hook_commands",
    "is_bare_python3",
    "is_pinned_interpreter",
    "is_system_python",
    "main",
    "render_hook_command",
    "render_launchd_plist",
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

# `python`, `python3`, `python3.13` — a name PATH has to resolve.
_BARE_NAME = re.compile(r"^python(\d+(\.\d+)*)?$")

# The same shape, used to confirm an ABSOLUTE path actually points at a python.
_PYTHON_NAME = _BARE_NAME

#: `bin` directories shared by everything on the machine, so a `.pth` dropped in their
#: `site-packages` reaches every caller. `/opt/homebrew/bin/python3` is Homebrew's own
#: python, not a brainlayer keg — the keg lives under `libexec/venv`.
_SYSTEM_BIN_DIRS = frozenset(
    {
        "/bin",
        "/sbin",
        "/usr/bin",
        "/usr/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/home/linuxbrew/.linuxbrew/bin",
    }
)


class HookPythonUnresolved(RuntimeError):
    """No keg python could be found, and PATH is not an acceptable answer."""


@dataclass(frozen=True)
class UnpinnedHookCommand:
    """A configured hook command that does not name an interpreter we can vouch for."""

    event: str
    script: str
    command: str
    interpreter: str
    reason: str = ""


def _tokens(value: str | None) -> list[str]:
    """Normalise a raw token, a shebang line, or an `env python3` pair into tokens."""
    if value is None:
        return []
    token = value.strip()
    if token.startswith("#!"):
        token = token[2:].strip()
    return token.split()


def is_bare_python3(value: str | None) -> bool:
    """True when PATH decides which interpreter runs.

    Exactly one question, and only this one: a bare name (`python3`, `python3.13`) or an
    `/usr/bin/env python3` form. **An absolute path is never bare** — it names something,
    whatever it names. Whether the thing it names is a *good* choice is `is_system_python`'s
    question. Conflating the two is what made the escape hatch and the linter disagree:
    `resolve_hook_python` accepted an operator's `/tmp/myvenv/bin/python` while this
    function called the resulting command unpinned.
    """
    parts = _tokens(value)
    if not parts:
        return False
    head = parts[0]
    if os.path.basename(head) == "env":
        # `/usr/bin/env python3` — env's whole job is to ask PATH.
        return len(parts) > 1 and bool(_BARE_NAME.match(os.path.basename(parts[1])))
    if os.sep in head:
        # Any directory component means something other than PATH resolves it: absolute, or
        # relative to the cwd. Both are wrong for a hook, but they are wrong differently, and
        # `_why_unpinned` has to be able to say which.
        return False
    return bool(_BARE_NAME.match(head))


def is_system_python(value: str | None) -> bool:
    """True for a site-wide interpreter — the class that carries global `.pth` files.

    This is not a style preference. `/Library/Frameworks/Python.framework/.../python3` is
    precisely the interpreter whose `site-packages/_brainlayer.pth` injected a live checkout
    onto every hook's import path. Naming it absolutely closes the PATH hazard and leaves
    the `.pth` hazard wide open, so a named site-wide python is still not an acceptable pin.
    A venv or keg python (`libexec/venv`, `.venv`, any per-project prefix) is not site-wide.
    """
    parts = _tokens(value)
    if not parts:
        return False
    head = parts[-1] if os.path.basename(parts[0]) == "env" else parts[0]
    if not os.path.isabs(head):
        return False
    directory = os.path.dirname(head)
    if any(segment in head for segment in ("/libexec/venv/", "/.venv/")):
        return False
    if "/Python.framework/" in head:
        return True
    return directory in _SYSTEM_BIN_DIRS


def is_pinned_interpreter(value: str | None) -> bool:
    """The affirmative gate: does this command name an interpreter we can vouch for?

    Fail CLOSED. Everything unrecognised — an empty token, `--`, `run`, a relative path, a
    non-python word — answers False. A gate that says "fine" to a command shape it does not
    understand is not a gate, and this one guards every future BrainLayer hook.
    """
    parts = _tokens(value)
    if not parts:
        return False
    head = parts[0]
    if not os.path.isabs(head):
        return False
    if is_bare_python3(value) or is_system_python(value):
        return False
    return bool(_PYTHON_NAME.match(os.path.basename(head)))


def shebang_of(path: str | os.PathLike[str]) -> str | None:
    """Return the file's shebang line (stripped), or None when it has none.

    `path` names a hook script this repo ships — it is never user input, and this reads
    the first line only. DeepSource's "external variable used in file path" audit fires on
    any non-literal `open()`, which is a false positive here; suppressed rather than
    contorted, because a caller that could not pass a path would make the helper useless.
    """
    with open(path, "rb") as handle:  # skipcq: PTC-W6004 - repo-local hook scripts, not user input
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
        # A relative override is refused outright, not resolved. `os.path.exists("python3")`
        # is true whenever the cwd happens to hold one, so an existence check alone would
        # accept `BRAINLAYER_HOOK_PYTHON=python3`, `render_hook_command` would emit
        # `python3 <script>`, and the hook process would resolve it through PATH — this
        # module's whole bug, arriving through its own escape hatch.
        if not os.path.isabs(override):
            raise HookPythonUnresolved(
                f"{HOOK_PYTHON_ENV}={override!r} is not an absolute path. A relative "
                "interpreter is resolved by PATH or the working directory at hook time, "
                "which is exactly what this pin exists to prevent. Give the full path."
            )
        if is_system_python(override):
            raise HookPythonUnresolved(
                f"{HOOK_PYTHON_ENV}={override!r} is a site-wide python. Naming it absolutely "
                "closes the PATH hazard and leaves the other one open: a site-wide "
                "interpreter's site-packages is where a global .pth lives, and a "
                "_brainlayer.pth there is what put a live checkout on every hook's import "
                "path. It is also the configuration this module's own settings lint rejects. "
                "Point this at a keg or venv python."
            )
        # Set-but-missing RAISES; it does not fall through to the keg candidates. Setting
        # this variable is a deliberate operator choice, and quietly substituting a
        # different interpreter for a typo'd one is the same silent-substitution failure
        # this module refuses everywhere else.
        if not os.path.exists(override):
            raise HookPythonUnresolved(
                f"{HOOK_PYTHON_ENV}={override!r} does not exist. Refusing to silently "
                "substitute another interpreter for an override that was set on purpose — "
                f"fix the path or unset {HOOK_PYTHON_ENV} to use the keg."
            )
        return override

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


def render_launchd_plist(
    template: str,
    *,
    python: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Render the prefix-aware keg interpreter into a launchd template.

    Templates stay portable across ARM and Intel Homebrew prefixes. Resolution
    uses the same fail-closed candidate order and override rules as hook command
    rendering; a caller-supplied interpreter is accepted only when the existing
    affirmative pin gate can vouch for it.
    """
    interpreter = python or resolve_hook_python(env=env)
    if not is_pinned_interpreter(interpreter):
        raise HookPythonUnresolved(f"launchd interpreter is not explicitly pinned: {interpreter!r}")
    return template.replace("__BRAINLAYER_PYTHON__", escape(interpreter))


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
        if os.path.basename(token) not in BRAINLAYER_HOOK_SCRIPTS:
            continue
        # Find the interpreter by what it LOOKS like, walking back from the script — not by
        # position. `python3 -u script.py` would make the adjacent token `-u`, and
        # `is_bare_python3("-u")` is False, so the lint would call a PATH-resolved hook
        # pinned. Skipping tokens that start with `-` is not enough either: `-X utf8` is an
        # option WITH an argument, and `utf8` does not start with `-`.
        interpreter = ""
        cursor = index - 1
        while cursor >= 0:
            candidate = tokens[cursor]
            if os.path.basename(candidate).startswith("python"):
                interpreter = candidate
                # `/usr/bin/env python3 …` — env's whole job is to ask PATH, so keep both.
                if cursor >= 1 and os.path.basename(tokens[cursor - 1]) == "env":
                    interpreter = f"{tokens[cursor - 1]} {candidate}"
                break
            cursor -= 1
        if not interpreter and index:
            # No python-shaped token: fall back to the adjacent one so an unrecognised
            # runner is still reported rather than silently passing as pinned.
            interpreter = tokens[index - 1]
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
        # Affirmative gate, not a blacklist. Anything this module cannot vouch for is
        # REPORTED — an empty token, `--`, `run`, a relative path, a site-wide python. A
        # lint that answers "fine" to a shape it does not understand is not a gate, and the
        # blacklist form let the Stop shim drop its pin and still read clean.
        if not is_pinned_interpreter(interpreter):
            findings.append(
                UnpinnedHookCommand(
                    event=event,
                    script=script,
                    command=command,
                    interpreter=interpreter,
                    reason=_why_unpinned(interpreter),
                )
            )
    return findings


def _why_unpinned(interpreter: str) -> str:
    """Say which failure it is, so the finding is actionable rather than a bare verdict."""
    if not interpreter.strip():
        return "no interpreter in the command — the script runs under its shebang, which PATH may resolve"
    if is_bare_python3(interpreter):
        return f"{interpreter!r} is resolved by PATH"
    if is_system_python(interpreter):
        return (
            f"{interpreter!r} is a site-wide python — its site-packages is where a global "
            ".pth lives, which is how a stale checkout reached the hooks"
        )
    if not _PYTHON_NAME.match(os.path.basename(interpreter)):
        return (
            f"{interpreter!r} is not a recognisable python interpreter — refusing to assume "
            "a command shape this lint does not understand is pinned"
        )
    if not os.path.isabs(interpreter):
        return f"{interpreter!r} is a relative path, resolved against the cwd at hook time"
    return f"{interpreter!r} is not an interpreter this lint can vouch for"


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
        print(f"  {finding.event}: {finding.script} — {finding.reason}")
        print(f"    {finding.command}")
    try:
        print(f"  pin them to: {resolve_hook_python()}")
    except HookPythonUnresolved as exc:
        print(f"  (and the keg python is missing: {exc})")
    return 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
