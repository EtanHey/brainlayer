"""Every BrainLayer Claude Code hook must name the keg python explicitly.

Hooks are wired into `~/.claude/settings.json` as `<python> <script>`. When that
`<python>` is bare `python3`, PATH decides which interpreter — and therefore which
`brainlayer` library — every hook fire executes. On the M4 that answer was the
framework python at `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`,
whose `site-packages/_brainlayer.pth` injects `~/Gits/brainlayer/src`: a live,
agent-editable checkout with no `__build_sha__`. A stale snapshot there ran 09-02
library code under a 1.5.15 CLI for days without a single error.

So: the interpreter is pinned, and a silent PATH fallback is refused the same way
`scripts/launchd/install.sh` refuses one when a keg is present.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from brainlayer.hook_python import (
    BRAINLAYER_HOOK_SCRIPTS,
    DEFAULT_KEG_PYTHON,
    HOOK_PYTHON_ENV,
    HookPythonUnresolved,
    find_unpinned_hook_commands,
    is_bare_python3,
    is_pinned_interpreter,
    is_system_python,
    main,
    render_hook_command,
    resolve_hook_python,
    shebang_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = REPO_ROOT / "hooks"


def _hook_scripts_with_shebangs() -> list[Path]:
    return sorted(p for p in HOOKS_DIR.glob("*.py") if shebang_of(p) is not None)


class TestShebangs:
    @staticmethod
    def test_hooks_dir_is_not_empty():
        assert _hook_scripts_with_shebangs(), f"no shebang-bearing hooks under {HOOKS_DIR}"

    @staticmethod
    @pytest.mark.parametrize("script", _hook_scripts_with_shebangs(), ids=lambda p: p.name)
    def test_hook_shebang_is_not_bare_python3(script: Path):
        shebang = shebang_of(script)
        assert is_pinned_interpreter(shebang), (
            f"{script.name} has a PATH-resolved shebang ({shebang!r}). "
            "PATH decides which brainlayer library the hook imports; pin the keg python."
        )

    @staticmethod
    @pytest.mark.parametrize("script", _hook_scripts_with_shebangs(), ids=lambda p: p.name)
    def test_hook_shebang_names_the_keg_python(script: Path):
        assert shebang_of(script) == f"#!{DEFAULT_KEG_PYTHON}", (
            f"{script.name} must invoke the keg python, not {shebang_of(script)!r}"
        )

    @staticmethod
    def test_every_shebang_bearing_hook_is_declared():
        """The shipped constant is what the settings.json linter matches on."""
        on_disk = {p.name for p in _hook_scripts_with_shebangs()}
        assert on_disk <= BRAINLAYER_HOOK_SCRIPTS, (
            f"undeclared hook scripts: {sorted(on_disk - BRAINLAYER_HOOK_SCRIPTS)} — "
            "add them to BRAINLAYER_HOOK_SCRIPTS or the settings.json lint will miss them"
        )


class TestIsBarePython3:
    """`is_bare_python3` answers exactly one question: does PATH decide?"""

    @pytest.mark.parametrize(
        "value",
        [
            "python3",
            "python",
            "  python3  ",
            "#!/usr/bin/env python3",
            "#!/usr/bin/env python",
            "/usr/bin/env python3",
        ],
    )
    @staticmethod
    def test_path_resolved_interpreters_are_bare(value):
        assert is_bare_python3(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            DEFAULT_KEG_PYTHON,
            f"#!{DEFAULT_KEG_PYTHON}",
            "/opt/homebrew/Cellar/brainlayer/1.5.15/libexec/venv/bin/python",
            # An absolute path is named, whatever it names. Whether it is a GOOD choice is
            # `is_system_python`'s question, not this one — conflating the two is what made
            # the escape hatch and the linter disagree (review round 1, medium).
            "/usr/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "/tmp/myvenv/bin/python",
        ],
    )
    @staticmethod
    def test_absolute_interpreters_are_not_path_resolved(value):
        assert is_bare_python3(value) is False

    @staticmethod
    def test_none_is_not_bare():
        assert is_bare_python3(None) is False

    @pytest.mark.parametrize("value", ["./bin/python", "bin/python3", "../venv/bin/python"])
    @staticmethod
    def test_cwd_relative_interpreters_are_not_path_resolved(value):
        """Both are wrong for a hook, but they are wrong differently.

        A directory component — absolute or relative — means something other than PATH
        resolves it. `is_pinned_interpreter` rejects these anyway (not absolute); keeping
        them out of `is_bare_python3` is what lets `_why_unpinned` say "resolved against
        the cwd" instead of the false "resolved by PATH".
        """
        assert is_bare_python3(value) is False
        assert is_pinned_interpreter(value) is False


class TestIsSystemPython:
    """Site-wide interpreters are the ones that carry a global `.pth` — this bug's origin."""

    @pytest.mark.parametrize(
        "value",
        [
            "/usr/bin/python3",
            "/usr/local/bin/python3",
            "/opt/homebrew/bin/python3",
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "#!/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "/System/Library/Frameworks/Python.framework/Versions/2.7/bin/python",
        ],
    )
    @staticmethod
    def test_site_wide_interpreters_are_system(value):
        assert is_system_python(value) is True

    @pytest.mark.parametrize(
        "value",
        [DEFAULT_KEG_PYTHON, "/tmp/myvenv/bin/python", "/Users/x/Gits/brainlayer/.venv/bin/python"],
    )
    @staticmethod
    def test_venv_and_keg_interpreters_are_not_system(value):
        assert is_system_python(value) is False


class TestIsPinnedInterpreter:
    """The affirmative gate the lint uses. Anything it does not recognise is NOT pinned."""

    @pytest.mark.parametrize(
        "value",
        [
            DEFAULT_KEG_PYTHON,
            "/usr/local/opt/brainlayer/libexec/venv/bin/python",
            # An operator's deliberate absolute venv python. `resolve_hook_python` accepts
            # this as an override, so the linter must accept it too, or the escape hatch and
            # the gate contradict each other (review round 1, medium).
            "/tmp/myvenv/bin/python",
            "'/Users/Jane Doe/.venv/bin/python'",
            "/Users/x/Gits/brainlayer/.venv/bin/python3.13",
        ],
    )
    @staticmethod
    def test_absolute_non_system_pythons_are_pinned(value):
        assert is_pinned_interpreter(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "",  # no interpreter token at all — the script was the whole command
            "   ",
            None,
            "--",  # the Stop shim's separator, with no python after it
            "run",  # `uv run <script>`
            "python3",
            "/usr/bin/env python3",
            "/usr/bin/python3",  # named, but site-wide: carries the .pth that caused this
            "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
            "./relative/python",
        ],
    )
    @staticmethod
    def test_everything_else_is_not_pinned(value):
        assert is_pinned_interpreter(value) is False


class TestResolveHookPython:
    @staticmethod
    def test_env_override_wins(tmp_path):
        override = tmp_path / "python"
        override.write_text("#!/bin/sh\n")
        override.chmod(0o755)
        assert resolve_hook_python(env={HOOK_PYTHON_ENV: str(override)}) == str(override)

    @staticmethod
    def test_env_override_must_exist(tmp_path):
        with pytest.raises(HookPythonUnresolved) as excinfo:
            resolve_hook_python(env={HOOK_PYTHON_ENV: str(tmp_path / "nope")}, candidates=())
        assert HOOK_PYTHON_ENV in str(excinfo.value)

    @staticmethod
    def test_a_missing_override_raises_even_when_a_keg_is_available(tmp_path):
        """A set-but-missing override must RAISE, not quietly fall through to the keg.

        Setting `BRAINLAYER_HOOK_PYTHON` is a deliberate operator choice. Ignoring a typo in
        it and silently using a different interpreter is the same fail-open this module
        refuses everywhere else — and the previous `test_env_override_must_exist` could not
        catch it, because it passed `candidates=()` so there was nothing to fall through to.
        """
        keg = tmp_path / "libexec" / "venv" / "bin" / "python"
        keg.parent.mkdir(parents=True)
        keg.write_text("#!/bin/sh\n")
        with pytest.raises(HookPythonUnresolved) as excinfo:
            resolve_hook_python(
                env={HOOK_PYTHON_ENV: str(tmp_path / "typo" / "python")},
                candidates=(str(keg),),
            )
        message = str(excinfo.value)
        assert HOOK_PYTHON_ENV in message
        assert str(keg) not in message, "it must not report the keg it declined to substitute"

    @staticmethod
    def test_a_system_python_override_is_refused(tmp_path):
        """Pointing the escape hatch at a site-wide python re-arms the `.pth` that caused this.

        It is also what keeps the hatch and the linter agreeing: `find_unpinned_hook_commands`
        would flag a command rendered from it, so accepting it here would let an operator
        create a configuration this module's own gate rejects.
        """
        fake_framework = tmp_path / "Library" / "Frameworks" / "Python.framework"
        target = fake_framework / "Versions" / "3.13" / "bin" / "python3"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\n")
        with pytest.raises(HookPythonUnresolved) as excinfo:
            resolve_hook_python(env={HOOK_PYTHON_ENV: str(target)}, candidates=())
        assert "site-wide" in str(excinfo.value)

    @staticmethod
    def test_an_absolute_venv_override_outside_a_keg_is_accepted(tmp_path):
        """The hatch is not "must be Homebrew-shaped" — it is "must be explicitly named"."""
        target = tmp_path / "myvenv" / "bin" / "python"
        target.parent.mkdir(parents=True)
        target.write_text("#!/bin/sh\n")
        target.chmod(0o755)
        resolved = resolve_hook_python(env={HOOK_PYTHON_ENV: str(target)}, candidates=())
        assert resolved == str(target)
        assert is_pinned_interpreter(resolved), "the linter must accept what the hatch returns"

    @staticmethod
    def test_first_existing_candidate_wins(tmp_path):
        missing = tmp_path / "missing" / "python"
        present = tmp_path / "present"
        present.write_text("#!/bin/sh\n")
        present.chmod(0o755)
        assert resolve_hook_python(env={}, candidates=(str(missing), str(present))) == str(present)

    @staticmethod
    def test_candidate_must_be_a_regular_executable_file(tmp_path):
        directory = tmp_path / "directory" / "python"
        directory.mkdir(parents=True)
        non_executable = tmp_path / "not-executable" / "python"
        non_executable.parent.mkdir()
        non_executable.write_text("#!/bin/sh\n")
        usable = tmp_path / "usable" / "python"
        usable.parent.mkdir()
        usable.write_text("#!/bin/sh\n")
        usable.chmod(0o755)

        assert resolve_hook_python(env={}, candidates=(str(directory), str(non_executable), str(usable))) == str(usable)

    @staticmethod
    def test_never_falls_back_to_path():
        """A silent `python3` fallback is the bug, not the remedy."""
        with pytest.raises(HookPythonUnresolved) as excinfo:
            resolve_hook_python(env={}, candidates=("/nonexistent/keg/bin/python",))
        message = str(excinfo.value)
        assert "/nonexistent/keg/bin/python" in message, "the error must name what it looked for"
        assert HOOK_PYTHON_ENV in message, "the error must name the explicit escape hatch"

    @staticmethod
    @pytest.mark.parametrize("relative", ["python3", "python", "./venv/bin/python", "bin/python"])
    def test_a_relative_override_is_refused(relative, monkeypatch, tmp_path):
        """`BRAINLAYER_HOOK_PYTHON=python3` would hand the choice straight back to PATH.

        `os.path.exists("python3")` is true whenever the cwd happens to contain one, so an
        existence check alone would return it, `render_hook_command` would emit
        `python3 <script>`, and the hook process would resolve it through PATH — the exact
        bug this module exists to close, arriving through the escape hatch.
        """
        (tmp_path / "python3").write_text("#!/bin/sh\n")
        (tmp_path / "python").write_text("#!/bin/sh\n")
        (tmp_path / "bin").mkdir()
        (tmp_path / "bin" / "python").write_text("#!/bin/sh\n")
        (tmp_path / "venv" / "bin").mkdir(parents=True)
        (tmp_path / "venv" / "bin" / "python").write_text("#!/bin/sh\n")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(HookPythonUnresolved) as excinfo:
            resolve_hook_python(env={HOOK_PYTHON_ENV: relative}, candidates=())
        assert "absolute" in str(excinfo.value)

    @staticmethod
    def test_default_candidate_is_the_opt_symlink():
        """`opt/` outlives the Cellar version a command was rendered against."""
        assert DEFAULT_KEG_PYTHON == "/opt/homebrew/opt/brainlayer/libexec/venv/bin/python"

    @staticmethod
    def test_non_python_executable_override_is_refused(tmp_path):
        target = tmp_path / "bin" / "bash"
        target.parent.mkdir()
        target.write_text("#!/bin/sh\n")
        target.chmod(0o755)
        with pytest.raises(HookPythonUnresolved, match="executable Python"):
            resolve_hook_python(env={HOOK_PYTHON_ENV: str(target)}, candidates=())


class TestRenderHookCommand:
    @staticmethod
    def test_renders_pinned_interpreter_and_script(tmp_path):
        # Shaped like a real keg: `is_bare_python3` clears a path only because of the
        # `libexec/venv` segment, so a fixture without it would be flagged correctly.
        python = tmp_path / "libexec" / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        rendered = render_hook_command(
            "/Users/x/.claude/hooks/brainlayer-prompt-search.py",
            env={HOOK_PYTHON_ENV: str(python)},
        )
        assert rendered == f"{python} /Users/x/.claude/hooks/brainlayer-prompt-search.py"
        assert not is_bare_python3(rendered.split()[0])


def _settings(command: str) -> dict:
    return {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": command}]},
            ]
        }
    }


class TestFindUnpinnedHookCommands:
    @staticmethod
    def test_flags_bare_python3_on_a_brainlayer_hook():
        settings = _settings("python3 /Users/x/.claude/hooks/brainlayer-prompt-search.py")
        findings = find_unpinned_hook_commands(settings)
        assert len(findings) == 1
        assert findings[0].script == "brainlayer-prompt-search.py"
        assert findings[0].event == "UserPromptSubmit"

    @staticmethod
    def test_accepts_a_pinned_command():
        settings = _settings(f"{DEFAULT_KEG_PYTHON} /Users/x/.claude/hooks/brainlayer-prompt-search.py")
        assert find_unpinned_hook_commands(settings) == []

    @staticmethod
    def test_ignores_hooks_this_repo_does_not_own():
        """Etan's other hooks are not ours to repin."""
        settings = _settings("python3 /Users/x/.claude/hooks/tdd-guard.py")
        assert find_unpinned_hook_commands(settings) == []

    @staticmethod
    def test_sees_through_a_wrapper_command():
        """The Stop hook runs through skill-creator's stop-telemetry shim."""
        settings = {
            "hooks": {
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    "/Users/x/Gits/skill-creator/hooks-lab/stop-telemetry.mjs "
                                    "brainbar-stop-index -- python3 "
                                    "/Users/x/Gits/brainlayer/hooks/brainbar-stop-index.py"
                                ),
                            }
                        ]
                    }
                ]
            }
        }
        findings = find_unpinned_hook_commands(settings)
        assert len(findings) == 1
        assert findings[0].script == "brainbar-stop-index.py"

    @staticmethod
    @pytest.mark.parametrize(
        "command",
        [
            "python3 -u /Users/x/.claude/hooks/brainlayer-prompt-search.py",
            "python3 -X utf8 -u /Users/x/.claude/hooks/brainlayer-prompt-search.py",
            "/usr/bin/env python3 -u /Users/x/.claude/hooks/brainlayer-prompt-search.py",
        ],
    )
    def test_interpreter_options_do_not_hide_a_path_resolved_python(command):
        """`python3 -u script.py` is still PATH-resolved.

        Reading only the token immediately before the script would record `-u` as the
        interpreter, `is_bare_python3("-u")` is False, and the lint would call a
        PATH-resolved hook pinned. Option tokens are skipped instead.
        """
        findings = find_unpinned_hook_commands(_settings(command))
        assert len(findings) == 1, f"{command!r} must be flagged"
        assert findings[0].script == "brainlayer-prompt-search.py"

    @staticmethod
    def test_options_after_a_pinned_interpreter_stay_clean():
        settings = _settings(f"{DEFAULT_KEG_PYTHON} -u /Users/x/.claude/hooks/brainlayer-prompt-search.py")
        assert find_unpinned_hook_commands(settings) == []

    @staticmethod
    def test_env_python3_is_flagged_too():
        settings = _settings("/usr/bin/env python3 /Users/x/.claude/hooks/session-cleanup.py")
        assert len(find_unpinned_hook_commands(settings)) == 1

    @staticmethod
    def test_empty_settings_is_clean():
        assert find_unpinned_hook_commands({}) == []

    @pytest.mark.parametrize(
        ("command", "why"),
        [
            (
                "/Users/x/.claude/hooks/brainlayer-prompt-search.py",
                "script alone: no interpreter token, so PATH resolves the shebang instead",
            ),
            (
                "/Users/x/hooks-lab/stop-telemetry.mjs brainbar-stop-index -- "
                "/Users/x/Gits/brainlayer/hooks/brainbar-stop-index.py",
                "the Stop shim with the pin dropped: the token before the script is `--`",
            ),
            (
                "uv run /Users/x/.claude/hooks/brainlayer-prompt-search.py",
                "an unrecognised runner: the token before the script is `run`",
            ),
            (
                "/usr/bin/python3 /Users/x/.claude/hooks/brainlayer-prompt-search.py",
                "named but site-wide — this is the interpreter carrying the .pth",
            ),
        ],
    )
    @staticmethod
    def test_a_shape_the_lint_cannot_recognise_is_reported_not_passed(command, why):
        """Fail CLOSED. A gate that answers "fine" to a shape it does not understand is not a gate.

        Each of these returned zero findings before review round 1. The Stop case is the worst:
        `_brainlayer_script_in` handles that wrapper specially, so the one command shape this
        module goes out of its way to parse could drop its pin and still lint clean.
        """
        findings = find_unpinned_hook_commands(_settings(command))
        assert len(findings) == 1, f"must be flagged ({why})"
        assert findings[0].reason, "a finding must say WHY, or it is not actionable"

    @staticmethod
    def test_the_reason_distinguishes_the_failure_modes():
        reasons = {
            find_unpinned_hook_commands(_settings(cmd))[0].reason
            for cmd in (
                "python3 /Users/x/.claude/hooks/brainlayer-prompt-search.py",
                "/Users/x/.claude/hooks/brainlayer-prompt-search.py",
                "/usr/bin/python3 /Users/x/.claude/hooks/brainlayer-prompt-search.py",
            )
        }
        assert len(reasons) == 3, f"each failure mode needs its own reason, got {reasons}"


class TestCli:
    """`python -m brainlayer.hook_python <settings.json>` — a hand-runnable lint."""

    @staticmethod
    def test_exits_one_on_an_unpinned_hook(tmp_path, capsys):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(_settings("python3 /x/hooks/brainlayer-prompt-search.py")))
        assert main([str(path)]) == 1
        assert "brainlayer-prompt-search.py" in capsys.readouterr().out

    @staticmethod
    def test_exits_zero_when_pinned(tmp_path, capsys):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(_settings(f"{DEFAULT_KEG_PYTHON} /x/hooks/brainlayer-prompt-search.py")))
        assert main([str(path)]) == 0
        assert "OK" in capsys.readouterr().out

    @staticmethod
    def test_exits_two_on_an_unreadable_file(tmp_path, capsys):
        assert main([str(tmp_path / "absent.json")]) == 2
        assert "cannot read" in capsys.readouterr().out

    @staticmethod
    def test_exits_two_on_invalid_json(tmp_path, capsys):
        path = tmp_path / "settings.json"
        path.write_text("{not json")
        assert main([str(path)]) == 2
        assert "cannot read" in capsys.readouterr().out

    @staticmethod
    def test_print_interpreter_uses_affirmative_resolver(tmp_path, monkeypatch, capsys):
        python = tmp_path / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        monkeypatch.setenv(HOOK_PYTHON_ENV, str(python))

        assert main(["--print-interpreter"]) == 0
        assert capsys.readouterr().out.strip() == str(python)


#: conftest sandboxes HOME for every test, so `~/.claude/settings.json` is not
#: reachable by default — deliberately: a unit suite must not read Etan's home. Point
#: this at the real file to run the deployment check:
#:   BRAINLAYER_CLAUDE_SETTINGS=~/.claude/settings.json pytest tests/test_hook_python_pin.py
SETTINGS_PATH_ENV = "BRAINLAYER_CLAUDE_SETTINGS"


class TestLiveSettings:
    """The deployment half. Skipped where there is no settings.json to check."""

    @staticmethod
    def test_installed_brainlayer_hooks_are_pinned():
        override = os.environ.get(SETTINGS_PATH_ENV)
        settings_path = Path(os.path.expanduser(override or "~/.claude/settings.json"))
        if not settings_path.exists():
            pytest.skip(f"no {settings_path} (set {SETTINGS_PATH_ENV} to check a real one)")
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError as exc:  # pragma: no cover - operator error, not ours
            pytest.skip(f"{settings_path} is not valid JSON: {exc}")
        findings = find_unpinned_hook_commands(settings)
        assert findings == [], "BrainLayer hooks still resolve their interpreter through PATH: " + "; ".join(
            f"{f.event}: {f.command}" for f in findings
        )


def test_launchd_plist_templates_pin_their_interpreter(tmp_path):
    """Render machine-specific placeholders, then apply the existing pin gate."""
    import plistlib

    from brainlayer.hook_python import render_launchd_plist

    plists = sorted((REPO_ROOT / "launchd").glob("*.plist"))
    assert plists, "expected launchd templates to exist"
    python = tmp_path / "intel" / "opt" / "brainlayer" / "libexec" / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    unpinned: list[str] = []
    for path in plists:
        template = path.read_text(encoding="utf-8")
        rendered = render_launchd_plist(template, python=str(python))
        args = plistlib.loads(rendered.encode()).get("ProgramArguments") or []
        if not args:
            continue
        interpreter = args[0]
        # A wrapper or installed CLI is not a direct interpreter claim.
        if "python" not in interpreter and not interpreter.endswith("/env"):
            continue
        if interpreter.endswith("/env"):
            interpreter = f"{interpreter} {args[1] if len(args) > 1 else ''}".strip()
        if not is_pinned_interpreter(interpreter):
            unpinned.append(f"{path.name}: {interpreter}")

    assert not unpinned, "launchd templates must name a pinned interpreter, not PATH: " + "; ".join(unpinned)


def test_launchd_plist_render_uses_available_intel_keg(monkeypatch):
    from brainlayer import hook_python

    template = "<string>__BRAINLAYER_PYTHON__</string>"
    intel = "/usr/local/opt/brainlayer/libexec/venv/bin/python"
    monkeypatch.setattr(hook_python.os.path, "exists", lambda path: path == intel)
    monkeypatch.setattr(hook_python.os.path, "isfile", lambda path: path == intel)
    monkeypatch.setattr(hook_python.os, "access", lambda path, mode: path == intel and mode == hook_python.os.X_OK)

    rendered = hook_python.render_launchd_plist(template, env={})

    assert rendered == f"<string>{intel}</string>"


def test_launchd_plist_render_rejects_non_executable_interpreter(tmp_path):
    from brainlayer.hook_python import HookPythonUnresolved, render_launchd_plist

    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")

    with pytest.raises(HookPythonUnresolved):
        render_launchd_plist("<string>__BRAINLAYER_PYTHON__</string>", python=str(python))


def test_launchd_plist_render_accepts_executable_interpreter_with_spaces(tmp_path):
    import xml.etree.ElementTree as ET

    from brainlayer.hook_python import render_launchd_plist

    python = tmp_path / "Jane Doe" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    rendered = render_launchd_plist("<string>__BRAINLAYER_PYTHON__</string>", python=str(python))

    assert ET.fromstring(rendered).text == str(python)


def test_launchd_plist_render_rejects_xml_forbidden_interpreter_path(tmp_path):
    from brainlayer.hook_python import HookPythonUnresolved, render_launchd_plist

    python = tmp_path / "bad\x01path" / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\n")
    python.chmod(0o755)

    with pytest.raises(HookPythonUnresolved):
        render_launchd_plist("<string>__BRAINLAYER_PYTHON__</string>", python=str(python))
