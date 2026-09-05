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
        assert not is_bare_python3(shebang), (
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
    @pytest.mark.parametrize(
        "value",
        [
            "python3",
            "python",
            "  python3  ",
            "#!/usr/bin/env python3",
            "#!/usr/bin/env python",
            "/usr/bin/env python3",
            "#!/usr/bin/python3",
            "#!/usr/local/bin/python3",
            "#!/Library/Frameworks/Python.framework/Versions/3.13/bin/python3",
        ],
    )
    @staticmethod
    def test_path_or_system_interpreters_are_bare(value):
        assert is_bare_python3(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            DEFAULT_KEG_PYTHON,
            f"#!{DEFAULT_KEG_PYTHON}",
            "/opt/homebrew/Cellar/brainlayer/1.5.15/libexec/venv/bin/python",
        ],
    )
    @staticmethod
    def test_keg_interpreters_are_pinned(value):
        assert is_bare_python3(value) is False

    @staticmethod
    def test_none_is_not_bare():
        assert is_bare_python3(None) is False


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
    def test_first_existing_candidate_wins(tmp_path):
        missing = tmp_path / "missing" / "python"
        present = tmp_path / "present"
        present.write_text("#!/bin/sh\n")
        present.chmod(0o755)
        assert resolve_hook_python(env={}, candidates=(str(missing), str(present))) == str(present)

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
