"""Launcher probes: intercept subprocess.run so TLC can never start."""

import importlib.util
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location("calendar_guard_check", Path(__file__).with_name("check.py"))
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (None, str(Path.home() / ".local/bin/tlc")),
        ("/tmp/t2 rig/bin/tlc", "/tmp/t2 rig/bin/tlc"),
        ("~/t2-rig/tlc", str(Path.home() / "t2-rig/tlc")),
    ],
)
def test_tlc_executable_selection(tmp_path, override, expected):
    entry = {
        "name": "launcher-probe",
        "module": "CalendarGuard",
        "constants": {},
        "invariants": [],
        "properties": [],
        "violation": None,
    }
    environment = dict(os.environ)
    environment.pop("TLC", None)
    if override is not None:
        environment["TLC"] = override
    result = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with (
        patch.dict(os.environ, environment, clear=True),
        patch.object(checker, "wait_for_load", return_value={}),
        patch.object(checker.subprocess, "run", return_value=result) as launch,
    ):
        assert checker.run_tlc(entry, tmp_path)["reproduced"]
    command = launch.call_args.args[0]
    assert command[0] == expected
    assert command[1:3] == ["-workers", "1"]
    assert launch.call_count == 1


def probe_entry():
    return dict(
        name="launcher-probe", module="CalendarGuard", constants={}, invariants=[], properties=[], violation=None
    )


def test_relative_tlc_uses_caller_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TLC", "rig/bin/tlc")
    with (
        patch.object(checker, "wait_for_load", return_value={}),
        patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as launch,
    ):
        checker.run_tlc(probe_entry(), tmp_path)
    assert launch.call_args.args[0][0] == str(tmp_path / "rig/bin/tlc")


@pytest.mark.parametrize("banner,version", [("TLC2 Version 9.9 of test\n", "9.9 of test"), ("", None)])
def test_version_is_observed(tmp_path, banner, version):
    with (
        patch.object(checker, "wait_for_load", return_value={}),
        patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, banner, "")),
    ):
        assert checker.run_tlc(probe_entry(), tmp_path)["tlc_version"] == version


@pytest.mark.parametrize("error", [NameError("pulse"), StopIteration(), KeyError("sources"), TypeError("signature")])
def test_probe_crash_is_incomplete(error, capsys):
    with patch.object(checker, "main", side_effect=error):
        assert checker.cli() == 2
    assert "Verification could not complete" in capsys.readouterr().err


def test_parser_exit_is_preserved():
    with patch.object(checker, "main", side_effect=SystemExit(2)), pytest.raises(SystemExit) as exc:
        checker.cli()
    assert exc.value.code == 2


@pytest.mark.parametrize("error", [FileNotFoundError("tlc"), subprocess.TimeoutExpired("tlc", 60)])
def test_launch_errors_propagate(tmp_path, error):
    with (
        patch.object(checker, "wait_for_load", return_value={}),
        patch.object(checker.subprocess, "run", side_effect=error),
        pytest.raises(type(error)),
    ):
        checker.run_tlc(probe_entry(), tmp_path)


def test_unexpected_exit_is_not_reproduced(tmp_path):
    with (
        patch.object(checker, "wait_for_load", return_value={}),
        patch.object(checker.subprocess, "run", return_value=subprocess.CompletedProcess([], 2, "", "tool error")),
    ):
        assert not checker.run_tlc(probe_entry(), tmp_path)["reproduced"]


@pytest.mark.parametrize("field", ["sync_sha", "base_sha", "historical_sources"])
def test_coverage_pin_drift_is_rejected(tmp_path, monkeypatch, field):
    import json

    coverage = json.loads((checker.HERE / "coverage.json").read_text())
    coverage[field] = "wrong"
    (tmp_path / "coverage.json").write_text(json.dumps(coverage))
    monkeypatch.setattr(checker, "HERE", tmp_path)
    monkeypatch.setattr(checker, "source", lambda *_: "")
    monkeypatch.setattr("sys.argv", ["check.py", "--prepare-only"])
    with pytest.raises(RuntimeError, match="pins differ"):
        checker.main()


def test_old_calendar_checkout_is_rejected(tmp_path, monkeypatch):
    old = tmp_path / checker.CALENDAR
    old.parent.mkdir(parents=True)
    old.write_text(checker.source(checker.MAIN))
    for path in (checker.GUARD, "src/brainlayer/supervisor_child.py"):
        target = tmp_path / path
        target.write_text(checker.source(checker.MAIN, path))
    monkeypatch.setattr(checker, "REPO", tmp_path)
    # source() still needs the real Git repository for its immutable object reads.
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(
        checker,
        "source",
        lambda ref, path=checker.CALENDAR: subprocess.check_output(
            ["git", "show", f"{ref}:{path}"], cwd=repo, text=True
        ),
    )
    monkeypatch.setattr("sys.argv", ["check.py", "--prepare-only"])
    with pytest.raises(RuntimeError, match="checkout drift"):
        checker.main()
