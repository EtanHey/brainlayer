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
