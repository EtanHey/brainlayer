"""Every scheduled plist must still address a parseable command."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import plistlib
import re
import subprocess
import sys
from pathlib import Path

import typer

from brainlayer.cli import app

ROOT = Path(__file__).resolve().parents[1]
PLISTS = ROOT / "scripts" / "launchd"
SCRIPT_SOURCES = {
    "backup-daily.sh": PLISTS / "backup-daily.sh",
    "jsonl-backup.sh": PLISTS / "jsonl-backup.sh",
    "tier0-watchdog.sh": ROOT / "scripts" / "tier0-watchdog.sh",
    "hotlane_brainbar_daemon.py": ROOT / "scripts" / "hotlane_brainbar_daemon.py",
    "throughput-watchdog.py": PLISTS / "throughput-watchdog.py",
}
RENDER = {
    "__BRAINLAYER_ENV_RUN__": "/tmp/brainlayer-fixture/env-run",
    "__BRAINLAYER_BIN__": "/tmp/brainlayer-fixture/bin/brainlayer",
    "__PYTHON_BIN__": sys.executable,
    "__BRAINLAYER_PYTHON__": sys.executable,
    "__HOME__": "/tmp/brainlayer-fixture",
    "__BRAINLAYER_DIR__": str(ROOT),
    "__BRAINLAYER_ENV_FILE__": "/tmp/brainlayer-fixture/brainlayer.env",
    "__BRAINLAYER_LAUNCHD_DIR__": str(PLISTS),
    "__HOTLANE_BRAINBAR_DAEMON__": str(SCRIPT_SOURCES["hotlane_brainbar_daemon.py"]),
    "__THROUGHPUT_WATCHDOG_SCRIPT__": str(SCRIPT_SOURCES["throughput-watchdog.py"]),
    "__TIER0_WATCHDOG_SCRIPT__": str(SCRIPT_SOURCES["tier0-watchdog.sh"]),
}


class _Parsed(Exception):
    pass


def _check_argparse(module, args, monkeypatch):
    original = argparse.ArgumentParser.parse_args

    def parse_and_stop(parser, argv=None, namespace=None):
        original(parser, argv, namespace)
        raise _Parsed

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", parse_and_stop)
        patch.setattr(sys, "argv", ["scheduled-job", *args])
        try:
            module.main()
        except _Parsed:
            return
    raise AssertionError(f"{module.__name__}.main did not parse arguments")


def _import_script(path):
    spec = importlib.util.spec_from_file_location(f"scheduled_{path.stem.replace('-', '_')}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _check_cli(args):
    command = typer.main.get_command(app)
    name, *remainder = args
    child = command.get_command(None, name)
    assert child is not None, f"unknown brainlayer command: {name}"
    with child.make_context(name, remainder):
        pass


def test_all_scheduled_launchd_program_arguments_parse(monkeypatch):
    paths = sorted(PLISTS.glob("com.brainlayer.*.plist"))
    assert paths
    checked = []
    for path in paths:
        rendered = path.read_text(encoding="utf-8")
        for token, value in RENDER.items():
            rendered = rendered.replace(token, value)
        assert not re.search(r"__[A-Z0-9_]+__", rendered), path.name
        plist = plistlib.loads(rendered.encode())
        assert "StartInterval" in plist or "StartCalendarInterval" in plist or "RunAtLoad" in plist
        args = list(plist["ProgramArguments"])
        if args[0] == RENDER["__BRAINLAYER_ENV_RUN__"]:
            args.pop(0)
        if args[0] == "/usr/bin/env":
            args.pop(0)
            while args and "=" in args[0]:
                args.pop(0)
        program, *rest = args
        if program == RENDER["__BRAINLAYER_BIN__"]:
            _check_cli(rest)
        elif program == sys.executable:
            if rest[0] == "-m":
                module_name, *module_args = rest[1:]
                if module_name == "brainlayer":
                    _check_cli(module_args)
                else:
                    _check_argparse(importlib.import_module(module_name), module_args, monkeypatch)
            else:
                script, *script_args = rest
                _check_argparse(_import_script(SCRIPT_SOURCES[Path(script).name]), script_args, monkeypatch)
        else:
            if program in {"/bin/sh", "/bin/bash"}:
                program, *rest = rest
            script = SCRIPT_SOURCES[Path(program).name]
            assert script.is_file()
            syntax = subprocess.run(["/bin/sh", "-n", str(script)], capture_output=True, text=True)
            assert syntax.returncode == 0, f"{path.name}: {syntax.stderr}"
            assert not rest, f"{path.name}: shell arguments need a parser check"
        checked.append(path.name)
    assert len(checked) == len(paths)
