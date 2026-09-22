"""Upgrade restart must use only fake launchctl/lsof and a disposable keg."""

import json
import plistlib
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from brainlayer.cli import app
from brainlayer.jobs import restart_loaded_jobs


class FakeCommands:
    def __init__(self, old_keg: Path, current_keg: Path):
        self.old_keg = old_keg
        self.current_keg = current_keg
        self.commands: list[list[str]] = []
        self.pids = {"watch": 101, "drain": 102, "backup-daily": 103, "enrichment": 104, "brainbar": 105}
        self.kegs = {name: old_keg for name in self.pids}

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        if args[:2] == ["launchctl", "print"]:
            name = args[-1].split("com.brainlayer.")[-1]
            if name == "unloaded":
                return subprocess.CompletedProcess(args, 113, "", "not loaded")
            return subprocess.CompletedProcess(args, 0, f"pid = {self.pids[name]}\n", "")
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            name = args[-1].split("com.brainlayer.")[-1]
            self.kegs[name] = self.current_keg
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["lsof", "-p"]:
            pid = int(args[2])
            name = next(name for name, value in self.pids.items() if value == pid)
            return subprocess.CompletedProcess(args, 0, f"p{pid}\nn{self.kegs[name]}/libexec/venv/bin/python\n", "")
        raise AssertionError(args)


def _plist(directory: Path, name: str, **options: object) -> None:
    (directory / f"com.brainlayer.{name}.plist").write_bytes(
        plistlib.dumps({"Label": f"com.brainlayer.{name}", **options})
    )


def test_restart_loaded_daemons_and_stale_inflight_interval(tmp_path: Path) -> None:
    old_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.35"
    current_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.36"
    old_keg.mkdir(parents=True)
    current_keg.mkdir(parents=True)
    opt = tmp_path / "opt" / "brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current_keg)
    for name, options in {
        "watch": {"KeepAlive": True},
        "drain": {"RunAtLoad": True},
        "backup-daily": {"StartInterval": 86400},
        "enrichment": {"KeepAlive": True},
        "unloaded": {"KeepAlive": True},
    }.items():
        _plist(tmp_path, name, **options)
    fake = FakeCommands(old_keg, current_keg)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=fake, uid=501)

    assert result["ok"] is True
    assert set(result["restarted"]) == {
        "com.brainlayer.watch",
        "com.brainlayer.drain",
        "com.brainlayer.backup-daily",
    }
    assert "com.brainlayer.enrichment" in result["skipped"]
    assert "com.brainlayer.unloaded" in result["skipped"]
    assert not any("enrichment" in " ".join(args) and "kickstart" in args for args in fake.commands)


def test_stale_mapping_fails_with_named_job(tmp_path: Path) -> None:
    old_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.35"
    current_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.36"
    old_keg.mkdir(parents=True)
    current_keg.mkdir(parents=True)
    opt = tmp_path / "opt" / "brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current_keg)
    _plist(tmp_path, "watch", KeepAlive=True)
    fake = FakeCommands(old_keg, current_keg)
    fake.kegs["watch"] = old_keg

    def failed_restart(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            return subprocess.CompletedProcess(args, 1, "", "failed")
        return fake(args)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=failed_restart, uid=501)

    assert result["ok"] is False
    assert "com.brainlayer.watch" in result["stale"]
    assert "com.brainlayer.watch" in result["errors"]


def test_cli_emits_json_and_exits_nonzero_for_stale_job(tmp_path: Path, monkeypatch) -> None:
    from brainlayer import jobs

    monkeypatch.setattr(jobs, "installed_opt_path", lambda: tmp_path)
    monkeypatch.setattr(
        jobs,
        "restart_loaded_jobs",
        lambda *_args, **_kwargs: {"ok": False, "stale": {"com.brainlayer.watch": "old keg"}},
    )

    result = CliRunner().invoke(app, ["jobs", "restart", "--all", "--verify"])

    assert result.exit_code == 1
    assert json.loads(result.stdout)["stale"] == {"com.brainlayer.watch": "old keg"}


def test_cask_owned_brainbar_is_left_to_cask_postflight(tmp_path: Path) -> None:
    old_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.35"
    current_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.36"
    old_keg.mkdir(parents=True)
    current_keg.mkdir(parents=True)
    opt = tmp_path / "opt" / "brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current_keg)
    _plist(tmp_path, "brainbar", KeepAlive=True, AssociatedBundleIdentifiers=["com.brainlayer.brainbar"])
    fake = FakeCommands(old_keg, current_keg)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=fake, uid=501)

    assert result["ok"] is True
    assert result["skipped"]["com.brainlayer.brainbar"] == "cask-owned BrainBar job"
    assert not any(args[0] in {"lsof"} or "kickstart" in args for args in fake.commands)
