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
        self.pids = {
            "watch": 101,
            "drain": 102,
            "backup-daily": 103,
            "enrichment": 104,
            "brainbar": 105,
            "gemini-loopback": 106,
            "health-check": 107,
        }
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


def _kegs(tmp_path: Path) -> tuple[Path, Path, Path]:
    old_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.35"
    current_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.36"
    old_keg.mkdir(parents=True)
    current_keg.mkdir(parents=True)
    opt = tmp_path / "opt" / "brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current_keg)
    return old_keg, current_keg, opt


def test_restart_loaded_daemons_and_stale_inflight_interval(tmp_path: Path) -> None:
    old_keg, current_keg, opt = _kegs(tmp_path)
    keg_cli = f"{opt}/bin/brainlayer"
    for name, options in {
        "watch": {"KeepAlive": True, "ProgramArguments": ["/tmp/BrainLayer Watcher", keg_cli, "watch"]},
        "drain": {"RunAtLoad": True, "ProgramArguments": ["/tmp/BrainLayer Drain", keg_cli, "drain"]},
        "health-check": {
            "StartInterval": 300,
            "ProgramArguments": ["/tmp/BrainLayer Health Check", keg_cli, "health-check"],
        },
        "backup-daily": {"StartCalendarInterval": {"Hour": 3}, "ProgramArguments": ["/bin/sh", "/tmp/backup-daily.sh"]},
        "gemini-loopback": {"KeepAlive": True, "ProgramArguments": ["/opt/homebrew/bin/socat", "TCP-LISTEN:48123"]},
        "enrichment": {"KeepAlive": True, "ProgramArguments": [keg_cli, "enrich"]},
        "unloaded": {"KeepAlive": True, "ProgramArguments": [keg_cli, "watch"]},
    }.items():
        _plist(tmp_path, name, AssociatedBundleIdentifiers=["com.brainlayer.brainbar"], **options)
    fake = FakeCommands(old_keg, current_keg)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=fake, uid=501)

    assert result["ok"] is True
    assert set(result["restarted"]) == {
        "com.brainlayer.watch",
        "com.brainlayer.drain",
        "com.brainlayer.health-check",
    }
    assert "com.brainlayer.enrichment" in result["skipped"]
    assert "com.brainlayer.unloaded" in result["skipped"]
    assert "com.brainlayer.backup-daily" in result["skipped"]
    assert "com.brainlayer.gemini-loopback" in result["skipped"]
    assert not any("enrichment" in " ".join(args) and "kickstart" in args for args in fake.commands)


def test_stale_mapping_fails_with_named_job(tmp_path: Path) -> None:
    old_keg, current_keg, opt = _kegs(tmp_path)
    _plist(tmp_path, "watch", KeepAlive=True, ProgramArguments=[f"{opt}/bin/brainlayer", "watch"])
    fake = FakeCommands(old_keg, current_keg)
    fake.kegs["watch"] = old_keg

    def failed_restart(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            return subprocess.CompletedProcess(args, 1, "", "failed")
        return fake(args)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=failed_restart, uid=501, sleep_fn=lambda _: None)

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
    old_keg, current_keg, opt = _kegs(tmp_path)
    _plist(
        tmp_path,
        "brainbar",
        KeepAlive=True,
        AssociatedBundleIdentifiers=["com.brainlayer.brainbar"],
        ProgramArguments=["/Applications/BrainBar.app/Contents/MacOS/BrainBar"],
    )
    fake = FakeCommands(old_keg, current_keg)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=fake, uid=501)

    assert result["ok"] is False
    assert result["skipped"]["com.brainlayer.brainbar"] == "cask-owned BrainBar job"
    assert "selection" in result["errors"]
    assert not any(args[0] in {"lsof"} or "kickstart" in args for args in fake.commands)


def test_discovery_and_launchctl_errors_fail_closed(tmp_path: Path) -> None:
    _, current_keg, opt = _kegs(tmp_path)
    fake = FakeCommands(current_keg, current_keg)
    assert restart_loaded_jobs(tmp_path, opt, command_runner=fake, uid=501)["ok"] is False
    assert restart_loaded_jobs(tmp_path / "missing", opt, command_runner=fake, uid=501)["ok"] is False
    _plist(tmp_path, "watch", KeepAlive=True, ProgramArguments=[f"{opt}/bin/brainlayer", "watch"])

    def denied_print(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:2] == ["launchctl", "print"]:
            return subprocess.CompletedProcess(args, 1, "", "operation not permitted")
        return fake(args)

    result = restart_loaded_jobs(tmp_path, opt, command_runner=denied_print, uid=501)
    assert result["ok"] is False
    assert "com.brainlayer.watch" in result["errors"]


def test_cold_start_retries_before_declaring_missing_pid(tmp_path: Path) -> None:
    old_keg, current_keg, opt = _kegs(tmp_path)
    _plist(tmp_path, "watch", KeepAlive=True, ProgramArguments=[f"{opt}/bin/brainlayer", "watch"])

    class ColdStart(FakeCommands):
        cold_prints = 0

        def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
            if args[:2] == ["launchctl", "print"] and self.kegs["watch"] == current_keg:
                self.cold_prints += 1
                if self.cold_prints <= 2:
                    return subprocess.CompletedProcess(args, 0, "state = waiting\n", "")
            return super().__call__(args)

    fake = ColdStart(old_keg, current_keg)
    result = restart_loaded_jobs(tmp_path, opt, command_runner=fake, uid=501, sleep_fn=lambda _: None)
    assert result["ok"] is True
    assert fake.cold_prints == 3
