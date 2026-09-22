import plistlib
import subprocess
from pathlib import Path

from brainlayer.job_lifecycle_health import scan_job_lifecycle


class FakeCommands:
    def __init__(self, current: Path, old: Path):
        self.current = current
        self.runs = {"watch": 1, "drain": 1}
        self.exit = {"watch": 1, "drain": 0}
        self.pid = {"watch": None, "drain": 42}
        self.keg = {42: old}
        self.ps_command = f"/Library/Frameworks/Python.framework/python {current}/bin/brainlayer"
        self.unloaded: set[str] = set()
        self.commands: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        if args[:2] == ["launchctl", "print-disabled"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["launchctl", "print"]:
            name = args[-1].split("com.brainlayer.")[-1]
            if name in self.unloaded:
                return subprocess.CompletedProcess(args, 113, "", "not loaded")
            output = f"runs = {self.runs[name]}\nlast exit code = {self.exit[name]}\n"
            if self.pid[name] is not None:
                output += f"pid = {self.pid[name]}\n"
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:2] == ["lsof", "-p"]:
            keg = self.keg.get(int(args[2]))
            output = f"n{keg}/libexec/venv/bin/python\n" if keg else "n/usr/lib/libSystem.B.dylib\n"
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[:2] == ["ps", "-p"]:
            return subprocess.CompletedProcess(args, 0, self.ps_command, "")
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


def _setup(tmp_path: Path, *jobs: tuple[str, dict]) -> tuple[Path, Path, Path]:
    current = tmp_path / "Cellar/brainlayer/1.5.36"
    current.mkdir(parents=True)
    opt = tmp_path / "opt/brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current)
    for name, options in jobs:
        payload = {"Label": f"com.brainlayer.{name}", "ProgramArguments": [f"{opt}/bin/brainlayer", name], **options}
        (tmp_path / f"com.brainlayer.{name}.plist").write_bytes(plistlib.dumps(payload))
    return tmp_path / "Cellar/brainlayer/1.5.35", current, opt


def _tick(tmp_path: Path, opt: Path, fake: FakeCommands, state: dict, now: int, **kwargs):
    return scan_job_lifecycle(tmp_path, opt, state, now_epoch=now, command_runner=fake, uid=501, **kwargs)


def test_daemon_backoff_badge_and_recovery(tmp_path: Path) -> None:
    old, current, opt = _setup(tmp_path, ("watch", {"KeepAlive": True}), ("enrichment", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    state: dict = {}
    for now in (0, 300, 301, 900, 2100):
        fake.runs["watch"] += 1
        tick = _tick(tmp_path, opt, fake, state, now)
        state = tick.state
    fake.runs["watch"] += 1
    tick = _tick(tmp_path, opt, fake, state, 2101)
    kicks = [args[-1] for args in fake.commands if args[:3] == ["launchctl", "kickstart", "-k"]]
    assert kicks == ["gui/501/com.brainlayer.watch"] * 3
    assert "3 heal attempts" in tick.escalations[0] and "last exit code 1" in tick.escalations[0]


def test_stale_keg_first_cycle_ps_fallback_pause_and_self_skip(tmp_path: Path) -> None:
    old, current, opt = _setup(tmp_path, ("drain", {"KeepAlive": True}), ("health-check", {"StartInterval": 300}))
    fake = FakeCommands(current, old)
    first = _tick(tmp_path, opt, fake, {}, 0)
    assert first.actions == ["kickstart:com.brainlayer.drain"]
    assert _tick(tmp_path, opt, fake, first.state, 900, paused_labels={"com.brainlayer.drain"}).actions == []
    fake.keg.clear()
    current_run = _tick(tmp_path, opt, fake, {}, 901)
    assert current_run.actions == [] and current_run.state["com.brainlayer.drain"]["consecutive"] == 0


def test_interval_sticky_exit_and_live_pid_crashloop(tmp_path: Path) -> None:
    old, current, opt = _setup(
        tmp_path, ("watch", {"StartCalendarInterval": {"Hour": 3}}), ("drain", {"KeepAlive": True})
    )
    fake = FakeCommands(current, old)
    fake.keg[42] = current
    fake.exit["drain"] = 1
    state: dict = {}
    for now in range(5):
        tick = _tick(tmp_path, opt, fake, state, now * 300)
        state = tick.state
        assert tick.actions == [] and tick.escalations == []
    for now in range(3):
        fake.runs["watch"] += 1
        fake.runs["drain"] += 1
        tick = _tick(tmp_path, opt, fake, state, 2000 + now * 300)
        state = tick.state
    assert "failed runs" in tick.escalations[0]
    assert "kickstart:com.brainlayer.drain" in tick.actions
    assert not any(
        args[-1].endswith("/com.brainlayer.watch")
        for args in fake.commands
        if args[:3] == ["launchctl", "kickstart", "-k"]
    )


def test_corrupt_counter_and_prune_unloaded_or_deleted(tmp_path: Path) -> None:
    old, current, opt = _setup(tmp_path, ("watch", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    bad = {"com.brainlayer.watch": {"consecutive": "x", "failed_heals": True}}
    corrupt = _tick(tmp_path, opt, fake, bad, 300)
    assert corrupt.scan_error and corrupt.escalations and corrupt.actions == []
    prior = {"com.brainlayer.watch": {"runs": 1}}
    fake.unloaded.add("watch")
    unloaded = _tick(tmp_path, opt, fake, prior, 600)
    assert unloaded.state == {} and unloaded.actions == ["pruned:com.brainlayer.watch"]
    (tmp_path / "com.brainlayer.watch.plist").unlink()
    deleted = _tick(tmp_path, opt, fake, prior, 900)
    assert deleted.state == {} and deleted.actions == ["pruned:com.brainlayer.watch"]


def test_unmapped_inflight_interval_does_not_abort_other_jobs(tmp_path: Path) -> None:
    old, current, opt = _setup(tmp_path, ("watch", {"StartInterval": 60}), ("drain", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    fake.pid["watch"] = 44
    fake.keg[42] = current
    fake.ps_command = "/Library/Frameworks/Python.framework/python /Users/test/.local/lib/brainlayer/watch.py"
    tick = _tick(tmp_path, opt, fake, {}, 0)
    assert tick.scan_error is None
    assert tick.escalations == []
    assert "com.brainlayer.drain" in tick.state


def test_interval_stale_keg_does_not_count_samples_as_failed_runs(tmp_path: Path) -> None:
    old, current, opt = _setup(tmp_path, ("watch", {"StartInterval": 60}))
    fake = FakeCommands(current, old)
    state = _tick(tmp_path, opt, fake, {}, 0).state
    fake.pid["watch"] = 44
    fake.keg[44] = old
    for now in (300, 600, 900):
        tick = _tick(tmp_path, opt, fake, state, now)
        state = tick.state
        assert tick.actions == []
        assert tick.escalations == []
    fake.runs["watch"] += 1
    for now in (1200, 1500, 1800):
        tick = _tick(tmp_path, opt, fake, state, now)
        state = tick.state
        assert tick.escalations == []
    fake.pid["watch"] = None
    completed = _tick(tmp_path, opt, fake, state, 2100)
    assert completed.state["com.brainlayer.watch"]["consecutive"] == 1


def test_healthy_daemon_sample_resets_prior_crash_streak(tmp_path: Path) -> None:
    old, current, opt = _setup(tmp_path, ("drain", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    fake.keg[42] = current
    fake.exit["drain"] = 1
    baseline = _tick(tmp_path, opt, fake, {}, 0)
    fake.runs["drain"] += 1
    failed = _tick(tmp_path, opt, fake, baseline.state, 300)
    assert failed.state["com.brainlayer.drain"]["consecutive"] == 1
    healthy = _tick(tmp_path, opt, fake, failed.state, 600)
    assert healthy.state["com.brainlayer.drain"]["consecutive"] == 0
