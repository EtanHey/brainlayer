import plistlib
import subprocess
from pathlib import Path

from brainlayer.job_lifecycle_health import scan_job_lifecycle


class FakeCommands:
    def __init__(self, old_keg: Path, current_keg: Path):
        self.commands: list[list[str]] = []
        self.watch_output = "state = waiting\nlast exit code = 1\n"
        self.drain_keg = old_keg
        self.drain_command = f"{current_keg}/libexec/venv/bin/python"

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        if args[:2] == ["launchctl", "print-disabled"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[:2] == ["launchctl", "print"]:
            name = args[-1].split("com.brainlayer.")[-1]
            if name == "watch":
                return subprocess.CompletedProcess(args, 0, self.watch_output, "")
            if name == "drain":
                return subprocess.CompletedProcess(args, 0, "pid = 42\nstate = running\n", "")
            raise AssertionError(args)
        if args[:2] == ["lsof", "-p"]:
            mapped = f"n{self.drain_keg}/libexec/venv/bin/python\n" if self.drain_keg else ""
            return subprocess.CompletedProcess(args, 0, f"p42\n{mapped}", "")
        if args[:2] == ["ps", "-p"]:
            return subprocess.CompletedProcess(args, 0, self.drain_command, "")
        if args[:3] == ["launchctl", "kickstart", "-k"]:
            return subprocess.CompletedProcess(args, 0, "", "")
        raise AssertionError(args)


def _plist(directory: Path, name: str, **options: object) -> None:
    options.setdefault("ProgramArguments", [f"{directory}/opt/brainlayer/bin/brainlayer", name])
    (directory / f"com.brainlayer.{name}.plist").write_bytes(
        plistlib.dumps(
            {"Label": f"com.brainlayer.{name}", "AssociatedBundleIdentifiers": ["com.brainlayer.brainbar"], **options}
        )
    )


def test_crashloop_heals_with_backoff_then_badges_until_measured_healthy(tmp_path: Path) -> None:
    old_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.35"
    current_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.36"
    current_keg.mkdir(parents=True)
    opt = tmp_path / "opt" / "brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current_keg)
    _plist(tmp_path, "watch", KeepAlive=True)
    _plist(tmp_path, "enrichment", KeepAlive=True)
    fake = FakeCommands(old_keg, current_keg)
    state: dict = {}
    escalations = []
    for now in (0, 300, 301, 900, 2100):
        tick = scan_job_lifecycle(tmp_path, opt, state, now_epoch=now, command_runner=fake, uid=501)
        state = tick.state
        escalations = tick.escalations
    assert escalations == []
    tick = scan_job_lifecycle(tmp_path, opt, state, now_epoch=2101, command_runner=fake, uid=501)
    state, escalations = tick.state, tick.escalations
    kickstarts = [args for args in fake.commands if args[:3] == ["launchctl", "kickstart", "-k"]]
    assert len(kickstarts) == 3
    assert all(args[-1].endswith("/com.brainlayer.watch") for args in kickstarts)
    assert len(escalations) == 1 and "com.brainlayer.watch" in escalations[0]
    assert "last exit code 1" in escalations[0]
    assert "3 heal attempts" in escalations[0]
    fake.watch_output = "pid = 43\nstate = running\n"
    fake.drain_keg = current_keg
    recovered = scan_job_lifecycle(tmp_path, opt, state, now_epoch=2400, command_runner=fake, uid=501)
    assert recovered.state == {} and recovered.escalations == []


def test_stale_keg_process_heals_and_pause_skips_drain(tmp_path: Path) -> None:
    old_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.35"
    current_keg = tmp_path / "Cellar" / "brainlayer" / "1.5.36"
    current_keg.mkdir(parents=True)
    opt = tmp_path / "opt" / "brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(current_keg)
    _plist(tmp_path, "drain", KeepAlive=True)
    fake = FakeCommands(old_keg, current_keg)
    first = scan_job_lifecycle(tmp_path, opt, {}, now_epoch=0, command_runner=fake, uid=501)
    assert first.actions == ["kickstart:com.brainlayer.drain"]
    second = scan_job_lifecycle(tmp_path, opt, first.state, now_epoch=299, command_runner=fake, uid=501)
    assert second.actions == []
    assert "stale keg" in second.state["com.brainlayer.drain"]["reason"]
    paused = scan_job_lifecycle(
        tmp_path,
        opt,
        second.state,
        now_epoch=900,
        command_runner=fake,
        uid=501,
        paused_labels={"com.brainlayer.drain"},
    )
    assert paused.actions == []
    fake.drain_keg = None
    tick = scan_job_lifecycle(tmp_path, opt, {}, now_epoch=901, command_runner=fake, uid=501)
    assert tick.state == {} and tick.actions == []
