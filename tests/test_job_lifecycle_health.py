import hashlib
import json
import plistlib
import subprocess
from pathlib import Path

import pytest

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
    return scan_job_lifecycle(
        tmp_path, opt, state, now_epoch=now, command_runner=fake, uid=kwargs.pop("uid", 501), **kwargs
    )


def test_daemon_backoff_badge_and_recovery(tmp_path: Path) -> None:
    old, current, opt = _setup(
        tmp_path, ("watch", {"KeepAlive": True}), ("enrichment", {"KeepAlive": True}), ("enrich", {"KeepAlive": True})
    )
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
    assert not any("com.brainlayer.enrich" in " ".join(args) for args in fake.commands)
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


@pytest.fixture
def supervisor_sample(tmp_path):
    home = tmp_path / "home"
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    prefix = tmp_path / "opt/homebrew"
    framework = prefix / (
        "Cellar/python@3.13/3.13.16/Frameworks/Python.framework/Versions/3.13/"
        "Resources/Python.app/Contents/MacOS/Python"
    )
    framework.parent.mkdir(parents=True)
    framework.write_text("fixture interpreter, never executed")
    launcher = framework.parents[4] / "bin/python3.13"
    launcher.parent.mkdir()
    launcher.write_text("fixture framework launcher, never executed")
    kegs = [prefix / f"Cellar/brainlayer/{version}" for version in ("1.5.56", "1.5.57")]
    for keg in kegs:
        venv = keg / "libexec/venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin/python").symlink_to(launcher)
        (venv / "pyvenv.cfg").write_text(f"home = {framework.parent}\n")
    opt = prefix / "opt/brainlayer"
    opt.parent.mkdir(parents=True)
    opt.symlink_to(kegs[1])
    control = home / ".local/share/brainlayer/supervisor"
    control.mkdir(parents=True)
    manifest = control / "jobs.json"
    manifest.write_text("{}\n")
    args = [
        str(home / ".local/lib/brainlayer/BrainLayer Daemon"),
        str(opt / "libexec/venv/bin/python"),
        "-m",
        "brainlayer",
        "daemon",
        "--manifest",
        str(manifest),
    ]
    definition = {"Label": "com.brainlayer.daemon", "ProgramArguments": args, "KeepAlive": True}
    path = agents / "com.brainlayer.daemon.plist"
    raw = plistlib.dumps(definition)
    path.write_bytes(raw)
    pinned = definition | {"ProgramArguments": [args[0], str(kegs[1] / "libexec/venv/bin/python"), *args[2:]]}
    receipt = {
        "phase": "migrated",
        "keg": "../Cellar/brainlayer/1.5.57",
        "daemon_pinned": plistlib.dumps(pinned).hex(),
        "published_sha256": {path.name: hashlib.sha256(raw).hexdigest()},
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    journal = control / "migration.json"
    journal.write_text(json.dumps(receipt))
    fake = FakeCommands(kegs[1], kegs[0])
    fake.runs["daemon"], fake.exit["daemon"], fake.pid["daemon"] = 1, 0, 6538
    fake.keg.clear()  # Real-shaped lsof output: no Cellar/brainlayer mappings.
    fake.ps_command = f"{framework} -m brainlayer daemon --manifest {manifest}"
    return agents, opt, fake, journal, receipt, kegs


def test_framework_supervisor_with_no_mapped_keg_is_healthy(supervisor_sample):
    agents, opt, fake, *_ = supervisor_sample
    tick = _tick(agents, opt, fake, {}, 0, heal=False, uid=424242)
    assert tick.scan_error is None
    assert tick.state["com.brainlayer.daemon"]["consecutive"] == 0
    assert tick.actions == []
    assert ["lsof", "-p", "6538", "-Fn"] in fake.commands
    assert ["ps", "-p", "6538", "-o", "command="] in fake.commands


def test_framework_supervisor_mapped_old_keg_is_stale(supervisor_sample):
    agents, opt, fake, journal, receipt, kegs = supervisor_sample
    receipt["keg"] = "../Cellar/brainlayer/1.5.56"
    pinned = plistlib.loads(bytes.fromhex(receipt["daemon_pinned"]))
    pinned["ProgramArguments"][1] = str(kegs[0] / "libexec/venv/bin/python")
    receipt["daemon_pinned"] = plistlib.dumps(pinned).hex()
    journal.write_text(json.dumps(receipt))
    fake.keg[6538] = kegs[0]  # Actual mapped old code remains affirmative evidence.
    tick = _tick(agents, opt, fake, {}, 0, heal=False, uid=424242)
    assert tick.scan_error is None
    assert tick.state["com.brainlayer.daemon"]["reason"] == f"stale keg for pid 6538: {kegs[0]}"
    assert tick.actions == []


@pytest.mark.parametrize("heal", [False, True])
def test_moved_opt_cannot_prove_framework_supervisor_is_stale(supervisor_sample, heal):
    agents, opt, fake, journal, receipt, kegs = supervisor_sample
    receipt["keg"] = "../Cellar/brainlayer/1.5.56"
    pinned = plistlib.loads(bytes.fromhex(receipt["daemon_pinned"]))
    pinned["ProgramArguments"][1] = str(kegs[0] / "libexec/venv/bin/python")
    receipt["daemon_pinned"] = plistlib.dumps(pinned).hex()
    journal.write_text(json.dumps(receipt))
    # Both venvs resolve to the same framework executable. This exact observation
    # describes BOTH the old process and a replacement already running the new keg.
    prior = {"com.brainlayer.daemon": {"runs": 1, "attempts": 0}}
    for now in (0, 1, 2):
        tick = _tick(agents, opt, fake, prior, now, uid=424242, heal=heal)
        assert tick.actions == []
        assert tick.scan_error == "com.brainlayer.daemon: keg mapping unavailable for pid 6538"
        assert tick.state == prior
    assert not any(args[:2] == ["launchctl", "kickstart"] for args in fake.commands)


@pytest.mark.parametrize(
    "keg", ["../outside/1.5.57", "../Cellar/brainlayer/../../outside/1.5.57", "../Cellar/brainlayer/foreign"]
)
def test_relative_supervisor_keg_cannot_escape_cellar(supervisor_sample, keg):
    agents, opt, fake, journal, receipt, _ = supervisor_sample
    outside = opt.parents[1] / "outside/1.5.57"
    outside.mkdir(parents=True)
    (opt.resolve().parent / "foreign").symlink_to(outside)
    # Keep the other identity evidence valid, so containment is the refusal.
    pinned = plistlib.loads(bytes.fromhex(receipt["daemon_pinned"]))
    python = outside / "libexec/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(Path(pinned["ProgramArguments"][1]).resolve())
    (python.parent.parent / "pyvenv.cfg").write_text("fixture venv\n")
    pinned["ProgramArguments"][1] = str(python)
    receipt["daemon_pinned"] = plistlib.dumps(pinned).hex()
    receipt["keg"] = keg
    journal.write_text(json.dumps(receipt))
    tick = _tick(agents, opt, fake, {}, 0, heal=False, uid=424242)
    assert tick.scan_error == "com.brainlayer.daemon: keg mapping unavailable for pid 6538"
    assert tick.actions == []


def test_non_keg_daemon_is_still_skipped(supervisor_sample):
    agents, opt, fake, *_ = supervisor_sample
    path = agents / "com.brainlayer.daemon.plist"
    definition = plistlib.loads(path.read_bytes())
    definition["ProgramArguments"][1] = "/Users/dev/brainlayer/.venv/bin/python"
    path.write_bytes(plistlib.dumps(definition))
    tick = _tick(agents, opt, fake, {}, 0, uid=424242)
    assert tick.scan_error is None and tick.state == {} and tick.actions == []
    assert fake.commands == [["launchctl", "print-disabled", "gui/424242"]]


@pytest.mark.parametrize("broken", ["journal", "phase", "manifest", "plist", "process", "prefix", "argv"])
def test_framework_supervisor_requires_all_identity_evidence(supervisor_sample, broken):
    agents, opt, fake, journal, receipt, kegs = supervisor_sample
    if broken == "journal":
        journal.unlink()
    elif broken == "phase":
        receipt["phase"] = "migrating"
        journal.write_text(json.dumps(receipt))
    elif broken == "manifest":
        (journal.parent / "jobs.json").write_text('{"foreign": true}')
    elif broken == "plist":
        receipt["published_sha256"]["com.brainlayer.daemon.plist"] = "0" * 64
        journal.write_text(json.dumps(receipt))
    elif broken == "process":
        fake.ps_command = "/usr/bin/python3 -m brainlayer daemon --manifest " + str(journal.parent / "jobs.json")
    elif broken == "argv":
        fake.ps_command += " --unreviewed"
    else:
        pinned = plistlib.loads(bytes.fromhex(receipt["daemon_pinned"]))
        pinned["ProgramArguments"][1] = str(kegs[0] / "libexec/venv/bin/python")
        receipt["daemon_pinned"] = plistlib.dumps(pinned).hex()
        journal.write_text(json.dumps(receipt))
    tick = _tick(agents, opt, fake, {}, 0, uid=424242)
    assert tick.scan_error == "com.brainlayer.daemon: keg mapping unavailable for pid 6538"
    assert tick.actions == []


def test_unmapped_resident_keeps_critical_and_allows_later_healing(supervisor_sample):
    agents, opt, fake, *_ = supervisor_sample
    daemon = agents / "com.brainlayer.daemon.plist"
    document = plistlib.loads(daemon.read_bytes())
    document["Label"] = "com.brainlayer.drain"
    (agents / "com.brainlayer.drain.plist").write_bytes(plistlib.dumps(document))
    daemon.unlink()
    watch = {"Label": "com.brainlayer.watch", "ProgramArguments": [str(opt / "bin/brainlayer"), "watch"]}
    (agents / "com.brainlayer.watch.plist").write_bytes(plistlib.dumps(watch))
    fake.runs["watch"] = 2
    prior = {"com.brainlayer.watch": {"runs": 1, "consecutive": 1}}
    tick = _tick(agents, opt, fake, prior, 0, uid=424242)
    assert tick.scan_error == "com.brainlayer.drain: keg mapping unavailable for pid 42"
    assert tick.actions == ["kickstart:com.brainlayer.watch"]
    assert tick.state["com.brainlayer.watch"]["attempts"] == 1
    assert any(
        args[:2] == ["launchctl", "print"] and args[-1].endswith("com.brainlayer.watch") for args in fake.commands
    )


@pytest.mark.parametrize("unknown", ["drain", "zzz"])
@pytest.mark.parametrize("heal", [False, True])
def test_unmapped_resident_preserves_episodes_and_known_job_progress(tmp_path: Path, unknown: str, heal: bool):
    old, current, opt = _setup(tmp_path, (unknown, {"KeepAlive": True}), ("watch", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    fake.runs.update({unknown: 7, "watch": 2})
    fake.exit[unknown] = 0
    fake.pid[unknown] = 42
    fake.keg.clear()
    fake.ps_command = "/fixture/framework/python -m brainlayer daemon"
    label = f"com.brainlayer.{unknown}"
    prior = {
        label: {"runs": 6, "attempts": 2, "consecutive": 2, "next_retry_epoch": 900},
        "com.brainlayer.watch": {"runs": 1, "consecutive": 1},
    }
    before = json.loads(json.dumps(prior))
    tick = _tick(tmp_path, opt, fake, prior, 300, heal=heal, uid=424242)
    assert tick.scan_error == f"{label}: keg mapping unavailable for pid 42"
    assert tick.state[label] == before[label] and prior == before
    assert tick.state["com.brainlayer.watch"]["runs"] == 2
    assert tick.state["com.brainlayer.watch"]["attempts"] == int(heal)
    assert tick.actions == (["kickstart:com.brainlayer.watch"] if heal else [])
    kicks = [args[-1] for args in fake.commands if args[:3] == ["launchctl", "kickstart", "-k"]]
    assert kicks == (["gui/424242/com.brainlayer.watch"] if heal else [])


def test_multiple_unmapped_residents_keep_each_critical_and_escalation(tmp_path: Path):
    old, current, opt = _setup(tmp_path, ("drain", {"KeepAlive": True}), ("watch", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    fake.pid["watch"] = 43
    fake.keg.clear()
    fake.ps_command = "/fixture/framework/python -m brainlayer daemon"
    prior = {"com.brainlayer.watch": {"attempts": 3, "failed_heals": True, "reason": "existing failure"}}
    tick = _tick(tmp_path, opt, fake, prior, 300)
    assert tick.scan_error == (
        "com.brainlayer.drain: keg mapping unavailable for pid 42; "
        "com.brainlayer.watch: keg mapping unavailable for pid 43"
    )
    assert tick.state == prior and tick.actions == []
    assert len(tick.escalations) == 1 and "existing failure" in tick.escalations[0]


@pytest.mark.parametrize("refusal", ["current_keg", "disabled", "persisted", "plist"])
def test_global_scan_refusals_still_prevent_healing(tmp_path: Path, refusal: str):
    old, current, opt = _setup(tmp_path, ("drain", {"KeepAlive": True}))
    fake = FakeCommands(current, old)
    prior = {}
    runner = fake
    if refusal == "current_keg":
        opt.unlink()
    elif refusal == "disabled":

        def runner(args):
            return subprocess.CompletedProcess(args, 1, "", "disabled query failed")
    elif refusal == "persisted":
        prior = {"com.brainlayer.drain": {"attempts": "unknown"}}
    else:
        (tmp_path / "com.brainlayer.zzz.plist").write_bytes(b"malformed plist")
    tick = scan_job_lifecycle(tmp_path, opt, prior, now_epoch=300, command_runner=runner, uid=424242)
    assert tick.scan_error and tick.state == prior and tick.actions == []
    assert not any(
        args[:2] == ["launchctl", "print"] or args[:2] == ["launchctl", "kickstart"] for args in fake.commands
    )


@pytest.fixture
def nightly_maintenance(tmp_path: Path):
    old, current, opt = _setup(tmp_path, ("maintenance", {"StartCalendarInterval": {"Hour": 3, "Minute": 15}}))
    fake = FakeCommands(current, old)
    fake.runs["maintenance"] = 1
    fake.exit["maintenance"] = 0
    fake.pid["maintenance"] = None
    baseline = _tick(tmp_path, opt, fake, {}, 0)
    assert baseline.scan_error is None and baseline.escalations == []
    return opt, fake, baseline


def _failed_maintenance_nights(tmp_path: Path, nightly_maintenance, nights: int):
    opt, fake, tick = nightly_maintenance
    fake.exit["maintenance"] = 1
    for night in range(1, nights + 1):
        fake.runs["maintenance"] += 1
        tick = _tick(tmp_path, opt, fake, tick.state, night * 86400)
        assert tick.scan_error is None and tick.actions == []
    return tick


@pytest.mark.parametrize("nights", [3, 4])
def test_nightly_maintenance_failure_message_uses_nights(tmp_path: Path, nightly_maintenance, nights: int):
    tick = _failed_maintenance_nights(tmp_path, nightly_maintenance, nights)
    assert tick.escalations == [f"com.brainlayer.maintenance: failed {nights} nights in a row; last exit 1"]


def test_nightly_maintenance_failure_message_has_one_exit_detail(tmp_path: Path, nightly_maintenance):
    tick = _failed_maintenance_nights(tmp_path, nightly_maintenance, 4)
    assert len(tick.escalations) == 1
    assert tick.escalations[0].count("last exit") == 1


@pytest.mark.parametrize("nights", [4, 7])
def test_nightly_maintenance_failure_message_reports_actual_count(tmp_path: Path, nightly_maintenance, nights: int):
    tick = _failed_maintenance_nights(tmp_path, nightly_maintenance, nights)
    assert tick.state["com.brainlayer.maintenance"]["consecutive"] == nights
    assert f"failed {nights} nights in a row" in tick.escalations[0]


@pytest.mark.parametrize("completed_failures", [2, 4])
def test_nightly_maintenance_in_progress_run_is_not_a_failure(
    tmp_path: Path, nightly_maintenance, completed_failures: int
):
    opt, fake, _ = nightly_maintenance
    prior = _failed_maintenance_nights(tmp_path, nightly_maintenance, completed_failures)
    fake.runs["maintenance"] += 1
    fake.pid["maintenance"] = 44
    fake.keg[44] = fake.current
    now = (completed_failures + 1) * 86400
    state = prior.state
    for offset in (0, 300):
        running = _tick(tmp_path, opt, fake, state, now + offset)
        assert running.scan_error is None and running.actions == []
        assert running.state["com.brainlayer.maintenance"]["consecutive"] == completed_failures
        assert running.escalations == prior.escalations
        state = running.state
    fake.pid["maintenance"] = None
    completed = _tick(tmp_path, opt, fake, state, now + 600)
    assert completed.state["com.brainlayer.maintenance"]["consecutive"] == completed_failures + 1
    assert len(completed.escalations) == 1
    repeated = _tick(tmp_path, opt, fake, completed.state, now + 900)
    assert repeated.state["com.brainlayer.maintenance"]["consecutive"] == completed_failures + 1
    assert repeated.escalations == completed.escalations


def test_nightly_maintenance_alerts_after_three_completed_failures(tmp_path: Path, nightly_maintenance):
    opt, fake, tick = nightly_maintenance
    fake.exit["maintenance"] = 1
    for failures in range(1, 5):
        fake.runs["maintenance"] += 1
        tick = _tick(tmp_path, opt, fake, tick.state, failures * 86400)
        episode = tick.state["com.brainlayer.maintenance"]
        assert episode["consecutive"] == failures
        assert episode["failed_heals"] is (failures >= 3)
        assert len(tick.escalations) == int(failures >= 3)
        assert tick.actions == [] and episode["attempts"] == 0
        # Health-check samples between nightly runs must not advance the threshold.
        repeated = _tick(tmp_path, opt, fake, tick.state, failures * 86400 + 300)
        assert repeated.state == tick.state and repeated.escalations == tick.escalations
    assert not any(args[:2] == ["launchctl", "kickstart"] for args in fake.commands)


def test_nightly_maintenance_first_success_clears_and_restarts_threshold(tmp_path: Path, nightly_maintenance):
    opt, fake, _ = nightly_maintenance
    failed = _failed_maintenance_nights(tmp_path, nightly_maintenance, 4)
    assert len(failed.escalations) == 1
    fake.runs["maintenance"] += 1
    fake.exit["maintenance"] = 0
    cleared = _tick(tmp_path, opt, fake, failed.state, 5 * 86400)
    episode = cleared.state["com.brainlayer.maintenance"]
    assert cleared.escalations == [] and cleared.actions == []
    assert episode["consecutive"] == 0 and episode["failed_heals"] is False
    fake.exit["maintenance"] = 1
    state = cleared.state
    for failures in range(1, 4):
        fake.runs["maintenance"] += 1
        tick = _tick(tmp_path, opt, fake, state, (5 + failures) * 86400)
        assert tick.state["com.brainlayer.maintenance"]["consecutive"] == failures
        assert len(tick.escalations) == int(failures == 3)
        assert tick.actions == []
        state = tick.state
