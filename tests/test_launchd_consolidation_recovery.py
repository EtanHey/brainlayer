"""Interrupted transactions recover with persisted scratch launchd state, never live launchctl."""

import hashlib
import json
import os
import plistlib
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.test_consolidation_plan import NEW, OLD, ROOT
from tests.test_consolidation_plan import setup as setup
from tests.test_launchd_consolidation import Launchctl, invoke

FLEET = "com.etanhey.brainlayer-fleet-watchdog"
CALENDAR = "com.brainlayer.maintenance"


@pytest.mark.parametrize("pinned", [False, True], ids=["opt-plist", "fallback-pinned-plist"])
def test_keg_restart_accepts_idle_empty_calendar_with_older_boot_stamp(setup, monkeypatch, pinned):
    from brainlayer import launchd_consolidation as migration
    from brainlayer import supervisor_guard

    monkeypatch.setattr(migration, "boot_time", lambda: "1791800119:895228")
    monkeypatch.setattr(supervisor_guard, "boot_time", lambda: "1791800119:895228")
    runner = Launchctl()
    previous = invoke(setup, runner)
    home, agents, _ = setup
    control = home / ".local/share/brainlayer/supervisor"
    calendar = control / "calendar"
    calendar.mkdir(mode=0o700, exist_ok=True)
    groups = calendar / "groups.json"
    groups.write_text(json.dumps({"boot_time": "1791800119:947167", "groups": {}}))
    groups.chmod(0o600)
    if pinned:
        daemon = agents / f"{NEW}.plist"
        raw = bytes.fromhex(previous["daemon_pinned"])
        daemon.write_bytes(raw)
        previous["published_files"][daemon.name] = raw.hex()
        previous["published_sha256"][daemon.name] = hashlib.sha256(raw).hexdigest()
        (control / "migration.json").write_text(json.dumps(previous))
    runner.calls.clear()
    result = invoke(setup, runner, "keg2")
    assert result["phase"] == "migrated" and result["keg"] == "keg2"
    assert runner.loaded == {NEW, CALENDAR}
    assert not any(call[0] == "kickstart" for call in runner.calls)
    assert json.loads(groups.read_text()) == {"boot_time": "1791800119:947167", "groups": {}}


@pytest.mark.parametrize("groups", [{"52": {"pgid": 52, "start_time": "older"}}, None, []])
def test_custody_refuses_nonempty_or_malformed_previous_boot_without_rewriting(tmp_path, groups):
    from brainlayer.launchd_consolidation import wait_custody

    tmp_path.chmod(0o700)
    journal = tmp_path / "groups.json"
    raw = json.dumps({"boot_time": "previous-boot", "groups": groups}).encode()
    journal.write_bytes(raw)
    journal.chmod(0o600)
    with pytest.raises(RuntimeError, match="custody identity refused"):
        wait_custody(tmp_path)
    assert journal.read_bytes() == raw


def test_custody_validates_journal_written_while_waiting_for_lock(tmp_path, monkeypatch):
    from brainlayer import supervisor_guard
    from brainlayer.launchd_consolidation import wait_custody

    tmp_path.chmod(0o700)
    journal = tmp_path / "groups.json"
    journal.write_text(json.dumps({"boot_time": "previous-boot", "groups": {}}))
    journal.chmod(0o600)
    replacement = json.dumps({"boot_time": "previous-boot", "groups": {"52": {"pgid": 52, "start_time": "older"}}})
    flock = supervisor_guard.fcntl.flock

    def replace_after_lock(fd, operation):
        flock(fd, operation)
        journal.write_text(replacement)

    monkeypatch.setattr(supervisor_guard.fcntl, "flock", replace_after_lock)
    with pytest.raises(RuntimeError, match="custody identity refused"):
        wait_custody(tmp_path)
    assert journal.read_text() == replacement


def test_empty_previous_boot_custody_waits_for_real_kernel_lock(tmp_path):
    from brainlayer.launchd_consolidation import wait_custody
    from brainlayer.supervisor_guard import Custody

    class Clock:
        now = 0

        def monotonic(self):
            return self.now

        def sleep(self, delay):
            self.now += delay

    tmp_path.chmod(0o700)
    with Custody(tmp_path):
        journal = tmp_path / "groups.json"
        raw = json.dumps({"boot_time": "previous-boot", "groups": {}}).encode()
        journal.write_bytes(raw)
        journal.chmod(0o600)
        with pytest.raises(RuntimeError, match="custody did not exit"):
            wait_custody(tmp_path, clock=Clock(), grace=0)
        assert journal.read_bytes() == raw
    wait_custody(tmp_path)
    assert journal.read_bytes() == raw


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("after_bootstrap", [False, True])
def test_base_exception_runs_rollback_then_reraises_original(setup, restart, failure, after_bootstrap):
    class Interrupted(Launchctl):
        interrupt = False

        def __call__(self, argv):
            if self.interrupt and argv[1] == "bootstrap" and Path(argv[3]).stem == NEW:
                self.interrupt = False
                if after_bootstrap:
                    super().__call__(argv)
                raise failure("interrupted bootstrap")
            return super().__call__(argv)

    runner = Interrupted()
    if restart:
        invoke(setup, runner)
    runner.interrupt = True
    with pytest.raises(failure, match="interrupted bootstrap"):
        invoke(setup, runner, "keg2" if restart else "keg1")
    journal = json.loads((setup[0] / ".local/share/brainlayer/supervisor/migration.json").read_bytes())
    assert journal["phase"] == ("migrated" if restart else "rolled_back")
    assert runner.loaded == ({NEW, CALENDAR} if restart else {OLD, FLEET})


def test_active_calendar_defers_restart_without_any_mutation(setup):
    runner = Launchctl()
    invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    original = receipt.read_bytes()
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="calendar is active"):
        invoke(setup, runner, "keg2", alive=lambda pid: True)
    assert receipt.read_bytes() == original
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)
    assert runner.loaded == {NEW, CALENDAR}
    assert (setup[1] / f"{NEW}.plist.loaded-keg").read_bytes() == b"keg1\n"


def test_journal_hashes_every_published_launcher_and_marker_through_restart(setup):
    runner = Launchctl()
    for keg in ("keg1", "keg2"):
        journal = invoke(setup, runner, keg)
        expected = {
            f"{label}{suffix}": hashlib.sha256((setup[1] / f"{label}{suffix}").read_bytes()).hexdigest()
            for label in (NEW, CALENDAR)
            for suffix in (".plist", ".plist.loaded-keg")
        }
        assert journal["published_sha256"] == expected


def pending_first(setup, runner):
    journal = invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    journal["phase"] = "migrating"
    receipt.write_text(json.dumps(journal))
    runner.calls.clear()
    return receipt, journal


@pytest.mark.parametrize("name", [f"{NEW}.plist", f"{NEW}.plist.loaded-keg", "jobs.json"])
def test_recovery_preserves_foreign_published_bytes_and_never_revives_legacy(setup, name):
    runner = Launchctl()
    receipt, _ = pending_first(setup, runner)
    path = receipt.parent / name if name == "jobs.json" else setup[1] / name
    foreign = b"foreign personal bytes\n"
    path.write_bytes(foreign)
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        invoke(setup, runner)
    assert path.read_bytes() == foreign
    assert OLD not in runner.loaded
    assert not any(call[0] == "bootstrap" for call in runner.calls)


def test_recovery_preserves_exact_operator_override_change(setup):
    runner = Launchctl()
    pending_first(setup, runner)
    runner.disabled.add(OLD)
    with pytest.raises(RuntimeError, match="disabled state changed"):
        invoke(setup, runner)
    assert OLD in runner.disabled
    assert runner.loaded == {NEW, CALENDAR}
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


@pytest.mark.parametrize("third_link", [False, True])
def test_recovery_finishes_exact_restore_link_pair_after_another_crash(setup, third_link):
    runner = Launchctl()
    _, journal = pending_first(setup, runner)
    runner.loaded.clear()
    source = setup[1] / f"{OLD}.plist"
    archived = Path(journal["archive"]) / source.name
    os.link(archived, source)  # SIGKILL between rollback link and archive unlink.
    assert source.stat().st_nlink == 2
    if third_link:
        personal = setup[0] / "personal"
        os.link(source, personal)
        with pytest.raises(RuntimeError, match="hardlink custody refused"):
            invoke(setup, runner)
        assert personal.read_bytes() == archived.read_bytes() == source.read_bytes()
        assert not runner.loaded
        return
    result = invoke(setup, runner)
    assert result["phase"] == "rolled_back"
    assert source.read_bytes() == bytes.fromhex(journal["files"][source.name])
    assert source.stat().st_nlink == 1 and not archived.exists()
    assert runner.loaded == {OLD, FLEET}


def test_failed_same_byte_foreign_publication_never_becomes_owned_on_rerun(setup, monkeypatch):
    from brainlayer import launchd_consolidation as module

    original = module.write_launch_agent
    foreign_path = setup[1] / f"{NEW}.plist"

    def arrival(path, raw):
        if path == foreign_path:
            path.write_bytes(raw)
        original(path, raw)

    runner = Launchctl()
    monkeypatch.setattr(module, "write_launch_agent", arrival)
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        invoke(setup, runner)
    foreign = foreign_path.read_bytes()
    monkeypatch.setattr(module, "write_launch_agent", original)
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="operator reconciliation"):
        invoke(setup, runner)
    assert foreign_path.read_bytes() == foreign
    assert not any(call[0] == "bootstrap" for call in runner.calls)


def test_interrupted_restart_fallback_pins_previous_keg_after_opt_moves(setup):
    home, agents, _ = setup
    old_keg, new_keg = home / "Cellar/brainlayer/keg1", home / "Cellar/brainlayer/keg2"
    old_keg.mkdir(parents=True)
    new_keg.mkdir(parents=True)
    opt = home / "opt/brainlayer"
    opt.parent.mkdir(parents=True)
    opt.symlink_to(old_keg, target_is_directory=True)
    runner = Launchctl()
    previous = invoke(setup, runner)
    opt.unlink()
    opt.symlink_to(new_keg, target_is_directory=True)
    new_python = new_keg / "libexec/venv/bin/python"
    new_python.parent.mkdir(parents=True)
    new_python.write_text("#!/bin/sh\nexit 0\n")
    new_python.chmod(0o755)
    (new_python.parents[1] / "lib/python3.12/site-packages/brainlayer").mkdir(parents=True)
    daemon = agents / f"{NEW}.plist"
    requested = plistlib.loads(daemon.read_bytes())
    interrupted = previous | dict(
        phase="migrating",
        requested_keg="keg2",
        restart_preimage=previous,
        restart_state={"loaded": True, "pid": 4242},
        requested_daemon=plistlib.dumps(requested, sort_keys=False).hex(),
    )
    receipt = home / ".local/share/brainlayer/supervisor/migration.json"
    receipt.write_text(json.dumps(interrupted))
    runner.loaded.remove(NEW)
    checked = []

    def verify():
        selected = Path(plistlib.loads(daemon.read_bytes())["ProgramArguments"][1])
        checked.append(selected)
        if selected == opt / "libexec/venv/bin/python":
            raise RuntimeError("requested keg not ready")
        assert selected == old_keg / "libexec/venv/bin/python"

    result = invoke(setup, runner, "keg2", verify=verify)
    assert result["phase"] == "migrated" and result["keg"] == "keg1"
    assert result["restart_fallback"] == {"requested_keg": "keg2", "restored_keg": "keg1"}
    assert len(checked) == 2 and runner.loaded == {NEW, CALENDAR}
    assert hashlib.sha256(daemon.read_bytes()).hexdigest() == result["published_sha256"][daemon.name]
    assert (agents / f"{NEW}.plist.loaded-keg").read_bytes() == b"keg1\n"
    recovered = invoke(setup, runner, "keg2")
    assert recovered["keg"] == "keg2" and "restart_fallback" not in recovered
    assert Path(plistlib.loads(daemon.read_bytes())["ProgramArguments"][1]) == opt / "libexec/venv/bin/python"


@pytest.mark.parametrize("interrupted_pin", [False, True], ids=["normal", "interrupted-cellar-pin"])
def test_successful_restart_survives_next_upgrade_and_cellar_cleanup(setup, monkeypatch, interrupted_pin):
    from brainlayer import launchd_consolidation as module

    home, agents, env = setup
    opt = home / "opt/brainlayer"
    opt.parent.mkdir(parents=True)
    kegs = [home / f"Cellar/brainlayer/v{version}" for version in (1, 2, 3)]
    for keg in kegs:
        venv = keg / "libexec/venv"
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=15)
        (next((venv / "lib").glob("python*/site-packages")) / "brainlayer").mkdir()
    opt.symlink_to(kegs[0], target_is_directory=True)
    python = opt / "libexec/venv/bin/python"
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    runner = Launchctl()

    def migrate(keg):
        return module.migrate(
            home=home,
            templates=ROOT / "scripts/launchd",
            env_file=env,
            python=python,
            keg=str(keg),
            domain="gui/424242",
            run=runner,
            alive=lambda pid: False,
            verify=lambda: None,
        )

    previous = migrate(kegs[0])
    opt.unlink()
    opt.symlink_to(kegs[1], target_is_directory=True)
    if interrupted_pin:
        # A pre-R3 crash journal can carry the old physically pinned requested bytes.
        requested = plistlib.loads((agents / f"{NEW}.plist").read_bytes())
        requested["ProgramArguments"][1] = str(kegs[1] / "libexec/venv/bin/python")
        requested["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"] = str(
            next((kegs[1] / "libexec/venv/lib").glob("python*/site-packages"))
        )
        pending = previous | dict(
            phase="migrating",
            requested_keg=str(kegs[1]),
            restart_preimage=previous,
            restart_state={"loaded": True, "pid": 4242},
            requested_daemon=plistlib.dumps(requested, sort_keys=False).hex(),
        )
        (home / ".local/share/brainlayer/supervisor/migration.json").write_text(json.dumps(pending))
    result = migrate(kegs[1])
    daemon = agents / f"{NEW}.plist"
    raw = daemon.read_bytes()
    opt.unlink()
    opt.symlink_to(kegs[2], target_is_directory=True)
    shutil.rmtree(kegs[1])  # Only this test's private Cellar, simulating brew cleanup.
    document = plistlib.loads(raw)
    selected = Path(document["ProgramArguments"][1])
    assert selected.is_file(), "successful restart published an interpreter removed by brew cleanup"
    assert selected == python and b"/Cellar/" not in raw
    assert Path(document["EnvironmentVariables"]["BRAINLAYER_REPO_ROOT"]).is_dir()
    prefix = subprocess.check_output([str(selected), "-c", "import sys;print(sys.prefix)"], text=True, timeout=5)
    assert Path(prefix.strip()).resolve() == kegs[2] / "libexec/venv"
    assert result["configuration_sha256"][NEW] == hashlib.sha256(raw).hexdigest()
    assert result["published_sha256"][daemon.name] == hashlib.sha256(raw).hexdigest()
    assert plistlib.loads(bytes.fromhex(result["daemon_pinned"]))["ProgramArguments"][1] == str(
        kegs[1] / "libexec/venv/bin/python"
    )


@pytest.mark.parametrize("failure", ["verify", "bootstrap", "persistent-bootstrap"])
def test_removed_previous_keg_never_publishes_missing_fallback_and_plain_rerun_recovers(setup, failure):
    home, agents, _ = setup
    old, current = [home / f"Cellar/brainlayer/{name}" for name in ("old", "current")]
    old.mkdir(parents=True)
    current.mkdir(parents=True)
    opt = home / "opt/brainlayer"
    opt.parent.mkdir(parents=True)
    opt.symlink_to(old, target_is_directory=True)

    class BootstrapFailure(Launchctl):
        refuse = False

        def __call__(self, argv):
            if self.refuse and argv[1] == "bootstrap" and Path(argv[3]).stem == NEW:
                self.fail = NEW
            return super().__call__(argv)

    runner = BootstrapFailure()
    invoke(setup, runner)
    opt.unlink()
    opt.symlink_to(current, target_is_directory=True)
    shutil.rmtree(old)  # This test created and owns the removed keg.
    runner.calls.clear()
    if failure == "bootstrap":
        runner.fail = NEW
    runner.refuse = failure == "persistent-bootstrap"

    def verify():
        raise RuntimeError("requested keg not ready")

    message = "previous keg removed by brew cleanup; re-run `install.sh consolidate` on the current keg"
    with pytest.raises(RuntimeError, match="rollback incomplete") as error:
        invoke(setup, runner, "keg2", verify=verify if failure == "verify" else lambda: None)
    assert message in str(error.value)
    daemon = agents / f"{NEW}.plist"
    raw = daemon.read_bytes()
    assert Path(plistlib.loads(raw)["ProgramArguments"][1]) == opt / "libexec/venv/bin/python"
    assert b"/Cellar/" not in raw and (opt / "libexec/venv/bin/python").is_file()
    assert runner.loaded == ({CALENDAR} if runner.refuse else {NEW, CALENDAR})
    receipt = home / ".local/share/brainlayer/supervisor/migration.json"
    pending = json.loads(receipt.read_bytes())
    assert pending["phase"] == "migrating" and pending["requested_keg"] == "keg2"
    assert "restart_fallback" not in pending and "recovery_refusal" not in pending
    runner.refuse = False
    recovered = invoke(setup, runner, "keg2")
    assert recovered["phase"] == "migrated" and recovered["keg"] == "keg2"
    assert "restart_fallback" not in recovered


def test_pinned_daemon_keeps_venv_python_symlink_and_actual_prefix(setup, monkeypatch):
    from brainlayer import launchd_consolidation as module

    home, _, env = setup
    keg = home / "Cellar/brainlayer/keg1"
    venv = keg / "libexec/venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=15)
    site = next((venv / "lib").glob("python*/site-packages"))
    (site / "brainlayer").mkdir()
    opt = home / "opt/brainlayer"
    opt.parent.mkdir(parents=True)
    opt.symlink_to(keg, target_is_directory=True)
    python = opt / "libexec/venv/bin/python"
    assert python.is_symlink()
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    # Do not use invoke(): its dummy-python writer would follow this real interpreter link.
    result = module.migrate(
        home=home,
        templates=ROOT / "scripts/launchd",
        env_file=env,
        python=python,
        keg=str(keg),
        domain="gui/424242",
        run=Launchctl(),
        alive=lambda pid: False,
        verify=lambda: None,
    )
    selected = plistlib.loads(bytes.fromhex(result["daemon_pinned"]))["ProgramArguments"][1]
    assert selected == str(venv / "bin/python")
    prefix = subprocess.check_output([selected, "-c", "import sys;print(sys.prefix)"], text=True, timeout=5)
    assert prefix.strip() == str(venv)


class Relaunched(Launchctl):
    daemon_pid = 10000

    def __call__(self, argv):
        if argv[1] == "bootstrap" and Path(argv[3]).stem == NEW:
            self.daemon_pid += 1
        result = super().__call__(argv)
        if argv[1] == "print" and argv[2].endswith("/" + NEW) and result.returncode == 0:
            result.stdout = f"state = running\npid = {self.daemon_pid}"
        return result


@pytest.mark.parametrize("restart", [False, True])
def test_readiness_failure_reaps_owned_launchd_replacement_after_pid_rotation(setup, restart):
    runner = Relaunched()
    if restart:
        invoke(setup, runner)
    checks = []

    def verify():
        checks.append(runner.daemon_pid)
        if len(checks) == 1:
            runner.daemon_pid += 1  # launchd KeepAlive restarted the same owned definition.
            raise RuntimeError("replacement not ready")

    with pytest.raises(RuntimeError, match="restored"):
        invoke(setup, runner, "keg2" if restart else "keg1", verify=verify)
    journal = json.loads((setup[0] / ".local/share/brainlayer/supervisor/migration.json").read_bytes())
    assert journal["phase"] == ("migrated" if restart else "rolled_back")
    assert "recovery_refusal" not in journal
    assert runner.loaded == ({NEW, CALENDAR} if restart else {OLD, FLEET})
    if restart:
        assert journal["keg"] == "keg1" and len(checks) == 2


def test_pid_rotation_still_waits_for_the_prior_owned_generation_before_legacy_bootstrap(setup):
    from tests.test_migration_runtime import Clock

    runner = Relaunched()
    prior = []

    def verify():
        prior.append(runner.daemon_pid)
        runner.daemon_pid += 1
        raise RuntimeError("replacement not ready")

    with pytest.raises(RuntimeError, match="rollback incomplete"):
        invoke(setup, runner, verify=verify, alive=lambda pid: pid in prior, clock=Clock())
    assert prior
    assert ("bootout", f"gui/424242/{NEW}") in runner.calls
    assert not any(call[0] == "bootstrap" and Path(call[2]).stem in {OLD, FLEET} for call in runner.calls)


PROBE = r"""
import json, os, plistlib, signal, subprocess, sys
from pathlib import Path
from brainlayer.launchd_consolidation import migrate
root, keg, cut = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
class Clock:
    now = 0
    def monotonic(self): return self.now
    def sleep(self, delay): self.now += delay
home = root / "home"
state_file = root / "launchd.json"
events = []
original_replace = os.replace
def boundary(name):
    events.append(name)
    if cut == len(events):
        os.kill(os.getpid(), signal.SIGKILL)
def run(argv):
    action, target = argv[1:3]
    assert target == "gui/424242" or target.startswith("gui/424242/"), argv
    state = json.loads(state_file.read_bytes())
    label = target.rsplit("/", 1)[-1]
    boundary("before:launchctl:" + action + ":" + label)
    rc, text = 0, ""
    if action == "print-disabled":
        text = 'disabled services = {\n"com.brainlayer.watch" => false\n"com.brainlayer.observability" => true\n}'
    elif action == "print":
        rc, text = ((0, "state = running\npid = 4242") if label in state["loaded"]
                    else (113, "Could not find service"))
    elif action == "bootout":
        state["loaded"] = [item for item in state["loaded"] if item != label]
    elif action == "bootstrap":
        label = plistlib.loads(Path(argv[3]).read_bytes())["Label"]
        if label in state["loaded"]:
            state["violations"].append("duplicate bootstrap: " + label)
        state["loaded"].append(label)
    else:
        raise AssertionError(argv)
    if "com.brainlayer.watch" in state["loaded"] and "com.brainlayer.daemon" in state["loaded"]:
        state["violations"].append("two writers")
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(state))
    original_replace(temporary, state_file)
    boundary("after:launchctl:" + action + ":" + label)
    return subprocess.CompletedProcess(argv, rc, text, text if rc else "")
def wrap(name, original):
    def call(*args, **kwargs):
        boundary("before:os." + name)
        result = original(*args, **kwargs)
        boundary("after:os." + name)
        return result
    return call
for name in ("open", "close", "fchmod", "fsync", "link", "unlink", "rename", "replace", "mkdir", "fstat", "stat", "listdir"):
    setattr(os, name, wrap(name, getattr(os, name)))
python = home / "opt/brainlayer/libexec/venv/bin/python"
os.environ["BRAINLAYER_HOOK_PYTHON"] = str(python)
result = migrate(home=home, templates=Path(sys.argv[4]) / "scripts/launchd",
                 env_file=home / ".config/brainlayer/brainlayer.env", python=python,
                 keg=keg, domain="gui/424242", run=run, alive=lambda pid: False, verify=lambda: None, clock=Clock())
print(json.dumps({"events": events, "phase": result["phase"]}))
"""


def scratch(root):
    agents = root / "home/Library/LaunchAgents"
    agents.mkdir(parents=True)
    for label in (OLD, FLEET):
        (agents / f"{label}.plist").write_bytes(
            plistlib.dumps({"Label": label, "ProgramArguments": ["/bin/true"], "KeepAlive": True})
        )
        (agents / f"{label}.plist.loaded-keg").write_bytes(b"previous-keg\n")
    env = root / "home/.config/brainlayer/brainlayer.env"
    env.parent.mkdir(parents=True)
    env.write_text("BRAINLAYER_LAUNCHD_WATCH_ENABLED=1\n")
    env.chmod(0o600)
    python = root / "home/opt/brainlayer/libexec/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    (python.parents[1] / "lib/python3.12/site-packages/brainlayer").mkdir(parents=True)
    (root / "launchd.json").write_text(json.dumps({"loaded": [OLD, FLEET], "violations": []}))
    return {path.name: path.read_bytes() for path in agents.iterdir()}


@pytest.mark.parametrize("streams", [(b"partial output", b"stuck stderr"), ("partial output", "stuck stderr")])
def test_probe_timeout_reports_exact_rerun_cut_and_journal(tmp_path, monkeypatch, streams):
    journal = tmp_path / "home/.local/share/brainlayer/supervisor/migration.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(json.dumps({"phase": "migrating"}))

    def timeout(argv, **kwargs):
        assert kwargs["timeout"] == 30
        raise subprocess.TimeoutExpired(argv, 30, output=streams[0], stderr=streams[1])

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(pytest.fail.Exception) as error:
        probe(tmp_path, "keg2", context="keg-restart rerun cut=356 after:os.rename")
    message = str(error.value)
    for detail in ("keg-restart rerun cut=356 after:os.rename", "30s", "migrating", "partial output", "stuck stderr"):
        assert detail in message


def probe(root, keg, cut=0, *, context="baseline"):
    try:
        return subprocess.run(
            [sys.executable, "-c", PROBE, str(root), keg, str(cut), str(ROOT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        journal = root / "home/.local/share/brainlayer/supervisor/migration.json"
        try:
            phase = json.loads(journal.read_bytes()).get("phase", "missing phase")
        except (OSError, ValueError) as journal_error:
            phase = f"unreadable: {journal_error}"

        def tail(stream):
            if isinstance(stream, bytes):
                stream = stream.decode("utf-8", errors="replace")
            return (stream or "")[-2000:]

        pytest.fail(
            f"probe exceeded 30s: {context}; journal phase={phase}; "
            f"stdout={tail(error.stdout)!r}; stderr={tail(error.stderr)!r}",
            pytrace=False,
        )


@pytest.mark.crash_sweep
@pytest.mark.parametrize("restart", [False, True], ids=["first-install", "keg-restart"])
@pytest.mark.parametrize("shard", range(32), ids=lambda shard: f"shard-{shard:02}")
def test_sigkill_every_os_and_launchctl_boundary_then_plain_rerun(tmp_path, restart, shard):
    mode = "keg-restart" if restart else "first-install"
    baseline_root = tmp_path / "baseline"
    scratch(baseline_root)
    if restart:
        first = probe(baseline_root, "keg1")
        assert first.returncode == 0, first.stderr
    baseline = probe(baseline_root, "keg2" if restart else "keg1")
    assert baseline.returncode == 0, baseline.stderr
    events = json.loads(baseline.stdout)["events"]
    assert len(events) > 100 and any("os.rename" in event for event in events) == (not restart)
    cuts = [(cut, event) for cut, event in enumerate(events, 1) if (cut - 1) % 32 == shard]
    assert 0 < len(cuts) <= 30, "increase shard count to retain the <=30-cut per-item bound"
    failures = []
    phases = {"rolled_back": 0, "migrated": 0}
    timings = []
    for cut, event in cuts:
        started = time.monotonic()
        context = f"{mode} cut={cut} {event}"
        root = tmp_path / str(cut)
        originals = scratch(root)
        if restart:
            first = probe(root, "keg1", context=f"{context} setup")
            assert first.returncode == 0, first.stderr
        killed = probe(root, "keg2" if restart else "keg1", cut, context=f"{context} kill")
        assert killed.returncode == -signal.SIGKILL, (cut, event, killed.stderr)
        rerun_started = time.monotonic()
        rerun = probe(root, "keg2" if restart else "keg1", context=f"{mode} rerun cut={cut} {event}")
        timing = {
            "mode": mode,
            "cut": cut,
            "boundary": event,
            "rerun_seconds": time.monotonic() - rerun_started,
            "total_seconds": time.monotonic() - started,
        }
        timings.append(timing)
        print("CRASH_CUT " + json.dumps(timing), flush=True)
        if rerun.returncode:
            failures.append((cut, event, rerun.stderr[-500:]))
            continue
        state = json.loads((root / "launchd.json").read_bytes())
        journal = json.loads((root / "home/.local/share/brainlayer/supervisor/migration.json").read_bytes())
        phases[journal["phase"]] += 1
        assert not state["violations"], (cut, event, state)
        agents = root / "home/Library/LaunchAgents"
        if journal["phase"] == "rolled_back":
            assert not restart and set(state["loaded"]) == {OLD, FLEET}, (cut, event, state)
            assert {p.name: p.read_bytes() for p in agents.iterdir() if p.is_file()} == originals
            assert not (root / "home/.local/share/brainlayer/supervisor/jobs.json").exists()
        else:
            assert journal["phase"] == "migrated" and set(state["loaded"]) == {NEW, CALENDAR}, (cut, event, state)
            assert journal["keg"] == ("keg2" if restart else "keg1")
            assert not any((agents / name).exists() for name in originals)
        assert journal["states"][OLD]["disabled_override"] == "enabled"
        assert journal["states"]["com.brainlayer.observability"]["disabled_override"] == "disabled"
        assert journal["states"][FLEET]["disabled_override"] is None
    assert not failures, f"{len(failures)}/{len(cuts)} unrecovered cuts: {failures[:12]}"
    print(
        "CRASH_SWEEP "
        + json.dumps(
            {
                "mode": mode,
                "shard": shard,
                "cuts": len(cuts),
                "all_cuts": len(events),
                "phases": phases,
                "violations": 0,
                "slowest_rerun": max(timings, key=lambda timing: timing["rerun_seconds"]),
            }
        ),
        flush=True,
    )
