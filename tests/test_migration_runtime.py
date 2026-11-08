"""Private launchctl stubs exercise bounded migration process and readiness gates."""

import subprocess

import pytest

from brainlayer.service_supervisor import Component
from tests.test_consolidation_plan import DOMAIN, OLD


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.now += delay


@pytest.mark.parametrize("override,ready_at", [(None, 30), ("10", 5), ("120", 100), ("900", 100)])
def test_readiness_grace_allows_slow_residents(monkeypatch, capsys, override, ready_at):
    from brainlayer.migration_runtime import wait_supervisor

    monkeypatch.delenv("BRAINLAYER_SUPERVISOR_READY_GRACE_S", raising=False)
    if override is not None:
        monkeypatch.setenv("BRAINLAYER_SUPERVISOR_READY_GRACE_S", override)
    clock = Clock()

    def query():
        if clock.now < 1:
            raise FileNotFoundError("socket not ready")
        return {
            "watch": {"configured": True, "paused": False, "pid": 4242 if clock.now >= ready_at else None},
            "idle": {"configured": True, "paused": False, "pid": None},
        }

    expected = {"watch": Component(("/bin/true",)), "idle": Component(("/bin/true",), interval=300)}
    wait_supervisor(query, expected, clock=clock, alive=lambda pid: True)
    assert ready_at <= clock.now < ready_at + 0.2
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == int((ready_at - 0.1) // 10)
    assert all("watch" in line and "not ready" in line and "idle" not in line for line in lines)


@pytest.mark.parametrize("value", ["", "9", "901", "30.5", "no", "-10", "1_0"])
def test_invalid_ready_grace_refused_before_query(monkeypatch, value):
    from brainlayer.migration_runtime import wait_supervisor

    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_READY_GRACE_S", value)
    with pytest.raises(ValueError, match="BRAINLAYER_SUPERVISOR_READY_GRACE_S.*integer.*10.*900"):
        wait_supervisor(lambda: pytest.fail("invalid grace reached query"), {"watch": Component(("/bin/true",))})


def test_stop_allows_slow_exit_and_reports_progress(monkeypatch, capsys):
    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_READY_GRACE_S", "90")
    clock = Clock()

    def run(argv):
        return (
            subprocess.CompletedProcess(argv, 0, "pid = 4242", "")
            if clock.now < 40
            else subprocess.CompletedProcess(argv, 113, "", "Could not find service")
        )

    instance = runtime(run, alive=lambda pid: clock.now < 40)
    instance.clock = clock
    instance.stop(OLD)
    assert 40 <= clock.now < 40.2
    assert OLD in capsys.readouterr().err


def test_custody_allows_slow_exit_and_reports_progress(tmp_path, monkeypatch, capsys):
    from brainlayer import launchd_consolidation as module

    monkeypatch.setenv("BRAINLAYER_SUPERVISOR_READY_GRACE_S", "90")
    clock, custody = Clock(), module.Custody
    tmp_path.chmod(0o700)

    def delayed_custody(*args, **kwargs):
        if clock.now < 40:
            raise BlockingIOError
        return custody(*args, **kwargs)

    monkeypatch.setattr(module, "Custody", delayed_custody)
    module.wait_custody(tmp_path, clock=clock)
    assert 40 <= clock.now < 40.2
    assert str(tmp_path) in capsys.readouterr().err


def runtime(run, alive=lambda pid: False):
    from brainlayer.migration_runtime import LaunchdRuntime

    clock = Clock()
    return LaunchdRuntime(DOMAIN, run, alive=alive, clock=clock)


@pytest.mark.parametrize("pid", ["invalid", "", "42\npid = 43"])
def test_runtime_refuses_ambiguous_pid_before_bootout(pid):
    calls = []

    def run(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "pid = " + pid, "")

    with pytest.raises(RuntimeError, match="process state"):
        runtime(run).stop(OLD)
    assert [argv[1] for argv in calls] == ["print"]


def test_bootout_absence_does_not_authorize_overlap_with_live_old_pid():
    calls = []

    def run(argv):
        calls.append(argv)
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 0, "pid = 4242", "")
        return (
            subprocess.CompletedProcess(argv, 113, "", "Could not find service")
            if argv[1] == "print"
            else subprocess.CompletedProcess(argv, 0, "", "")
        )

    with pytest.raises(RuntimeError, match="did not exit"):
        runtime(run, alive=lambda pid: True).stop(OLD, grace=0)
    assert all(argv[1] in {"print", "bootout"} and argv[2] == f"{DOMAIN}/{OLD}" for argv in calls)


def test_bootout_preserves_disabled_tristate_by_never_writing_overrides():
    calls = []

    def run(argv):
        calls.append(argv)
        loaded = len(calls) == 1
        rc = 0 if loaded or argv[1] == "bootout" else 113
        return subprocess.CompletedProcess(
            argv, rc, "state = waiting" if loaded else "", "Could not find service" if rc else ""
        )

    runtime(run).stop(OLD)
    assert [argv[1] for argv in calls] == ["print", "bootout", "print"]


def test_changed_service_refused_before_bootout():
    calls = []

    def run(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "pid = 43", "")

    with pytest.raises(RuntimeError, match="changed"):
        runtime(run).stop(OLD, expected={"loaded": True, "pid": 42})
    assert len(calls) == 1


def test_readiness_retries_startup_and_checks_effective_config():
    from brainlayer.migration_runtime import wait_supervisor

    clock, calls = Clock(), []

    def query():
        calls.append(1)
        if len(calls) == 1:
            raise FileNotFoundError("socket not ready")
        return {"watch": {"configured": True, "paused": False, "pid": 4242}}

    wait_supervisor(query, {"watch": Component(("/bin/true",))}, clock=clock, alive=lambda pid: True)
    assert len(calls) == 2 and clock.now > 0


@pytest.mark.parametrize(
    "states", [{}, {"watch": {"configured": False, "paused": False}}, {"watch": {"configured": True, "paused": True}}]
)
def test_readiness_refuses_false_green(states):
    from brainlayer.migration_runtime import wait_supervisor

    with pytest.raises(RuntimeError, match="not ready"):
        wait_supervisor(lambda: states, {"watch": Component(("/bin/true",))}, grace=0, clock=Clock())


@pytest.mark.parametrize("grace", [-1, float("nan"), float("inf"), True, "10"])
def test_invalid_deadline_refused_before_any_service_probe(grace):
    with pytest.raises(ValueError, match="deadline"):
        runtime(lambda argv: pytest.fail("invalid bound reached launchctl")).stop(OLD, grace=grace)


def test_foreign_service_refused_before_probe_or_bootout():
    with pytest.raises(ValueError, match="non-owned"):
        runtime(lambda argv: pytest.fail("foreign job reached launchctl")).stop("com.example.personal")


def test_idle_interval_readiness_does_not_require_pid():
    from brainlayer.migration_runtime import wait_supervisor

    state = {"observability": {"configured": True, "paused": False, "pid": None}}
    assert wait_supervisor(lambda: state, {"observability": Component(("/bin/true",), interval=300)}) == state


def test_configured_off_is_not_ready_while_a_process_is_reported():
    from brainlayer.migration_runtime import wait_supervisor

    state = {"watch": {"configured": False, "paused": False, "pid": 4242}}
    with pytest.raises(RuntimeError, match="not ready"):
        wait_supervisor(lambda: state, {"watch": Component(("/bin/true",), enabled=False)}, grace=0, clock=Clock())


@pytest.mark.parametrize("initial_pid", [None, 4242])
def test_stop_waits_for_pid_observed_after_bootout(initial_pid):
    prints = iter([initial_pid, 4343, "absent"])
    observed = []

    def run(argv):
        if argv[1] == "bootout":
            return subprocess.CompletedProcess(argv, 0, "", "")
        pid = next(prints, "absent")
        if pid == "absent":
            return subprocess.CompletedProcess(argv, 113, "", "Could not find service")
        return subprocess.CompletedProcess(argv, 0, "state = waiting" if pid is None else f"pid = {pid}", "")

    def alive(pid):
        observed.append(pid)
        return pid == 4343

    with pytest.raises(RuntimeError, match="did not exit"):
        runtime(run, alive).stop(OLD, grace=0)
    assert 4343 in observed
