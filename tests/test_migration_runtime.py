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
