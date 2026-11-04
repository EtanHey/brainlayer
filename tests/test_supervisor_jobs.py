"""Manifest parity and real isolated interval workers, never production jobs."""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

from brainlayer.service_supervisor import Component, Supervisor
from brainlayer.supervisor_jobs import load_components


@pytest.fixture
def short_path():
    with tempfile.TemporaryDirectory(prefix="bl-j-", dir="/tmp") as directory:
        yield Path(directory)


def manifest(tmp_path, **overrides):
    plist = dict(
        Label="com.brainlayer.drain",
        ProgramArguments=[sys.executable, "-c", "pass"],
        EnvironmentVariables={"BRAINLAYER_DRAIN_EMBED": "0"},
        StandardOutPath=str(tmp_path / "out"),
        StandardErrorPath=str(tmp_path / "err"),
        WorkingDirectory=str(tmp_path),
        KeepAlive=True,
        ExitTimeOut=30,
    )
    plist.update(overrides)
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"com.brainlayer.drain": {"enabled": False, "plist": plist}}))
    path.chmod(0o600)
    return path


def test_manifest_preserves_argv_env_logs_cwd_and_off(tmp_path):
    child = load_components(manifest(tmp_path), tmp_path / "control.sock")["drain"]
    assert child.argv == (sys.executable, "-c", "pass")
    assert child.env["BRAINLAYER_DRAIN_EMBED"] == "0"
    assert child.env["BRAINLAYER_SUPERVISOR_SOCKET"] == str(tmp_path / "control.sock")
    assert child.stdout == tmp_path / "out" and child.stderr == tmp_path / "err"
    assert child.cwd == tmp_path and not child.enabled and child.exit_timeout == 30


def test_independent_wal_fallback_registered_once_without_calendar(tmp_path, monkeypatch):
    from brainlayer.supervisor_adapters import ALIASES, ARCHIVED

    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "parent-db"))
    monkeypatch.setenv("BRAINLAYER_BACKUP_STAGING_DIR", str(tmp_path / "backups"))
    path = manifest(tmp_path, EnvironmentVariables={"BRAINLAYER_DB": str(tmp_path / "db")})
    components = load_components(path, tmp_path / "control.sock")
    child = components["wal-checkpoint-fallback"]
    assert ALIASES["com.brainlayer.wal-checkpoint-fallback"] == "wal-checkpoint-fallback"
    assert "com.brainlayer.wal-checkpoint" in ARCHIVED
    assert child.interval == 900 and child.run_timeout == 30
    assert child.argv == (sys.executable, "-m", "brainlayer.wal_checkpoint")
    assert child.env["BRAINLAYER_DB"] == str(tmp_path / "db")
    assert child.env["BRAINLAYER_BACKUP_STAGING_DIR"] == str(tmp_path / "backups")
    assert sum(c.argv == child.argv for c in components.values()) == 1
    inherited = load_components(manifest(tmp_path), tmp_path / "control.sock")["wal-checkpoint-fallback"]
    assert inherited.env["BRAINLAYER_DB"] == str(tmp_path / "parent-db")


@pytest.mark.parametrize(
    "override",
    [
        dict(Label="com.brainlayer.enrich"),
        dict(ProgramArguments=[]),
        dict(StartInterval=0),
        dict(StartInterval=True),
        dict(RunAtLoad="false"),
        dict(MysteryPolicy=True),
        dict(KeepAlive=False),
        dict(EnvironmentVariables={"BAD": 3}),
        dict(StartCalendarInterval={"Hour": 3}),
    ],
)
def test_manifest_refuses_unknown_malformed_or_calendar_jobs(tmp_path, override):
    with pytest.raises(ValueError):
        load_components(manifest(tmp_path, **override), tmp_path / "control.sock")


def test_manifest_refuses_readable_or_symlink_file(tmp_path):
    path = manifest(tmp_path)
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        load_components(path, tmp_path / "control.sock")
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_components(link, tmp_path / "control.sock")


@pytest.mark.parametrize("field,value", [("interval", -1), ("interval", float("nan")), ("run_timeout", 0)])
def test_scheduler_rejects_invalid_bounds(tmp_path, field, value):
    with pytest.raises(ValueError):
        Supervisor({"tick": Component((sys.executable, "-c", "pass"), **{field: value})}, tmp_path / "s")


def test_intervals_reap_descendants_bound_time_and_keep_sibling_live(short_path):
    tmp_path = short_path

    async def scenario():
        output = tmp_path / "ticks"
        code = (
            "import os,subprocess,sys,time; "
            "open('ticks','a').write(str(os.getpid())+'\\n'); "
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"
        )
        tick = Component((sys.executable, "-c", code), cwd=tmp_path, interval=0.25, run_timeout=0.2, exit_timeout=0.02)
        sibling = Component((sys.executable, "-c", "import time;time.sleep(60)"), exit_timeout=0.02)
        s = Supervisor(
            {"tick": tick, "sibling": sibling},
            tmp_path / "s",
            guard_timeout=2,
            shutdown_timeout=0.5,
            stable_uptime=0.05,
        )
        task = asyncio.create_task(s.run())
        try:
            deadline = time.monotonic() + 5
            while (not output.exists() or len(output.read_text().splitlines()) < 2) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            pids = [int(p) for p in output.read_text().splitlines()]
            assert len(pids) >= 2
            with pytest.raises(ProcessLookupError):
                os.killpg(pids[0], 0)
            assert sibling.starts == 1 and sibling.process.poll() is None
            assert tick.last_exit_code == 124
            assert tick.failures > 0
            assert tick.status()["last_run_at"] is not None
        finally:
            s.stop.set()
            await task
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.killpg(pid, 0)

    asyncio.run(scenario())


def test_successful_interval_waits_without_failure_backoff_or_catchup(short_path):
    tmp_path = short_path

    async def scenario():
        child = Component((sys.executable, "-c", "pass"), interval=0.6, run_timeout=1, exit_timeout=0.02)
        s = Supervisor({"tick": child}, tmp_path / "s", shutdown_timeout=0.5)
        task = asyncio.create_task(s.run())
        try:
            deadline = time.monotonic() + 3
            while child.last_exit_code is None and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert child.last_exit_code == 0 and child.failures == 0
            await asyncio.sleep(0.2)
            assert child.starts == 1 and child.process is None
            # A missed interval starts once; its next deadline is in the future.
            child.retry_at = child.schedule_clock() - 100
            s._tick()
            await s._wait_start(child)
            assert child.starts == 2
            assert child.retry_at > child.schedule_clock()
        finally:
            s.stop.set()
            await task

    asyncio.run(scenario())


def test_daemon_entrypoint_serves_real_control_client(short_path):
    import subprocess

    from brainlayer.migration_runtime import wait_supervisor
    from brainlayer.supervisor_adapters import SupervisorCommands

    path = manifest(short_path, ProgramArguments=[sys.executable, "-c", "import time;time.sleep(60)"])
    payload = json.loads(path.read_text())
    payload["com.brainlayer.drain"]["enabled"] = True
    path.write_text(json.dumps(payload))
    endpoint = short_path / "control.sock"
    process = subprocess.Popen(
        [sys.executable, "-m", "brainlayer.supervisor_jobs", "--manifest", str(path), "--socket", str(endpoint)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = SupervisorCommands(endpoint, lambda args: None)
    pid = None
    try:
        state = wait_supervisor(client.request, load_components(path, endpoint), grace=5)["drain"]
        pid = state["pid"]
        assert pid and state["configured"] is True and state["starts"] == 1
        assert endpoint.stat().st_mode & 0o777 == 0o600
    finally:
        process.terminate()
        stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, (stdout, stderr)
    assert not endpoint.exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)


def test_loader_preserves_child_resource_policy(tmp_path):
    child = load_components(
        manifest(
            tmp_path,
            SoftResourceLimits={"NumberOfFiles": 4096},
            Nice=10,
            LowPriorityIO=True,
            ProcessType="Background",
            ThrottleInterval=5,
        ),
        tmp_path / "s",
    )["drain"]
    assert child.policy == dict(
        SoftResourceLimits={"NumberOfFiles": 4096}, Nice=10, LowPriorityIO=True, ProcessType="Background"
    )
    assert child.backoff == 5


def test_repository_managed_plist_keys_and_policy_are_preserved(short_path):
    import plistlib
    import re

    from brainlayer.supervisor_adapters import ALIASES

    seen = set()
    for path in (Path(__file__).resolve().parents[1] / "scripts/launchd").glob("*.plist"):
        plist = plistlib.loads(re.sub(r"__[A-Z0-9_]+__", "/private/fixture", path.read_text()).encode())
        label = plist["Label"]
        if label not in ALIASES:
            continue
        seen.add(label)
        snapshot = short_path / "snapshot.json"
        snapshot.write_text(json.dumps({label: {"enabled": False, "plist": plist}}))
        snapshot.chmod(0o600)
        child = load_components(snapshot, short_path / "s")[ALIASES[label]]
        assert child.argv == tuple(plist["ProgramArguments"])
        assert child.policy == {
            k: v for k, v in plist.items() if k in {"SoftResourceLimits", "Nice", "LowPriorityIO", "ProcessType"}
        }
    assert seen == set(ALIASES) - {"com.brainlayer.wal-checkpoint-fallback"}  # Supervisor-only.


def test_cleanup_error_is_recorded_and_paused_deadline_absent(short_path):
    async def scenario():
        child = Component((sys.executable, "-c", "pass"), enabled=False, interval=10)
        child.cleanup = asyncio.get_running_loop().create_future()
        child.cleanup.set_exception(RuntimeError("private reap refusal"))
        supervisor = Supervisor({"off": child}, short_path / "s")
        supervisor._tick()
        assert "private reap refusal" in child.error and child.cleanup_retry_at > time.monotonic()
        assert child.status()["next_run_at"] is None
        child.enabled, child.paused = True, True
        assert child.status()["next_run_at"] is None

    asyncio.run(scenario())


def test_cleanup_retry_preserves_original_timeout_accounting(short_path):
    async def scenario():
        child = Component((sys.executable, "-c", "pass"), enabled=False, interval=10)
        from types import SimpleNamespace

        child.process = SimpleNamespace(poll=lambda: 7, returncode=7)
        child.last_exit_code, child.failures = 124, 1
        child.cleanup = asyncio.get_running_loop().create_future()
        child.cleanup.set_exception(RuntimeError("private cleanup failure"))
        supervisor = Supervisor({"tick": child}, short_path / "s")
        supervisor._tick()
        child.cleanup_retry_at = time.monotonic() - 1
        supervisor._tick()
        try:
            assert child.last_exit_code == 124 and child.failures == 1
            assert child.cleanup is not None
        finally:
            if child.cleanup is not None:
                child.cleanup.cancel()
                await asyncio.gather(child.cleanup, return_exceptions=True)

    asyncio.run(scenario())


def test_cleanup_retry_keeps_inflight_control_reap(short_path, monkeypatch):
    async def scenario():
        child = Component(("unused",), enabled=False)
        supervisor = Supervisor({"probe": child}, short_path / "control.sock")
        child.cleanup = asyncio.get_running_loop().create_future()
        child.cleanup.set_exception(RuntimeError("initial cleanup failure"))
        supervisor._tick()
        release, entered = asyncio.Event(), []

        async def held_reap(component):
            entered.append(asyncio.current_task())
            await release.wait()

        monkeypatch.setattr(supervisor, "_reap", held_reap)
        control = asyncio.create_task(supervisor._stop_child(child))
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            original = child.cleanup
            assert len(entered) == 1 and original is not None
            child.cleanup_retry_at = time.monotonic() - 1
            supervisor._tick()
            await asyncio.sleep(0)
            assert child.cleanup is original and len(entered) == 1
        finally:
            release.set()
            await control
            await asyncio.gather(*entered)

    asyncio.run(scenario())


def test_successful_control_reap_clears_retry_before_restart(short_path, monkeypatch):
    from types import SimpleNamespace

    async def scenario():
        child = Component(("unused",), enabled=False)
        child.process = SimpleNamespace(pid=42)
        child.cleanup_retry_at = time.monotonic() - 1
        supervisor = Supervisor({"probe": child}, short_path / "control.sock")
        forgotten = []
        supervisor.custody = SimpleNamespace(forget=lambda guard, pid: forgotten.append(pid))

        async def stopped(*args):
            pass

        monkeypatch.setattr("brainlayer.service_supervisor.stop_group", stopped)
        await supervisor._stop_child(child)
        assert forgotten == [42] and child.process is None and child.cleanup_retry_at == 0
        replacement = SimpleNamespace(poll=lambda: None)
        child.process = replacement
        supervisor._tick()
        assert child.cleanup is None and child.process is replacement

    asyncio.run(scenario())


def test_hotlane_job_resets_inherited_io_throttle():
    """S23: the hotlane's supervisor policy carries an explicit LowPriorityIO=False.

    apply_policy resets I/O to IOPOL_DEFAULT only when the key is present, and the supervisor
    itself runs I/O-throttled, so a missing key would leave the child inheriting the throttle.
    test_supervisor_child.test_explicit_policy_clears_inherited_io_in_private_child proves the
    reset in a real child; this pins that the repository plist actually requests it.
    """
    import plistlib
    from pathlib import Path

    from brainlayer.supervisor_jobs import resource_policy

    plist = plistlib.loads(
        (Path(__file__).resolve().parents[1] / "scripts/launchd/com.brainlayer.hotlane-brainbar.plist").read_bytes()
    )
    policy = resource_policy(plist)
    assert policy["LowPriorityIO"] is False
    assert policy["ProcessType"] == "Standard"


@pytest.mark.parametrize("exit_observed", [False, True])
def test_periodic_job_runs_once_after_sleep_without_waiting_another_interval(short_path, monkeypatch, exit_observed):
    """S36: 05:08 health tick, clamshell sleep, full wake at 06:34; uptime barely advanced."""
    from types import SimpleNamespace

    from brainlayer import service_supervisor as module

    wall = [1793768893.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: wall[0], monotonic=time.monotonic))

    async def scenario():
        child = Component((sys.executable, "-c", "pass"), interval=300, exit_timeout=0.02)
        s = Supervisor({"health-check": child}, short_path / "s", shutdown_timeout=0.5)
        rows = []
        tick = s._tick
        allow_ticks = [True]
        s._tick = lambda: tick() if allow_ticks[0] else None
        task = asyncio.create_task(s.run())
        try:
            deadline = time.monotonic() + 5
            while child.starts == 0:
                assert time.monotonic() < deadline
                await asyncio.sleep(0.01)
            if exit_observed:
                while child.process is not None or child.cleanup is not None:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.01)
            else:
                allow_ticks[0] = False
                while child.process.poll() is None:
                    assert time.monotonic() < deadline
                    await asyncio.sleep(0.01)
            s.diagnostics.close()
            s.diagnostics = SimpleNamespace(emit=lambda **row: rows.append(row), close=lambda: None)
            wall[0] += 86 * 60 + 13  # Sleep advances wall time, not the process uptime clock.
            tick()
            if child.cleanup:
                await child.cleanup
                tick()
            await s._wait_start(child)
            assert child.starts == 2, "An overdue health check must run immediately after wake"
            assert child.status()["next_run_at"] > wall[0]
            tick()
            assert child.starts == 2, "Missed intervals coalesce; never replay 17 health checks"
            assert any(
                r["event"] == "interval-delayed"
                and r["component"] == "health-check"
                and r["reason"] == "scheduler-gap"
                and r["delay_s"] >= 4800
                for r in rows
            )
        finally:
            s.stop.set()
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "blocked", ["paused", "configured-off", "startup-pending", "cleanup-pending", "cleanup-backoff"]
)
def test_overdue_periodic_job_reports_blocker_without_log_storm(tmp_path, monkeypatch, blocked, caplog):
    from types import SimpleNamespace

    from brainlayer import service_supervisor as module

    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[0]))

    async def scenario():
        child = Component((sys.executable, "-c", "pass"), interval=300)
        child.retry_at = 900
        child.paused = blocked == "paused"
        child.enabled = blocked != "configured-off"
        if blocked == "startup-pending":
            child.startup = asyncio.create_task(asyncio.Event().wait())
        if blocked == "cleanup-pending":
            child.cleanup = asyncio.create_task(asyncio.Event().wait())
            child.cleanup_retry_at = 1
        if blocked == "cleanup-backoff":
            child.cleanup_retry_at = 2000
        s = Supervisor({"health-check": child}, tmp_path / "s")
        s.recovery_at = 3000
        rows = []
        s.diagnostics = SimpleNamespace(emit=lambda **row: rows.append(row))
        try:
            for _ in range(10):
                s._tick()
            assert [(r["component"], r["reason"], r["delay_s"]) for r in rows] == [("health-check", blocked, 100)]
            assert f"periodic component health-check delayed 100.0s: {blocked}" in caplog.text
            clock[0] += 60
            s._tick()
            assert len(rows) == 2
        finally:
            for task in [child.startup, child.cleanup]:
                if task:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_periodic_launch_logs_maintenance_lease_delay(short_path):
    from types import SimpleNamespace

    from brainlayer.supervisor_guard import component_gate

    child = Component((sys.executable, "-c", "pass"), interval=300)
    s = Supervisor({"health-check": child}, short_path / "s")
    rows = []
    s.diagnostics = SimpleNamespace(emit=lambda **row: rows.append(row))
    with component_gate(short_path, exclusive=True):
        s._start(child)
        s._start(child)
    assert child.process is None and child.starts == 0
    assert [(r["component"], r["reason"]) for r in rows] == [("health-check", "maintenance-lease-held")]


def test_periodic_launch_failure_reports_retry_reason(short_path):
    from types import SimpleNamespace

    child = Component(("unused",), interval=300, stdout=short_path / "absent" / "out")
    s = Supervisor({"health-check": child}, short_path / "s")
    rows = []
    s.diagnostics = SimpleNamespace(emit=lambda **row: rows.append(row))
    s._start(child)
    assert child.process is None and child.failures == 1
    assert rows[0]["reason"] == "startup-failed" and rows[0]["error"] == child.error
