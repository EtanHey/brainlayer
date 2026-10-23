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
            child.retry_at = time.monotonic() - 100
            s._tick()
            await s._wait_start(child)
            assert child.starts == 2
            assert child.retry_at > time.monotonic()
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
    assert seen == set(ALIASES)


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
