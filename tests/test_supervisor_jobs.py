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
        tick = Component((sys.executable, "-c", code), cwd=tmp_path, interval=0.25, run_timeout=0.1, exit_timeout=0.02)
        sibling = Component((sys.executable, "-c", "import time;time.sleep(60)"), exit_timeout=0.02)
        s = Supervisor({"tick": tick, "sibling": sibling}, tmp_path / "s", guard_timeout=2, shutdown_timeout=0.5)
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
            assert child.starts == 2
            assert child.retry_at > time.monotonic()
        finally:
            s.stop.set()
            await task

    asyncio.run(scenario())


def test_daemon_entrypoint_serves_real_control_client(short_path):
    import subprocess

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
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and process.poll() is None:
            if endpoint.exists():
                try:
                    state = client.request()["drain"]
                    pid = state["pid"]
                    if pid:
                        break
                except ConnectionRefusedError:
                    pass  # bind precedes the guard readiness acknowledgment/listen.
            time.sleep(0.02)
        assert pid and state["configured"] is True and state["starts"] == 1
        assert endpoint.stat().st_mode & 0o777 == 0o600
    finally:
        process.terminate()
        stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == 0, (stdout, stderr)
    assert not endpoint.exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
