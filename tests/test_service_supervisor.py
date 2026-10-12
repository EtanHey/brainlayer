"""Inactive supervisor acceptance with real owned processes and Unix clients."""

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from brainlayer.service_supervisor import Component, Supervisor


@pytest.fixture
def control_dir():
    with tempfile.TemporaryDirectory(prefix="bl-supervisor-", dir="/tmp") as directory:
        yield Path(directory)


async def request(path, action="status", component="*"):
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write(json.dumps({"action": action, "component": component}).encode() + b"\n")
    await writer.drain()
    reply = json.loads(await asyncio.wait_for(reader.readline(), 2))
    writer.close()
    await writer.wait_closed()
    return reply


async def eventually(probe):
    async with asyncio.timeout(5):
        while not await probe():
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_crash_backoff_control_and_off_state(tmp_path, control_dir):
    path = control_dir / "control.sock"
    sleeper = (sys.executable, "-c", "import time; time.sleep(60)")
    supervisor = Supervisor(
        {"watch": Component(sleeper), "drain": Component(sleeper), "off": Component(sleeper, enabled=False)},
        path,
        backoff=0.4,
    )
    task = asyncio.create_task(supervisor.run(install_signals=True))
    try:

        async def running():
            return path.exists() and (await request(path))["components"]["watch"]["pid"] is not None

        await eventually(running)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        initial = (await request(path))["components"]
        assert initial["off"]["configured"] is False and initial["off"]["pid"] is None
        assert "error" in await request(path, "resume", "off")
        os.kill(initial["watch"]["pid"], signal.SIGKILL)

        async def failed():
            return (await request(path))["components"]["watch"]["failures"] == 1

        await eventually(failed)
        await asyncio.sleep(0.1)
        snapshot = (await request(path))["components"]
        assert snapshot["watch"]["starts"] == 1 and snapshot["watch"]["pid"] is None
        assert snapshot["drain"]["pid"] == initial["drain"]["pid"]

        async def restarted():
            return (await request(path))["components"]["watch"]["starts"] == 2

        await eventually(restarted)
        paused = await request(path, "pause", "watch")
        assert paused["components"]["watch"]["paused"] and paused["components"]["watch"]["pid"] is None
        await asyncio.sleep(0.5)
        assert (await request(path))["components"]["watch"]["starts"] == 2
        assert (await request(path, "resume", "watch"))["components"]["watch"]["pid"]
        before = (await request(path))["components"]["watch"]["pid"]
        assert (await request(path, "restart", "watch"))["components"]["watch"]["pid"] != before
        assert "error" in await request(path, "delete", "watch")
        duplicate = Supervisor({}, path)
        with pytest.raises(FileExistsError):
            await duplicate.run()
        final = (await request(path))["components"]
        assert final["drain"]["pid"] == initial["drain"]["pid"]
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.wait_for(task, 5)
        for name in ("watch", "drain"):
            with pytest.raises(ProcessLookupError):
                os.killpg(final[name]["pid"], 0)
    finally:
        supervisor.stop.set()
        await asyncio.wait_for(task, 5)
    assert not path.exists()


@pytest.mark.asyncio
async def test_pause_reaps_term_resistant_group_and_keeps_logs(tmp_path, control_dir):
    pidfile = tmp_path / "grandchild.pid"
    log = tmp_path / "child.log"
    resistant = (
        "import signal,time,os; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"
    )
    code = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{resistant!r}]); print('started',flush=True); time.sleep(60)"
    path = control_dir / "control.sock"
    supervisor = Supervisor({"watch": Component((sys.executable, "-c", code), stdout=log, exit_timeout=0.2)}, path)
    task = asyncio.create_task(supervisor.run())
    try:

        async def ready():
            return pidfile.exists() and log.exists() and "started" in log.read_text()

        await eventually(ready)
        group = (await request(path))["components"]["watch"]["pid"]
        reply = await request(path, "pause", "watch")
        assert "error" not in reply and reply["components"]["watch"]["pid"] is None
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)
        assert log.read_text() == "started\n"
    finally:
        supervisor.stop.set()
        await asyncio.wait_for(task, 5)


@pytest.mark.asyncio
async def test_guard_failure_is_fatal_and_shutdown_reaps_children(tmp_path, control_dir):
    path = control_dir / "control.sock"
    supervisor = Supervisor(
        {"watch": Component((sys.executable, "-c", "import time; time.sleep(60)"))}, path, guard_timeout=0.2
    )
    task = asyncio.create_task(supervisor.run())

    async def ready():
        return path.exists() and (await request(path))["components"]["watch"]["pid"] is not None

    try:
        await eventually(ready)
        group = (await request(path))["components"]["watch"]["pid"]
        wake = subprocess.Popen(
            (sys.executable, "-c", f"import os,signal,time; time.sleep(0.5); os.kill({os.getpid()},signal.SIGCONT)")
        )
        os.kill(os.getpid(), signal.SIGSTOP)  # Suspend this test process; the separate guard must expire.
        wake.wait(timeout=2)
        with pytest.raises(RuntimeError, match="guard"):
            await asyncio.wait_for(task, 5)
        with pytest.raises(ProcessLookupError):
            os.killpg(group, 0)
        assert not path.exists()
    finally:
        supervisor.stop.set()
        if not task.done():
            await asyncio.wait_for(task, 5)


def test_retired_component_refused_and_import_does_not_activate(tmp_path):
    for name in ("enrichment", "com.brainlayer.enrichment"):
        with pytest.raises(ValueError, match="retired"):
            Supervisor({name: Component((sys.executable,))}, tmp_path / "unused.sock")
    inactive = Supervisor({"watch": Component((sys.executable,))}, tmp_path / "unused.sock")
    assert inactive.components["watch"].process is None
    assert not (tmp_path / "unused.sock").exists()
