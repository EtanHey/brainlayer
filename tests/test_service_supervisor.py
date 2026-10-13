"""Real-process lifecycle acceptance; all signals target privately owned PIDs."""

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from brainlayer.service_supervisor import Component, Supervisor

BOOT = """
import asyncio,sys
from pathlib import Path
from brainlayer.service_supervisor import Component,Supervisor
p=Path(sys.argv[1])
resistant="import signal,time,os; signal.signal(signal.SIGTERM,signal.SIG_IGN); open(%r,'w').write(str(os.getpid())); time.sleep(60)"
leaf=resistant % str(p/'descendant')
watch="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); open(%r,'w').write('ready'); time.sleep(60)" % (leaf,str(p/'watch'))
drain=resistant % str(p/'drain')
s=Supervisor({'watch':Component((sys.executable,'-c',watch),exit_timeout=1),
 'drain':Component((sys.executable,'-c',drain),exit_timeout=.15),
 'off':Component((sys.executable,'-c',drain),enabled=False)},p/'control.sock',backoff=.3,guard_timeout=.4)
s.shutdown_timeout=.6
s.stable_uptime=.3
async def main():
 task=asyncio.create_task(s.run(install_signals=True))
 while s.guard is None and not task.done(): await asyncio.sleep(.01)
 if s.guard: (p/'guard').write_text(str(s.guard.pid))
 await task
asyncio.run(main())
"""


def exists(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def eventually(probe, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe():
            return
        time.sleep(0.02)
    assert probe()


@pytest.fixture
def native():
    with tempfile.TemporaryDirectory(prefix="bl-s-", dir="/tmp") as directory:
        path, processes, groups = Path(directory), [], set()

        def request(action="status", component="*"):
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(0.6 if action == "status" else 2)
                client.connect(str(path / "control.sock"))
                client.sendall(json.dumps(dict(action=action, component=component)).encode() + b"\n")
                response = json.loads(client.makefile("rb").readline())
                groups.update(c["pid"] for c in response.get("components", {}).values() if c["pid"])
                return response

        def start():
            for name in ("watch", "drain", "descendant", "guard"):
                (path / name).unlink(missing_ok=True)
            output = (path / "errors").open("ab")
            process = subprocess.Popen((sys.executable, "-c", BOOT, str(path)), stderr=output)
            output.close()
            processes.append(process)

            def ready():
                assert process.poll() is None
                return all((path / n).exists() for n in ("descendant", "drain", "guard"))

            try:
                # Interpreter/import startup under CI load is not a lifecycle deadline.
                eventually(ready, timeout=10)
            except AssertionError:
                missing = [n for n in ("descendant", "drain", "guard") if not (path / n).exists()]
                pytest.fail(
                    f"supervisor startup failed: returncode={process.poll()}, missing={missing}, "
                    f"stderr={(path / 'errors').read_text()}"
                )
            snapshot = request()["components"]
            return process, snapshot, int((path / "guard").read_text())

        try:
            yield path, start, request
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()  # Never SIGCONT a frozen supervisor.
                process.wait(timeout=3)
            for group in groups:
                if exists(group):
                    os.killpg(group, signal.SIGKILL)
            eventually(lambda: not any(exists(g) for g in groups))


def test_startup_reports_early_supervisor_death(native, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "BOOT", "raise RuntimeError('injected startup failure')")
    with pytest.raises(pytest.fail.Exception, match="(?s)returncode=1.*injected startup failure"):
        native[1]()


@pytest.mark.parametrize("death", [signal.SIGSTOP, signal.SIGKILL])
def test_guard_recovers_permanent_freeze_or_parent_death(native, death):
    path, start, request = native
    process, before, _ = start()
    os.kill(process.pid, death)
    assert process.wait(timeout=2) == -signal.SIGKILL
    eventually(lambda: not any(exists(c["pid"]) for c in before.values() if c["pid"]))
    replacement, after, _ = start()
    assert all(after[n]["starts"] == 1 and after[n]["pid"] != before[n]["pid"] for n in ("watch", "drain"))
    replacement.terminate()
    assert replacement.wait(timeout=2) == 0


def test_journal_recovers_if_guard_and_parent_both_die(native):
    path, start, request = native
    process, before, guard = start()
    os.kill(guard, signal.SIGKILL)
    process.kill()
    process.wait(timeout=2)
    assert any(exists(c["pid"]) for c in before.values() if c["pid"])
    replacement, after, _ = start()
    assert not any(exists(c["pid"]) for c in before.values() if c["pid"])
    assert after["off"]["pid"] is None
    replacement.terminate()
    assert replacement.wait(timeout=2) == 0


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, 0])
def test_total_shutdown_disarms_guard_and_stale_socket_restarts(native, stop_signal):
    path, start, request = native
    if not stop_signal:
        with socket.socket(socket.AF_UNIX) as stale:
            stale.bind(str(path / "control.sock"))
    process, before, _ = start()
    began = time.monotonic()
    os.kill(process.pid, stop_signal or signal.SIGTERM)
    assert process.wait(timeout=2) == 0
    assert time.monotonic() - began < 1.5
    assert not any(exists(c["pid"]) for c in before.values() if c["pid"])
    (path / "control.sock").touch()  # Stale endpoint must not cause a launchd crash loop.
    replacement, _, _ = start()
    duplicate = subprocess.run((sys.executable, "-c", BOOT, str(path)), capture_output=True, timeout=2)
    assert duplicate.returncode != 0 and b"lock" in duplicate.stderr.lower()
    assert request()["components"]["drain"]["starts"] == 1
    replacement.terminate()
    assert replacement.wait(timeout=2) == 0


def test_crashed_group_cleanup_does_not_block_controls_and_off_is_skipped(native):
    path, start, request = native
    process, before, _ = start()
    os.kill(before["watch"]["pid"], signal.SIGKILL)
    eventually(lambda: request()["components"]["watch"]["pid"] is None)
    assert request()["components"]["drain"]["pid"] == before["drain"]["pid"]
    restarted = request("restart", "drain")
    assert restarted["components"]["drain"]["pid"] != before["drain"]["pid"]
    eventually(lambda: request()["components"]["watch"]["starts"] == 2)
    eventually(lambda: request()["components"]["watch"]["failures"] == 0)
    assert "error" not in request("resume")
    assert "error" not in request("restart")
    paused = request("pause", "watch")["components"]["watch"]
    assert paused["paused"] and paused["pid"] is None
    time.sleep(0.4)
    assert request()["components"]["watch"]["starts"] == paused["starts"]
    assert "error" in request("resume", "off")
    process.terminate()
    assert process.wait(timeout=2) == 0


def test_retirement_and_constructor_remain_inactive(tmp_path):
    for name in ("enrichment", "com.brainlayer.enrichment"):
        with pytest.raises(ValueError, match="retired"):
            Supervisor({name: Component((sys.executable,))}, tmp_path / "unused.sock")
    inactive = Supervisor({"watch": Component((sys.executable,))}, tmp_path / "unused.sock")
    assert inactive.components["watch"].process is None and not inactive.path.exists()
