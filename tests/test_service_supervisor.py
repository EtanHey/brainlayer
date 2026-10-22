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
 'off':Component((sys.executable,'-c',drain),enabled=False)},p/'control.sock',backoff=.3)
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


def test_healthy_supervisor_survives_slow_custody_write(native, monkeypatch):
    slow_journal = """
import time
from brainlayer.supervisor_guard import Custody
save=Custody._save
delayed=False
def slow_save(custody):
 global delayed
 save(custody)
 if custody.groups and not delayed:
  delayed=True
  time.sleep(.8)
Custody._save=slow_save
"""
    monkeypatch.setattr(sys.modules[__name__], "BOOT", BOOT.replace("s=Supervisor(", slow_journal + "\ns=Supervisor("))
    _, start, request = native
    process, initial, _ = start()
    assert request()["components"]["watch"]["pid"] == initial["watch"]["pid"]
    process.terminate()
    assert process.wait(timeout=2) == 0


@pytest.mark.parametrize("death", [signal.SIGSTOP, signal.SIGKILL])
def test_guard_recovers_permanent_freeze_or_parent_death(native, death, monkeypatch):
    # Only intentional freeze acceptance needs a subsecond heartbeat deadline.
    monkeypatch.setattr(sys.modules[__name__], "BOOT", BOOT.replace("backoff=.3)", "backoff=.3,guard_timeout=.4)"))
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


@pytest.mark.parametrize(
    "stage,timeout,delay", [("spawn", 0.4, 0.9), ("record", 0.4, 0.9), ("record", 30, 31), ("forget", 0.4, 0.9)]
)
def test_slow_control_io_keeps_loop_and_guard_live(native, monkeypatch, stage, timeout, delay):
    from concurrent.futures import ThreadPoolExecutor

    from brainlayer.supervisor_adapters import SupervisorCommands

    injection = f"""
import time
import brainlayer.service_supervisor as module
from brainlayer.supervisor_guard import Custody
original = module.gated_spawn if {stage!r} == 'spawn' else getattr(Custody, {stage!r})
def slow(*args, **kwargs):
 if (p/'slow').exists() and ({stage!r} != 'record' or args[2] is not None):
  (p/'slow').unlink()
  (p/'entered').touch()
  time.sleep({delay})
 return original(*args, **kwargs)
if {stage!r} == 'spawn': module.gated_spawn = slow
else: setattr(Custody, {stage!r}, slow)
"""
    boot = BOOT.replace("s=Supervisor(", injection + "\ns=Supervisor(")
    monkeypatch.setattr(
        sys.modules[__name__], "BOOT", boot.replace("backoff=.3)", f"backoff=.3,guard_timeout={timeout})")
    )
    path, start, request = native
    process, before, _ = start()
    client = SupervisorCommands(path / "control.sock", lambda args: None)
    (path / "slow").touch()
    with ThreadPoolExecutor() as executor:
        pending = executor.submit(client.request, "restart", "drain", timeout=delay + 5)
        eventually(lambda: (path / "entered").exists())
        try:
            # A status client must still run while the mutation is waiting on real blocking I/O.
            during = request()
            assert "components" in during
            assert during["components"]["drain"]["pid"] is None  # Gated startup is not readiness.
            after = pending.result(timeout=delay + 5)
        except (OSError, ValueError):
            # Before the fix the real guard kills this exact parent at its unchanged deadline.
            if process.poll() is None:
                process.wait(timeout=timeout + 3)
            assert process.returncode != -signal.SIGKILL, f"guard killed parent during {stage}"
            raise
    assert after["drain"]["starts"] == before["drain"]["starts"] + 1
    assert request()["components"]["drain"]["pid"] == after["drain"]["pid"]
    process.terminate()
    assert process.wait(timeout=2) == 0


def test_disconnected_control_runs_mutation_once(native):
    path, start, request = native
    process, before, _ = start()
    with socket.socket(socket.AF_UNIX) as client:
        client.connect(str(path / "control.sock"))
        client.sendall(b'{"action":"restart","component":"drain"}\n')
        client.shutdown(socket.SHUT_RDWR)
    eventually(lambda: request()["components"]["drain"]["starts"] == before["drain"]["starts"] + 1)
    assert request()["components"]["watch"]["pid"] == before["watch"]["pid"]
    process.terminate()
    assert process.wait(timeout=2) == 0
    assert "Unhandled exception" not in (path / "errors").read_text()


def test_cancelled_start_retains_writer_lock_until_worker_finishes(tmp_path, monkeypatch):
    import asyncio
    import threading

    async def scenario():
        supervisor = Supervisor({}, tmp_path / "unused")
        entered, release, finished = threading.Event(), threading.Event(), []

        def write(child):
            entered.set()
            release.wait(5)
            finished.append(True)

        monkeypatch.setattr(supervisor, "_start", write)
        task = supervisor._schedule_start(Component(("unused",)))
        try:
            while not entered.is_set():
                await asyncio.sleep(0.01)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done() and supervisor.io_lock.locked() and not finished
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert finished and not supervisor.io_lock.locked() and not supervisor.operations

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["birth", "unexpected"])
def test_start_failure_keeps_heartbeating_without_releasing_or_orphaning_child(native, monkeypatch, caplog, failure):
    import asyncio

    import brainlayer.service_supervisor as module

    path, _, _ = native
    spawned, records, sent = [], [], []
    spawn, record = module.gated_spawn, module.Custody.record

    def tracked_spawn(*args, **kwargs):
        process, gate = spawn(*args, **kwargs)
        spawned.append(process)
        return process, gate

    def tracked_record(*args):
        records.append(args[1:])
        return record(*args)

    monkeypatch.setattr(module, "gated_spawn", tracked_spawn)
    monkeypatch.setattr(module.Custody, "record", tracked_record)
    monkeypatch.setattr(module, "start_time", lambda pid: None)

    async def scenario():
        marker = path / "executed"
        child = Component((sys.executable, "-c", f"open({str(marker)!r},'w').close()"))
        supervisor = Supervisor({"test": child}, path / "control.sock", backoff=60)
        send = supervisor._send

        def tracked_send(line):
            sent.append(line)
            send(line)

        def unexpected(component):
            raise ValueError("injected startup explosion")

        monkeypatch.setattr(supervisor, "_send", tracked_send)
        if failure == "unexpected":
            monkeypatch.setattr(supervisor, "_start", unexpected)
        task = asyncio.create_task(supervisor.run())

        async def until(probe):
            while not probe():
                if task.done():
                    await task
                    pytest.fail("supervisor stopped during startup failure")
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(until(lambda: child.failures == 1 and child.startup is None), 5)
            assert child.error == (
                "cannot establish child birth identity"
                if failure == "birth"
                else "startup failed: injected startup explosion"
            )
            assert child.retry_at > time.monotonic() and child.starts == 0 and child.process is None
            assert not marker.exists() and not records and not supervisor.custody.groups
            assert not any(line.startswith("+") for line in sent)
            assert len(spawned) == (failure == "birth")
            for process in spawned:
                assert process.returncode == 1 and not exists(process.pid)
                with pytest.raises(ChildProcessError):
                    os.waitpid(process.pid, os.WNOHANG)
            pulses = sent.count("H")
            await asyncio.wait_for(until(lambda: sent.count("H") >= pulses + 2), 5)
            if failure == "unexpected":
                assert "supervisor component startup failed" in caplog.text
                assert "ValueError: injected startup explosion" in caplog.text
        finally:
            supervisor.stop.set()
            try:
                await task
            finally:
                for process in spawned:
                    process.wait(timeout=2)

    asyncio.run(scenario())


@pytest.mark.parametrize("wedge", ["loop", "worker"])
def test_guard_kills_wedged_loop_or_overdue_io_and_reaps(native, monkeypatch, wedge):
    injection = f"""
import time
import brainlayer.service_supervisor as module
original = module.gated_spawn
def stuck(*args, **kwargs):
 if (p/'wedge').exists():
  while True: time.sleep(.05)
 return original(*args, **kwargs)
if {wedge!r} == 'worker': module.gated_spawn = stuck
else:
 tick = Supervisor._tick
 def blocked(self):
  if (p/'wedge').exists():
   while True: time.sleep(.05)
  return tick(self)
 Supervisor._tick = blocked
"""
    boot = BOOT.replace("s=Supervisor(", injection + "\ns=Supervisor(")
    boot = boot.replace("backoff=.3)", "backoff=.3,guard_timeout=.4,operation_timeout=.5)")
    monkeypatch.setattr(sys.modules[__name__], "BOOT", boot)
    path, start, _ = native
    process, before, _ = start()
    (path / "wedge").touch()
    if wedge == "worker":
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(path / "control.sock"))
            client.sendall(b'{"action":"restart","component":"drain"}\n')
    assert process.wait(timeout=3) == -signal.SIGKILL
    eventually(lambda: not any(exists(c["pid"]) for c in before.values() if c["pid"]))


@pytest.mark.parametrize("failure", ["write", "drain", "wait_closed"])
def test_control_disconnect_is_expected_but_other_errors_surface(tmp_path, monkeypatch, failure):
    import asyncio

    monkeypatch.setattr("brainlayer.service_supervisor.peer_uid", lambda sock: os.getuid())

    class Writer:
        closed = False

        def get_extra_info(self, key):
            return None

        def write(self, data):
            if failure == "write":
                raise BrokenPipeError("peer left")

        async def drain(self):
            if failure == "drain":
                raise ConnectionResetError("peer left")

        def close(self):
            self.closed = True

        async def wait_closed(self):
            raise BrokenPipeError("peer left")

    async def scenario():
        reader, writer = asyncio.StreamReader(), Writer()
        reader.feed_data(b'{"action":"status"}\n')
        await Supervisor({}, tmp_path / "unused")._control(reader, writer)
        assert writer.closed
        writer.wait_closed = unexpected
        reader.feed_data(b'{"action":"status"}\n')
        with pytest.raises(RuntimeError, match="unexpected"):
            await Supervisor({}, tmp_path / "unused")._control(reader, writer)

    async def unexpected():
        raise RuntimeError("unexpected")

    asyncio.run(scenario())
