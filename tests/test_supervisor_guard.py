"""Custody primitives use private directories and real process groups."""

import json
import os
import signal
import socket
import subprocess
import sys

import pytest

from brainlayer import supervisor_guard as guard
from brainlayer.supervisor_guard import Custody, group_exists, peer_uid, start_time


def test_lock_journal_and_birth_identity_recovery(tmp_path):
    child = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(60)"), start_new_session=True)
    try:
        with Custody(tmp_path) as custody:
            birth = start_time(child.pid)
            assert birth is not None
            custody.record(child.pid, birth)
            with pytest.raises(BlockingIOError):
                with Custody(tmp_path):
                    pass
        with Custody(tmp_path) as custody:
            custody.recover(0.2)
        child.wait(timeout=2)
        assert not group_exists(child.pid)
        assert json.loads((tmp_path / "groups.json").read_text())["groups"] == {}
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


@pytest.mark.parametrize("journal_kind", ["legacy", "current", "previous-boot"])
def test_stale_birth_or_previous_boot_starts_and_preserves_unrelated_process(tmp_path, journal_kind):
    child = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(60)"), start_new_session=True)
    try:
        boot = getattr(guard, "boot_time", lambda: "current")()
        groups = {
            str(child.pid): {
                "pgid": child.pid,
                "start_time": start_time(child.pid) if journal_kind == "previous-boot" else "different-birth",
            }
        }
        document = (
            groups
            if journal_kind == "legacy"
            else {"boot_time": boot if journal_kind == "current" else "previous-boot", "groups": groups}
        )
        (tmp_path / "groups.json").write_text(json.dumps(document))
        with Custody(tmp_path) as custody:
            custody.recover(0.1)
        assert child.poll() is None
        assert json.loads((tmp_path / "groups.json").read_text()) == {"boot_time": boot, "groups": {}}
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def test_directory_and_peer_identity(tmp_path):
    left, right = socket.socketpair()
    with left, right:
        assert peer_uid(left) == os.getuid()
    tmp_path.chmod(0o755)
    with pytest.raises(PermissionError):
        with Custody(tmp_path):
            pass


@pytest.mark.parametrize("quiesced", [False, True])
def test_guard_kills_frozen_or_overdue_quiesced_parent_and_reaps_before_unlock(tmp_path, quiesced):
    import select
    import time

    code = """
import inspect,os,sys,time
from pathlib import Path
from brainlayer.supervisor_guard import Custody,launch_guard,gated_spawn,start_time
with Custody(Path(sys.argv[1])) as custody:
 guard=launch_guard(custody,.3,.1,**({"shutdown_timeout":.1} if "shutdown_timeout" in inspect.signature(launch_guard).parameters else {}))
 child,gate=gated_spawn((sys.executable,'-c','import time; time.sleep(60)'),os.environ,None,None)
 custody.record(child.pid,start_time(child.pid))
 guard.stdin.write(('+%s\\nH\\n'%child.pid).encode()); guard.stdin.flush()
 os.write(gate,b'R'); os.close(gate)
 if sys.argv[2]=='True': guard.stdin.write(b'Q\\n'); guard.stdin.flush()
 print(child.pid,flush=True)
 while True:
  guard.stdin.write(b'H\\n'); guard.stdin.flush(); time.sleep(.05)
"""
    parent = subprocess.Popen((sys.executable, "-c", code, str(tmp_path), str(quiesced)), stdout=subprocess.PIPE)
    group = None
    try:
        assert select.select([parent.stdout], [], [], 3)[0]
        group = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGSTOP)  # No SIGCONT, including in cleanup.
        assert parent.wait(timeout=3) == -signal.SIGKILL
        deadline = time.monotonic() + 3
        while group_exists(group) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not group_exists(group)
        # The guard inherits flock until every group is gone.
        while True:
            try:
                with Custody(tmp_path) as custody:
                    custody.recover(0.1)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline
                time.sleep(0.02)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()
        if group and group_exists(group):
            os.killpg(group, signal.SIGKILL)


def test_gate_eof_prevents_execution_and_release_preserves_env_and_append_log(tmp_path):
    import time

    from brainlayer.supervisor_guard import gated_spawn

    output = tmp_path / "child.log"
    output.write_text("existing\n")
    argv = (sys.executable, "-c", "import os; print(os.environ['BL_GATE_TEST'],flush=True)")
    with output.open("ab") as log:
        child, gate = gated_spawn(argv, os.environ | {"BL_GATE_TEST": "released"}, log, None)
        time.sleep(0.1)
        assert output.read_text() == "existing\n"
        os.close(gate)
        assert child.wait(timeout=2) != 0
        assert output.read_text() == "existing\n"
        child, gate = gated_spawn(argv, os.environ | {"BL_GATE_TEST": "released"}, log, None)
        os.write(gate, b"R")
        os.close(gate)
        assert child.wait(timeout=2) == 0
    assert output.read_text() == "existing\nreleased\n"


def test_unreadable_live_birth_fails_without_signal(tmp_path, monkeypatch):
    child = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(60)"), start_new_session=True)
    try:
        with Custody(tmp_path) as custody:
            custody.record(child.pid, start_time(child.pid))
            monkeypatch.setattr(guard, "start_time", lambda pid: None)
            with pytest.raises(RuntimeError, match="identity"):
                custody.recover(0.1)
            assert str(child.pid) in custody.groups and child.poll() is None
    finally:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def test_unkillable_group_reap_has_deadline_and_reports_it(tmp_path):
    import select

    code = """
import inspect,os,sys,subprocess
from brainlayer.supervisor_guard import reap_groups
child=subprocess.Popen((sys.executable,'-c','import time; time.sleep(60)'),start_new_session=True)
print(child.pid,flush=True)
killpg=os.killpg
os.killpg=lambda pgid,sig: killpg(pgid,0) if sig==0 else None
reap_groups({child.pid},.01,**({'timeout':.1} if 'timeout' in inspect.signature(reap_groups).parameters else {}))
"""
    parent = subprocess.Popen((sys.executable, "-c", code), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    group = None
    try:
        assert select.select([parent.stdout], [], [], 3)[0]
        group = int(parent.stdout.readline())
        assert parent.wait(timeout=2) != 0
        assert b"deadline" in parent.stderr.readline().lower()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()
        if group:
            os.killpg(group, signal.SIGKILL)


@pytest.mark.parametrize("delay,budget,ready", [(5.2, None, True), (0.3, 0.1, False)])
def test_guard_startup_delayed_readiness_is_bounded(tmp_path, monkeypatch, delay, budget, ready):
    import time

    original = subprocess.Popen
    guards = []

    def delayed(argv, **kwargs):
        command = list(argv)
        command[2] = f"import time; time.sleep({delay}); " + command[2]
        process = original(command, **kwargs)
        guards.append(process)
        return process

    monkeypatch.setattr(guard.subprocess, "Popen", delayed)
    options = {} if budget is None else {"startup_timeout": budget}
    with Custody(tmp_path) as custody:
        began = time.monotonic()
        try:
            if ready:
                child = guard.launch_guard(custody, 30, 0.1, **options)
                child.stdin.close()
                assert child.wait(timeout=2) == 0
                assert time.monotonic() - began >= delay
            else:
                with pytest.raises(RuntimeError, match="guard startup failed.*timeout"):
                    guard.launch_guard(custody, 30, 0.1, **options)
                assert time.monotonic() - began < 2
        finally:
            for child in guards:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)
