"""Custody primitives use private directories and real process groups."""

import os
import signal
import socket
import subprocess
import sys

import pytest

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
        assert (tmp_path / "groups.json").read_text() == "{}"
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def test_mismatched_birth_never_signals_unrelated_process(tmp_path):
    child = subprocess.Popen((sys.executable, "-c", "import time; time.sleep(60)"), start_new_session=True)
    try:
        with Custody(tmp_path) as custody:
            custody.record(child.pid, "different-birth")
            with pytest.raises(RuntimeError, match="identity"):
                custody.recover(0.1)
        assert child.poll() is None
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


def test_guard_kills_frozen_exact_parent_and_reaps_before_unlock(tmp_path):
    import select
    import time

    code = """
import os,sys,time
from pathlib import Path
from brainlayer.supervisor_guard import Custody,launch_guard,gated_spawn,start_time
with Custody(Path(sys.argv[1])) as custody:
 guard=launch_guard(custody,.3,.1)
 child,gate=gated_spawn((sys.executable,'-c','import time; time.sleep(60)'),os.environ,None,None)
 custody.record(child.pid,start_time(child.pid))
 guard.stdin.write(('+%s\\nH\\n'%child.pid).encode()); guard.stdin.flush()
 os.write(gate,b'R'); os.close(gate)
 print(child.pid,flush=True)
 while True:
  guard.stdin.write(b'H\\n'); guard.stdin.flush(); time.sleep(.05)
"""
    parent = subprocess.Popen((sys.executable, "-c", code, str(tmp_path)), stdout=subprocess.PIPE)
    group = None
    try:
        assert select.select([parent.stdout], [], [], 3)[0]
        group = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGSTOP)  # No SIGCONT, including in cleanup.
        assert parent.wait(timeout=2) == -signal.SIGKILL
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
