from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("registration_race", [False, True])
def test_parent_watcher_keeps_original_pid_during_registration(monkeypatch, registration_race):
    import errno

    from brainlayer import parent_death

    monkeypatch.delenv("BRAINLAYER_DISABLE_PARENT_DEATH_WATCH", raising=False)
    parent_reads = []
    registrations = []
    threads = []

    def getppid():
        parent_reads.append(True)
        return 12345 if len(parent_reads) == 1 else 1

    def self_exit(code):
        raise SystemExit(code)

    class Queue:
        def control(self, changes, count, timeout):
            if changes:
                registrations.extend(changes)
                assert (count, timeout) == (0, 0)
                if registration_race:
                    raise ProcessLookupError(errno.ESRCH, "synthetic original parent exited")
            else:
                assert (count, timeout) == (1, None)

    class Thread:
        def __init__(self, *, target, name, daemon):
            assert name == "brainlayer-parent-death" and daemon
            self.target = target

        def start(self):
            threads.append(self)

    monkeypatch.setattr(parent_death.os, "getppid", getppid)
    monkeypatch.setattr(parent_death.os, "_exit", self_exit)
    monkeypatch.setattr(parent_death.select, "kqueue", Queue, raising=False)
    monkeypatch.setattr(parent_death.select, "kevent", lambda pid, **kwargs: pid, raising=False)
    for name in ("KQ_FILTER_PROC", "KQ_EV_ADD", "KQ_NOTE_EXIT"):
        monkeypatch.setattr(parent_death.select, name, 1, raising=False)
    monkeypatch.setattr(parent_death.threading, "Thread", Thread)

    if registration_race:
        with pytest.raises(SystemExit) as result:
            parent_death.install_parent_death_watcher()
        assert result.value.code == 0
        assert threads == []
    else:
        assert parent_death.install_parent_death_watcher() is True
        with pytest.raises(SystemExit) as result:
            threads[0].target()
        assert result.value.code == 0
    assert len(parent_reads) == 1
    assert registrations == [12345]


def test_parent_death_watcher_noops_without_kqueue(monkeypatch):
    from brainlayer import parent_death

    monkeypatch.setattr(parent_death.select, "kqueue", None, raising=False)
    assert parent_death.install_parent_death_watcher() is False


def test_long_running_python_sidecar_entrypoints_install_parent_death_watcher():
    entrypoints = {
        "src/brainlayer/drain.py",
        "src/brainlayer/brainbar_hybrid_helper.py",
        "src/brainlayer/mcp_stdio_bridge.py",
        "src/brainlayer/cli/__init__.py",
        "brain-bar/Scripts/brainbar_stdio_adapter.py",
    }

    for relative in entrypoints:
        text = (REPO_ROOT / relative).read_text()
        assert "install_parent_death_watcher" in text, relative


@pytest.mark.skipif(not hasattr(select, "kqueue"), reason="kqueue is only available on BSD/macOS")
def test_parent_death_watcher_exits_when_parent_dies(tmp_path):
    ready_path = tmp_path / "ready"
    parent_script = tmp_path / "parent.py"
    child_script = tmp_path / "child.py"

    child_script.write_text(
        textwrap.dedent(
            """
            import os
            import sys
            import time

            from brainlayer.parent_death import install_parent_death_watcher

            ready_path = sys.argv[1]
            installed = install_parent_death_watcher()
            if not installed:
                raise SystemExit("watcher did not install")
            with open(ready_path, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            while True:
                time.sleep(0.1)
            """
        )
    )
    parent_script.write_text(
        textwrap.dedent(
            """
            import os
            import subprocess
            import sys
            import time

            child = subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
            deadline = time.monotonic() + 5
            while not os.path.exists(sys.argv[2]):
                if time.monotonic() > deadline:
                    child.kill()
                    raise SystemExit("child did not become ready")
                time.sleep(0.01)
            print(child.pid, flush=True)
            """
        )
    )

    env = os.environ.copy()
    src_path = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    parent = subprocess.Popen(
        [sys.executable, str(parent_script), str(child_script), str(ready_path)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert parent.stdout is not None
    child_pid = int(parent.stdout.readline().strip())
    assert parent.wait(timeout=5) == 0

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    else:
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("child did not exit after parent death")
