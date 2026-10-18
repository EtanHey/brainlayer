"""Natural-exit fixtures: no signals, production sockets, or unbounded children."""

from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    import threading
    from pathlib import Path

    def expire():
        try:
            Path("bridge-expired").write_text("self-expiry")
        finally:
            os._exit(90)

    watchdog = threading.Timer(6, expire)
    watchdog.daemon = True
    watchdog.start()

    from brainlayer import mcp_stdio_bridge as bridge

    real_run_bridge = bridge.run_bridge
    def ready(*args, **kwargs):
        Path("ready").write_text(str(os.getppid()))
        return real_run_bridge(*args, **kwargs)

    bridge.run_bridge = ready
    bridge.main()
    """
)
PARENT_SCRIPT = textwrap.dedent(
    """
    import os
    import subprocess
    import sys
    import threading
    from pathlib import Path

    def expire():
        try:
            Path("parent-expired").write_text("self-expiry")
        finally:
            os._exit(91)

    watchdog = threading.Timer(10, expire)
    watchdog.daemon = True
    watchdog.start()
    child = subprocess.Popen(
        [sys.executable, "-c", sys.argv[1]],
        stdin=int(sys.argv[2]), stdout=int(sys.argv[3]),
    )
    os.close(int(sys.argv[2]))
    os.close(int(sys.argv[3]))
    if sys.stdin.buffer.read(1) == b"w":
        Path("child-status").write_text(str(child.wait(timeout=7)))
    """
)


def _wait_for_eof(fd: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        readable, _, _ = select.select([fd], [], [], min(0.05, max(0, deadline - time.monotonic())))
        if readable and os.read(fd, 4096) == b"":
            return True
    return False


@pytest.mark.parametrize("mode", ["connected", "reconnecting", "normal-eof", "disabled"])
def test_bridge_natural_parent_exit(tmp_path, monkeypatch, mode, record_property):
    if mode != "normal-eof" and not hasattr(select, "kqueue"):
        pytest.skip("parent-exit watcher requires BSD/macOS kqueue")

    # Relative Unix paths avoid AF_UNIX's short pathname limit, even with a long
    # operator-selected pytest artifact directory. Every child inherits this cwd.
    monkeypatch.chdir(tmp_path)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if mode != "reconnecting":
        listener.bind("backend.sock")
        listener.listen(1)
        listener.settimeout(2)
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    env = {
        "HOME": str(tmp_path),
        "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "tests/socket_hygiene"), str(REPO_ROOT / "src"))),
        "BRAINLAYER_FORBID_BRAINBAR_SOCKET": "1",
        "BRAINLAYER_FORBID_EMBEDDING_MODEL": "1",
        "BRAINLAYER_MCP_SOCKET": "backend.sock",
        "BRAINLAYER_MCP_RECONNECT_MS": "10",
        "BRAINLAYER_MCP_MAX_RECONNECT_MS": "20",
        "BRAINLAYER_MCP_STDIN_EOF_DRAIN_MS": "50",
        "BRAINLAYER_DISABLE_PARENT_DEATH_WATCH": "1" if mode == "disabled" else "0",
    }
    peer = None
    parent = None
    observation = {"mode": mode}
    with (tmp_path / "fixture.log").open("wb") as log:
        try:
            parent = subprocess.Popen(
                [sys.executable, "-c", PARENT_SCRIPT, BRIDGE_SCRIPT, str(stdin_r), str(stdout_w)],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=log,
                pass_fds=(stdin_r, stdout_w),
                env=env,
            )
            os.close(stdin_r)
            stdin_r = -1
            os.close(stdout_w)
            stdout_w = -1
            deadline = time.monotonic() + 3
            ready_path = tmp_path / "ready"
            while time.monotonic() < deadline:
                if ready_path.exists() and ready_path.read_text() == str(parent.pid):
                    break
                time.sleep(0.01)
            assert ready_path.exists(), (tmp_path / "fixture.log").read_text()
            assert int(ready_path.read_text()) == parent.pid

            if mode != "reconnecting":
                peer, _ = listener.accept()
                peer.settimeout(2)
                notification = b'{"jsonrpc":"2.0","method":"notifications/initialized"}\n'
                os.write(stdin_w, notification)
                received = bytearray()
                while len(received) < len(notification):
                    chunk = peer.recv(len(notification) - len(received))
                    assert chunk, "bridge closed before forwarding the notification"
                    received.extend(chunk)
                assert received == notification
            else:
                # Keep the original parent alive through several reconnect turns.
                time.sleep(0.1)
            assert not _wait_for_eof(stdout_r, 0.1), "bridge exited while its parent was alive"
            assert parent.stdin is not None

            if mode == "normal-eof":
                os.close(stdin_w)
                stdin_w = -1
                exited = _wait_for_eof(stdout_r, 1)
                assert parent.poll() is None
                parent.stdin.write(b"w")
                parent.stdin.flush()
                assert parent.wait(timeout=8) == 0
                assert (tmp_path / "child-status").read_text() == "0"
            else:
                # The test retains the write end of the bridge's stdin. Parent
                # exit therefore produces no EOF and the backend stays connected.
                parent.stdin.close()
                assert parent.wait(timeout=8) == 0
                exited = _wait_for_eof(stdout_r, 1)
                observation["bridge_eof_within_1s"] = exited
                if mode == "disabled":
                    assert not exited, "explicit watcher opt-out was ignored"
                else:
                    assert exited, "bridge survived natural parent exit with stdin held open"
                    if peer is not None:
                        assert peer.recv(1) == b"", "backend connection survived bridge exit"
            record_property("mode", mode)
            record_property("parent_exit_code", parent.returncode)
            record_property("bridge_eof_within_1s", exited)
            observation["bridge_eof_within_1s"] = exited
            if mode == "normal-eof":
                assert exited, "bridge did not drain normal stdin EOF"
        finally:
            # Natural EOF is the first cleanup route, including on a RED base.
            # A self-expiring child independently bounds any regression here.
            if stdin_w != -1:
                os.close(stdin_w)
            if parent is not None:
                if parent.stdin is not None and not parent.stdin.closed:
                    parent.stdin.close()
                parent.wait(timeout=11)
                cleanup_eof = _wait_for_eof(stdout_r, 7)
                observation.update(
                    parent_exit_code=parent.returncode,
                    cleanup_eof=cleanup_eof,
                    bridge_self_expired=(tmp_path / "bridge-expired").exists(),
                    parent_self_expired=(tmp_path / "parent-expired").exists(),
                )
                (tmp_path / "observation.json").write_text(json.dumps(observation, indent=2) + "\n")
                assert cleanup_eof, "self-expiring bridge did not close stdout"
            if peer is not None:
                peer.close()
            listener.close()
            os.close(stdout_r)
            for fd in (stdin_r, stdout_w):
                if fd != -1:
                    os.close(fd)
    assert not (tmp_path / "bridge-expired").exists(), "bridge required self-expiry rescue"
    assert not (tmp_path / "parent-expired").exists(), "parent required self-expiry rescue"


def test_main_arms_parent_watcher_before_config_or_io(monkeypatch):
    from brainlayer import mcp_stdio_bridge as bridge
    from brainlayer import parent_death

    calls = []
    monkeypatch.setattr(parent_death, "install_parent_death_watcher", lambda: calls.append("watcher"))
    # Accommodate a module-level import while testing the entrypoint's ordering.
    if hasattr(bridge, "install_parent_death_watcher"):
        monkeypatch.setattr(bridge, "install_parent_death_watcher", lambda: calls.append("watcher"))
    config = object()

    def configure():
        assert calls == ["watcher"]
        calls.append("config")
        return config

    def run(actual_config):
        assert actual_config is config
        assert calls == ["watcher", "config"]
        calls.append("run")
        return 0

    monkeypatch.setattr(bridge, "config_from_env", configure)
    monkeypatch.setattr(bridge, "run_bridge", run)
    with pytest.raises(SystemExit) as result:
        bridge.main()
    assert result.value.code == 0
    assert calls == ["watcher", "config", "run"]
