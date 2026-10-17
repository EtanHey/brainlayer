"""Signals are intercepted except for the hook's explicitly spawned scratch child."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "hooks/session-cleanup.py"


@pytest.fixture
def hook(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("session_cleanup_owned", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OWNERSHIP_DIR", tmp_path / "owners", raising=False)
    return mod


@pytest.mark.parametrize("parent", [1, 500])
@pytest.mark.parametrize(
    "command",
    [
        "bun fixture.ts VOICELAYER_MCP_SOCKET_PATH=/scratch/voicelayer-mcp.sock",
        "socat STDIO UNIX-CONNECT:/tmp/voicelayer-mcp.sock",
        "/opt/homebrew/Cellar/brainlayer/1.5.59/bin/brainlayer-mcp-stdio-bridge",
    ],
)
def test_unrecorded_processes_never_signalled(hook, monkeypatch, parent, command):
    # Old selector calls this for the table, pgrep and each child's command.
    def output(argv, **kwargs):
        if argv[0] == "pgrep":
            return "901\n"
        if "-eo" in argv:
            return f"PID PPID ELAPSED COMMAND\n901 {parent} 08:00:00 {command}\n"
        return command

    monkeypatch.setattr(hook.subprocess, "check_output", output)
    sends = []
    monkeypatch.setattr(hook.os, "kill", lambda *args: sends.append(args))
    assert hook.cleanup_stale_mcp() == 0
    assert sends == []


def identity(pid=901, ppid=1, start="123:456", executable="/owned/mcp", uid=None):
    return dict(pid=pid, ppid=ppid, uid=os.getuid() if uid is None else uid, start=start, executable=executable)


def record(hook, data):
    hook.OWNERSHIP_DIR.mkdir(mode=0o700, exist_ok=True)
    path = hook.OWNERSHIP_DIR / "901.json"
    path.write_text(json.dumps(data))
    path.chmod(0o600)
    return path


@pytest.mark.parametrize(
    "change", [dict(start="new-birth"), dict(executable="/other/mcp"), dict(ppid=500), dict(uid=-1)]
)
def test_record_does_not_authorize_changed_or_attached_process(hook, monkeypatch, change):
    record(hook, identity())
    monkeypatch.setattr(hook, "_process_identity", lambda pid: identity(**change))
    sends = []
    monkeypatch.setattr(hook.os, "kill", lambda *args: sends.append(args))
    assert hook.cleanup_stale_mcp() == 0
    assert sends == []


def test_identity_revalidated_immediately_before_signal(hook, monkeypatch):
    record(hook, identity())
    snapshots = iter([identity(), identity(start="reused")])
    monkeypatch.setattr(hook, "_process_identity", lambda pid: next(snapshots))
    monkeypatch.setattr(hook.os, "kill", lambda *args: pytest.fail("reused PID signalled"))
    assert hook.cleanup_stale_mcp() == 0


@pytest.mark.parametrize("unsafe", ["malformed", "symlink", "public"])
def test_unsafe_record_is_ignored(hook, monkeypatch, tmp_path, unsafe):
    path = record(hook, identity())
    if unsafe == "malformed":
        path.write_text("{")
    elif unsafe == "public":
        path.chmod(0o644)
    else:
        other = tmp_path / "foreign"
        path.rename(other)
        path.symlink_to(other)
    monkeypatch.setattr(hook.os, "kill", lambda *args: pytest.fail("unsafe record signalled"))
    assert hook.cleanup_stale_mcp() == 0


def test_cannot_adopt_foreign_process(hook, monkeypatch):
    monkeypatch.setattr(hook, "_process_identity", lambda pid: identity(ppid=999999))
    assert hook.record_owned_child(901) is False
    assert not hook.OWNERSHIP_DIR.exists()


def test_hook_spawned_orphan_is_terminated(hook, tmp_path):
    # The helper imports the real hook and launches its own child. After the helper
    # exits, only that recorded child can receive a real signal from cleanup.
    helper = """import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('cleanup', sys.argv[1])
hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
hook.OWNERSHIP_DIR = pathlib.Path(sys.argv[2])
p = hook.start_owned_mcp(['/bin/sleep', '30'])
print(p.pid, flush=True)
"""
    launched = subprocess.run(
        [sys.executable, "-c", helper, str(HOOK), str(hook.OWNERSHIP_DIR)], check=True, text=True, capture_output=True
    )
    pid = int(launched.stdout.strip())
    saved = json.loads((hook.OWNERSHIP_DIR / f"{pid}.json").read_text())
    try:
        deadline = time.monotonic() + 3
        while (current := hook._process_identity(pid)) and current["ppid"] != 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert current is not None
        assert hook.cleanup_stale_mcp() == 1
        deadline = time.monotonic() + 3
        while hook._process_identity(pid) is not None:
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        # Never signal a PID whose current birth differs from this test's child.
        actual = hook._process_identity(pid)
        if actual and all(actual[key] == saved[key] for key in ("pid", "uid", "start", "executable")):
            os.kill(pid, signal.SIGTERM)
