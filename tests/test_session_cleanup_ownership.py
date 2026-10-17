"""Record reconciliation must never signal any process."""

import ast
import errno
import importlib.util
import json
import os
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

    def forbidden(*args, **kwargs):
        pytest.fail("session cleanup must never send signals")

    monkeypatch.setattr(mod.os, "kill", forbidden)
    monkeypatch.setattr(mod.os, "killpg", forbidden)
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

    monkeypatch.setattr(subprocess, "check_output", output)
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


def test_live_owned_orphan_is_logged_without_signal(hook, monkeypatch, caplog):
    path = record(hook, identity())
    monkeypatch.setattr(hook, "_process_identity", lambda pid: identity())
    assert hook.cleanup_stale_mcp() == 1
    assert path.exists()
    assert "901" in caplog.text
    assert "no signal" in caplog.text


def assert_no_signal_calls(source):
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    forbidden = {"subprocess", "signal", "pty", "multiprocessing", "asyncio.subprocess", "shlex"}
    os_names = {"os"}

    def forbidden_os(name):
        return name in {"system", "popen", "startfile"} or name.startswith(("exec", "spawn", "posix_spawn", "fork"))

    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
                if node.module == "os":
                    assert all(alias.name != "*" and not forbidden_os(alias.name) for alias in node.names)
            else:
                os_names.update(alias.asname or "os" for alias in node.names if alias.name == "os")
            assert not any(name == bad or name.startswith(bad + ".") for name in modules for bad in forbidden)
        for field, value in ast.iter_fields(node):
            if field in {"id", "attr", "name", "arg", "asname", "names", "rest", "module"}:
                values = value if isinstance(value, list) else [value]
                assert not any(isinstance(name, str) and "kill" in name.casefold() for name in values)
    for node in nodes:
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in os_names:
            assert not forbidden_os(node.attr)


@pytest.mark.parametrize("source", [None, "print('kill')", "import os; os.getpid()", "import json; json.dumps({})"])
def test_no_signal_calls_exist_in_hook(source):
    assert_no_signal_calls(HOOK.read_text() if source is None else source)


@pytest.mark.parametrize(
    "source",
    [
        "import math as kill; os.kill(123, 15)",
        "os.killpg(123, 15)",
        "from os import kill; kill(123, 15)",
        "from os import kill as stop; stop(123, 15)",
        "signal.pthread_kill(123, 15)",
        "from subprocess import run as launch; launch(['/bin/kill', '123'])",
        "import math as run; subprocess.run(args=['/bin/kill', '123'])",
        "os.system('true;kill 123')",
    ]
    + [f"subprocess.run([{command!r}, '123'])" for command in ("kill", "/usr/bin/pkill", "/usr/bin/killall")]
    + [f"os.system({(command + ' 123')!r})" for command in ("/bin/kill", "pkill", "killall")]
    + [
        "from os import kill\ndef dangerous():\n    kill(123, 15)\ndef unrelated():\n    import math as kill",
        "from os import kill as stop\ndef dangerous():\n    stop(123, 15)\ndef unrelated():\n    import math as stop",
        "from subprocess import run as launch\ndef dangerous():\n    launch(['kill', '123'])\ndef unrelated():\n    import math as launch",
        "def dangerous():\n    subprocess.getoutput('kill 123')",
        "def dangerous():\n    subprocess.getstatusoutput('pkill fixture')",
    ],
)
def test_signal_guard_rejects_negative_controls(source):
    # Preserve all 14 original snippets and Sol's five; supply their required imports.
    names = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    imports = "".join(f"import {name}\n" for name in ("os", "signal", "subprocess") if name in names)
    with pytest.raises(AssertionError):
        assert_no_signal_calls(imports + source)


@pytest.mark.parametrize(
    "source",
    [
        form.format(module=module)
        for module in ("subprocess", "signal", "pty", "multiprocessing", "asyncio.subprocess", "shlex")
        for form in ("import {module} as helper", "from {module} import member as helper")
    ]
    + ["from asyncio import subprocess", "from os import system as launch", "from os import *"]
    + [
        f"import os as operating; operating.{name}"
        for name in ("system", "popen", "execve", "spawnv", "posix_spawn", "fork", "forkpty", "startfile")
    ]
    + [
        "KILLER = 0",
        "obj.KilL_switch",
        "import math as KillAlias",
        "def overkill(): pass",
        "def f(kill_flag): pass",
        "class Skill: pass",
        "f(kill_flag=True)",
        "try: pass\nexcept Exception as kill_error: pass",
        "def f():\n    global kill_state",
    ],
)
def test_scope_free_policy_rejects_forbidden_ast(source):
    with pytest.raises(AssertionError):
        assert_no_signal_calls(source)


def test_hook_has_no_optional_launcher(hook):
    assert not hasattr(hook, "start_owned_mcp")
    assert not hasattr(hook, "subprocess")


@pytest.mark.parametrize("state", ["absent", "changed-birth"])
def test_stale_record_removed_and_pid_can_register_again(hook, monkeypatch, state):
    path = record(hook, identity())

    def current(pid):
        if state == "absent":
            raise ProcessLookupError(errno.ESRCH, "gone")
        return identity(start="new-birth")

    monkeypatch.setattr(hook, "_process_identity", current)
    assert hook.cleanup_stale_mcp() == 0
    assert not path.exists()
    monkeypatch.setattr(hook, "_process_identity", lambda pid: identity(ppid=os.getpid(), start="new-birth"))
    assert hook.record_owned_child(901) is True


@pytest.mark.parametrize("state", ["unknown", "permission"])
def test_unknown_process_identity_preserves_record(hook, monkeypatch, state):
    path = record(hook, identity())

    def current(pid):
        if state == "permission":
            raise PermissionError(errno.EPERM, "not proof of absence")
        return None

    monkeypatch.setattr(hook, "_process_identity", current)
    assert hook.cleanup_stale_mcp() == 0
    assert path.exists()


@pytest.mark.parametrize("operation", ["fsync", "dump"])
def test_failed_registration_removes_its_partial_record(hook, monkeypatch, operation):
    monkeypatch.setattr(hook, "_process_identity", lambda pid: identity(ppid=os.getpid()))

    def failed(*args, **kwargs):
        raise OSError("injected registration failure")

    monkeypatch.setattr(hook.os if operation == "fsync" else hook.json, operation, failed)
    assert hook.record_owned_child(901) is False
    assert not (hook.OWNERSHIP_DIR / "901.json").exists()


def test_failed_registration_does_not_remove_an_existing_record(hook, monkeypatch):
    path = record(hook, identity())
    before = path.read_bytes()
    monkeypatch.setattr(hook, "_process_identity", lambda pid: identity(ppid=os.getpid()))
    assert hook.record_owned_child(901) is False
    assert path.read_bytes() == before


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


def test_recorded_orphan_exits_naturally_without_signals(hook):
    # The scratch helper launches and records its own child using the real hook. After it
    # exits, cleanup reports it and the child expires naturally, without signals.
    helper = """import importlib.util, pathlib, subprocess, sys
spec = importlib.util.spec_from_file_location('cleanup', sys.argv[1])
hook = importlib.util.module_from_spec(spec); spec.loader.exec_module(hook)
hook.OWNERSHIP_DIR = pathlib.Path(sys.argv[2])
p = subprocess.Popen(['/bin/sleep', '1'], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, start_new_session=True)
assert hook.record_owned_child(p.pid)
print(p.pid, flush=True)
"""
    launched = subprocess.run(
        [sys.executable, "-c", helper, str(HOOK), str(hook.OWNERSHIP_DIR)], check=True, text=True, capture_output=True
    )
    pid = int(launched.stdout.strip())
    deadline = time.monotonic() + 3
    while (current := hook._process_identity(pid)) and current["ppid"] != 1:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert current is not None
    assert hook.cleanup_stale_mcp() == 1
    assert hook._process_identity(pid) is not None
    while True:
        try:
            if hook._process_identity(pid) is None:
                break
        except ProcessLookupError:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.parametrize("error", [errno.ESRCH, errno.EPERM, errno.EINVAL])
def test_darwin_only_esrch_proves_absence(hook, monkeypatch, error):
    from types import SimpleNamespace

    def failed(*args):
        hook.ctypes.set_errno(error)
        return 0

    monkeypatch.setattr(hook, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(hook.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(proc_pidinfo=failed))
    if error == errno.ESRCH:
        with pytest.raises(ProcessLookupError):
            hook._process_identity(901)
    else:
        assert hook._process_identity(901) is None
