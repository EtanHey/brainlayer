"""Native app adapters must use existing transports and never launch authentication."""

import importlib.util
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("native_probe", SCRIPTS / "harness_probe_native.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_agy_keyring_metadata_checks_presence_without_reading_secret(monkeypatch, tmp_path):
    module = load(monkeypatch)
    from types import SimpleNamespace

    calls = []

    def metadata(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module, "subprocess", SimpleNamespace(run=metadata, DEVNULL=-3))
    assert module.agy_auth_error(tmp_path) is None
    assert calls == [
        (
            ["/usr/bin/security", "find-generic-password", "-s", "gemini", "-a", "antigravity"],
            {"stdout": module.subprocess.DEVNULL, "stderr": module.subprocess.DEVNULL, "timeout": 5},
        )
    ]


def test_agy_auth_flow_output_is_detected_without_returning_codes(monkeypatch):
    module = load(monkeypatch)
    assert module.agy_auth_flow("Open your browser to sign in: https://accounts.google.com/o/oauth2/auth?code=private")
    assert module.agy_auth_flow("Please enter the authorization code")
    assert not module.agy_auth_flow('{"type":"result","text":"probe completed"}')


def test_native_app_mcp_items_become_existing_contract_events(monkeypatch):
    module = load(monkeypatch)
    thread = {
        "id": "fresh",
        "turns": [
            {
                "status": "completed",
                "items": [
                    {
                        "type": "mcpToolCall",
                        "server": "brainlayer",
                        "tool": "brain_store",
                        "arguments": {"content": "hp123"},
                        "status": "completed",
                        "result": {"content": [{"type": "text", "text": "STORED"}]},
                    }
                ],
            }
        ],
    }
    from harness_probe_contract import parse_calls, session_id

    output = module.app_events(thread)
    assert session_id(output) == "fresh"
    assert parse_calls(output) == [{"tool": "brain_store", "args": {"content": "hp123"}, "ok": True}]


def test_missing_agy_credentials_never_launch_cli_or_auth(monkeypatch, tmp_path):
    module = load(monkeypatch)
    from types import SimpleNamespace

    import harness_probe

    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(
        harness_probe.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not start auth"))
    )
    row = harness_probe.probe("agy", SimpleNamespace(output=tmp_path))
    assert row["status"] == "FAIL"
    assert "FAIL(auth)" in row["reason"]


def test_app_socket_handles_ping_and_fragmented_json_without_real_socket(monkeypatch):
    module = load(monkeypatch)

    class Wire:
        def __init__(self):
            self.data = b'\x01\x03{"a' + b"\x89\x04ping" + b'\x80\x04":1}'
            self.sent = []

        def recv(self, n):
            result, self.data = self.data[:n], self.data[n:]
            return result

        def sendall(self, data):
            self.sent.append(data)

    app = object.__new__(module.AppSocket)
    app.sock = Wire()
    app.buffer = b""
    assert app.recv() == {"a": 1}
    pong = app.sock.sent[0]
    assert pong[0] == 0x8A and pong[1] & 128
    mask = pong[2:6]
    assert bytes(c ^ mask[i % 4] for i, c in enumerate(pong[6:])) == b"ping"


def test_app_failure_interrupts_only_its_own_turn(monkeypatch, tmp_path):
    import pytest

    module = load(monkeypatch)
    home = tmp_path / ".codex/app-server-daemon"
    home.mkdir(parents=True)
    (home / "daemon.pid").write_text('{"pid":7}')
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(module.Path, "is_socket", lambda _: True)
    monkeypatch.setattr(module, "observe", lambda _: [])
    called = []

    class App:
        sock = type("Socket", (), {"settimeout": lambda *_: None})()

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def rpc(self, method, params):
            called.append((method, params))
            if method == "thread/start":
                return {"thread": {"id": "fresh"}}
            if method == "turn/start":
                return {"turn": {"id": "own-turn"}}
            if method == "thread/read":
                raise RuntimeError("fixture connection failure")
            return {}

    monkeypatch.setattr(module, "AppSocket", App)
    with pytest.raises(RuntimeError, match="fixture"):
        module.app_probe("probe", module.time.monotonic() + 5)
    assert called[-1] == ("turn/interrupt", {"threadId": "fresh", "turnId": "own-turn"})


def test_agy_auth_flow_stops_only_owned_child_before_verification(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import harness_probe
    import harness_probe_native

    monkeypatch.setattr(harness_probe_native, "agy_auth_error", lambda _: None)
    monkeypatch.setattr(harness_probe.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(harness_probe.shutil, "which", lambda _: "/installed/agy")
    killed = []

    class Child:
        pid = 424242
        returncode = None

        def __init__(self, *a, **kw):
            kw["stderr"].write("Please enter the authorization code: private-fixture")
            kw["stderr"].flush()

        def poll(self):
            return self.returncode

        def wait(self, **kw):
            self.returncode = -15

    monkeypatch.setattr(harness_probe.subprocess, "Popen", Child)
    monkeypatch.setattr(harness_probe.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(harness_probe, "observe", lambda _: (_ for _ in ()).throw(AssertionError("must stop first")))
    monkeypatch.setattr(harness_probe, "Bridge", lambda: (_ for _ in ()).throw(AssertionError("must not verify")))
    row = harness_probe.probe("agy", SimpleNamespace(output=tmp_path, timeout=1))
    assert killed == [(424242, harness_probe.signal.SIGTERM)]
    assert row["status"] == "FAIL" and "FAIL(auth)" in row["reason"]
    assert "private-fixture" not in json.dumps(row)


def test_agy_deadline_never_signals_an_exited_child(monkeypatch):
    from types import SimpleNamespace

    module = load(monkeypatch)
    callbacks, killed = [], []
    child = SimpleNamespace(pid=1234, returncode=None)
    child.poll = lambda: child.returncode

    class Timer:
        def __init__(self, seconds, callback):
            assert seconds == 30
            callbacks.append(callback)

        def start(self):
            pass

    monkeypatch.setattr(module.threading, "Timer", Timer)
    monkeypatch.setattr(module.os, "killpg", lambda *args: killed.append(args))
    module.agy_deadline(child, 30)
    callbacks[0]()
    child.returncode = 0
    callbacks[0]()
    assert killed == [(1234, module.signal.SIGTERM)]


def test_agy_advertised_browser_tool_is_not_authentication_flow(monkeypatch):
    module = load(monkeypatch)
    inventory = json.dumps({"event": "init", "init": {"tools": ["open_browser_url"]}})
    assert not module.agy_auth_flow("", inventory)
    assert not module.agy_auth_flow("", inventory[:-3])
    assert module.agy_auth_flow("", json.dumps({"event": "authentication_required"}))
