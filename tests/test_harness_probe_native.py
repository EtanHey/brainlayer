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


def test_agy_missing_and_expired_tokens_fail_without_process(monkeypatch, tmp_path):
    module = load(monkeypatch)
    assert "missing" in module.agy_auth_error(tmp_path)
    p = tmp_path / ".gemini/antigravity-cli/antigravity-oauth-token"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"token": {"expiry": "2000-01-01T00:00:00Z", "access_token": "fixture"}}))
    assert "expired" in module.agy_auth_error(tmp_path)


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
