"""T3 proof binds a fresh native timeline to its own stored provider reference."""

import importlib.util
import json
import sqlite3
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("t3_probe", SCRIPTS / "harness_probe_t3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_timeline_rejects_prose_foreign_threads_and_truncation(monkeypatch):
    module = load(monkeypatch)
    item = {
        "sourceThreadId": "fresh",
        "type": "dynamic_tool",
        "status": "completed",
        "text": json.dumps(
            {
                "toolName": "brainlayer.brain_store",
                "input": {"content": "hp123"},
                "output": [{"type": "text", "text": "STORED"}],
            }
        ),
    }
    data = {
        "thread": {"threadId": "fresh"},
        "items": [
            dict(item, sourceThreadId="foreign"),
            dict(item, textTruncated=True),
            dict(item, type="assistant_message"),
            item,
        ],
    }
    from harness_probe_contract import parse_calls

    assert parse_calls(module.events(data, "native")) == [
        {"tool": "brain_store", "args": {"content": "hp123"}, "ok": True}
    ]


def test_native_binding_is_readonly_and_exact_thread(monkeypatch, tmp_path):
    module = load(monkeypatch)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    p = tmp_path / ".t3/userdata/statev2.sqlite"
    p.parent.mkdir(parents=True)
    identity = "01a141a1-4046-7590-aae1-95f13768d974"
    with sqlite3.connect(p) as conn:
        conn.execute(
            "CREATE TABLE orchestration_v2_projection_provider_threads(thread_id TEXT,driver TEXT,payload_json TEXT)"
        )
        payload = json.dumps({"nativeThreadRef": {"driver": "codex", "nativeId": identity, "strength": "strong"}})
        conn.execute(
            "INSERT INTO orchestration_v2_projection_provider_threads VALUES (?,?,?)", ("fresh", "codex", payload)
        )
    before = p.read_bytes()
    assert module.native_id("fresh") == identity
    assert module.native_id("foreign") is None
    assert p.read_bytes() == before


def test_no_existing_t3_auth_never_starts_bridge(monkeypatch):
    import pytest

    module = load(monkeypatch)
    monkeypatch.delenv("T3_ACP_MCP_ENDPOINT", raising=False)
    monkeypatch.delenv("T3_ACP_MCP_AUTHORIZATION", raising=False)
    monkeypatch.setattr(
        module.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no server or auth"))
    )
    with pytest.raises(RuntimeError, match=r"FAIL\(auth\)"):
        module.probe("probe", module.time.monotonic() + 5)


def test_endpoint_mismatch_never_sends_credential(monkeypatch, tmp_path):
    import pytest

    module = load(monkeypatch)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    p = tmp_path / ".t3/userdata/server-runtime.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"pid": 1, "port": 3773}))
    monkeypatch.setenv("T3_ACP_MCP_ENDPOINT", "https://foreign.invalid:3773/mcp")
    monkeypatch.setenv("T3_ACP_MCP_AUTHORIZATION", "fixture-only")
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("must not send credential"))
    with pytest.raises(RuntimeError, match="does not match"):
        module.probe("probe", module.time.monotonic() + 5)


def test_failure_interrupts_only_the_launched_thread(monkeypatch, tmp_path):
    import pytest

    module = load(monkeypatch)
    monkeypatch.setattr(module.Path, "home", lambda: tmp_path)
    p = tmp_path / ".t3/userdata/server-runtime.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"pid": 1, "port": 3773}))
    monkeypatch.setenv("T3_ACP_MCP_ENDPOINT", "http://127.0.0.1:3773/mcp")
    monkeypatch.setenv("T3_ACP_MCP_AUTHORIZATION", "fixture-only")
    monkeypatch.setattr(module, "observe", lambda *_: [])
    calls = []

    class MCP:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def call(self, name, args):
            calls.append((name, args))
            if name == "t3_thread_launch":
                return {"threadId": "owned", "runId": "owned-run"}
            if name == "t3_thread_read":
                raise RuntimeError("fixture error")

    monkeypatch.setattr(module, "ExistingMCP", MCP)
    with pytest.raises(RuntimeError, match="fixture error"):
        module.probe("probe", module.time.monotonic() + 5)
    assert calls[-1] == (
        "t3_thread_interrupt",
        {"threadId": "owned", "runId": "owned-run", "clientRequestId": "probe-stop-owned"},
    )
