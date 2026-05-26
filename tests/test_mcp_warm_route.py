import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

import pytest
from mcp.types import TextContent

from brainlayer.mcp.search_handler import _brain_search


def test_find_warm_helper_socket_skips_disappearing_socket(monkeypatch):
    mtimes = {
        "/tmp/brainbar-hybrid-old.sock": 1.0,
        "/tmp/brainbar-hybrid-new.sock": 3.0,
    }

    def fake_getmtime(path):
        if path == "/tmp/brainbar-hybrid-gone.sock":
            raise FileNotFoundError(path)
        return mtimes[path]

    monkeypatch.setattr(
        "brainlayer.mcp.search_handler.glob.glob",
        lambda _pattern: [
            "/tmp/brainbar-hybrid-old.sock",
            "/tmp/brainbar-hybrid-gone.sock",
            "/tmp/brainbar-hybrid-new.sock",
        ],
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler.os.path.getmtime", fake_getmtime)

    from brainlayer.mcp.search_handler import _find_warm_helper_socket

    assert _find_warm_helper_socket() == "/tmp/brainbar-hybrid-new.sock"


async def _cold_result(**kwargs):
    return (
        [TextContent(type="text", text=f"cold result for {kwargs['query']}")],
        {"query": kwargs["query"], "total": 0, "results": [], "path": "cold"},
    )


def _serve_one_json_line(sock_path: Path, response: dict):
    ready = threading.Event()
    received = []

    def run():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(str(sock_path))
            server.listen(1)
            ready.set()
            conn, _ = server.accept()
            with conn:
                data = b""
                while b"\n" not in data:
                    data += conn.recv(65536)
                received.append(json.loads(data.split(b"\n", 1)[0].decode("utf-8")))
                conn.sendall(json.dumps(response).encode("utf-8") + b"\n")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(1)
    return received, thread


@pytest.mark.asyncio
async def test_brain_search_uses_cold_path_when_helper_route_disabled(monkeypatch, tmp_path):
    calls = []

    async def cold_dispatch(**kwargs):
        calls.append(kwargs)
        return await _cold_result(**kwargs)

    def helper_should_not_be_called(*_args, **_kwargs):
        raise AssertionError("helper should not be called when opt-in is disabled")

    monkeypatch.delenv("BRAINLAYER_MCP_USE_HELPER", raising=False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_sentinel_path", lambda: tmp_path / "missing")
    monkeypatch.setattr("brainlayer.mcp.search_handler._find_warm_helper_socket", lambda: "/tmp/helper.sock")
    monkeypatch.setattr("brainlayer.mcp.search_handler._forward_to_helper", helper_should_not_be_called)
    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search_dispatch", cold_dispatch)

    content, structured = await _brain_search(query="warm route disabled", project="brainlayer")

    assert content[0].text == "cold result for warm route disabled"
    assert structured["path"] == "cold"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_brain_search_uses_helper_path_when_enabled_with_valid_socket(monkeypatch, tmp_path):
    sock_path = Path(f"/tmp/bl-warm-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
    received, thread = _serve_one_json_line(
        sock_path,
        {
            "ok": True,
            "text": "warm helper result",
            "metadata": {
                "structuredContent": {
                    "query": "warm route enabled",
                    "total": 1,
                    "results": [{"chunk_id": "warm-1"}],
                }
            },
        },
    )

    async def cold_dispatch(**_kwargs):
        raise AssertionError("cold path should not be called when helper succeeds")

    monkeypatch.setenv("BRAINLAYER_MCP_USE_HELPER", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._find_warm_helper_socket", lambda: str(sock_path))
    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search_dispatch", cold_dispatch)

    content, structured = await _brain_search(
        query="warm route enabled",
        project="brainlayer",
        source="all",
        tag="demo",
        importance_min=8,
        num_results=3,
        detail="compact",
    )

    thread.join(1)
    sock_path.unlink(missing_ok=True)
    assert content == [TextContent(type="text", text="warm helper result")]
    assert structured["results"] == [{"chunk_id": "warm-1"}]
    assert structured["via_helper"] is True
    assert isinstance(structured["helper_latency_ms"], float)
    assert received == [
        {
            "method": "brain_search",
            "arguments": {
                "query": "warm route enabled",
                "project": "brainlayer",
                "source": "all",
                "tag": "demo",
                "importance_min": 8,
                "num_results": 3,
                "detail": "compact",
            },
        }
    ]


@pytest.mark.asyncio
async def test_brain_search_uses_helper_path_when_sentinel_file_exists(monkeypatch, tmp_path):
    sock_path = Path(f"/tmp/bl-warm-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
    sentinel = tmp_path / "use-helper-socket"
    sentinel.touch()
    _received, thread = _serve_one_json_line(
        sock_path,
        {
            "ok": True,
            "text": "sentinel helper result",
            "metadata": {"structuredContent": {"query": "sentinel route", "total": 0, "results": []}},
        },
    )

    async def cold_dispatch(**_kwargs):
        raise AssertionError("cold path should not be called when sentinel route succeeds")

    monkeypatch.delenv("BRAINLAYER_MCP_USE_HELPER", raising=False)
    monkeypatch.setattr("brainlayer.mcp.search_handler._helper_sentinel_path", lambda: sentinel)
    monkeypatch.setattr("brainlayer.mcp.search_handler._find_warm_helper_socket", lambda: str(sock_path))
    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search_dispatch", cold_dispatch)

    content, structured = await _brain_search(query="sentinel route", project="brainlayer")

    thread.join(1)
    sock_path.unlink(missing_ok=True)
    assert content == [TextContent(type="text", text="sentinel helper result")]
    assert structured["via_helper"] is True


@pytest.mark.asyncio
async def test_brain_search_falls_back_when_helper_enabled_but_socket_missing(monkeypatch):
    calls = []

    async def cold_dispatch(**kwargs):
        calls.append(kwargs)
        return await _cold_result(**kwargs)

    monkeypatch.setenv("BRAINLAYER_MCP_USE_HELPER", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._find_warm_helper_socket", lambda: None)
    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search_dispatch", cold_dispatch)

    content, structured = await _brain_search(query="missing helper socket", project="brainlayer")

    assert content[0].text == "cold result for missing helper socket"
    assert structured["path"] == "cold"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_brain_search_falls_back_when_helper_raises(monkeypatch):
    calls = []

    async def cold_dispatch(**kwargs):
        calls.append(kwargs)
        return await _cold_result(**kwargs)

    monkeypatch.setenv("BRAINLAYER_MCP_USE_HELPER", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._find_warm_helper_socket", lambda: "/tmp/helper.sock")
    monkeypatch.setattr(
        "brainlayer.mcp.search_handler._forward_to_helper",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("helper timed out")),
    )
    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search_dispatch", cold_dispatch)

    content, structured = await _brain_search(query="helper timeout fallback", project="brainlayer")

    assert content[0].text == "cold result for helper timeout fallback"
    assert structured["path"] == "cold"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_brain_search_helper_path_returns_under_100ms_with_mock_helper(monkeypatch, tmp_path):
    sock_path = Path(f"/tmp/bl-warm-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock")
    _received, thread = _serve_one_json_line(
        sock_path,
        {
            "ok": True,
            "text": "fast warm helper result",
            "metadata": {"structuredContent": {"query": "fast helper", "total": 0, "results": []}},
        },
    )

    async def cold_dispatch(**_kwargs):
        raise AssertionError("cold path should not be called during latency-budget helper test")

    monkeypatch.setenv("BRAINLAYER_MCP_USE_HELPER", "1")
    monkeypatch.setattr("brainlayer.mcp.search_handler._find_warm_helper_socket", lambda: str(sock_path))
    monkeypatch.setattr("brainlayer.mcp.search_handler._brain_search_dispatch", cold_dispatch)

    started = time.perf_counter()
    _content, structured = await _brain_search(query="fast helper", project="brainlayer")
    elapsed_ms = (time.perf_counter() - started) * 1000

    thread.join(1)
    sock_path.unlink(missing_ok=True)
    assert elapsed_ms < 100
    assert structured["via_helper"] is True
    assert structured["helper_latency_ms"] < 100
