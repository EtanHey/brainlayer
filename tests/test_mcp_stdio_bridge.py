from __future__ import annotations

import json
import os
import select
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from brainlayer.mcp_stdio_bridge import BridgeConfig, _connect_timed_out


class RestartableLineServer:
    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.generation = 0
        self._listener: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop()
        self.generation += 1
        self._stop = threading.Event()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.socket_path))
        listener.listen()
        listener.settimeout(0.05)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        for client in self._clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        self._clients.clear()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            self._clients.append(client)
            threading.Thread(target=self._serve_client, args=(client, self.generation), daemon=True).start()

    def _serve_client(self, client: socket.socket, generation: int) -> None:
        with client:
            file = client.makefile("rwb", buffering=0)
            while not self._stop.is_set():
                try:
                    line = file.readline()
                except OSError:
                    break
                if not line:
                    break
                request = json.loads(line.decode("utf-8"))
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"generation": generation},
                }
                try:
                    file.write(json.dumps(response).encode("utf-8") + b"\n")
                except OSError:
                    break


def _read_json_line(process: subprocess.Popen[bytes], timeout: float = 5.0) -> dict[str, Any]:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout
    data = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.05)
        if process.stdout not in ready:
            if process.poll() is not None:
                raise AssertionError(f"bridge exited early with code {process.returncode}")
            continue
        chunk = os.read(process.stdout.fileno(), 1)
        if not chunk:
            raise AssertionError("bridge stdout closed before response")
        data.extend(chunk)
        if chunk == b"\n":
            return json.loads(data.decode("utf-8"))
    raise AssertionError(f"timed out waiting for bridge response, partial={data!r}")


def _write_json_line(process: subprocess.Popen[bytes], payload: dict[str, Any]) -> None:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
    process.stdin.flush()


def test_connect_timeout_uses_configured_milliseconds() -> None:
    config = BridgeConfig(socket_path="/tmp/missing.sock", connect_timeout_ms=100)

    assert not _connect_timed_out(None, 10.2, config)
    assert not _connect_timed_out(10.0, 10.099, config)
    assert _connect_timed_out(10.0, 10.101, config)


def test_write_all_preserves_complete_frame_after_partial_stdout_write(monkeypatch) -> None:
    import brainlayer.mcp_stdio_bridge as bridge

    writes: list[bytes] = []
    limits = iter([7, 3, 1024])

    def short_write(fd: int, data: bytes | memoryview) -> int:
        assert fd == 99
        chunk = bytes(data)
        written = min(next(limits), len(chunk))
        writes.append(chunk[:written])
        return written

    monkeypatch.setattr(bridge.os, "write", short_write)

    bridge._write_all(99, b'{"jsonrpc":"2.0","id":1,"result":"large-response"}\n')

    assert b"".join(writes) == b'{"jsonrpc":"2.0","id":1,"result":"large-response"}\n'


def test_stdio_bridge_reconnects_without_process_restart(tmp_path: Path) -> None:
    del tmp_path
    socket_path = Path("/tmp") / f"brainlayer-bridge-test-{os.getpid()}-{time.monotonic_ns()}.sock"
    server = RestartableLineServer(socket_path)
    server.start()

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "BRAINLAYER_MCP_SOCKET": str(socket_path),
        "BRAINLAYER_MCP_RECONNECT_MS": "25",
        "BRAINLAYER_MCP_MAX_RECONNECT_MS": "50",
        "BRAINLAYER_MCP_CONNECT_TIMEOUT_MS": "100",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "brainlayer.mcp_stdio_bridge"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        _write_json_line(process, {"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert _read_json_line(process)["result"] == {"generation": 1}

        server.stop()
        time.sleep(0.1)
        assert process.poll() is None

        _write_json_line(process, {"jsonrpc": "2.0", "id": 2, "method": "ping"})
        time.sleep(0.1)
        server.start()

        assert _read_json_line(process)["result"] == {"generation": 2}
        assert process.poll() is None
    finally:
        server.stop()
        if process.stdin is not None:
            process.stdin.close()
        process.wait(timeout=5)
