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
    def __init__(self, socket_path: Path, *, response_delay: float = 0.0) -> None:
        self.socket_path = socket_path
        self.response_delay = response_delay
        self.generation = 0
        self._listener: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._clients_lock = threading.Lock()
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
        with self._clients_lock:
            clients = list(self._clients)
            self._clients.clear()
        for client in clients:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with self._clients_lock:
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
                if self.response_delay:
                    time.sleep(self.response_delay)
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


def test_stdio_bridge_exits_cleanly_when_stdout_pipe_closes(tmp_path: Path) -> None:
    del tmp_path
    socket_path = Path("/tmp") / f"brainlayer-bridge-stdout-closed-{os.getpid()}-{time.monotonic_ns()}.sock"
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
        assert process.stdout is not None
        process.stdout.close()
        assert process.wait(timeout=5) == 0
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        assert "Traceback" not in stderr
    finally:
        server.stop()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stdio_bridge_exits_after_stdin_eof_without_pending_frames(tmp_path: Path) -> None:
    del tmp_path
    socket_path = Path("/tmp") / f"brainlayer-bridge-empty-eof-{os.getpid()}-{time.monotonic_ns()}.sock"
    server = RestartableLineServer(socket_path)
    server.start()

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "BRAINLAYER_MCP_SOCKET": str(socket_path),
        "BRAINLAYER_MCP_RECONNECT_MS": "25",
        "BRAINLAYER_MCP_MAX_RECONNECT_MS": "50",
        "BRAINLAYER_MCP_CONNECT_TIMEOUT_MS": "100",
        "BRAINLAYER_MCP_STDIN_EOF_DRAIN_MS": "50",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "brainlayer.mcp_stdio_bridge"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        deadline = time.monotonic() + 2
        while server.client_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.client_count() > 0

        assert process.stdin is not None
        process.stdin.close()

        assert process.wait(timeout=5) == 0
    finally:
        server.stop()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stdio_bridge_flushes_buffered_request_after_stdin_eof(tmp_path: Path) -> None:
    del tmp_path
    socket_path = Path("/tmp") / f"brainlayer-bridge-eof-test-{os.getpid()}-{time.monotonic_ns()}.sock"
    server = RestartableLineServer(socket_path)

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
        assert process.stdin is not None
        process.stdin.close()

        time.sleep(0.1)
        server.start()

        assert _read_json_line(process)["result"] == {"generation": 1}
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        server.stop()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stdio_bridge_waits_for_delayed_backend_response_after_stdin_eof(tmp_path: Path) -> None:
    del tmp_path
    socket_path = Path("/tmp") / f"brainlayer-delayed-eof-{os.getpid()}-{time.monotonic_ns()}.sock"
    server = RestartableLineServer(socket_path, response_delay=0.2)
    server.start()

    env = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        "BRAINLAYER_MCP_SOCKET": str(socket_path),
        "BRAINLAYER_MCP_RECONNECT_MS": "25",
        "BRAINLAYER_MCP_MAX_RECONNECT_MS": "50",
        "BRAINLAYER_MCP_CONNECT_TIMEOUT_MS": "100",
        "BRAINLAYER_MCP_STDIN_EOF_DRAIN_MS": "50",
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
        assert process.stdin is not None
        process.stdin.close()

        assert _read_json_line(process, timeout=2)["result"] == {"generation": 1}
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        server.stop()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_stdio_bridge_replays_full_frame_after_partial_socket_send_then_reconnect(monkeypatch) -> None:
    import brainlayer.mcp_stdio_bridge as bridge

    real_socketpair = socket.socketpair
    first_connection_prefixes: list[bytes] = []
    second_connection_payload = bytearray()
    response = b'{"jsonrpc":"2.0","id":7,"result":{"generation":2}}\n'

    class FlakySocket:
        def __init__(self, generation: int) -> None:
            self.generation = generation
            self.client, self.peer = real_socketpair()
            self.send_calls = 0
            self.responded = False

        def setblocking(self, flag: bool) -> None:
            self.client.setblocking(flag)

        def connect(self, _path: str) -> None:
            return None

        def fileno(self) -> int:
            return self.client.fileno()

        def getsockopt(self, _level: int, _option: int) -> int:
            return 0

        def send(self, data: bytes | memoryview) -> int:
            payload = bytes(data)
            if self.generation == 1:
                self.send_calls += 1
                if self.send_calls == 1:
                    written = min(9, len(payload))
                    first_connection_prefixes.append(payload[:written])
                    return written
                raise OSError("socket dropped after partial frame")

            second_connection_payload.extend(payload)
            if len(second_connection_payload) >= len(request) and not self.responded:
                self.peer.sendall(response)
                self.responded = True
            return len(payload)

        def recv(self, size: int) -> bytes:
            return self.client.recv(size)

        def close(self) -> None:
            self.client.close()
            self.peer.close()

    sockets = iter([FlakySocket(1), FlakySocket(2)])

    def fake_socket(*_args, **_kwargs):
        return next(sockets)

    monkeypatch.setattr(bridge.socket, "socket", fake_socket)

    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    stderr_r, stderr_w = os.pipe()
    stdin_reader = os.fdopen(stdin_r, "rb", buffering=0)
    stdout_writer = os.fdopen(stdout_w, "wb", buffering=0)
    stderr_writer = os.fdopen(stderr_w, "wb", buffering=0)
    exit_codes: list[int] = []

    def run() -> None:
        exit_codes.append(
            bridge.run_bridge(
                BridgeConfig(
                    socket_path="/tmp/brainlayer-bridge-flaky.sock",
                    reconnect_ms=1,
                    max_reconnect_ms=1,
                    connect_timeout_ms=50,
                    stdin_eof_drain_ms=10,
                ),
                stdin=stdin_reader,
                stdout=stdout_writer,
                stderr=stderr_writer,
            )
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    body = b'{"jsonrpc":"2.0","id":7,"method":"ping"}'
    request = b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
    try:
        os.write(stdin_w, request)
        os.close(stdin_w)
        stdin_w = -1

        deadline = time.monotonic() + 5
        received = bytearray()
        while time.monotonic() < deadline and b"\n" not in received:
            readable, _, _ = select.select([stdout_r], [], [], 0.05)
            if stdout_r in readable:
                received.extend(os.read(stdout_r, 1024))
        thread.join(timeout=5)

        assert json.loads(bytes(received).decode("utf-8"))["result"] == {"generation": 2}
        assert first_connection_prefixes == [request[:9]]
        assert bytes(second_connection_payload) == request
        assert exit_codes == [0]
    finally:
        if stdin_w != -1:
            os.close(stdin_w)
        os.close(stdout_r)
        os.close(stderr_r)
        stdout_writer.close()
        stderr_writer.close()
        stdin_reader.close()
