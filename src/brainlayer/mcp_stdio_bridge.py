"""Reconnecting stdio-to-Unix-socket bridge for BrainLayer MCP clients.

Agent hosts usually spawn an MCP process once and keep its stdio transport for
the lifetime of the thread. A plain `socat STDIO UNIX-CONNECT:/tmp/brainbar.sock`
process exits when BrainBar or the proxy restarts, which leaves the agent with a
dead MCP transport. This bridge keeps stdio alive, buffers outbound bytes while
the socket is down, and reconnects to the socket.
"""

from __future__ import annotations

import errno
import json
import os
import select
import socket
import sys
import time
from collections import deque
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import BinaryIO

DEFAULT_SOCKET_PATH = "/tmp/brainbar.sock"
DEFAULT_RECONNECT_MS = 250
DEFAULT_MAX_RECONNECT_MS = 2000
DEFAULT_CONNECT_TIMEOUT_MS = 5000
DEFAULT_MAX_PENDING_BYTES = 16 * 1024 * 1024
DEFAULT_STDIN_EOF_DRAIN_MS = 1000
CONTENT_LENGTH_SEPARATOR = b"\r\n\r\n"


@dataclass(frozen=True)
class BridgeConfig:
    socket_path: str
    reconnect_ms: int = DEFAULT_RECONNECT_MS
    max_reconnect_ms: int = DEFAULT_MAX_RECONNECT_MS
    connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES
    stdin_eof_drain_ms: int = DEFAULT_STDIN_EOF_DRAIN_MS


@dataclass
class PendingFrame:
    data: bytes
    offset: int = 0

    def remaining(self) -> memoryview:
        return memoryview(self.data)[self.offset :]

    def advance(self, count: int) -> bool:
        self.offset += count
        return self.offset >= len(self.data)

    def reset(self) -> None:
        self.offset = 0


def _positive_int(value: str | None, default: int) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def config_from_env(env: MutableMapping[str, str] | None = None) -> BridgeConfig:
    env = env if env is not None else os.environ
    return BridgeConfig(
        socket_path=env.get("BRAINLAYER_MCP_SOCKET") or env.get("MCPLAYER_BRAINLAYER_FRONT") or DEFAULT_SOCKET_PATH,
        reconnect_ms=_positive_int(env.get("BRAINLAYER_MCP_RECONNECT_MS"), DEFAULT_RECONNECT_MS),
        max_reconnect_ms=_positive_int(env.get("BRAINLAYER_MCP_MAX_RECONNECT_MS"), DEFAULT_MAX_RECONNECT_MS),
        connect_timeout_ms=_positive_int(
            env.get("BRAINLAYER_MCP_CONNECT_TIMEOUT_MS"),
            DEFAULT_CONNECT_TIMEOUT_MS,
        ),
        max_pending_bytes=_positive_int(env.get("BRAINLAYER_MCP_MAX_PENDING_BYTES"), DEFAULT_MAX_PENDING_BYTES),
        stdin_eof_drain_ms=_positive_int(
            env.get("BRAINLAYER_MCP_STDIN_EOF_DRAIN_MS"),
            DEFAULT_STDIN_EOF_DRAIN_MS,
        ),
    )


def _schedule_reconnect(now: float, delay: float, config: BridgeConfig) -> tuple[float, float]:
    next_connect_at = now + delay
    next_delay = min(delay * 2, config.max_reconnect_ms / 1000)
    return next_connect_at, next_delay


def _connect_timed_out(connect_started_at: float | None, now: float, config: BridgeConfig) -> bool:
    if connect_started_at is None:
        return False
    return now - connect_started_at >= config.connect_timeout_ms / 1000


def _close_socket(sock: socket.socket | None) -> None:
    if sock is None:
        return
    try:
        sock.close()
    except OSError:
        pass


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        except BlockingIOError:
            select.select([], [fd], [])
            continue
        if written == 0:
            raise BrokenPipeError("stdout write returned 0 bytes")
        view = view[written:]


def _frame_expects_response(data: bytes) -> bool:
    try:
        _headers, body = data.split(CONTENT_LENGTH_SEPARATOR, 1)
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return True
    return isinstance(payload, dict) and "id" in payload


def _content_length_frame_size(buffer: bytearray) -> int | None:
    if not bytes(buffer[:32]).lower().startswith(b"content-length:"):
        return None
    separator = buffer.find(CONTENT_LENGTH_SEPARATOR)
    if separator == -1:
        return None
    headers = bytes(buffer[:separator]).decode("ascii", errors="ignore")
    content_length: int | None = None
    for line in headers.split("\r\n"):
        name, _, value = line.partition(":")
        if name.lower() != "content-length":
            continue
        try:
            content_length = int(value.strip())
        except ValueError:
            return separator + len(CONTENT_LENGTH_SEPARATOR)
        break
    if content_length is None or content_length <= 0:
        return separator + len(CONTENT_LENGTH_SEPARATOR)
    frame_size = separator + len(CONTENT_LENGTH_SEPARATOR) + content_length
    return frame_size if len(buffer) >= frame_size else None


def _consume_complete_backend_frames(buffer: bytearray) -> int:
    complete = 0
    while buffer:
        frame_size = _content_length_frame_size(buffer)
        if frame_size is None:
            if bytes(buffer[:32]).lower().startswith(b"content-length:"):
                return complete
            newline = buffer.find(b"\n")
            if newline == -1:
                return complete
            frame_size = newline + 1
        complete += 1
        del buffer[:frame_size]
    return complete


def _enqueue_complete_frames(buffer: bytearray, pending: deque[PendingFrame], *, stdin_eof: bool = False) -> None:
    while buffer:
        frame_size = _content_length_frame_size(buffer)
        if frame_size is None:
            if bytes(buffer[:32]).lower().startswith(b"content-length:"):
                if not stdin_eof:
                    return
                frame_size = len(buffer)
            else:
                newline = buffer.find(b"\n")
                if newline == -1:
                    if not stdin_eof:
                        return
                    frame_size = len(buffer)
                else:
                    frame_size = newline + 1
        pending.append(PendingFrame(bytes(buffer[:frame_size])))
        del buffer[:frame_size]


def run_bridge(
    config: BridgeConfig,
    *,
    stdin: BinaryIO | None = None,
    stdout: BinaryIO | None = None,
    stderr: BinaryIO | None = None,
) -> int:
    stdin = stdin or sys.stdin.buffer
    stdout = stdout or sys.stdout.buffer
    stderr = stderr or sys.stderr.buffer

    stdin_fd = stdin.fileno()
    stdout_fd = stdout.fileno()
    previous_stdin_blocking = os.get_blocking(stdin_fd)
    os.set_blocking(stdin_fd, False)

    pending: deque[PendingFrame] = deque()
    input_buffer = bytearray()
    pending_bytes = 0
    backend_response_buffer = bytearray()
    sock: socket.socket | None = None
    connected = False
    connecting = False
    connect_started_at: float | None = None
    next_connect_at = 0.0
    reconnect_delay = config.reconnect_ms / 1000
    stdin_eof = False
    eof_idle_deadline: float | None = None
    eof_response_deadline: float | None = None
    socket_data_seen_after_stdin_eof = False
    socket_data_sent_to_backend = False
    pending_responses = 0

    def disconnect(schedule: bool = True) -> None:
        nonlocal sock, connected, connecting, connect_started_at, next_connect_at, reconnect_delay, pending_responses
        _close_socket(sock)
        sock = None
        connected = False
        connecting = False
        connect_started_at = None
        pending_responses = 0
        for frame in pending:
            frame.reset()
        if schedule:
            next_connect_at, reconnect_delay = _schedule_reconnect(time.monotonic(), reconnect_delay, config)

    def start_connect(now: float) -> None:
        nonlocal sock, connected, connecting, connect_started_at, next_connect_at, reconnect_delay
        if sock is not None or now < next_connect_at:
            return
        candidate = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        candidate.setblocking(False)
        try:
            candidate.connect(config.socket_path)
        except BlockingIOError:
            sock = candidate
            connected = False
            connecting = True
            connect_started_at = now
            return
        except OSError:
            candidate.close()
            next_connect_at, reconnect_delay = _schedule_reconnect(now, reconnect_delay, config)
            return
        sock = candidate
        connected = True
        connecting = False
        connect_started_at = None
        reconnect_delay = config.reconnect_ms / 1000

    def mark_connected() -> None:
        nonlocal connected, connecting, connect_started_at, reconnect_delay
        connected = True
        connecting = False
        connect_started_at = None
        reconnect_delay = config.reconnect_ms / 1000

    try:
        while True:
            now = time.monotonic()
            if (
                stdin_eof
                and not pending
                and pending_responses == 0
                and eof_idle_deadline is not None
                and now >= eof_idle_deadline
            ):
                return 0
            if (
                stdin_eof
                and not pending
                and pending_responses > 0
                and eof_response_deadline is not None
                and now >= eof_response_deadline
            ):
                return 0
            if connecting and _connect_timed_out(connect_started_at, now, config):
                disconnect(schedule=True)
                now = time.monotonic()
            start_connect(now)

            readers = [] if stdin_eof else [stdin_fd]
            writers: list[int] = []
            sock_fd: int | None = None
            if sock is not None:
                sock_fd = sock.fileno()
                readers.append(sock_fd)
                if connecting or pending:
                    writers.append(sock_fd)

            timeout = 0.05
            if sock is None:
                timeout = max(0.0, min(0.05, next_connect_at - now))
            if stdin_eof and not pending and pending_responses == 0:
                eof_idle_deadline = eof_idle_deadline or (now + config.stdin_eof_drain_ms / 1000)
                timeout = max(0.0, min(timeout, eof_idle_deadline - now))
                eof_response_deadline = None
            elif stdin_eof and not pending and pending_responses > 0:
                response_wait_seconds = max(
                    config.connect_timeout_ms / 1000,
                    (config.stdin_eof_drain_ms / 1000) * 10,
                )
                eof_response_deadline = eof_response_deadline or (now + response_wait_seconds)
                timeout = max(0.0, min(timeout, eof_response_deadline - now))
                eof_idle_deadline = None
            else:
                eof_idle_deadline = None
                eof_response_deadline = None

            try:
                readable, writable, _ = select.select(readers, writers, [], timeout)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                raise

            if stdin_fd in readable:
                try:
                    chunk = os.read(stdin_fd, 64 * 1024)
                except BlockingIOError:
                    chunk = None
                if chunk is None:
                    continue
                if chunk == b"":
                    stdin_eof = True
                    _enqueue_complete_frames(input_buffer, pending, stdin_eof=True)
                    continue
                if pending_bytes + len(chunk) > config.max_pending_bytes:
                    stderr.write(
                        (
                            f"brainlayer-mcp-stdio-bridge: pending buffer exceeded {config.max_pending_bytes} bytes\n"
                        ).encode("utf-8")
                    )
                    stderr.flush()
                    return 1
                input_buffer.extend(chunk)
                pending_bytes += len(chunk)
                _enqueue_complete_frames(input_buffer, pending)

            if sock is None or sock_fd is None:
                continue

            if sock_fd in writable and connecting:
                error = sock.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                if error:
                    disconnect(schedule=True)
                    continue
                mark_connected()

            if sock is not None and connected and sock_fd in writable:
                while pending:
                    frame = pending[0]
                    try:
                        sent = sock.send(frame.remaining())
                    except (BlockingIOError, InterruptedError):
                        break
                    except OSError:
                        disconnect(schedule=True)
                        break
                    if sent == 0:
                        disconnect(schedule=True)
                        break
                    socket_data_sent_to_backend = True
                    if frame.advance(sent):
                        if _frame_expects_response(frame.data):
                            pending_responses += 1
                        pending_bytes -= len(frame.data)
                        pending.popleft()
                    else:
                        break

            if sock is None or sock_fd is None:
                continue

            if sock_fd in readable:
                try:
                    data = sock.recv(64 * 1024)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    disconnect(schedule=True)
                    continue
                if not data:
                    disconnect(schedule=True)
                    if stdin_eof and not pending:
                        eof_idle_deadline = time.monotonic() + config.stdin_eof_drain_ms / 1000
                    continue
                try:
                    _write_all(stdout_fd, data)
                except OSError:
                    return 0
                backend_response_buffer.extend(data)
                completed_responses = _consume_complete_backend_frames(backend_response_buffer)
                if completed_responses:
                    pending_responses = max(0, pending_responses - completed_responses)
                if stdin_eof:
                    socket_data_seen_after_stdin_eof = True
                    if pending_responses == 0:
                        eof_idle_deadline = time.monotonic() + config.stdin_eof_drain_ms / 1000
    finally:
        disconnect(schedule=False)
        os.set_blocking(stdin_fd, previous_stdin_blocking)


def main() -> None:
    raise SystemExit(run_bridge(config_from_env()))


if __name__ == "__main__":
    main()
