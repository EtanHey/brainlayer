"""Reconnecting stdio-to-Unix-socket bridge for BrainLayer MCP clients.

Agent hosts usually spawn an MCP process once and keep its stdio transport for
the lifetime of the thread. A plain `socat STDIO UNIX-CONNECT:/tmp/brainbar.sock`
process exits when BrainBar or the proxy restarts, which leaves the agent with a
dead MCP transport. This bridge keeps stdio alive, buffers outbound bytes while
the socket is down, and reconnects to the socket.
"""

from __future__ import annotations

import errno
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


@dataclass(frozen=True)
class BridgeConfig:
    socket_path: str
    reconnect_ms: int = DEFAULT_RECONNECT_MS
    max_reconnect_ms: int = DEFAULT_MAX_RECONNECT_MS
    connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS
    max_pending_bytes: int = DEFAULT_MAX_PENDING_BYTES


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

    pending: deque[memoryview] = deque()
    pending_bytes = 0
    sock: socket.socket | None = None
    connected = False
    connecting = False
    connect_started_at: float | None = None
    next_connect_at = 0.0
    reconnect_delay = config.reconnect_ms / 1000

    def disconnect(schedule: bool = True) -> None:
        nonlocal sock, connected, connecting, connect_started_at, next_connect_at, reconnect_delay
        _close_socket(sock)
        sock = None
        connected = False
        connecting = False
        connect_started_at = None
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
            if connecting and _connect_timed_out(connect_started_at, now, config):
                disconnect(schedule=True)
                now = time.monotonic()
            start_connect(now)

            readers = [stdin_fd]
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
                    return 0
                if pending_bytes + len(chunk) > config.max_pending_bytes:
                    stderr.write(
                        (
                            f"brainlayer-mcp-stdio-bridge: pending buffer exceeded {config.max_pending_bytes} bytes\n"
                        ).encode("utf-8")
                    )
                    stderr.flush()
                    return 1
                pending.append(memoryview(chunk))
                pending_bytes += len(chunk)

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
                    view = pending[0]
                    try:
                        sent = sock.send(view)
                    except (BlockingIOError, InterruptedError):
                        break
                    except OSError:
                        disconnect(schedule=True)
                        break
                    if sent == 0:
                        disconnect(schedule=True)
                        break
                    pending_bytes -= sent
                    if sent == len(view):
                        pending.popleft()
                    else:
                        pending[0] = view[sent:]
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
                    continue
                _write_all(stdout_fd, data)
    finally:
        disconnect(schedule=False)
        os.set_blocking(stdin_fd, previous_stdin_blocking)


def main() -> None:
    raise SystemExit(run_bridge(config_from_env()))


if __name__ == "__main__":
    main()
