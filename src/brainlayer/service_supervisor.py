"""Explicitly started, inactive-by-default resident process supervisor.

No production commands, launchd wiring, database access, TCP or MCP endpoint.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

_GUARD = """
import os, select, sys
while True:
    if not select.select([0], [], [], float(sys.argv[1]))[0]:
        sys.exit(1)
    if not os.read(0, 65536):
        sys.exit(0)
"""


@dataclass
class Component:
    """Explicit resident command and the supervisor's custody of its process group."""

    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    stdout: Path | None = None
    stderr: Path | None = None
    enabled: bool = True
    exit_timeout: float = 60
    process: subprocess.Popen | None = field(default=None, init=False)
    paused: bool = field(default=False, init=False)
    starts: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    retry_at: float = field(default=0, init=False)
    error: str = field(default="", init=False)


class Supervisor:
    """Isolate child failures; serialize control mutations and acknowledge quiescence."""

    def __init__(self, components: dict[str, Component], path: Path, *, backoff: float = 10, guard_timeout: float = 30):
        if any(name.removeprefix("com.brainlayer.") in {"enrich", "enrichment"} for name in components):
            raise ValueError("enrichment is retired")
        if not all(math.isfinite(v) and v > 0 for v in (backoff, guard_timeout)) or any(
            not c.argv or not math.isfinite(c.exit_timeout) or c.exit_timeout < 0 for c in components.values()
        ):
            raise ValueError("commands and timing bounds must be valid")
        self.components, self.path, self.backoff, self.guard_timeout = components, path, backoff, guard_timeout
        self.stop = asyncio.Event()
        self.lock = asyncio.Lock()
        self.guard: subprocess.Popen | None = None

    def status(self) -> dict:
        """Configured OFF and paused states remain separate from observed process liveness."""
        return {
            "components": {
                name: {
                    "configured": c.enabled,
                    "paused": c.paused,
                    "pid": c.process.pid if c.process and c.process.poll() is None else None,
                    "starts": c.starts,
                    "failures": c.failures,
                    "error": c.error,
                }
                for name, c in self.components.items()
            }
        }

    @staticmethod
    def _group_exists(pid: int) -> bool:
        try:
            os.killpg(pid, 0)
            return True
        except ProcessLookupError:
            return False

    async def _stop_child(self, child: Component) -> None:
        process = child.process
        if process is None:
            return
        for sig, grace in ((signal.SIGTERM, child.exit_timeout), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                process.poll()  # Reap the owned leader even if a descendant ignores SIGTERM.
                if not self._group_exists(process.pid):
                    break
                await asyncio.sleep(0.02)
            if not self._group_exists(process.pid):
                process.wait()
                child.process = None
                return
        raise RuntimeError(f"owned process group {process.pid} did not exit")

    def _failed(self, child: Component, error: str) -> None:
        child.failures += 1
        child.error = error
        child.retry_at = time.monotonic() + min(300, self.backoff * 2 ** min(child.failures - 1, 30))

    def _start(self, child: Component) -> None:
        try:
            with ExitStack() as files:
                stdout = files.enter_context(child.stdout.open("ab")) if child.stdout else None
                stderr = files.enter_context(child.stderr.open("ab")) if child.stderr else None
                child.process = subprocess.Popen(
                    child.argv, env=os.environ | child.env, stdout=stdout, stderr=stderr, start_new_session=True
                )
            child.starts += 1
            child.error = ""
        except OSError as exc:
            self._failed(child, str(exc))

    async def _tick(self) -> None:
        for child in self.components.values():
            if child.process and child.process.poll() is not None:
                code = child.process.returncode
                await self._stop_child(child)
                self._failed(child, f"exited {code}")
            if child.enabled and not child.paused and child.process is None and time.monotonic() >= child.retry_at:
                self._start(child)

    async def _locked_tick(self) -> None:
        async with self.lock:
            if not self.stop.is_set():
                await self._tick()

    async def _control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            message = json.loads(await asyncio.wait_for(reader.readline(), 2))
            action, name = message["action"], message.get("component", "*")
            if action not in {"status", "pause", "resume", "restart"} or (name != "*" and name not in self.components):
                raise ValueError("unknown action or component")
            async with self.lock:
                if self.stop.is_set():
                    raise RuntimeError("supervisor stopping")
                selected = list(self.components.values()) if name == "*" else [self.components[name]]
                if action in {"resume", "restart"} and any(
                    not c.enabled or (action == "restart" and c.paused) for c in selected
                ):
                    raise ValueError("component is configured OFF or paused")
                if action != "status":
                    for child in selected:
                        child.paused = True  # No revival until every requested group is quiescent.
                    await asyncio.gather(*(self._stop_child(c) for c in selected))
                    if action != "pause":
                        for child in selected:
                            child.paused, child.retry_at = False, 0
                            self._start(child)
                response = self.status()
        except Exception as exc:
            response = {"error": str(exc)}
        try:
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    async def run(self, *, install_signals: bool = False) -> None:
        """Start only when explicitly called; refuse existing endpoints and unsafe directories."""
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = self.path.parent.lstat()
        if not stat.S_ISDIR(directory.st_mode) or directory.st_uid != os.getuid() or directory.st_mode & 0o077:
            raise PermissionError("control directory must be owner-only")
        if os.path.lexists(self.path):
            raise FileExistsError(self.path)
        sock = socket.socket(socket.AF_UNIX)
        try:
            sock.bind(str(self.path))  # Atomic bind; asyncio's path mode may unlink another server.
            os.chmod(self.path, 0o600)
            server = await asyncio.start_unix_server(self._control, sock=sock, limit=4096, start_serving=False)
        except BaseException:
            sock.close()
            raise
        identity = self.path.stat()
        previous = signal.getsignal(signal.SIGTERM)
        tick = None
        try:
            if install_signals:
                signal.signal(signal.SIGTERM, lambda *_args: self.stop.set())
            self.guard = subprocess.Popen(
                (sys.executable, "-c", _GUARD, str(self.guard_timeout)), stdin=subprocess.PIPE, start_new_session=True
            )
            os.set_blocking(self.guard.stdin.fileno(), False)
            await server.start_serving()
            while not self.stop.is_set():
                if self.guard.poll() is not None:
                    raise RuntimeError("supervisor heartbeat guard exited")
                os.write(self.guard.stdin.fileno(), b".")
                if tick is None or tick.done():
                    if tick:
                        tick.result()
                    tick = asyncio.create_task(self._locked_tick())
                await asyncio.sleep(0.1)
        finally:
            self.stop.set()
            server.close()
            await server.wait_closed()
            if tick:
                tick.cancel()
                await asyncio.gather(tick, return_exceptions=True)
            try:
                async with self.lock:
                    await asyncio.gather(*(self._stop_child(c) for c in self.components.values()))
            finally:
                if self.guard:
                    self.guard.stdin.close()
                    try:
                        self.guard.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.guard.kill()
                        self.guard.wait()
                if install_signals:
                    signal.signal(signal.SIGTERM, previous)
            if (
                self.path.exists()
                and self.path.stat().st_ino == identity.st_ino
                and self.path.stat().st_dev == identity.st_dev
            ):
                self.path.unlink()
