"""Explicitly started resident supervisor; no production commands or launchd wiring."""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import socket
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from brainlayer.supervisor_guard import Custody, gated_spawn, launch_guard, peer_uid, start_time, stop_group


@dataclass
class Component:
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
    started_at: float = field(default=0, init=False)
    cleanup: asyncio.Task | None = field(default=None, init=False)

    def status(self):
        return dict(
            configured=self.enabled,
            paused=self.paused,
            pid=self.process.pid if self.process and self.process.poll() is None else None,
            starts=self.starts,
            failures=self.failures,
            error=self.error,
        )


class Supervisor:
    def __init__(
        self,
        components: dict[str, Component],
        path: Path,
        *,
        backoff: float = 10,
        guard_timeout: float = 30,
        shutdown_timeout: float = 15,
        stable_uptime: float = 60,
    ):
        if any(name.removeprefix("com.brainlayer.") in {"enrich", "enrichment"} for name in components):
            raise ValueError("enrichment is retired")
        if not all(
            math.isfinite(v) and v > 0 for v in (backoff, guard_timeout, shutdown_timeout, stable_uptime)
        ) or any(not c.argv or not math.isfinite(c.exit_timeout) or c.exit_timeout < 0 for c in components.values()):
            raise ValueError("commands and timing bounds must be valid")
        self.components, self.path = components, path
        self.backoff, self.guard_timeout = backoff, guard_timeout
        self.shutdown_timeout, self.stable_uptime = shutdown_timeout, stable_uptime
        self.stop, self.lock = asyncio.Event(), asyncio.Lock()
        self.guard: subprocess.Popen | None = None
        self.custody, self.deadline = None, None

    def status(self) -> dict:
        return {"components": {name: child.status() for name, child in self.components.items()}}

    def _send(self, line: str):
        os.write(self.guard.stdin.fileno(), (line + "\n").encode())

    async def _reap(self, child: Component):
        process = child.process
        if process is None:
            return
        await stop_group(
            process,
            child.exit_timeout,
            lambda: (self.deadline, self.shutdown_timeout / 2) if self.deadline is not None else None,
        )
        self.custody.forget(self.guard, process.pid)
        child.process = None

    async def _stop_child(self, child: Component):
        if child.cleanup is None:
            child.cleanup = asyncio.create_task(self._reap(child))
        await asyncio.shield(child.cleanup)
        child.cleanup = None

    def _failed(self, child: Component, error: str):
        child.failures += 1
        child.error = error
        child.retry_at = time.monotonic() + min(300, self.backoff * 2 ** min(child.failures - 1, 30))

    def _start(self, child: Component):
        gate = None
        try:
            with ExitStack() as files:
                stdout = files.enter_context(child.stdout.open("ab")) if child.stdout else None
                stderr = files.enter_context(child.stderr.open("ab")) if child.stderr else None
                child.process, gate = gated_spawn(child.argv, os.environ | child.env, stdout, stderr)
            birth = start_time(child.process.pid)
            if birth is None:
                raise RuntimeError("cannot establish child birth identity")
            self.custody.record(child.process.pid, birth)
            self._send(f"+{child.process.pid}")
            os.write(gate, b"R")
            child.starts += 1
            child.started_at, child.error = time.monotonic(), ""
        except OSError as exc:
            self._failed(child, str(exc))
        finally:
            if gate is not None:
                os.close(gate)  # On failure the wrapper exits without running the command.

    def _tick(self):
        for child in self.components.values():
            if child.cleanup and child.cleanup.done():
                child.cleanup.result()
                child.cleanup = None
            if child.process and child.process.poll() is not None and child.cleanup is None:
                self._failed(child, f"exited {child.process.returncode}")
                child.cleanup = asyncio.create_task(self._reap(child))  # Never under the control lock.
            if (
                child.process
                and child.process.poll() is None
                and time.monotonic() - child.started_at >= self.stable_uptime
            ):
                child.failures = 0
            if child.enabled and not child.paused and child.process is None and time.monotonic() >= child.retry_at:
                self._start(child)

    async def _control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            if peer_uid(writer.get_extra_info("socket")) != os.getuid():
                raise PermissionError("control peer uid differs")
            message = json.loads(await asyncio.wait_for(reader.readline(), 2))
            action, name = message["action"], message.get("component", "*")
            if action not in {"status", "pause", "resume", "restart"} or (name != "*" and name not in self.components):
                raise ValueError("unknown action or component")
            if action != "status":
                async with self.lock:
                    if self.stop.is_set():
                        raise RuntimeError("supervisor stopping")
                    selected = list(self.components.values()) if name == "*" else [self.components[name]]
                    if action in {"resume", "restart"}:
                        if name != "*" and (not selected[0].enabled or (action == "restart" and selected[0].paused)):
                            raise ValueError("component is configured OFF or paused")
                        selected = [c for c in selected if c.enabled and (action == "resume" or not c.paused)]
                    if action == "resume":
                        selected = [c for c in selected if c.paused]
                    for child in selected:
                        child.paused = True
                    results = await asyncio.gather(*(self._stop_child(c) for c in selected), return_exceptions=True)
                    if any(isinstance(r, BaseException) for r in results):
                        raise RuntimeError("control could not quiesce every selected group")
                    if action != "pause" and not self.stop.is_set():
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

    async def run(self, *, install_signals: bool = False):
        """Kernel exclusion precedes stale endpoint cleanup and recorded-group recovery."""
        loop, handlers, server = asyncio.get_running_loop(), {}, None
        grace = min(1, self.shutdown_timeout / 2)
        with Custody(self.path.parent) as self.custody:
            self.custody.recover(grace)
            self.path.unlink(missing_ok=True)
            sock = socket.socket(socket.AF_UNIX)
            try:
                sock.bind(str(self.path))
                os.chmod(self.path, 0o600)
                server = await asyncio.start_unix_server(self._control, sock=sock, limit=4096, start_serving=False)
                if install_signals:
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                        handlers[sig] = signal.getsignal(sig)
                        loop.add_signal_handler(sig, self.stop.set)
                self.guard = launch_guard(self.custody, self.guard_timeout, grace, self.shutdown_timeout)
                await server.start_serving()
                while not self.stop.is_set():
                    if self.guard.poll() is not None:
                        raise RuntimeError("supervisor heartbeat guard exited")
                    self._send("H")
                    self._tick()
                    await asyncio.sleep(min(0.1, self.guard_timeout / 4))
            finally:
                self.stop.set()
                self.deadline = time.monotonic() + self.shutdown_timeout
                if self.guard and self.guard.poll() is None:
                    try:
                        self._send("Q")  # Disarm before bounded shutdown.
                    except BrokenPipeError:
                        pass  # Still clean every child if the guard died.
                if server:
                    server.close()
                    await server.wait_closed()
                else:
                    sock.close()
                # Shorten already-running cleanup too; all groups share the same total deadline.
                for child in self.components.values():
                    child.exit_timeout = min(child.exit_timeout, self.shutdown_timeout / 2)
                results = await asyncio.gather(
                    *(self._stop_child(c) for c in self.components.values()), return_exceptions=True
                )
                if self.guard:
                    self.guard.stdin.close()  # EOF still cleans any remaining registered groups.
                    while self.guard.poll() is None and time.monotonic() < self.deadline:
                        await asyncio.sleep(0.02)
                for sig, previous in handlers.items():
                    loop.remove_signal_handler(sig)
                    signal.signal(sig, previous)
                self.path.unlink(missing_ok=True)
                if any(isinstance(r, BaseException) for r in results):
                    raise RuntimeError("shutdown deadline exceeded; guard retains orphan custody")
