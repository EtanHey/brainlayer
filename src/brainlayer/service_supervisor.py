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

from brainlayer import supervisor_guard
from brainlayer.paths import supervisor_socket_path
from brainlayer.supervisor_guard import Custody, gated_spawn, launch_guard, peer_uid, start_time, stop_group


def child_environment(env):
    """Launchd-like base; unrelated supervisor configuration never reaches children."""
    keys = ("HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "__CF_USER_TEXT_ENCODING")
    return {k: os.environ[k] for k in keys if k in os.environ} | {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"} | env


@dataclass
class Component:
    argv: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)
    stdout: Path | None = None
    stderr: Path | None = None
    enabled: bool = True
    exit_timeout: float = 60
    cwd: Path | None = None
    interval: float = 0
    run_timeout: float = 300
    policy: dict = field(default_factory=dict)
    backoff: float | None = None
    process: subprocess.Popen | None = field(default=None, init=False)
    paused: bool = field(default=False, init=False)
    pause_owner: str = field(default="", init=False)
    starts: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)
    retry_at: float = field(default=0, init=False)
    error: str = field(default="", init=False)
    started_at: float = field(default=0, init=False)
    cleanup: asyncio.Task | None = field(default=None, init=False)
    last_exit_code: int | None = field(default=None, init=False)
    last_run_at: float | None = field(default=None, init=False)
    cleanup_retry_at: float = field(default=0, init=False)

    def status(self):
        return dict(
            configured=self.enabled,
            exit_timeout=self.exit_timeout,
            paused=self.paused,
            pause_owner=self.pause_owner,
            pid=self.process.pid if self.process and self.process.poll() is None else None,
            starts=self.starts,
            failures=self.failures,
            error=self.error,
            interval=self.interval,
            last_exit_code=self.last_exit_code,
            last_run_at=self.last_run_at,
            next_run_at=time.time() + max(0, self.retry_at - time.monotonic())
            if self.interval and self.enabled and not self.paused
            else None,
        )


class Supervisor:
    def __init__(
        self,
        components: dict[str, Component],
        path: Path | None = None,
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
        ) or any(
            not c.argv
            or not math.isfinite(c.exit_timeout)
            or c.exit_timeout < 0
            or not math.isfinite(c.interval)
            or c.interval < 0
            or not math.isfinite(c.run_timeout)
            or c.run_timeout <= 0
            for c in components.values()
        ):
            raise ValueError("commands and timing bounds must be valid")
        self.components, self.path = components, path if path is not None else supervisor_socket_path()
        self.backoff, self.guard_timeout = backoff, guard_timeout
        self.shutdown_timeout, self.stable_uptime = shutdown_timeout, stable_uptime
        self.stop, self.lock = asyncio.Event(), asyncio.Lock()
        self.guard: subprocess.Popen | None = None
        self.custody, self.deadline = None, None
        self.recovery = None
        self.recovery_at = 0

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
        child.cleanup_retry_at = 0
        child.process = None

    async def _stop_child(self, child: Component):
        if child.cleanup is None:
            child.cleanup = asyncio.create_task(self._reap(child))
        await asyncio.shield(child.cleanup)
        child.cleanup = None

    def _failed(self, child: Component, error: str):
        child.failures += 1
        child.error = error
        child.retry_at = time.monotonic() + min(300, (child.backoff or self.backoff) * 2 ** min(child.failures - 1, 30))

    def _start(self, child: Component):
        gate = None
        try:
            with ExitStack() as files:
                files.enter_context(supervisor_guard.component_gate(self.path.parent))
                stdout = files.enter_context(child.stdout.open("ab")) if child.stdout else None
                stderr = files.enter_context(child.stderr.open("ab")) if child.stderr else None
                child.process, gate = gated_spawn(
                    child.argv, child_environment(child.env), stdout, stderr, cwd=child.cwd, policy=child.policy
                )
            birth = start_time(child.process.pid)
            if birth is None:
                raise RuntimeError("cannot establish child birth identity")
            self.custody.record(child.process.pid, birth)
            self._send(f"+{child.process.pid}")
            os.write(gate, b"R")
            child.starts += 1
            child.started_at, child.error = time.monotonic(), ""
            child.last_run_at = time.time()
            if child.interval:
                child.retry_at = child.started_at + child.interval
        except BlockingIOError:
            child.error = "maintenance lease held"
            child.retry_at = time.monotonic() + 0.5
        except OSError as exc:
            self._failed(child, str(exc))
        finally:
            if gate is not None:
                os.close(gate)  # On failure the wrapper exits without running the command.

    def _tick(self):
        if self.recovery and self.recovery.done():
            try:
                self.recovery.result()
            except BlockingIOError:
                pass  # A live holder or inherited child still owns the lease.
            except Exception as exc:
                for child in self.components.values():
                    if child.pause_owner:
                        child.error = f"maintenance recovery failed: {exc}"
            self.recovery = None
        # The journal alone drives recovery and expiry: a restarted supervisor has no pause owners.
        if (
            self.recovery is None
            and time.monotonic() >= self.recovery_at
            and self.path.with_name("maintenance.json").exists()
        ):
            from .maintenance_control import recover_hold

            self.recovery = asyncio.create_task(asyncio.to_thread(recover_hold, self.path))
            self.recovery_at = time.monotonic() + 1
        for child in self.components.values():
            if child.cleanup and child.cleanup.done():
                try:
                    child.cleanup.result()
                except Exception as exc:
                    child.error = f"cleanup failed: {exc}"
                    child.cleanup_retry_at = time.monotonic() + self.backoff
                child.cleanup = None
            if time.monotonic() < child.cleanup_retry_at:
                continue  # Keep custody; failed reaping never authorizes another child.
            if child.cleanup_retry_at:
                if child.cleanup is not None:
                    continue  # A control-initiated reap still owns this cleanup.
                child.cleanup_retry_at = 0
                child.cleanup = asyncio.create_task(self._reap(child))
                continue  # Retry cleanup without recounting/replacing the original exit.
            if child.process and child.process.poll() is not None and child.cleanup is None:
                child.last_exit_code = child.process.returncode
                if child.interval:
                    child.retry_at = time.monotonic() + child.interval
                    if child.last_exit_code not in {0, 75}:
                        child.failures += 1
                        child.error = f"exited {child.last_exit_code}"
                    else:
                        child.failures, child.error = 0, ""
                else:
                    self._failed(child, f"exited {child.process.returncode}")
                child.cleanup = asyncio.create_task(self._reap(child))  # Never under the control lock.
            if (
                child.interval
                and child.process
                and child.process.poll() is None
                and child.cleanup is None
                and time.monotonic() - child.started_at >= child.run_timeout
            ):
                child.last_exit_code, child.error = 124, "interval worker timed out"
                child.failures += 1
                child.retry_at = time.monotonic() + child.interval
                child.cleanup = asyncio.create_task(self._reap(child))
            if (
                not child.interval
                and child.process
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
            owner = message.get("owner", "")
            if not isinstance(owner, str) or len(owner) > 128:
                raise ValueError("invalid pause owner")
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
                    if action == "pause" and owner:
                        selected = [c for c in selected if c.enabled and not c.paused]
                    if action == "resume":
                        selected = [c for c in selected if c.paused]
                        if any(c.pause_owner != owner for c in selected):
                            raise ValueError("pause belongs to another owner")
                    for child in selected:
                        child.paused = True
                        child.pause_owner = owner
                    results = await asyncio.gather(*(self._stop_child(c) for c in selected), return_exceptions=True)
                    if any(isinstance(r, BaseException) for r in results):
                        raise RuntimeError("control could not quiesce every selected group")
                    if action != "pause" and not self.stop.is_set():
                        for child in selected:
                            child.paused, child.retry_at, child.pause_owner = False, 0, ""
                            self._start(child)
                        if action == "restart" and any(c.process is None for c in selected):
                            raise RuntimeError(
                                "restart did not start every component: " + "; ".join(c.error for c in selected)
                            )
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
                if self.recovery:
                    await asyncio.gather(self.recovery, return_exceptions=True)
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
