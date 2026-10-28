"""Inactive supervisor custody: kernel lock, birth journal and parent watchdog."""

from __future__ import annotations

import array
import asyncio
import ctypes
import fcntl
import json
import logging
import math
import os
import select
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .supervisor_diagnostics import DiagnosticSink, emit

STALL_OVERSHOOT_S = 5.0
MAX_STALL_REARMS = 3
MAX_STALL_GRACE_S = 300.0
GUARD_TIMEOUT_ENV = "BRAINLAYER_SUPERVISOR_GUARD_TIMEOUT_SECONDS"
# Background launchd jobs under host load stalled 31-36 s on 2026-10-28; 30 s killed live parents.
# A dead parent is still caught at once by pipe EOF, so this only bounds an alive-but-silent one.
DEFAULT_GUARD_TIMEOUT_SECONDS = 120.0
GUARD_TIMEOUT_BOUNDS = (10.0, 900.0)


def configured_guard_timeout() -> float:
    """Heartbeat budget for the daemon and calendar guards; an invalid override raises, never falls back."""
    from .config import _split_launchd_env_assignment, configured_brainlayer_env_value, get_user_env_path

    path = Path(os.environ.get("BRAINLAYER_ENV_FILE", get_user_env_path()))
    configured = configured_brainlayer_env_value(GUARD_TIMEOUT_ENV, path)
    if path.exists():
        # The shared parser skips a substitution line as absent; here that would be a silent fallback.
        for line in path.read_text(encoding="utf-8").splitlines():
            assignment = _split_launchd_env_assignment(line)
            if assignment and assignment[0] == GUARD_TIMEOUT_ENV and ("$(" in assignment[1] or "`" in assignment[1]):
                raise ValueError(f"{GUARD_TIMEOUT_ENV} in {path} uses unsupported command substitution")
    raw = configured if configured is not None else os.environ.get(GUARD_TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_GUARD_TIMEOUT_SECONDS
    low, high = GUARD_TIMEOUT_BOUNDS
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and low <= value <= high):
        raise ValueError(f"{GUARD_TIMEOUT_ENV} must be a finite number of seconds in [{low:g}, {high:g}]: {raw!r}")
    return value


@contextmanager
def component_gate(directory: Path, *, exclusive=False):
    """Close rather than LOCK_UN: inherited descriptors retain the maintenance lease."""
    fd = os.open(directory / "maintenance.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1:
            raise PermissionError("unsafe maintenance lease")
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Unknown is not proof of quiescence.


def start_time(pid: int) -> str | None:
    """Kernel birth identity, preserved across exec; refuse unsupported platforms."""
    if sys.platform == "darwin":
        # sys/proc_info.h: PROC_PIDTBSDINFO, 136-byte proc_bsdinfo, timeval at 120.
        info = ctypes.create_string_buffer(136)
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pidinfo(pid, 3, 0, info, len(info)) != len(info):
            return None
        return ":".join(map(str, struct.unpack_from("QQ", info.raw, 120)))
    if sys.platform == "linux":
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            return None
    raise RuntimeError("unsupported process identity platform")


def boot_time() -> str:
    if sys.platform == "darwin":
        stamp = (ctypes.c_long * 2)()
        size = ctypes.c_size_t(ctypes.sizeof(stamp))
        if ctypes.CDLL(None, use_errno=True).sysctlbyname(b"kern.boottime", stamp, ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), "kern.boottime unreadable")
        return ":".join(map(str, struct.unpack_from("qi", stamp)))  # Darwin timeval: long + int.
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def peer_uid(sock: socket.socket) -> int:
    if sys.platform == "darwin":
        uid, gid = ctypes.c_uint(), ctypes.c_uint()
        if ctypes.CDLL(None, use_errno=True).getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid)):
            raise OSError(ctypes.get_errno(), "getpeereid")
        return uid.value
    return struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]


def reap_groups(groups: set[int], grace: float, *, timeout: float = 20) -> None:
    """TERM/KILL with a total deadline; never acknowledge unresolved custody."""
    end = time.monotonic() + timeout
    for sig, delay in ((signal.SIGTERM, min(grace, timeout / 2)), (signal.SIGKILL, 0)):
        for pgid in groups:
            try:
                os.killpg(pgid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and any(group_exists(p) for p in groups):
            time.sleep(0.02)
    while any(group_exists(p) for p in groups):
        if time.monotonic() >= end:
            logging.error("process group reap deadline exceeded: %s", sorted(groups))
            raise RuntimeError(f"process group reap deadline exceeded: {sorted(groups)}")
        for pgid in groups:
            try:
                os.waitpid(pgid, os.WNOHANG)  # Reap if this process owns the leader.
            except ChildProcessError:
                pass
        time.sleep(0.02)


async def stop_group(process: subprocess.Popen, grace: float, shutdown_budget):
    """A dynamic shared deadline also bounds cleanup started before shutdown."""
    for sig, delay in ((signal.SIGTERM, grace), (signal.SIGKILL, 2)):
        process.poll()
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + delay
        while True:
            process.poll()
            if not group_exists(process.pid):
                return
            budget = shutdown_budget()
            bound = deadline
            if budget is not None:
                end, reserve = budget
                bound = min(bound, end - reserve if sig == signal.SIGTERM else end)
            if time.monotonic() >= bound:
                break
            await asyncio.sleep(0.02)
    raise RuntimeError(f"owned process group {process.pid} did not exit")


class Custody:
    """The guard inherits this lock until orphan cleanup ends. Probes can disable journal reset."""

    def __init__(self, directory: Path, *, reset_previous_boot=True):
        self.directory, self.fd = directory, None
        self.reset_previous_boot = reset_previous_boot
        self.journal = directory / "groups.json"
        self.groups: dict = {}
        self.boot = boot_time()

    def __enter__(self):
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("control directory must be owner-only")
        self.fd = os.open(self.directory / "control.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(self.fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_nlink != 1
            ):
                raise PermissionError("unsafe control lock")
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.journal.exists():
                document = json.loads(self.journal.read_text())
                self.groups = document.get("groups", document)  # Legacy journals still get birth checks.
                if self.reset_previous_boot and "boot_time" in document and document["boot_time"] != self.boot:
                    self.groups = {}
                    self._save()  # Rewrite previous-boot custody before any spawn.
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def record(self, pgid: int, birth: str | None):
        if birth is None:
            self.groups.pop(str(pgid), None)
        else:
            self.groups[str(pgid)] = {"pgid": pgid, "start_time": birth}
        self._save()

    def _save(self):
        pending = self.journal.with_suffix(".pending")
        fd = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump(dict(boot_time=self.boot, groups=self.groups), output)
            output.flush()
            os.fsync(output.fileno())
        pending.replace(self.journal)
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def forget(self, guard: subprocess.Popen, pgid: int):
        """Remove custody only after ESRCH; a dead guard cannot block that proof."""
        self.record(pgid, None)
        try:
            os.write(guard.stdin.fileno(), f"-{pgid}\n".encode())
        except BrokenPipeError:
            pass

    def recover(self, grace: float):
        owned = set()
        for item in self.groups.values():
            pgid, birth = item["pgid"], item["start_time"]
            if not isinstance(pgid, int) or pgid <= 1 or not isinstance(birth, str):
                raise RuntimeError("invalid group identity journal")
            if group_exists(pgid):
                actual = start_time(pgid)
                if actual is None:
                    raise RuntimeError(f"cannot establish recorded group identity: {pgid}")
                if actual == birth:
                    owned.add(pgid)
        reap_groups(owned, grace)
        self.groups = {}
        self._save()


def run_guard(
    parent: int,
    timeout: float,
    ready: int,
    grace: float,
    shutdown_timeout: float = 20,
    lease_control=None,
    *,
    parent_birth=None,
    diagnostics=None,
    diagnostic_fd=None,
):
    """H / +pgid / -pgid / Q, then EOF. Only the exact live parent can be killed."""
    if os.getppid() != parent:
        raise RuntimeError("guard parent identity mismatch")
    owned_sink = None
    if diagnostic_fd is not None:
        try:
            owned_sink = DiagnosticSink(diagnostic_fd)
        except (OSError, RuntimeError):
            os.close(diagnostic_fd)  # A diagnostics-thread startup failure cannot disarm the guard.
    diagnostics = owned_sink if owned_sink is not None else diagnostics
    parent_birth = parent_birth if parent_birth is not None else start_time(parent)
    emit(diagnostics, event="guard-started", parent_pid=parent, parent_start_time=parent_birth, guard_pid=os.getpid())
    os.write(ready, b"R")
    os.close(ready)
    groups, buffer, quiesced = set(), b"", False
    timed_out = False
    stall_rearms = 0
    stall_grace_used = 0.0
    last_heartbeat = time.monotonic()
    deadline = last_heartbeat + timeout
    channel = socket.socket(fileno=lease_control) if lease_control is not None else None
    lease = None
    try:
        while True:
            wait_started = time.monotonic()
            remaining = max(0, deadline - wait_started)
            readable = select.select([0, channel] if channel else [0], [], [], remaining)[0]
            now = time.monotonic()
            overshoot = now - wait_started - remaining
            data = os.read(0, 65536) if 0 in readable else None
            if data == b"":
                break  # A dead parent needs cleanup, never stall grace.
            # A starved guard cannot judge parent liveness. Give it a fresh window,
            # bounded by both incident caps; only a real heartbeat resets them.
            if (
                not quiesced
                and overshoot > STALL_OVERSHOOT_S
                and stall_rearms < MAX_STALL_REARMS
                and stall_grace_used < MAX_STALL_GRACE_S
            ):
                stall_grace = min(timeout, MAX_STALL_GRACE_S - stall_grace_used)
                stall_rearms += 1
                stall_grace_used += stall_grace
                deadline = now + stall_grace
                emit(
                    diagnostics,
                    event="guard-stall-rearm",
                    parent_pid=parent,
                    parent_start_time=parent_birth,
                    guard_pid=os.getpid(),
                    wait_overshoot_s=overshoot,
                    stall_rearms=stall_rearms,
                    grace_used_s=stall_grace_used,
                )
                if not readable:
                    continue
            timeout_event = dict(
                event="guard-timeout",
                health_event="supervisor_guard_timeout",
                parent_pid=parent,
                parent_start_time=parent_birth,
                guard_pid=os.getpid(),
                mode="shutdown" if quiesced else "heartbeat",
                guard_mode="Q" if quiesced else "H",
                last_heartbeat_age_s=now - last_heartbeat,
                registered_pgids=sorted(groups),
                reason="shutdown-timeout" if quiesced else "heartbeat-timeout",
            )
            if not readable:
                emit(diagnostics, **timeout_event)
                if os.getppid() == parent:
                    os.kill(parent, signal.SIGKILL)
                    timed_out = True
                break
            if channel and channel in readable:
                message, ancillary, _, _ = channel.recvmsg(1, socket.CMSG_SPACE(4))
                if not message:
                    break
                if message == b"L" and lease is None:
                    level, kind, payload = ancillary[0]
                    fds = array.array("i")
                    fds.frombytes(payload)
                    if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS or len(fds) != 1:
                        raise RuntimeError("unknown lease transfer")
                    lease = fds[0]
                elif message == b"U" and lease is not None:
                    os.close(lease)
                    lease = None
                else:
                    raise RuntimeError("unknown lease action")
                channel.sendall(b"R")
                if 0 not in readable:
                    continue
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line == b"H" and not quiesced:
                    stall_rearms = 0
                    stall_grace_used = 0.0
                    last_heartbeat = time.monotonic()
                    deadline = last_heartbeat + timeout
                elif line == b"Q" and not quiesced:
                    quiesced = True
                    deadline = time.monotonic() + shutdown_timeout + 2
                elif line in (b"H", b"Q"):
                    pass  # Repeated pulses cannot extend quiescence forever.
                elif line[:1] in (b"+", b"-"):
                    pgid = int(line[1:])
                    if pgid <= 1:
                        raise RuntimeError("invalid registered group")
                    (groups.add if line[:1] == b"+" else groups.discard)(pgid)
                else:
                    raise RuntimeError("invalid guard protocol")
    finally:
        try:
            reap_groups(groups, grace, timeout=shutdown_timeout)
        finally:
            if lease is not None:
                os.close(lease)
            if channel:
                channel.close()
            if owned_sink:
                owned_sink.close(drain=2 if timed_out else 0.02)


def guard_lease(guard, fd=None):
    """Transfer the held open-file description; no reopen or unlock while a writer can remain."""
    channel = guard.lease_channel
    if fd is None:
        channel.sendall(b"U")
    else:
        channel.sendmsg([b"L"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])
    if channel.recv(1) != b"R":
        raise RuntimeError("lease transfer unacknowledged")


def launch_guard(
    custody: Custody,
    timeout: float,
    grace: float,
    shutdown_timeout: float = 20,
    *,
    extra_fds=(),
    lease_control=False,
    startup_timeout: float = 30,
) -> subprocess.Popen:
    if not math.isfinite(startup_timeout) or startup_timeout <= 0:
        raise ValueError("guard startup timeout must be finite and positive")
    diagnostic_fd = None
    try:
        diagnostic_fd = DiagnosticSink.open_fd(custody.directory / "guard-events.jsonl")
    except OSError:
        logging.warning("guard diagnostic sink unavailable; timeout enforcement remains active")
    read, write = os.pipe()
    channel, receiver = socket.socketpair() if lease_control else (None, None)
    try:
        command = (
            "from brainlayer.supervisor_guard import run_guard; import sys; "
            "run_guard(int(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),float(sys.argv[5]),"
            "None if sys.argv[6]=='-' else int(sys.argv[6]),parent_birth=sys.argv[7],"
            "diagnostic_fd=None if sys.argv[8]=='-' else int(sys.argv[8]))"
        )
        guard = subprocess.Popen(
            (
                sys.executable,
                "-c",
                command,
                str(os.getpid()),
                str(timeout),
                str(write),
                str(grace),
                str(shutdown_timeout),
                str(receiver.fileno()) if receiver else "-",
                start_time(os.getpid()) or "unknown",
                str(diagnostic_fd) if diagnostic_fd is not None else "-",
            ),
            stdin=subprocess.PIPE,
            pass_fds=(
                custody.fd,
                write,
                *extra_fds,
                *((receiver.fileno(),) if receiver else ()),
                *((diagnostic_fd,) if diagnostic_fd is not None else ()),
            ),
            start_new_session=True,
        )
        os.close(write)
        write = None
        ready = os.read(read, 1) if select.select([read], [], [], startup_timeout)[0] else None
        if ready != b"R":
            guard.kill()
            guard.wait()
            reason = "timeout" if ready is None else "EOF" if not ready else f"unexpected byte {ready!r}"
            raise RuntimeError(f"guard startup failed: {reason}; parent={os.getpid()} guard={guard.pid}")
        os.set_blocking(guard.stdin.fileno(), False)
        if channel:
            channel.settimeout(2)
            guard.lease_channel = channel
            channel = None
        return guard
    finally:
        if diagnostic_fd is not None:
            os.close(diagnostic_fd)
        os.close(read)
        if write is not None:
            os.close(write)
        if receiver:
            receiver.close()
        if channel:
            channel.close()


def gated_spawn(argv, env, stdout, stderr, *, cwd=None, policy=None, extra_fds=()) -> tuple[subprocess.Popen, int]:
    """EOF aborts startup; commands run only after durable custody registration."""
    read, write = os.pipe()
    try:
        process = subprocess.Popen(
            (
                sys.executable,
                "-I",  # Stdlib-only helper: exclude package types.py and configured PYTHONPATH.
                str(Path(__file__).with_name("supervisor_child.py")),
                str(read),
                json.dumps(policy or {}),
                *argv,
            ),
            env=env,
            stdout=stdout,
            stderr=stderr,
            cwd=cwd,
            pass_fds=(read, *extra_fds),
            start_new_session=True,
        )
        return process, write
    except BaseException:
        os.close(write)
        raise
    finally:
        os.close(read)
