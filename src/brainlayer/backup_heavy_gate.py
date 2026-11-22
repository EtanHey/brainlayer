"""Fleet admission for scheduled DB backups; never reclaim another lane's lock."""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class BackupGateDeferred(RuntimeError):
    pass


def _number(name: str, default: float) -> float:
    value = float(os.environ.get(f"BRAINLAYER_BACKUP_HEAVY_{name}", default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"BRAINLAYER_BACKUP_HEAVY_{name} must be finite and positive")
    return value


def _sysctl(key: str) -> str:
    return subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True, check=True, timeout=5).stdout


def _host() -> str:
    configured = os.environ.get("BRAINLAYER_BACKUP_HEAVY_HOST")
    if configured is not None:
        if configured not in {"m1", "m4"}:
            raise ValueError("BRAINLAYER_BACKUP_HEAVY_HOST must be m1 or m4")
        return configured
    if sys.platform != "darwin":
        return "m4"
    # Same CPU-brand detection as mcp._shared's M1 read-pool default.
    return "m1" if "Apple M1" in _sysctl("machdep.cpu.brand_string") else "m4"


def _pressure_reason(host: str, swap_limit: float, load_limit: float, disk_floor: float) -> str | None:
    try:
        if sys.platform == "darwin":
            match = re.search(r"used\s*=\s*([\d.]+)([KMGT])", _sysctl("vm.swapusage"))
            if match is None:
                return "swap usage unavailable"
            swap_gib = float(match[1]) * 1024 ** ("KMGT".index(match[2]) - 2)
        else:
            values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            swap_gib = (int(values["SwapTotal"].split()[0]) - int(values["SwapFree"].split()[0])) / 1024**2
        load = os.getloadavg()[0]
        if not math.isfinite(swap_gib) or not math.isfinite(load):
            return "non-finite swap/load measurement"
        if swap_gib >= swap_limit or load >= load_limit:
            return f"swap {swap_gib:.2f} GiB (under {swap_limit:g}), load {load:.2f} (under {load_limit:g})"
        if host == "m1":
            free_gib = shutil.disk_usage(Path.home()).free / 1024**3
            if free_gib < disk_floor:
                return f"free disk {free_gib:.2f} GiB (needs {disk_floor:g})"
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return f"resource probe unavailable: {type(exc).__name__}"
    return None


def _log(path: Path, action: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"brainlayer backup_daily {action} {datetime.now():%H:%M}\n")


@contextmanager
def backup_heavy_gate(*, deadline: float | None = None):
    """Wait before spawning the heavy child, retaining custody until it is reaped.

    A calendar snapshot's absolute deadline also bounds admission. SIGTERM/INT
    unwind the owned lock; the supervisor temporarily replaces these handlers
    and restores them only after its child has stopped.
    """
    host = _host()
    lock = Path(os.environ.get("BRAINLAYER_BACKUP_HEAVY_LOCK_PATH", str(Path.home() / f".{host}-heavy.lock")))
    log = Path(os.environ.get("BRAINLAYER_BACKUP_HEAVY_LOG_PATH", str(Path.home() / f".{host}-heavy.log")))
    window = _number("WINDOW_SECONDS", 6 * 3600)
    retry = _number("RETRY_SECONDS", 600)
    swap_limit = _number("SWAP_GIB", 9)
    load_limit = _number("LOAD", 12)
    disk_floor = _number("FREE_GIB", 44)
    started = time.monotonic()
    expires = min(started + window, deadline) if deadline is not None else started + window
    owned = False
    previous = {}

    def stopped(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stopped)
        while True:
            if time.monotonic() >= expires:
                hours = (time.monotonic() - started) / 3600
                raise BackupGateDeferred(f"backup deferred {hours:g} h by the heavy-work gate")
            reason = "heavy lock held by another lane (never cleared by backup)"
            try:
                lock.mkdir()
            except FileExistsError:
                pass
            else:
                owned = True
                try:
                    _log(log, "took")
                    reason = _pressure_reason(host, swap_limit, load_limit, disk_floor)
                    if reason is None and time.monotonic() < expires:
                        yield
                        return
                    if reason is None:
                        reason = "admission window elapsed during resource probe"
                finally:
                    try:
                        _log(log, "released")
                    finally:
                        lock.rmdir()
                        owned = False
            _log(log, f"deferred: {reason}; retry in {retry:g}s")
            print(f"brainlayer backup deferred: {reason}", flush=True)
            time.sleep(min(retry, max(0, expires - time.monotonic())))
    finally:
        try:
            if owned:
                try:
                    _log(log, "released")
                finally:
                    lock.rmdir()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
