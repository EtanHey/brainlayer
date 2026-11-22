"""Fleet admission for scheduled DB backups; never reclaim another lane's lock."""

from __future__ import annotations

import math
import os
import re
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


def _pressure_reason(swap_limit: float, load_limit: float) -> str | None:
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
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return f"resource probe unavailable: {type(exc).__name__}"
    return None


def _log(path: Path, action: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"brainlayer backup_daily {action} {datetime.now():%H:%M}\n")


@contextmanager
def backup_heavy_gate(*, deadline: float | None = None, job_key: str = "backup-daily"):
    """Wait before spawning the heavy child, retaining custody until it is reaped.

    A calendar snapshot's absolute deadline also bounds admission. SIGTERM/INT
    unwind the owned lock; the supervisor temporarily replaces these handlers
    and restores them only after its child has stopped.
    """
    from .job_alerts import report

    try:
        host = _host()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise BackupGateDeferred(f"backup heavy-work gate host detection failed: {type(exc).__name__}") from exc
    lock = Path(os.environ.get("BRAINLAYER_BACKUP_HEAVY_LOCK_PATH", str(Path.home() / f".{host}-heavy.lock")))
    log = Path(os.environ.get("BRAINLAYER_BACKUP_HEAVY_LOG_PATH", str(Path.home() / f".{host}-heavy.log")))
    window = _number("WINDOW_SECONDS", 6 * 3600)
    retry = _number("RETRY_SECONDS", 600)
    swap_limit = _number("SWAP_GIB", 9)
    load_limit = _number("LOAD", 12)
    started = time.monotonic()
    since = datetime.now().astimezone().isoformat(timespec="seconds")
    pressure_window_end = started + window
    expires = deadline if deadline is not None else float("inf")
    last_reason = "pressure admission window elapsed"
    owned = False
    admitted = False
    alerted = False
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
            past_window = time.monotonic() >= pressure_window_end
            reason = None if past_window else _pressure_reason(swap_limit, load_limit)
            if reason is None:
                try:
                    lock.mkdir()
                except FileExistsError:
                    reason = "heavy lock held by another lane (never cleared by backup)"
                else:
                    owned = True
                    reason = None if past_window else _pressure_reason(swap_limit, load_limit)
                    if reason is None and time.monotonic() < expires:
                        admitted = True
                        _log(log, "took")
                        if past_window:
                            _log(log, f"running past window after {window:g}s; ignoring swap/load: {last_reason}")
                        yield
                        return
                    if reason is None:
                        reason = "calendar admission deadline elapsed during resource probe"
                    lock.rmdir()
                    owned = False
            if not alerted:
                report(job_key, f"backup waiting behind the heavy-work gate since {since}: {reason}")
                alerted = True
            last_reason = reason
            _log(log, f"deferred: {reason}; retry in {retry:g}s")
            print(f"brainlayer backup deferred: {reason}", flush=True)
            wake_at = expires if past_window else min(expires, pressure_window_end)
            time.sleep(min(retry, max(0, wake_at - time.monotonic())))
    finally:
        try:
            if owned:
                try:
                    if admitted:
                        _log(log, "released")
                finally:
                    lock.rmdir()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
