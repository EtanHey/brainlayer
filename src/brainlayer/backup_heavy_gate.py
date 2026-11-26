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


def _host(*, strict: bool = False) -> str:
    configured = os.environ.get("BRAINLAYER_BACKUP_HEAVY_HOST")
    if configured is not None:
        if configured not in {"m1", "m4"}:
            raise ValueError("BRAINLAYER_BACKUP_HEAVY_HOST must be m1 or m4")
        return configured
    if sys.platform != "darwin":
        if strict:
            raise ValueError("Unknown backup retention host: platform is not darwin; keeping everything")
        return "m4"
    # Same CPU-brand detection as mcp._shared's M1 read-pool default.
    brand = _sysctl("machdep.cpu.brand_string").strip()
    if re.fullmatch(r"Apple M1(?: Pro| Max| Ultra)?", brand):
        return "m1"
    if strict and not re.fullmatch(r"Apple M4(?: Pro| Max| Ultra)?", brand):
        raise ValueError(f"Unknown backup retention host: {brand!r}; keeping everything")
    return "m4"


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


def _disk_reason(host: str, failed_proofs: set[tuple] | None = None) -> str | None:
    """Measure first; prove and reclaim dated raws only when that could admit, then measure again.

    `failed_proofs` is the gate's per-process memo of raw/archive identities whose proof failed,
    so a retry never re-hashes an unchanged pair.
    """
    from . import backup_daily as backup

    floor_gib = float(os.environ.get("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB", 40 if host == "m4" else 0))
    if not math.isfinite(floor_gib) or floor_gib < 0:
        raise ValueError("BRAINLAYER_BACKUP_HEAVY_HOST_FREE_GIB must be finite and nonnegative")
    try:
        staging = Path(os.environ.get("BRAINLAYER_BACKUP_STAGING_DIR", str(backup.DEFAULT_STAGING_DIR))).expanduser()
        backup._sweep_clones_before_admission(staging)
        phase = os.environ.get("BRAINLAYER_BACKUP_PHASE", "all")
        full = backup._should_run_full_verify(None)
        if phase == "upload":
            peak = _upload_peak_bytes(staging, full=full)
        else:
            db_size = backup._database_logical_size_bytes(backup.get_db_path())
            compressed = backup._estimated_gzip_bytes(staging, db_size) if staging.exists() else db_size
            clone_mode = backup.backup_clone.supported(backup.get_db_path(), staging)
            # Snapshot-only has no Drive download/restore phase.
            peak = backup._peak_backup_required_bytes(
                db_size,
                compressed,
                full=full and phase != "snapshot",
                clone_mode=clone_mode,
                **(backup.backup_clone.budget(backup.get_db_path()) if clone_mode else {}),
            )
        volume_path = staging
        while not volume_path.exists():
            volume_path = volume_path.parent

        def measure() -> tuple[list[tuple[Path, int]], int, int]:
            reserves = backup._clone_growth_reserves(staging, 0)
            required = peak + sum(size for _, size in reserves) + math.ceil(floor_gib * 1024**3)
            free = backup.disk_space.measure_disk_space(volume_path, include_available=False).free_now_bytes
            return reserves, required, free

        reserves, required, free = measure()
        if free < required:
            # Upper bound of what reclaiming the dated raws can return: each raw's COW reserve plus
            # its allocated blocks (a byte copy frees them; a clone shares them and frees less).
            reclaimable = sum(size + path.lstat().st_blocks * 512 for path, size in reserves if path.parent == staging)
            if free + reclaimable >= required and backup._reclaim_verified_raws_before_admission(
                staging, failed_proofs
            ):
                reserves, required, free = measure()
    except (OSError, RuntimeError, ValueError) as exc:
        return f"disk space measurement unavailable: {type(exc).__name__}"
    if free < required:
        reserved = ", ".join(f"{path.relative_to(staging)}={size / 1024**3:.3f} GiB" for path, size in reserves)
        return (
            f"disk space needs {required / 1024**3:.3f} GiB, has {free / 1024**3:.3f} GiB "
            f"(peak plus {floor_gib:g} GiB host floor)" + (f"; retained COW reserves: {reserved}" if reserved else "")
        )
    return None


def _upload_peak_bytes(staging: Path, *, full: bool) -> int:
    """Count only NEW verification files; existing local archives already occupy free space.

    This estimator only inspects receipt metadata; it never hashes payloads or mutates
    the journal. The caller sweeps stale clones before measuring, and proves and reclaims
    dated raws under the staging lock only when that could admit.
    Budget all possible pending candidates because upload-only can drain them sequentially.
    """
    from . import backup_daily as backup

    db = backup.get_db_path()
    log = backup._backup_log_path(None, db_path=db)
    rows = backup._read_backup_receipts(log) if log.exists() else []
    latest = {row["snapshot"]: row for row in rows if isinstance(row.get("snapshot"), str)}
    target = os.environ.get("BRAINLAYER_BACKUP_UPLOAD_SNAPSHOT")
    extra = []
    for row in latest.values():
        path = Path(row["snapshot"])
        if (
            row.get("db") != str(db)
            or row.get("local_verified") is not True
            or (row.get("uploaded") is True and row.get("verified") is True)
            or path.parent.resolve() != staging.resolve()
            or (target is not None and path.name != target)
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        if row.get("raw_sha256") is not None:
            extra.append(0)  # verify_sqlite_backup_artifact streams hashes, including Drive.
        else:
            extra.append(backup._database_logical_size_bytes(db) + (path.stat().st_size if full else 0))
    return max(extra, default=0) + backup.MIN_RAW_FREE_BYTES + backup.COPY_CHUNK_BYTES


def expected_phase_seconds() -> float:
    phase = os.environ.get("BRAINLAYER_BACKUP_PHASE", "all")
    defaults = {"snapshot": 2400, "upload": 1800, "all": 4200}
    if phase not in defaults:
        raise ValueError("backup phase must be all, snapshot, or upload")
    return _number(f"EXPECTED_{phase.upper()}_SECONDS", defaults[phase])


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
    previous = {}
    failed_proofs: set[tuple] = set()  # Raw/archive identities whose reclaim proof failed this gate run.

    def stopped(signum, _frame):
        raise SystemExit(128 + signum)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stopped)
        while True:
            if time.monotonic() >= expires:
                hours = (time.monotonic() - started) / 3600
                raise BackupGateDeferred(
                    f"backup deferred {hours:g} h by the heavy-work gate since {since}: remaining phase window exhausted; "
                    f"last cause: {last_reason}"
                )
            past_window = time.monotonic() >= pressure_window_end
            # Pressure first: the disk check may prove and reclaim raws, which is heavy I/O.
            reason = (None if past_window else _pressure_reason(swap_limit, load_limit)) or _disk_reason(
                host, failed_proofs
            )
            if reason is None:
                try:
                    lock.mkdir()
                except FileExistsError:
                    reason = "heavy lock held by another lane (never cleared by backup)"
                else:
                    owned = True
                    reason = (None if past_window else _pressure_reason(swap_limit, load_limit)) or _disk_reason(
                        host, failed_proofs
                    )
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
            report(
                job_key,
                f"backup waiting behind the heavy-work gate since {since}: {reason}",
                replace_reason=True,
            )
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
