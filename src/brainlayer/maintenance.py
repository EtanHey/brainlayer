"""Safety-gated recurring maintenance for the local BrainLayer database."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import functools
import json
import os
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic, sleep
from typing import Any, Callable, Mapping, Sequence

import apsw

from .backup_daily import (
    BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV,
    DAILY_RETENTION,
    DEFAULT_STAGING_DIR,
    WEEKLY_RETENTION,
    BackupAlreadyRunningError,
    _backup_log_path,
    _recent_verified_backup_for_reuse,
)
from .drain import BurnDrainResult, burn_drain_once
from .drive_credentials import _atomic_write
from .launchd_primitive import is_launchd_label_disabled, is_launchd_label_loaded
from .paths import get_db_path
from .pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state
from .pipeline.code_intelligence import _pid_is_alive, _pid_start_time
from .queue_io import get_queue_dir
from .wal_checkpoint import checkpoint

PAUSE_SENTINEL_PATH = DEFAULT_PAUSE_SENTINEL_PATH

MAINTENANCE_DATASET = "brainlayer-maintenance"
DEFAULT_SERVICES = ("watch", "index", "drain")
REFEED_SERVICES = ("watch", "index")
EXPECTED_WRITER_PATTERNS = (
    "BrainBar",
    "brainlayer watch",
    "brainlayer enrich",
    "brainlayer index",
    "brainlayer.drain",
    "drain_daemon.py",
    "com.brainlayer.",
)
QUIESCE_EXIT_TIMEOUT_SECONDS = 45.0
_WATCHDOG_DISABLED_BEFORE = "_fleet_watchdog_disabled_before"
MAINTENANCE_LOCK_TIMEOUT_SECONDS = 4 * 60 * 60
SEARCH_LATENCY_TARGET_MS = 50.0
# Ten times the target distinguishes sustained pathological delay from scheduler jitter.
SEARCH_LATENCY_FAILURE_MS = 500.0
SEARCH_LATENCY_SAMPLES = 5
# Keep in parity with BrainLayerMaintenanceExit.deliberateDeferrals in BrainBar.
# Exit 75 is also used for real failures; only these whole reasons are deliberate skips.
DELIBERATE_DEFERRALS = (
    r"^outside quiet window: now=\S+ start_hour=\d+ duration_minutes=\d+$",
    r"^recent queue write activity: \d+ file\(s\) modified recently$",
    r"^queue depth growing: before=\d+ after=\d+$",
    r"^unexpected writer holds brainlayer db: pid=\d+ command=.+ fd=\S+$",
)


def _is_deliberate_deferral(reason: str) -> bool:
    return "failed to resume" not in reason.casefold() and any(
        re.fullmatch(pattern, reason, flags=re.IGNORECASE) for pattern in DELIBERATE_DEFERRALS
    )


class MaintenanceAbort(RuntimeError):
    def __init__(self, reason: str, *, code: int = 75, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class LsofEntry:
    pid: int
    command: str
    fd: str
    path: str


@dataclass
class StaleQueueResult:
    scanned_files: int = 0
    candidate_files: int = 0
    quarantined_files: int = 0
    kept_files: int = 0
    invalid_files: int = 0
    dry_run: bool = False
    quarantine_dir: str | None = None


@dataclass
class MaintenanceResult:
    mode: str
    dry_run: bool
    stale_queue: StaleQueueResult = field(default_factory=StaleQueueResult)
    burn: BurnDrainResult | None = None
    checkpoint: tuple[int, int, int] | None = None
    db_before_bytes: int | None = None
    db_after_bytes: int | None = None
    queue_before_files: int | None = None
    queue_after_files: int | None = None
    data_dir_before_bytes: int | None = None
    data_dir_after_bytes: int | None = None
    search_latency_ms: float | None = None
    warnings: list[str] = field(default_factory=list)
    vacuum_before_bytes: int | None = None
    vacuum_after_bytes: int | None = None
    backup_status: str | None = None
    actions: list[str] = field(default_factory=list)


@dataclass
class MaintenanceConfig:
    db_path: Path = field(default_factory=get_db_path)
    queue_dir: Path = field(default_factory=get_queue_dir)
    quarantine_root: Path = field(default_factory=lambda: Path.home() / ".brainlayer" / "quarantine" / "stale-queue")
    log_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "BRAINLAYER_MAINTENANCE_LOG_PATH",
                Path.home() / ".local" / "share" / "brainlayer" / "logs" / "maintenance.log",
            )
        )
    )
    repo_root: Path = field(default_factory=lambda: Path(os.environ.get("BRAINLAYER_REPO_ROOT", Path.cwd())))
    now_fn: Callable[[], dt.datetime] = field(default_factory=lambda: lambda: dt.datetime.now().astimezone())
    quiet_window_start_hour: int = 4
    quiet_window_duration_minutes: int = 120
    idle_sample_seconds: float = 5.0
    recent_write_grace_seconds: float = 180.0
    expected_writer_patterns: Sequence[str] = EXPECTED_WRITER_PATTERNS
    backup_staging_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("BRAINLAYER_BACKUP_STAGING_DIR", DEFAULT_STAGING_DIR))
    )
    backup_log_path: Path | None = None
    backup_wait_timeout_seconds: float = 2 * 60 * 60
    backup_wait_poll_seconds: float = 30
    backup_reuse_max_age_hours: float = 6


def run_command(args: Sequence[str], *, check: bool = True, timeout=None) -> subprocess.CompletedProcess[str]:
    if timeout is None and args and args[0] == "launchctl":
        timeout = QUIESCE_EXIT_TIMEOUT_SECONDS
    return subprocess.run(list(args), text=True, capture_output=True, check=check, timeout=timeout)


def _write_log(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": dt.datetime.now(dt.UTC).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _emit_telemetry(event: dict[str, Any]) -> None:
    try:
        from .telemetry import emit

        emit(MAINTENANCE_DATASET, event)
    except Exception:
        return


def _minutes_since_midnight(now: dt.datetime) -> int:
    return now.hour * 60 + now.minute


def _check_quiet_window(config: MaintenanceConfig) -> None:
    now = config.now_fn()
    start = config.quiet_window_start_hour * 60
    end = (start + config.quiet_window_duration_minutes) % (24 * 60)
    current = _minutes_since_midnight(now)
    if start <= end:
        in_window = start <= current < end
    else:
        in_window = current >= start or current < end
    if not in_window:
        raise MaintenanceAbort(
            f"outside quiet window: now={now.isoformat()} start_hour={config.quiet_window_start_hour} "
            f"duration_minutes={config.quiet_window_duration_minutes}"
        )


def _queue_files(queue_dir: Path) -> list[Path]:
    if not queue_dir.exists():
        return []
    return sorted(queue_dir.glob("*.jsonl"))


def _check_idle(config: MaintenanceConfig) -> None:
    files_before = _queue_files(config.queue_dir)
    if config.recent_write_grace_seconds > 0:
        now_ts = config.now_fn().timestamp()
        recent = [
            path
            for path in files_before
            if path.exists() and now_ts - path.stat().st_mtime < config.recent_write_grace_seconds
        ]
        if recent:
            raise MaintenanceAbort(f"recent queue write activity: {len(recent)} file(s) modified recently")

    if config.idle_sample_seconds > 0:
        before = len(files_before)
        time.sleep(config.idle_sample_seconds)
        after = len(_queue_files(config.queue_dir))
        if after > before:
            raise MaintenanceAbort(f"queue depth growing: before={before} after={after}")


def _process_command_line(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    command = result.stdout.strip()
    return command or None


def collect_lsof_entries(paths: Sequence[Path]) -> list[LsofEntry]:
    existing = [str(path) for path in paths if path.exists()]
    if not existing:
        return []
    try:
        result = subprocess.run(
            ["lsof", "-F", "pcfn", "--", *existing],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MaintenanceAbort("lsof not found; cannot prove database writer cleanliness") from exc
    if result.returncode not in {0, 1}:
        raise MaintenanceAbort(f"lsof failed: {result.stderr.strip()}")

    entries: list[LsofEntry] = []
    pid: int | None = None
    command = ""
    fd = ""
    ps_cache: dict[int, str | None] = {}
    for line in result.stdout.splitlines():
        if not line:
            continue
        kind, value = line[0], line[1:]
        if kind == "p":
            try:
                pid = int(value)
            except ValueError:
                pid = None
            command = ""
            fd = ""
        elif kind == "c":
            command = value
        elif kind == "f":
            fd = value
        elif kind == "n" and pid is not None:
            if pid not in ps_cache:
                ps_cache[pid] = _process_command_line(pid)
            full_command = ps_cache[pid] or command
            entries.append(LsofEntry(pid=pid, command=full_command, fd=fd, path=value))
    return entries


def _is_write_fd(fd: str) -> bool:
    return "w" in fd or "u" in fd


def _is_expected_writer(entry: LsofEntry, patterns: Sequence[str]) -> bool:
    haystack = f"{entry.command} {entry.path}"
    return any(pattern in haystack for pattern in patterns)


def _check_lsof_clean(config: MaintenanceConfig) -> None:
    paths = [config.db_path, Path(f"{config.db_path}-wal"), Path(f"{config.db_path}-shm")]
    entries = collect_lsof_entries(paths)
    unexpected = [
        entry
        for entry in entries
        if _is_write_fd(entry.fd) and not _is_expected_writer(entry, config.expected_writer_patterns)
    ]
    if unexpected:
        details = ", ".join(f"pid={entry.pid} command={entry.command!r} fd={entry.fd}" for entry in unexpected)
        raise MaintenanceAbort(f"unexpected writer holds BrainLayer DB: {details}")


def _read_queue_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object queue event in {path.name}")
        events.append(value)
    return events


def _event_kind(event: dict[str, Any]) -> str:
    return str(event.get("kind") or "")


def _already_enriched(enrich_status: str | None, enriched_at: str | None) -> bool:
    return enrich_status == "success" or bool(enriched_at)


def _chunk_states(
    conn: apsw.Connection, chunk_ids: Sequence[str]
) -> dict[str, tuple[str | None, str | None, str | None]]:
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"SELECT id, content_hash, enrich_status, enriched_at FROM chunks WHERE id IN ({placeholders})",
        list(chunk_ids),
    )
    return {str(row[0]): (row[1], row[2], row[3]) for row in rows}


def _file_is_redundant_enrichment(conn: apsw.Connection, events: list[dict[str, Any]]) -> bool:
    if not events or any(_event_kind(event) != "enrichment_update" for event in events):
        return False
    chunk_ids = [str(event.get("chunk_id")) for event in events if event.get("chunk_id")]
    if len(chunk_ids) != len(events):
        return False
    states = _chunk_states(conn, chunk_ids)
    for event in events:
        if "entities" in event or str(event.get("provenance_class") or "").strip():
            return False
        chunk_id = str(event["chunk_id"])
        expected_hash = event.get("content_hash")
        if not expected_hash:
            return False
        state = states.get(chunk_id)
        if state is None:
            return False
        content_hash, enrich_status, enriched_at = state
        if content_hash != expected_hash or not _already_enriched(enrich_status, enriched_at):
            return False
    return True


def quarantine_stale_queue_files(
    *,
    db_path: Path,
    queue_dir: Path,
    quarantine_root: Path,
    dry_run: bool,
    now: dt.datetime | None = None,
) -> StaleQueueResult:
    result = StaleQueueResult(dry_run=dry_run)
    files = _queue_files(queue_dir)
    result.scanned_files = len(files)
    if not files:
        return result

    stamp = (now or dt.datetime.now(dt.UTC)).strftime("%Y%m%d-%H%M%S")
    quarantine_dir = quarantine_root / stamp
    result.quarantine_dir = str(quarantine_dir)
    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        for path in files:
            try:
                events = _read_queue_events(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                result.invalid_files += 1
                result.kept_files += 1
                continue
            if not _file_is_redundant_enrichment(conn, events):
                result.kept_files += 1
                continue
            result.candidate_files += 1
            if dry_run:
                continue
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            target = quarantine_dir / path.name
            counter = 1
            while target.exists():
                target = quarantine_dir / f"{path.stem}.{counter}{path.suffix}"
                counter += 1
            path.replace(target)
            result.quarantined_files += 1
    finally:
        conn.close()
    return result


def _launchd_label(service: str) -> str:
    if service == "fleet-watchdog":
        return "com.etanhey.brainlayer-fleet-watchdog"
    return f"com.brainlayer.{service}"


def _bootout_service(service: str) -> bool:
    result = run_command(
        ["launchctl", "bootout", f"gui/{os.getuid()}/{_launchd_label(service)}"],
        check=False,
    )
    return result.returncode == 0


def _service_is_loaded(service: str, *, timeout: float | None = None) -> bool:
    try:
        state = is_launchd_label_loaded(
            _launchd_label(service),
            command_runner=lambda args: run_command(args, check=False, timeout=timeout),
        )
    except Exception as exc:
        raise MaintenanceAbort(
            f"cannot determine whether launchd service {service} is loaded ({type(exc).__name__})",
            detail=f"state:{_launchd_label(service)}",
        ) from None
    if state is None:
        raise MaintenanceAbort(
            f"cannot determine whether launchd service {service} is loaded",
            detail=f"state:{_launchd_label(service)}",
        )
    return state


def _as_launchd_dir(path: Path) -> Path:
    if (path / "install.sh").exists() or (path / "brainlayer.env.example").exists():
        return path
    return path / "scripts" / "launchd"


def _launchd_dir_for_resume(repo_root: Path) -> Path:
    configured = os.environ.get("BRAINLAYER_LAUNCHD_DIR")
    if configured:
        return Path(configured)

    repo_launchd_dir = _as_launchd_dir(repo_root)
    if (repo_launchd_dir / "install.sh").exists():
        return repo_launchd_dir

    from .setup import get_launchd_dir

    return get_launchd_dir()


def _resume_service(repo_root: Path, service: str) -> None:
    if service in {"enrich", "enrichment"}:
        raise MaintenanceAbort(f"service {service} is retired; refusing to resume")
    if service in {"brainbar", "brainbar-daemon", "fleet-watchdog"}:
        # These already-installed jobs have no normal install.sh service option
        # (or own a different namespace). Restore their existing configuration.
        plist = Path.home() / "Library" / "LaunchAgents" / f"{_launchd_label(service)}.plist"
        run_command(["launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist)], check=True)
        return
    launchd_dir = _launchd_dir_for_resume(repo_root)
    run_command([str(launchd_dir / "install.sh"), service], check=True)


def _wait_for_service_exit(service: str) -> None:
    deadline = monotonic() + QUIESCE_EXIT_TIMEOUT_SECONDS
    while _service_is_loaded(service, timeout=max(0.001, deadline - monotonic())):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise MaintenanceAbort(
                f"failed to quiesce launchd service {service}; it remains loaded",
                detail=f"loaded:{_launchd_label(service)}",
            )
        sleep(min(0.1, remaining))


def _watchdog_hold_path() -> Path:
    return PAUSE_SENTINEL_PATH.with_name("fleet-watchdog-hold.json").expanduser()


@contextmanager
def _watchdog_hold_lock(path: Path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise MaintenanceAbort(
                "watchdog hold busy", detail=f"hold-active:{_launchd_label('fleet-watchdog')}"
            ) from None
        yield
    finally:
        os.close(fd)


def _watchdog_hold_owner_alive(path: Path) -> bool | None:
    try:
        record = json.loads(path.read_text())
        pid, start = record["pid"], record["pid_start_time"]
        if path.is_symlink() or type(pid) is not int or pid <= 0 or not isinstance(start, str) or not start:
            raise ValueError
        if not _pid_is_alive(pid):
            return False
        current = _pid_start_time(pid)
        if not current:
            raise ValueError
        return current == start
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        raise MaintenanceAbort(
            "watchdog hold state unknown", detail=f"hold-state:{_launchd_label('fleet-watchdog')}"
        ) from None


def _watchdog_recovery_paused(path: Path) -> bool:
    payload, active, _ = pause_sentinel_state(path.with_name("pause.sentinel"), dt.datetime.now(dt.UTC))
    return bool(active and pause_applies_to_label(payload, _launchd_label("fleet-watchdog"))) or (
        "fleet-watchdog" in _maintenance_keep_down_services()
    )


def _recover_watchdog_hold(path: Path, command_runner=None, alert_path: Path | None = None) -> bool:
    owner_alive = _watchdog_hold_owner_alive(path)
    if owner_alive is None:
        return False
    if owner_alive:
        raise MaintenanceAbort("watchdog hold owner alive", detail=f"hold-active:{_launchd_label('fleet-watchdog')}")
    if _watchdog_recovery_paused(path):
        raise MaintenanceAbort("watchdog recovery paused", detail=f"hold-state:{_launchd_label('fleet-watchdog')}")
    runner = command_runner or (lambda args: run_command(args, check=False))
    target = f"gui/{os.getuid()}"
    label = _launchd_label("fleet-watchdog")
    plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    for args in (["launchctl", "enable", f"{target}/{label}"], ["launchctl", "bootstrap", target, str(plist)]):
        result = runner(args)
        if getattr(result, "returncode", None) != 0:
            if args[1] == "bootstrap" and is_launchd_label_loaded(label, command_runner=runner) is True:
                continue  # An interrupted prior recovery may have already restored it.
            raise MaintenanceAbort("watchdog recovery failed", detail=f"hold-state:{label}")
    from .job_alerts import report

    report(
        "fleet-watchdog-hold", "BrainLayer recovered a stale fleet-watchdog hold; supervisor restored", path=alert_path
    )
    path.unlink()
    return True


def recover_fleet_watchdog_hold(
    *, path: Path | None = None, command_runner=None, alert_path: Path | None = None
) -> bool:
    """Recover only a dead/reused owner's hold; a live or unknown owner refuses."""
    path = path or _watchdog_hold_path()
    if not path.exists():
        return False
    with _watchdog_hold_lock(path):
        return _recover_watchdog_hold(path, command_runner, alert_path)


@contextmanager
def sigterm_cleanup():
    """Unwind CLI work on SIGTERM, allowing its finally to restore services."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def terminate(*_):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(143)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _quiesce_services(services: Sequence[str], booted_out: dict[str, bool]) -> None:
    if not services:
        return
    path = _watchdog_hold_path()
    with _watchdog_hold_lock(path):
        _recover_watchdog_hold(path)
        disabled = is_launchd_label_disabled(
            _launchd_label("fleet-watchdog"), command_runner=lambda args: run_command(args, check=False)
        )
        if disabled is None:
            raise MaintenanceAbort(
                "watchdog disabled state unknown", detail=f"state:{_launchd_label('fleet-watchdog')}"
            )
        if not disabled:
            start = _pid_start_time(os.getpid())
            if not start:
                raise MaintenanceAbort(
                    "watchdog owner unknown", detail=f"hold-state:{_launchd_label('fleet-watchdog')}"
                )
            _atomic_write(
                path, {"pid": os.getpid(), "pid_start_time": start, "started_at": dt.datetime.now(dt.UTC).isoformat()}
            )
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        booted_out[_WATCHDOG_DISABLED_BEFORE] = disabled
        if not disabled:
            try:
                run_command(
                    ["launchctl", "disable", f"gui/{os.getuid()}/{_launchd_label('fleet-watchdog')}"], check=True
                )
            except Exception:
                raise MaintenanceAbort(
                    "watchdog hold failed", detail=f"disable:{_launchd_label('fleet-watchdog')}"
                ) from None
    services = ("fleet-watchdog", *(service for service in services if service != "fleet-watchdog"))
    for service in services:
        try:
            booted_out[service] = bool(_bootout_service(service))
        except Exception as exc:
            raise MaintenanceAbort(
                f"failed to quiesce launchd service {service} ({type(exc).__name__})",
                detail=f"bootout:{_launchd_label(service)}",
            ) from None
        if not booted_out[service] and _service_is_loaded(service):
            raise MaintenanceAbort(
                f"failed to quiesce launchd service {service}; it remains loaded",
                detail=f"bootout:{_launchd_label(service)}",
            )
        _wait_for_service_exit(service)
        if service == "fleet-watchdog" and not disabled:
            booted_out[service] = True  # An owned hold must restore even an already-absent supervisor.


def _clean_git_env() -> dict[str, str]:
    """Ignore caller Git routing so ``-C repo_root`` remains authoritative."""
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _git_head_sha(repo_root: Path) -> str | None:
    """SHA of the code actually on disk — the code an editable install will import."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            env=_clean_git_env(),
            timeout=30,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def _git_toplevel(repo_root: Path) -> Path | None:
    """Return the checkout root only when Git can classify ``repo_root``."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            env=_clean_git_env(),
            timeout=30,
            check=True,
        )
    except Exception:
        return None
    value = out.stdout.strip()
    return Path(value).resolve() if value else None


def _git_merged_head_sha(repo_root: Path) -> str | None:
    """SHA of the merged code. Best-effort fetch; falls back to the cached ref."""
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "fetch", "--quiet", "origin", "main"],
            capture_output=True,
            text=True,
            env=_clean_git_env(),
            timeout=60,
            check=False,
        )
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "origin/main"],
            capture_output=True,
            text=True,
            env=_clean_git_env(),
            timeout=30,
            check=True,
        )
    except Exception:
        return None
    return out.stdout.strip() or None


def _assert_running_merged_code(repo_root: Path, *, strict: bool = False) -> None:
    """Refuse to run code that is not the merged code.

    A source-installed nightly job resolves imports to the working tree. On 2026-08-05
    that tree was 8 commits behind origin/main and would have run stale code. A packaged
    install is different: it has no BrainLayer checkout, and Git must not walk upward
    from site-packages into an unrelated ancestor repository such as Homebrew's tap.

    Source checkouts either match merged code or abort loudly. Installed artifacts skip
    only this checkout-specific comparison; their integrity is covered at build/release.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Under pytest the working tree is legitimately a feature branch, which is not
        # staleness. Production behaviour is unchanged; the freshness contract itself is
        # covered by tests/test_maintenance_code_freshness.py, which calls the guard
        # directly with injected SHAs.
        return
    if os.environ.get("BRAINLAYER_MAINTENANCE_ALLOW_STALE") == "1":
        print("maintenance: BRAINLAYER_MAINTENANCE_ALLOW_STALE=1 -- skipping freshness check", file=sys.stderr)
        return

    git_toplevel = _git_toplevel(repo_root)
    if git_toplevel != repo_root.resolve():
        print(
            "maintenance: installed artifact is not an editable BrainLayer checkout; "
            "skipping working-tree freshness check",
            file=sys.stderr,
        )
        return

    head = _git_head_sha(repo_root)
    merged = _git_merged_head_sha(repo_root)

    if merged is None:
        # Cannot prove freshness. Never treat "unverifiable" as "fresh".
        if strict:
            raise MaintenanceAbort(
                "maintenance aborted: cannot verify the working tree matches merged code "
                f"(HEAD={head or 'unknown'}, origin/main unavailable). "
                "Set BRAINLAYER_MAINTENANCE_ALLOW_STALE=1 to override deliberately."
            )
        print(
            f"maintenance: WARNING cannot reach origin/main to verify freshness (HEAD={head or 'unknown'})",
            file=sys.stderr,
        )
        return

    if head != merged:
        raise MaintenanceAbort(
            f"maintenance aborted: refusing to run STALE code. "
            f"working tree HEAD={head or 'unknown'} but merged origin/main={merged}. "
            "The nightly job imports the working tree (editable install), so merging is "
            "not deploying -- run `git pull` in the repo, or set "
            "BRAINLAYER_MAINTENANCE_ALLOW_STALE=1 to override deliberately."
        )


def _service_is_deliberately_paused(service: str) -> bool:
    """True when a pause sentinel names this service's launchd label.

    Maintenance must never resume a service someone deliberately stopped. Before this
    check, teardown re-installed every entry in DEFAULT_SERVICES unconditionally, so a
    human STOP was byte-identical to a maintenance pause -- which re-created the
    enrichment plist four times and cost 5,137 rows on 2026-08-04.
    """
    from datetime import UTC, datetime

    payload, active, _stale = pause_sentinel_state(PAUSE_SENTINEL_PATH, datetime.now(UTC))
    if not active:
        return False
    return pause_applies_to_label(payload, _launchd_label(service))


def _maintenance_keep_down_services() -> set[str]:
    """Services launchd reads from ``~/.config/brainlayer/brainlayer.env`` and leaves down."""
    prefix = "com.brainlayer."
    entries = os.environ.get("BRAINLAYER_MAINTENANCE_KEEP_DOWN", "").split(",")
    return {entry.removeprefix(prefix) for raw_entry in entries if (entry := raw_entry.strip())}


def _resume_services(
    repo_root: Path,
    services: Sequence[str],
    loaded_before: Mapping[str, bool] | None = None,
) -> list[tuple[str, Exception]]:
    failures: list[tuple[str, Exception]] = []
    keep_down = _maintenance_keep_down_services()
    held = loaded_before is not None and _WATCHDOG_DISABLED_BEFORE in loaded_before
    if held:
        services = (*(service for service in services if service != "fleet-watchdog"), "fleet-watchdog")
    for service in services:
        if service in {"enrich", "enrichment"}:
            print(f"service {service} is retired; leaving it down", file=sys.stderr)
            continue
        if service == "fleet-watchdog" and held:
            if loaded_before[_WATCHDOG_DISABLED_BEFORE]:
                continue  # Never undo an operator disable.
            try:
                run_command(["launchctl", "enable", f"gui/{os.getuid()}/{_launchd_label(service)}"], check=True)
            except Exception as exc:
                failures.append((service, exc))
                continue
        if loaded_before is not None and not loaded_before.get(service, False):
            print(f"maintenance did not boot out service {service}; leaving it down", file=sys.stderr)
            continue
        if service in keep_down:
            print(
                f"service {service} is listed in BRAINLAYER_MAINTENANCE_KEEP_DOWN; leaving it down",
                file=sys.stderr,
            )
            continue
        if _service_is_deliberately_paused(service):
            print(f"skipping resume of {service}: pause sentinel is active", file=sys.stderr)
            continue
        try:
            _resume_service(repo_root, service)
            if service == "fleet-watchdog" and held:
                _watchdog_hold_path().unlink(missing_ok=True)
        except Exception as exc:
            failures.append((service, exc))
    return failures


def _format_resume_failures(failures: Sequence[tuple[str, Exception]]) -> str:
    details = "; ".join(f"{service}: {failure}" for service, failure in failures)
    count = len(failures)
    noun = "service" if count == 1 else "services"
    return f"failed to resume {count} launchd {noun}: {details}"


def _checkpoint_full(db_path: Path) -> tuple[int, int, int]:
    return checkpoint(str(db_path), "FULL")


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += _file_size(child)
    return total


def _verify_search_latency(db_path: Path, *, threshold_ms: float = SEARCH_LATENCY_FAILURE_MS) -> float | None:
    """Measure warm query latency; connection setup and one cold query are excluded."""
    if not db_path.exists():
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
    try:
        has_fts = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts' LIMIT 1"
        ).fetchone()
        query, params = (
            ("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1", ("brainlayer",))
            if has_fts
            else ("SELECT id FROM chunks LIMIT 1", ())
        )
        conn.execute(query, params).fetchall()
        samples = []
        for _ in range(SEARCH_LATENCY_SAMPLES):
            started = time.perf_counter()
            conn.execute(query, params).fetchall()
            samples.append((time.perf_counter() - started) * 1000)
    finally:
        conn.close()
    latency_ms = statistics.median(samples)
    if latency_ms > threshold_ms:
        raise MaintenanceAbort(
            f"post-maintenance search latency pathological: {latency_ms:.1f}ms > {threshold_ms:.1f}ms",
            code=1,
        )
    return latency_ms


def _vacuum(db_path: Path) -> tuple[int, int]:
    before = db_path.stat().st_size
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("VACUUM")
    finally:
        conn.close()
    return before, db_path.stat().st_size


def _run_burn_until_empty(config: MaintenanceConfig, *, max_batches: int = 1000) -> BurnDrainResult:
    total = BurnDrainResult()
    for _ in range(max_batches):
        batch = burn_drain_once(
            db_path=config.db_path, queue_dir=config.queue_dir, batch_size=5000, log_path=config.log_path
        )
        total.scanned_files += batch.scanned_files
        total.applied_events += batch.applied_events
        total.skipped_verified_stale += batch.skipped_verified_stale
        total.files_deleted += batch.files_deleted
        total.failed_files += batch.failed_files
        total.checkpoints += batch.checkpoints
        if batch.failed_files or batch.files_deleted == 0:
            break
    return total


def _run_gates(config: MaintenanceConfig) -> None:
    _check_quiet_window(config)
    _check_idle(config)
    _check_lsof_clean(config)


@contextmanager
def _maintenance_lock(db_path: Path):
    """Serialize scheduled maintenance and FTS repair on the same database."""
    lock_path = db_path.parent / ".maintenance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + MAINTENANCE_LOCK_TIMEOUT_SECONDS
    with lock_path.open("a+b") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MaintenanceAbort("maintenance lock timed out", code=77) from None
                print("maintenance: waiting for another maintenance job", file=sys.stderr, flush=True)
                time.sleep(min(30, remaining))
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _serialize_maintenance(func: Callable[..., MaintenanceResult]) -> Callable[..., MaintenanceResult]:
    @functools.wraps(func)
    def wrapped(mode: str, *, config: MaintenanceConfig | None = None, dry_run: bool = False) -> MaintenanceResult:
        resolved = config or MaintenanceConfig()
        if dry_run:
            return func(mode, config=resolved, dry_run=True)
        with _maintenance_lock(resolved.db_path):
            return func(mode, config=resolved, dry_run=False)

    return wrapped


@sigterm_cleanup()
def run_coordinated_fts_repair(db_path: Path, *, repo_root: Path | None = None) -> dict[str, int]:
    """Repair the configured DB only after resident writers have been quiesced."""
    with _maintenance_lock(db_path):
        return _run_coordinated_fts_repair_unlocked(db_path, repo_root=repo_root)


def _run_coordinated_fts_repair_unlocked(db_path: Path, *, repo_root: Path | None = None) -> dict[str, int]:
    from .runtime_store import WriterRuntimeStore

    services = DEFAULT_SERVICES
    loaded_before = {service: _service_is_loaded(service) for service in services}
    booted_out: dict[str, bool] = {}
    body_error: BaseException | None = None
    try:
        _quiesce_services(services, booted_out)
        with WriterRuntimeStore(db_path) as store:
            result = store.repair_fts(rebuild_trigram=True)
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        root = repo_root or Path(os.environ.get("BRAINLAYER_REPO_ROOT", Path.cwd()))
        resume_failures = _resume_services(root, services, booted_out)
        if resume_failures and body_error is not None:
            body_error.add_note(_format_resume_failures(resume_failures))
    if resume_failures:
        raise MaintenanceAbort(_format_resume_failures(resume_failures))
    return result


def _backup_lock_is_held(staging_dir: Path) -> bool:
    lock_path = staging_dir / ".backup.lock"
    if not lock_path.exists():
        return False
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def _recent_verified_backup(config: MaintenanceConfig) -> dict[str, Any] | None:
    log_path = config.backup_log_path or _backup_log_path(None, db_path=config.db_path)
    return _recent_verified_backup_for_reuse(
        log_path,
        config.db_path,
        now=config.now_fn(),
        max_age_hours=config.backup_reuse_max_age_hours,
    )


def _remaining_quiet_window_seconds(config: MaintenanceConfig) -> int:
    now = config.now_fn()
    start = now.replace(hour=config.quiet_window_start_hour, minute=0, second=0, microsecond=0)
    if now < start:
        start -= dt.timedelta(days=1)
    end = start + dt.timedelta(minutes=config.quiet_window_duration_minutes)
    return max(0, int((end - now).total_seconds()))


def _run_bounded_weekly_backup(config: MaintenanceConfig, timeout_seconds: int) -> dict[str, Any]:
    """Use the daily process-group supervisor with a deadline inside the quiet window."""
    if WEEKLY_RETENTION != DAILY_RETENTION:
        raise MaintenanceAbort("weekly retention differs from supervised daily backup; VACUUM skipped", code=76)
    env = os.environ.copy()
    env.update(
        {
            "BRAINLAYER_DB": str(config.db_path),
            "BRAINLAYER_BACKUP_STAGING_DIR": str(config.backup_staging_dir),
            "BRAINLAYER_BACKUP_LOG_PATH": str(config.backup_log_path or _backup_log_path(None, db_path=config.db_path)),
            "BRAINLAYER_BACKUP_TIMEOUT_SECONDS": str(timeout_seconds),
            BACKUP_REUSE_VERIFIED_MAX_AGE_HOURS_ENV: str(config.backup_reuse_max_age_hours),
        }
    )
    env.pop("BRAINLAYER_BACKUP_SUPERVISED_CHILD", None)
    process = subprocess.Popen(
        [sys.executable, "-m", "brainlayer.backup_daily"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds + 10)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise MaintenanceAbort("fresh weekly backup timed out; VACUUM skipped", code=76) from exc
    if process.returncode != 0:
        if _backup_lock_is_held(config.backup_staging_dir):
            raise BackupAlreadyRunningError("backup lock acquired during fresh weekly backup")
        raise MaintenanceAbort(f"fresh weekly backup exited {process.returncode}; VACUUM skipped", code=76)
    for line in reversed(stdout.splitlines()):
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "verified" in result:
            return result
    raise MaintenanceAbort(
        f"fresh weekly backup returned no receipt ({len(stderr)} stderr bytes); VACUUM skipped", code=76
    )


def _weekly_backup(config: MaintenanceConfig) -> dict[str, Any]:
    deadline = time.monotonic() + config.backup_wait_timeout_seconds
    waited = False
    while True:
        try:
            held = _backup_lock_is_held(config.backup_staging_dir)
        except OSError as exc:
            raise MaintenanceAbort(f"backup lock unreadable ({type(exc).__name__}); VACUUM skipped", code=76) from exc
        if not held:
            try:
                receipt = _recent_verified_backup(config)
            except OSError as exc:
                raise MaintenanceAbort(
                    f"backup receipt unreadable ({type(exc).__name__}); VACUUM skipped", code=76
                ) from exc
            if receipt is not None and WEEKLY_RETENTION == DAILY_RETENTION:
                _write_log(config.log_path, {"mode": "full", "backup_status": "reused_verified", "waited": waited})
                return receipt
            remaining_window = _remaining_quiet_window_seconds(config)
            if remaining_window < 1:
                raise MaintenanceAbort("no quiet-window time remains for fresh backup; VACUUM skipped", code=76)
            try:
                backup = _run_bounded_weekly_backup(config, remaining_window)
            except BackupAlreadyRunningError:
                held = True
            except Exception as exc:
                raise MaintenanceAbort(f"weekly backup failed ({type(exc).__name__}); VACUUM skipped", code=76) from exc
            else:
                drive_file = backup.get("drive_file")
                if (
                    backup.get("verified") is not True
                    or backup.get("uploaded") is not True
                    or not isinstance(drive_file, dict)
                    or not isinstance(drive_file.get("id"), str)
                ):
                    raise MaintenanceAbort("weekly backup was not verified; VACUUM skipped", code=76)
                return backup
        waited = True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _write_log(config.log_path, {"mode": "full", "backup_status": "wait_timeout"})
            raise MaintenanceAbort("backup wait timed out; VACUUM skipped", code=76)
        _write_log(config.log_path, {"mode": "full", "backup_status": "waiting", "remaining_seconds": round(remaining)})
        time.sleep(min(config.backup_wait_poll_seconds, remaining))


@_serialize_maintenance
def run_maintenance(mode: str, *, config: MaintenanceConfig | None = None, dry_run: bool = False) -> MaintenanceResult:
    _assert_running_merged_code((config or MaintenanceConfig()).repo_root)
    if mode not in {"light", "full", "burn"}:
        raise ValueError(f"unsupported maintenance mode: {mode}")
    config = config or MaintenanceConfig()
    _run_gates(config)

    result = MaintenanceResult(mode=mode, dry_run=dry_run)
    result.db_before_bytes = _file_size(config.db_path)
    result.queue_before_files = len(_queue_files(config.queue_dir))
    result.data_dir_before_bytes = _directory_size(config.db_path.parent)
    result.stale_queue = quarantine_stale_queue_files(
        db_path=config.db_path,
        queue_dir=config.queue_dir,
        quarantine_root=config.quarantine_root,
        dry_run=True if dry_run else False,
        now=config.now_fn(),
    )
    if dry_run:
        result.actions.append(f"would run {mode} maintenance")
        if mode == "full" and _backup_lock_is_held(config.backup_staging_dir):
            result.actions.append("would wait for running backup")
        _write_log(config.log_path, {"mode": mode, "dry_run": True, "stale_queue": asdict(result.stale_queue)})
        return result

    # The daily backup is online. Wait while services still run, before quiescing
    # writers or starting the destructive VACUUM phase.
    backup = None
    backup_abort = None
    post_backup_gates_failed = False
    if mode == "full":
        try:
            backup = _weekly_backup(config)
            result.backup_status = "verified"
        except MaintenanceAbort as exc:
            if exc.code != 76:
                raise
            backup_abort = exc
            result.backup_status = "unavailable"
            result.actions.append(f"vacuum_skipped={exc.reason}")
        try:
            _run_gates(config)
        except MaintenanceAbort as exc:
            post_backup_gates_failed = True
            backup_abort = MaintenanceAbort(f"post-backup gate failed: {exc.reason}; VACUUM skipped", code=76)
            result.backup_status = "gates_failed"
            result.actions.append(f"vacuum_skipped={backup_abort.reason}")
            backup = None

    services = DEFAULT_SERVICES
    if mode == "burn":
        services = (*REFEED_SERVICES, "drain")
    resume_failures: list[tuple[str, Exception]] = []
    body_error: BaseException | None = None
    if post_backup_gates_failed:
        services = ()
    loaded_before = {service: _service_is_loaded(service) for service in services}
    booted_out: dict[str, bool] = {}
    try:
        _quiesce_services(tuple(loaded_before), booted_out)
        if not post_backup_gates_failed:
            result.checkpoint = _checkpoint_full(config.db_path)
        if mode == "full" and backup is not None:
            result.actions.append(f"verified_drive_backup={backup.get('drive_file', {}).get('id')}")
            result.vacuum_before_bytes, result.vacuum_after_bytes = _vacuum(config.db_path)
            result.checkpoint = _checkpoint_full(config.db_path)
        if mode == "burn":
            result.burn = _run_burn_until_empty(config)
            if result.burn.failed_files:
                raise MaintenanceAbort("burn drain failed; queue files preserved")
        elif not post_backup_gates_failed:
            result.stale_queue = quarantine_stale_queue_files(
                db_path=config.db_path,
                queue_dir=config.queue_dir,
                quarantine_root=config.quarantine_root,
                dry_run=False,
                now=config.now_fn(),
            )
    except BaseException as exc:
        body_error = exc
        raise
    finally:
        resume_failures = _resume_services(config.repo_root, services, booted_out)
        if resume_failures and body_error is not None:
            resume_failure_reason = _format_resume_failures(resume_failures)
            body_error.add_note(resume_failure_reason)
            if isinstance(body_error, MaintenanceAbort):
                body_error.reason = f"{body_error.reason}; {resume_failure_reason}"
                body_error.args = (body_error.reason,)

    if resume_failures:
        raise MaintenanceAbort(_format_resume_failures(resume_failures))

    result.db_after_bytes = _file_size(config.db_path)
    result.queue_after_files = len(_queue_files(config.queue_dir))
    result.data_dir_after_bytes = _directory_size(config.db_path.parent)
    result.search_latency_ms = _verify_search_latency(config.db_path)
    if result.search_latency_ms is not None and result.search_latency_ms > SEARCH_LATENCY_TARGET_MS:
        result.warnings.append(
            f"post-maintenance search latency above target: {result.search_latency_ms:.1f}ms > "
            f"{SEARCH_LATENCY_TARGET_MS:.1f}ms"
        )
    event = {
        "mode": mode,
        "dry_run": False,
        "stale_queue": asdict(result.stale_queue),
        "burn": asdict(result.burn) if result.burn else None,
        "checkpoint": result.checkpoint,
        "db_before_bytes": result.db_before_bytes,
        "db_after_bytes": result.db_after_bytes,
        "queue_before_files": result.queue_before_files,
        "queue_after_files": result.queue_after_files,
        "data_dir_before_bytes": result.data_dir_before_bytes,
        "data_dir_after_bytes": result.data_dir_after_bytes,
        "search_latency_ms": result.search_latency_ms,
        "warnings": result.warnings,
        "vacuum_before_bytes": result.vacuum_before_bytes,
        "vacuum_after_bytes": result.vacuum_after_bytes,
        "backup_status": result.backup_status,
    }
    _write_log(config.log_path, event)
    _emit_telemetry(event)
    if backup_abort is not None:
        # Exit 76 means light maintenance completed, but the required verified
        # backup was unavailable and the destructive VACUUM phase was skipped.
        raise backup_abort
    return result


def _result_to_dict(result: MaintenanceResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "dry_run": result.dry_run,
        "stale_queue": asdict(result.stale_queue),
        "burn": asdict(result.burn) if result.burn else None,
        "checkpoint": result.checkpoint,
        "db_before_bytes": result.db_before_bytes,
        "db_after_bytes": result.db_after_bytes,
        "queue_before_files": result.queue_before_files,
        "queue_after_files": result.queue_after_files,
        "data_dir_before_bytes": result.data_dir_before_bytes,
        "data_dir_after_bytes": result.data_dir_after_bytes,
        "search_latency_ms": result.search_latency_ms,
        "warnings": result.warnings,
        "vacuum_before_bytes": result.vacuum_before_bytes,
        "vacuum_after_bytes": result.vacuum_after_bytes,
        "backup_status": result.backup_status,
        "actions": result.actions,
    }


def _failure_alert(mode: str, reason: str, log_path: Path) -> str:
    return f"BrainLayer {mode} maintenance failed: {reason}. Retry after resolving the failure; inspect {log_path}"


@sigterm_cleanup()
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run safety-gated BrainLayer maintenance.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--light", action="store_true", help="Run the nightly light pass")
    modes.add_argument("--full", action="store_true", help="Run the weekly full pass")
    modes.add_argument("--burn", action="store_true", help="Run the single-writer bulk queue drain")
    parser.add_argument("--dry-run", action="store_true", help="Run gates and report actions without touching state")
    args = parser.parse_args(argv)
    mode = "full" if args.full else "burn" if args.burn else "light"
    log_path = MaintenanceConfig().log_path
    try:
        result = run_maintenance(mode, dry_run=args.dry_run)
    except MaintenanceAbort as exc:
        deferred = exc.code == 75 and _is_deliberate_deferral(exc.reason)
        if not args.dry_run:
            _write_log(
                log_path,
                {"status": "deferred" if deferred else "aborted", "mode": mode, "reason": exc.reason},
            )
            if not deferred:
                from .job_alerts import report

                report(f"maintenance-{mode}", _failure_alert(mode, exc.reason, log_path))
        print(json.dumps({"status": "aborted", "reason": exc.reason}, sort_keys=True), flush=True)
        return exc.code
    except Exception as exc:
        if not args.dry_run:
            reason = type(exc).__name__
            _write_log(log_path, {"status": "failed", "mode": mode, "reason": reason})
            from .job_alerts import report

            report(
                f"maintenance-{mode}",
                f"BrainLayer {mode} maintenance hit an unexpected error ({reason}); see {log_path}",
            )
        raise
    if not args.dry_run:
        from .job_alerts import report

        report(f"maintenance-{mode}", None)
    print(json.dumps({"status": "ok", **_result_to_dict(result)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
