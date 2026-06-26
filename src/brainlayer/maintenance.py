"""Safety-gated recurring maintenance for the local BrainLayer database."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import apsw

from .backup_daily import WEEKLY_RETENTION, run_backup
from .drain import BurnDrainResult, burn_drain_once
from .paths import get_db_path
from .queue_io import get_queue_dir
from .wal_checkpoint import checkpoint

MAINTENANCE_DATASET = "brainlayer-maintenance"
DEFAULT_SERVICES = ("watch", "enrichment", "index", "drain")
REFEED_SERVICES = ("watch", "enrichment", "index")
EXPECTED_WRITER_PATTERNS = (
    "BrainBar",
    "brainlayer watch",
    "brainlayer enrich",
    "brainlayer index",
    "brainlayer.drain",
    "drain_daemon.py",
    "com.brainlayer.",
)


class MaintenanceAbort(RuntimeError):
    def __init__(self, reason: str, *, code: int = 75) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


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
    vacuum_before_bytes: int | None = None
    vacuum_after_bytes: int | None = None
    actions: list[str] = field(default_factory=list)


@dataclass
class MaintenanceConfig:
    db_path: Path = field(default_factory=get_db_path)
    queue_dir: Path = field(default_factory=get_queue_dir)
    quarantine_root: Path = field(default_factory=lambda: Path.home() / ".brainlayer" / "quarantine" / "stale-queue")
    log_path: Path = field(
        default_factory=lambda: Path.home() / ".local" / "share" / "brainlayer" / "logs" / "maintenance.log"
    )
    repo_root: Path = field(default_factory=lambda: Path(os.environ.get("BRAINLAYER_REPO_ROOT", Path.cwd())))
    now_fn: Callable[[], dt.datetime] = field(default_factory=lambda: lambda: dt.datetime.now().astimezone())
    quiet_window_start_hour: int = 4
    quiet_window_duration_minutes: int = 120
    idle_sample_seconds: float = 5.0
    recent_write_grace_seconds: float = 180.0
    expected_writer_patterns: Sequence[str] = EXPECTED_WRITER_PATTERNS


def run_command(args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), text=True, capture_output=True, check=check)


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
    return f"com.brainlayer.{service}"


def _bootout_service(service: str) -> None:
    run_command(["launchctl", "bootout", f"gui/{os.getuid()}/{_launchd_label(service)}"], check=False)


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


def _verify_enrichment_template_flex_backend(repo_root_or_launchd_dir: Path) -> None:
    launchd_dir = _as_launchd_dir(repo_root_or_launchd_dir)
    env_template_path = launchd_dir / "brainlayer.env.example"
    if not env_template_path.exists():
        raise MaintenanceAbort(f"BrainLayer env template not found: {env_template_path}")
    env_template = env_template_path.read_text(encoding="utf-8")
    service_tier = None
    for raw_line in env_template.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "BRAINLAYER_GEMINI_SERVICE_TIER":
            service_tier = value.strip().strip("'\"").lower()
    if service_tier != "flex":
        raise MaintenanceAbort("BrainLayer env template no longer uses Gemini Flex backend")


def _resume_service(repo_root: Path, service: str) -> None:
    launchd_dir = _launchd_dir_for_resume(repo_root)
    if service == "enrichment":
        _verify_enrichment_template_flex_backend(launchd_dir)
    run_command([str(launchd_dir / "install.sh"), service], check=True)


def _quiesce_services(services: Sequence[str]) -> None:
    for service in services:
        _bootout_service(service)


def _resume_services(repo_root: Path, services: Sequence[str]) -> list[tuple[str, Exception]]:
    failures: list[tuple[str, Exception]] = []
    for service in services:
        try:
            _resume_service(repo_root, service)
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


def _verify_search_latency(db_path: Path, *, threshold_ms: float = 50.0) -> float | None:
    if not db_path.exists():
        return None
    started = time.perf_counter()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=1)
    try:
        has_fts = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts' LIMIT 1"
        ).fetchone()
        if has_fts:
            conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT 1", ("brainlayer",)).fetchall()
        else:
            conn.execute("SELECT id FROM chunks LIMIT 1").fetchall()
    finally:
        conn.close()
    latency_ms = (time.perf_counter() - started) * 1000
    if latency_ms > threshold_ms:
        raise MaintenanceAbort(f"post-maintenance search latency too high: {latency_ms:.1f}ms > {threshold_ms:.1f}ms")
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


def run_maintenance(mode: str, *, config: MaintenanceConfig | None = None, dry_run: bool = False) -> MaintenanceResult:
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
        _write_log(config.log_path, {"mode": mode, "dry_run": True, "stale_queue": asdict(result.stale_queue)})
        return result

    services = DEFAULT_SERVICES
    if mode == "burn":
        services = (*REFEED_SERVICES, "drain")
    resume_failures: list[tuple[str, Exception]] = []
    body_error: BaseException | None = None
    _quiesce_services(services)
    try:
        result.checkpoint = _checkpoint_full(config.db_path)
        if mode == "full":
            backup = run_backup(retention_policy=WEEKLY_RETENTION)
            result.actions.append(f"verified_drive_backup={backup.get('drive_file', {}).get('id')}")
            result.vacuum_before_bytes, result.vacuum_after_bytes = _vacuum(config.db_path)
            result.checkpoint = _checkpoint_full(config.db_path)
        if mode == "burn":
            result.burn = _run_burn_until_empty(config)
            if result.burn.failed_files:
                raise MaintenanceAbort("burn drain failed; queue files preserved")
        else:
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
        resume_failures = _resume_services(config.repo_root, services)
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
        "vacuum_before_bytes": result.vacuum_before_bytes,
        "vacuum_after_bytes": result.vacuum_after_bytes,
    }
    _write_log(config.log_path, event)
    _emit_telemetry(event)
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
        "vacuum_before_bytes": result.vacuum_before_bytes,
        "vacuum_after_bytes": result.vacuum_after_bytes,
        "actions": result.actions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run safety-gated BrainLayer maintenance.")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--light", action="store_true", help="Run the nightly light pass")
    modes.add_argument("--full", action="store_true", help="Run the weekly full pass")
    modes.add_argument("--burn", action="store_true", help="Run the single-writer bulk queue drain")
    parser.add_argument("--dry-run", action="store_true", help="Run gates and report actions without touching state")
    args = parser.parse_args(argv)
    mode = "full" if args.full else "burn" if args.burn else "light"
    try:
        result = run_maintenance(mode, dry_run=args.dry_run)
    except MaintenanceAbort as exc:
        print(json.dumps({"status": "aborted", "reason": exc.reason}, sort_keys=True), flush=True)
        return exc.code
    print(json.dumps({"status": "ok", **_result_to_dict(result)}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
