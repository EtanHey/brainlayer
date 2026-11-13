"""Collectors for the outcome verdict: one health tick's ``Facts`` from files that already exist.

Read-only and cheap. Every source read runs in one bounded stage under a single absolute
deadline; a collector that fails, times out or meets a malformed source yields UNOBSERVED and
never raises past this module. What follows that stage is pure computation over what it read.
The policy that turns these facts into the dot lives in ``health_verdict``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any

from .health_verdict import UNOBSERVED, Facts, Note, backup_facts, transcript_facts

if TYPE_CHECKING:
    from .health_check import HealthCheckConfig, HealthCheckResult

COLLECTOR_TIMEOUT_S = 5.0
RECEIPT_TAIL_BYTES = 4 * 1024 * 1024
STORE_ROUNDTRIP_MAX_AGE_S = 36 * 3600
STORE_ROUNDTRIP_FILE = "store-roundtrip.json"
EMBEDDING_WINDOW_HOURS = 24

# "Couldn't look" codes: the monitor's own blind spots, never an outcome.
BLIND_CODES = frozenset(
    {
        "process_snapshot_failed",
        "job_scan_failed",
        "slow_check",
        "health_check_busy",
        "job_state_unknown",
        "supervisor_control_unavailable",
    }
)
# Outcome codes that already have a verdict check: their message becomes context on that check,
# and the check's own facts decide its state.
CHECK_FOR_CODE = {
    **dict.fromkeys(
        (
            "hotlane_dead",
            "hotlane_backlog_disabled",
            "hotlane_unloaded",
            "missing_embeddings_climbing",
            "missing_embeddings_not_draining",
            "missing_embeddings_backlog",
            "missing_embeddings_count_failed",
            "missing_embeddings_scan_incomplete",
        ),
        "embedding",
    ),
    **dict.fromkeys(("watcher_stalled", "watch_unloaded", "t3_ingest_unhealthy"), "ingest"),
    **dict.fromkeys(
        (
            "queue_backed_up",
            "drain_no_progress",
            "drain_liveness_stalled",
            "drain_progress_stalled",
            "drain_progress_unknown",
            "lock_holder_wedge",
            "drain_unloaded",
            "pending_stores_count_failed",
        ),
        "stores",
    ),
    "brain_search_canary_failed": "search",
    **dict.fromkeys(
        (
            "jsonl_backup_attempt_missing",
            "jsonl_backup_attempt_invalid",
            "jsonl_backup_attempt_absent_multiple_nights",
            "jsonl_backup_attempt_stale",
            "jsonl_backup_attempt_failed",
        ),
        "transcripts",
    ),
    "backup_daily_verification_failed": "backup_local",
}


def _aware(value: Any) -> datetime | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value is not None else None


def read_receipts(path: Path, *, tail_bytes: int = RECEIPT_TAIL_BYTES) -> list[dict[str, Any]]:
    """JSONL receipts from the end of ``path``. Missing file = observed, none on record.

    A torn or malformed line is skipped; a non-empty file with no readable receipt raises,
    so the caller reports UNOBSERVED rather than "no copy on record".
    """
    try:
        with path.expanduser().open("rb") as handle:
            size = handle.seek(0, os.SEEK_END)
            handle.seek(max(0, size - tail_bytes))
            raw = handle.read()
    except FileNotFoundError:
        return []
    lines = raw.splitlines()[1:] if size > tail_bytes else raw.splitlines()
    receipts = []
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            receipts.append(value)
    if not receipts and any(line.strip() for line in lines):
        raise ValueError(f"no readable receipt in {path}")
    return receipts


def read_drive_token(token_path: Path) -> dict[str, Any] | None:
    """The Drive token file, offline. Missing = None; anything but an object raises."""
    try:
        token = json.loads(token_path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(token, dict):
        raise ValueError("Drive token is not an object")
    return token


def drive_auth_fact(
    token: Mapping[str, Any] | None, now: datetime, events: Iterable[tuple[datetime, str] | None]
) -> str:
    """Offline Drive authorization: token shape plus the newest Drive outcome in receipts.

    No network refresh. A missing token, a token without refresh authorization or an expired
    consent is lost access. Otherwise the newest receipt event after the last consent decides.
    """
    if token is None or not token.get("refresh_token"):
        return "consent_needed"
    consented = _aware(token.get("consented_at"))
    expires = _aware(token.get("refresh_expires_at"))
    if expires is not None and now >= expires:
        return "consent_needed"
    known = [event for event in events if event is not None and (consented is None or event[0] > consented)]
    return max(known, key=lambda event: event[0])[1] if known else "ok"


WATCHER_STALL_REASONS = ("coverage_drop", "offset_lag", "file_ingestion_failure")
WATCHER_DB_PROBE_GRACE_S = 300.0


def watcher_stall_codes(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Stall signals the watcher published: its debounced alerts, a DB probe failing for
    ``WATCHER_DB_PROBE_GRACE_S`` (one failed probe is a momentary lock), and missing roots."""
    reasons = payload.get("alert_reasons", [])
    if not isinstance(reasons, list):
        raise ValueError("watcher-health.json alert_reasons is not a list")
    codes = [r for r in WATCHER_STALL_REASONS if r in reasons]
    probe_for = payload.get("db_probe_failed_for_s")
    if payload.get("db_probe_failed") is True and isinstance(probe_for, (int, float)):
        if probe_for >= WATCHER_DB_PROBE_GRACE_S:
            codes.append("db_probe_failed")
    roots = payload.get("roots_missing") or []
    codes += [f"root_missing:{r}" for r in roots if isinstance(r, str)]
    return tuple(codes)


def watcher_health_fact(path: Path) -> tuple[datetime | None, tuple[str, ...]]:
    """(heartbeat, stall codes) from one read of watcher-health.json; missing = (None, ())."""
    try:
        payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, ()
    updated = _aware(payload.get("updated_at")) if isinstance(payload, dict) else None
    if updated is None:
        raise ValueError("watcher-health.json has no updated_at")
    return updated, watcher_stall_codes(payload)


def first_pending_store(pending_path: Path) -> datetime | None:
    """``queued_at`` of the oldest store in BrainBar's append-only replay file; missing = None."""
    try:
        with pending_path.expanduser().open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                queued = _aware(record.get("queued_at")) or _aware(record.get("queued_at_epoch"))
                if queued is None:
                    raise ValueError("pending store without queued_at")
                return queued  # Append-only: the first record is the oldest.
    except FileNotFoundError:
        pass
    return None


def oldest_queue_file(queue_dir: Path) -> datetime | None:
    """mtime of the oldest file in the durable drain queue; missing or empty = None."""
    try:
        mtimes = [path.stat().st_mtime for path in queue_dir.expanduser().glob("*.jsonl")]
    except FileNotFoundError:
        return None
    return datetime.fromtimestamp(min(mtimes), UTC) if mtimes else None


def oldest_pending_store(
    replay: datetime | None, queue: datetime | None, *, not_before: Iterable[datetime | None] = ()
) -> datetime | None:
    """Oldest store still waiting: BrainBar's replay file, then the durable drain queue.

    ``queue=None`` leaves the durable queue out (empty, or its lane is paused or held). The queue
    ages from the latest of its oldest mtime and ``not_before`` (drain progress, pause end), so a
    pipe that is moving is not reported as stuck behind one old file.
    """
    oldest = [replay] if replay is not None else []
    if queue is not None:
        floor = max((t for t in not_before if t is not None), default=None)
        oldest.append(max(queue, floor) if floor is not None else queue)
    return min(oldest, default=None)


def store_roundtrip_path(db_path: Path) -> Path:
    return db_path.expanduser().resolve().parent / STORE_ROUNDTRIP_FILE


def write_store_roundtrip_receipt(path: Path, *, ok: bool, reason: str, latency_seconds: float) -> dict[str, Any]:
    """Atomically persist the newest persisted store-then-retrieve result (doctor's round trip)."""
    receipt = {
        "checked_at": datetime.now(UTC).isoformat(),
        "ok": bool(ok),
        "reason": str(reason)[:500],
        "latency_seconds": round(float(latency_seconds), 4),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        json.dump(receipt, tmp, sort_keys=True)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp.name, path)
    return receipt


def store_canary_fact(path: Path, now: datetime, seen: str | None) -> tuple[Any, str | None]:
    """The newest round-trip receipt as the store canary; older than 36 h is UNOBSERVED.

    A failure counts once per receipt (a nightly failure must not be counted again every tick);
    afterwards it is UNOBSERVED and the verdict's latch holds it. A success is current positive
    evidence for as long as the receipt is fresh, so it is reported every tick.
    """
    try:
        receipt = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return UNOBSERVED, seen
    checked = _aware(receipt.get("checked_at")) if isinstance(receipt, dict) else None
    if checked is None or not isinstance(receipt.get("ok"), bool):
        raise ValueError("store round-trip receipt is malformed")
    if (now - checked).total_seconds() > STORE_ROUNDTRIP_MAX_AGE_S or (
        not receipt["ok"] and receipt["checked_at"] == seen
    ):
        return UNOBSERVED, seen
    return receipt["ok"], receipt["checked_at"]


_UNVECTORED_SQL = """
    SELECT COUNT(*) FROM chunks c
    LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
    WHERE r.id IS NULL AND c.created_at >= ? AND c.content IS NOT NULL AND c.content != ''
      AND c.archived_at IS NULL AND c.superseded_by IS NULL AND c.aggregated_into IS NULL
"""


def embedding_window_start(prior: Mapping[str, Any], now: datetime) -> datetime:
    """Lower ``created_at`` bound of the backlog being watched.

    While a backlog is outstanding the bound stays where it was when the backlog appeared, so a
    row leaves the count only by getting its vector or by a lifecycle change (archive, supersede,
    aggregate, delete), never by ageing past a rolling window. With no backlog it is the last day.
    """
    anchored = _aware(prior.get("window_start"))
    count = prior.get("count")
    if anchored is not None and isinstance(count, int) and count > 0:
        return anchored
    return now - timedelta(hours=EMBEDDING_WINDOW_HOURS)


def recent_unvectored(db_path: Path, window_start: datetime, deadline: float) -> int:
    """Eligible chunks created since ``window_start`` still waiting for a vector (``created_at`` is indexed)."""
    conn = sqlite3.connect(f"file:{db_path.expanduser()}?mode=ro", uri=True, timeout=0.5)
    try:
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        return int(conn.execute(_UNVECTORED_SQL, (window_start.isoformat(),)).fetchone()[0])
    finally:
        conn.close()


def embedding_fact(
    count: Any, prior: Mapping[str, Any], now: datetime, window_start: datetime
) -> tuple[Any, dict[str, Any]]:
    """Backlog not shrinking while non-empty: the stall starts at the last tick it shrank or appeared.

    ``count`` is measured over the anchored window from ``embedding_window_start``, so a shrink is
    real progress and zero is a real recovery.
    """
    if count is UNOBSERVED:
        return UNOBSERVED, dict(prior)
    if count == 0:
        return None, {"count": 0, "since": None, "window_start": None}
    before, since = prior.get("count"), _aware(prior.get("since"))
    if not isinstance(before, int) or before == 0 or since is None or count < before:
        since = now
    return since, {"count": count, "since": _iso(since), "window_start": _iso(window_start)}


def embedding_activity(cpu: Any, budget: Any, db: Path, now: datetime) -> tuple[datetime | None, str | None]:
    """Only fresh daemon evidence can explain a non-draining net queue."""
    if not isinstance(cpu, dict) or cpu.get("status") != "measured" or cpu.get("db") != str(db):
        return None, None
    measured = _aware(cpu.get("measured_at"))
    if measured is None or not 0 <= (now - measured).total_seconds() <= 120:
        return None, None
    last = _aware(cpu.get("backlog_last_batch_at"))
    encoded, seconds = cpu.get("backlog_encoded"), cpu.get("backlog_cpu_seconds")
    progressing = type(encoded) is int and encoded > 0 and isinstance(seconds, (int, float)) and seconds >= 0
    # Older daemons publish only rolling ten-minute completion evidence. Never
    # invent an exact last-batch time from the file's heartbeat timestamp.
    if last is not None and not 0 <= (now - last).total_seconds():
        return None, None
    recent = last if last is not None and (now - last).total_seconds() <= 600 else None
    gate = budget.get("backlog_max_load") if isinstance(budget, dict) else None
    paused = cpu.get("backlog_state") == "high load" and type(gate) in (int, float) and gate > 0
    if not paused and not progressing and recent is None:
        return None, None
    clock = last.astimezone().strftime("%H:%M") + " (local time)" if last else "time unavailable"
    label = (
        "Embedding paused by machine load"
        if paused
        else "Embedding progressing but incoming work is keeping the queue full"
    )
    if paused:
        due = _aware(cpu.get("backlog_next_forced_batch_at"))
        label += (
            f"; next forced batch {due.astimezone().strftime('%H:%M:%S')} (local time)"
            if due is not None
            else "; next forced batch time unavailable"
        )
    return recent or (
        now - timedelta(seconds=600) if progressing else None
    ), f"{label}: {{count}} queued, last batch {clock}"


def disk_fact(db_path: Path, staging_dir: Path) -> tuple[int, int, int, int]:
    """(Available bytes, legacy-mode requirement, raw-reclaim-mode requirement, raw floor).

    Both requirements are backup_daily's own peak budget for the next backup; which one applies
    is decided by ``reclaim_capability`` evidence, never by running the write-based probe here.
    """
    from . import backup_daily as bd
    from .disk_space import measure_disk_space

    db_size = bd._database_logical_size_bytes(db_path)
    gzip_size = bd._estimated_gzip_bytes(staging_dir, db_size)
    full = bd._should_run_full_verify(None)
    legacy, reclaim = (
        bd._peak_backup_required_bytes(db_size, gzip_size, full=full, reclaim_raw=r) for r in (False, True)
    )
    return measure_disk_space(staging_dir).available_bytes, legacy, reclaim, bd.MIN_RAW_FREE_BYTES


def reclaim_capability(receipts: Iterable[Mapping[str, Any]], staging_dir: Path) -> bool | None:
    """Whether the next backup on ``staging_dir`` will use raw reclamation, from receipts only.

    backup_daily chooses the mode per run from a write-based hole-punch probe on the staging
    filesystem, and every artifact receipt records the outcome as ``reclaim_basis`` (a basis when
    the raw was reclaimed, null for the legacy path). The newest artifact receipt for this same
    staging directory is the evidence. None when there is no such receipt or it is unreadable,
    so the caller budgets the larger requirement.
    """
    try:
        target = staging_dir.expanduser().resolve()
        for receipt in reversed(list(receipts)):
            snapshot = receipt.get("snapshot")
            if "reclaim_basis" not in receipt or not isinstance(snapshot, str) or not snapshot:
                continue
            if Path(snapshot).expanduser().resolve().parent != target:
                return None  # The newest artifact is from another staging volume: not applicable.
            basis = receipt["reclaim_basis"]
            return bool(basis) if basis is None or isinstance(basis, str) else None
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError):
        return None
    return None


def disk_budget(legacy: int, reclaim: int, capability: bool | None) -> int:
    """The requirement of the mode the next backup will run; without evidence, the larger one."""
    if capability is True:
        return reclaim
    if capability is False:
        return legacy
    return max(legacy, reclaim)


def _run_bounded(jobs: Mapping[str, Callable[[], Any]], deadline: float) -> dict[str, Any]:
    """Run every job at once and wait until the absolute ``deadline`` (``time.monotonic``), no later.

    Workers are daemon threads: one stuck in a read the OS will not interrupt (a stalled mount,
    a FIFO) is abandoned at the deadline and can never hold the producer process open at exit.
    A job that is late, raises, or has not finished yields UNOBSERVED.
    """
    lock = threading.Lock()
    finished: dict[str, Any] = {}

    def run(name: str, job: Callable[[], Any]) -> None:
        try:
            value = job()
        except Exception:
            value = UNOBSERVED
        with lock:
            finished[name] = value

    threads = [
        threading.Thread(target=run, args=(name, job), name=f"brainlayer-health-facts-{name}", daemon=True)
        for name, job in jobs.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    with lock:
        return {name: finished.get(name, UNOBSERVED) for name in jobs}


def _pure(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return UNOBSERVED


def classify_issues(result: HealthCheckResult) -> tuple[tuple[str, ...], tuple[Note, ...], dict[str, str]]:
    """Split health-check issues into blind codes, notes, and context for the check each one feeds."""
    blind: list[str] = []
    notes: list[Note] = []
    context: dict[str, list[str]] = {}
    for issue in result.issues:
        if issue.code in BLIND_CODES:
            blind.append(issue.code)
        elif check := CHECK_FOR_CODE.get(issue.code):
            context.setdefault(check, []).append(issue.message[:200])
        else:
            notes.append(Note(issue.code, issue.message))
    return tuple(blind), tuple(notes), {name: "; ".join(dict.fromkeys(msgs)) for name, msgs in context.items()}


def codes_by_check(result: HealthCheckResult) -> dict[str, tuple[str, ...]]:
    """The ``CHECK_FOR_CODE`` codes that fed each check's context, codes only."""
    codes: dict[str, list[str]] = {}
    for issue in result.issues:
        if issue.code not in BLIND_CODES and (check := CHECK_FOR_CODE.get(issue.code)):
            codes.setdefault(check, []).append(issue.code)
    return {name: tuple(dict.fromkeys(found)) for name, found in codes.items()}


def search_canary_fact(result: HealthCheckResult, now: datetime) -> Any:
    """The canary health_check already ran. A failure excused by a valid snapshot episode is unobserved."""
    from .badge_state import _expected_episode

    if result.canary_status is None:
        return UNOBSERVED
    if result.canary_ok:
        return True
    excused = any(
        issue.code == "brain_search_canary_failed" and _expected_episode(issue, now) for issue in result.issues
    )
    return UNOBSERVED if excused else False


def collect_facts(
    config: HealthCheckConfig,
    result: HealthCheckResult,
    prior: Mapping[str, Any] | None,
    now: datetime,
    *,
    timeout: float = COLLECTOR_TIMEOUT_S,
) -> tuple[Facts, dict[str, Any]]:
    """Build one tick's Facts. ``prior`` is the collectors' own persisted state; returns its successor.

    ``timeout`` is the whole budget: every source read, path resolution and enumeration runs in
    the single bounded stage below under one absolute deadline.
    """
    from .backup_daily import _backup_log_path
    from .drive_credentials import resolve_paths
    from .health_check import _default_command_runner
    from .pause import pause_applies_to_label, pause_sentinel_state

    prior = prior if isinstance(prior, Mapping) else {}
    deadline = time.monotonic() + timeout
    embedding_prior = prior.get("embedding") if isinstance(prior.get("embedding"), Mapping) else {}
    window_start = embedding_window_start(embedding_prior, now)
    staging = config.backup_staging_dir

    def backup_job() -> tuple[Any, bool | None]:
        receipts = read_receipts(config.backup_log_path or _backup_log_path(None, db_path=config.db_path))
        return backup_facts(receipts), reclaim_capability(receipts, staging)

    wall_lag = (datetime.now(UTC) - now).total_seconds()
    awake_sample = time.monotonic() - wall_lag
    got = _run_bounded(
        {
            "boot": lambda: (
                _default_command_runner(["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"])
                if sys.platform == "darwin" and abs(wall_lag) < 600
                else None
            ),
            "backup": backup_job,
            "transcripts": lambda: transcript_facts(read_receipts(config.jsonl_backup_log_path)),
            "drive_token": lambda: read_drive_token(config.drive_token_path or resolve_paths()[0]),
            "watcher": lambda: watcher_health_fact(config.watcher_health_path),
            "pause": lambda: pause_sentinel_state(config.pause_sentinel_path, now),
            "drain": lambda: json.loads(config.drain_health_path.expanduser().read_text(encoding="utf-8")),
            "replay": lambda: first_pending_store(config.pending_stores_path),
            "queue": lambda: oldest_queue_file(config.queue_dir),
            "canary": lambda: store_canary_fact(
                config.store_roundtrip_path or store_roundtrip_path(config.db_path), now, prior.get("store_canary_seen")
            ),
            "hotlane_cpu": lambda: json.loads((config.db_path.parent / "hotlane-cpu.json").read_text()),
            "hotlane_budget": lambda: json.loads((config.db_path.parent / "hotlane-budget.json").read_text()),
            "unvectored": lambda: recent_unvectored(config.db_path, window_start, deadline),
            "disk": lambda: disk_fact(config.db_path, staging),
        },
        deadline,
    )
    # Pure from here on: no reads, so the deadline above is the whole budget.
    backup, capability = got["backup"] if got["backup"] is not UNOBSERVED else (UNOBSERVED, None)
    transcripts = got["transcripts"]
    drive_auth = (
        _pure(
            lambda: drive_auth_fact(
                got["drive_token"],
                now,
                [
                    backup.drive if backup is not UNOBSERVED else None,
                    transcripts[1] if transcripts is not UNOBSERVED else None,
                ],
            )
        )
        if got["drive_token"] is not UNOBSERVED
        else UNOBSERVED
    )

    pause_payload, pause_active, _ = got["pause"] if got["pause"] is not UNOBSERVED else ({}, False, False)
    paused = lambda label: pause_active and pause_applies_to_label(pause_payload, label)  # noqa: E731
    drain_paused = paused(config.drain_label)
    pause_seen = now if drain_paused else _aware(prior.get("drain_paused_seen_at"))
    drain = got["drain"] if isinstance(got["drain"], dict) else {}
    queue = None if drain_paused else got["queue"]
    pending = (
        UNOBSERVED
        if got["replay"] is UNOBSERVED or queue is UNOBSERVED
        else _pure(
            lambda: oldest_pending_store(
                got["replay"], queue, not_before=(_aware(drain.get("last_progress_at")), pause_seen)
            )
        )
    )
    canary, canary_seen = (
        got["canary"] if got["canary"] is not UNOBSERVED else (UNOBSERVED, prior.get("store_canary_seen"))
    )
    stalled_since, embedding_state = embedding_fact(got["unvectored"], embedding_prior, now, window_start)
    observed_activity = _pure(
        lambda: embedding_activity(got["hotlane_cpu"], got["hotlane_budget"], config.db_path, now)
    )
    progress_at, activity = observed_activity if observed_activity is not UNOBSERVED else (None, None)
    activity = (
        activity.format(count=got["unvectored"])
        if activity and got["unvectored"] is not UNOBSERVED and got["unvectored"] > 0
        else None
    )
    disk_state = prior.get("disk_measurement", {})
    disk_state = disk_state if isinstance(disk_state, Mapping) else {}
    measured_at = _aware(disk_state.get("measured_at"))
    reused_age = None
    if got["disk"] is not UNOBSERVED:
        disk_state = {"measured_at": now.isoformat(), "values": list(got["disk"])}
    elif measured_at is not None and 0 <= (now - measured_at).total_seconds() <= 1800:
        values = disk_state.get("values")
        if isinstance(values, list) and len(values) == 4 and all(type(v) is int and v >= 0 for v in values):
            got["disk"] = tuple(values)
            reused_age = int((now - measured_at).total_seconds())
    if got["disk"] is not UNOBSERVED:
        free, legacy_peak, reclaim_peak, floor = got["disk"]
        peak = disk_budget(legacy_peak, reclaim_peak, capability)
    else:
        free = peak = floor = UNOBSERVED
    blind, notes, context = classify_issues(result)
    if reused_age is not None:
        context["disk"] = f"using earlier measurement ({reused_age}s old; maximum age 1800s)"
    if drain_paused:
        context["stores"] = "; ".join(filter(None, ("drain paused by maintenance", context.get("stores"))))

    # Identity and revocation travel with the clocks, so an invalidated copy cannot stay the clock.
    backup_kwargs = (
        backup.facts_kwargs()
        if backup is not UNOBSERVED
        else dict.fromkeys(("backup_local_verified_at", "backup_offsite_verified_at", "integrity"), UNOBSERVED)
    )
    boot = got["boot"]
    boot_session = boot.stdout.strip() if getattr(boot, "returncode", None) == 0 else None
    awake_clock = awake_sample if boot_session else None
    heartbeat, stall_codes = got["watcher"] if got["watcher"] is not UNOBSERVED else (UNOBSERVED, UNOBSERVED)
    facts = Facts(
        now=now,
        awake_clock_s=awake_clock,
        boot_session=boot_session,
        **backup_kwargs,
        transcripts_verified_at=transcripts[0] if transcripts is not UNOBSERVED else UNOBSERVED,
        drive_auth=drive_auth,
        watcher_updated_at=heartbeat,
        watcher_stall_codes=stall_codes,
        ingest_paused=paused(config.watch_label),
        stores_paused=drain_paused,
        pause_ends_at=_aware(pause_payload.get("expires_at")) if pause_active else None,
        oldest_pending_store_at=pending,
        store_canary_ok=canary,
        search_canary_ok=search_canary_fact(result, now),
        embedding_stalled_since=stalled_since,
        embedding_activity=activity,
        embedding_progress_recent=progress_at is not None,
        embedding_paused_by_load=bool(activity and got["hotlane_cpu"].get("backlog_state") == "high load"),
        disk_free_bytes=free,
        disk_peak_bytes=peak,
        disk_floor_bytes=floor,
        blind_codes=blind,
        notes=notes,
        check_notes=context,
        check_codes=codes_by_check(result),
    )
    state = {"store_canary_seen": canary_seen, "embedding": embedding_state, "drain_paused_seen_at": _iso(pause_seen)}
    state["disk_measurement"] = disk_state
    return facts, state
