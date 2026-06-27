"""Read-mostly BrainLayer doctor gate.

`brainlayer health-check` is allowed to heal; doctor is a loud gate. It reports
fatal issues with a non-zero exit and never kickstarts, bootstraps, or rebuilds
indexes.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .alarm import BrainLayerAlarm, emit_alarm, raise_alarm
from .deploy_drift import DEFAULT_DEPLOY_DRIFT_LABELS, default_deploy_provenance_dir, detect_deploy_drift
from .drain_liveness import (
    DEFAULT_DRAIN_LIVENESS_STALE_SECONDS,
    ENRICH_DAILY_COST_COUNTER_FILENAME,
    STALLED_CODE,
    check_drain_liveness,
)
from .health_check import (
    DEFAULT_DRAIN_LABEL,
    DEFAULT_ENRICHMENT_LABEL,
    DEFAULT_HOTLANE_LABEL,
    DEFAULT_WATCH_LABEL,
    CommandRunner,
    _default_command_runner,
    _default_ps_output,
    _load_json,
    _queue_stats,
    count_missing_embeddings,
    parse_hotlane_processes,
)
from .launchd_primitive import (
    LaunchdLabelNotLoadedError,
    LaunchdVerificationError,
    is_launchd_label_loaded,
    verify_launchd_label_loaded,
)
from .paths import get_db_path
from .search_repo import clear_hybrid_search_cache
from .vector_store import VectorStore

DEFAULT_RECENT_WINDOW_HOURS = 24
DEFAULT_ROUNDTRIP_TIMEOUT_SECONDS = 60.0
DEFAULT_QUEUE_WARNING_COUNT = 25
DEFAULT_QUEUE_MOVEMENT_SAMPLE_SECONDS = 10.0
DOCTOR_PROBE_PROJECT = "brainlayer-doctor"


@dataclass(frozen=True)
class DoctorIssue:
    code: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DoctorConfig:
    db_path: Path = field(default_factory=get_db_path)
    queue_dir: Path = field(default_factory=lambda: Path("~/.brainlayer/queue").expanduser())
    watcher_health_path: Path = field(
        default_factory=lambda: Path("~/.local/share/brainlayer/watcher-health.json").expanduser()
    )
    drain_health_path: Path = field(
        default_factory=lambda: Path("~/.local/share/brainlayer/drain-health.json").expanduser()
    )
    hotlane_label: str = DEFAULT_HOTLANE_LABEL
    watch_label: str = DEFAULT_WATCH_LABEL
    drain_label: str = DEFAULT_DRAIN_LABEL
    enrichment_label: str = DEFAULT_ENRICHMENT_LABEL
    recent_window_hours: int = DEFAULT_RECENT_WINDOW_HOURS
    roundtrip_timeout_seconds: float = DEFAULT_ROUNDTRIP_TIMEOUT_SECONDS
    queue_warning_count: int = DEFAULT_QUEUE_WARNING_COUNT
    queue_movement_sample_seconds: float = DEFAULT_QUEUE_MOVEMENT_SAMPLE_SECONDS
    drain_liveness_stale_seconds: float = DEFAULT_DRAIN_LIVENESS_STALE_SECONDS
    deploy_provenance_dir: Path = field(default_factory=default_deploy_provenance_dir)
    deploy_drift_labels: tuple[str, ...] = DEFAULT_DEPLOY_DRIFT_LABELS
    deploy_drift_enabled: bool = True


@dataclass
class DoctorResult:
    checked_at: str
    ok: bool
    exit_code: int
    issues: list[DoctorIssue] = field(default_factory=list)
    chunk_count: int | None = None
    recent_unvectored_chunks: int | None = None
    missing_vectors: int | None = None
    enrichment_backlog: int | None = None
    queue_count: int | None = None
    queue_bytes: int | None = None
    hotlane_running: bool = False
    roundtrip_latency_seconds: float | None = None
    fts5_health: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ro_conn(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.expanduser()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _recent_unvectored_chunks(db_path: Path, now: datetime, window_hours: int) -> int:
    cutoff = (now - timedelta(hours=window_hours)).isoformat()
    with _ro_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks c
            LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
            WHERE r.id IS NULL
              AND c.created_at >= ?
              AND c.content IS NOT NULL
              AND c.content != ''
              AND c.archived_at IS NULL
              AND c.superseded_by IS NULL
              AND c.aggregated_into IS NULL
              AND COALESCE(c.archived, 0) = 0
              AND COALESCE(c.status, 'active') = 'active'
            """,
            (cutoff,),
        ).fetchone()
    return int(row[0] if row else 0)


def _enrichment_backlog(db_path: Path) -> int:
    with _ro_conn(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*)
            FROM chunks
            WHERE enriched_at IS NULL
              AND enrich_status IS NULL
              AND COALESCE(char_count, length(content), 0) >= 50
              AND content IS NOT NULL
              AND content != ''
              AND archived_at IS NULL
              AND superseded_by IS NULL
              AND aggregated_into IS NULL
              AND COALESCE(archived, 0) = 0
              AND COALESCE(status, 'active') = 'active'
            """
        ).fetchone()
    return int(row[0] if row else 0)


def _probe_embedding(text: str) -> list[float]:
    if "doctor vector roundtrip probe" in text:
        return [1.0] + [0.0] * 1023
    return [0.0, 1.0] + [0.0] * 1022


def _cleanup_probe(store: VectorStore, chunk_id: str) -> None:
    cursor = store.conn.cursor()
    for table in ("chunk_vectors_binary", "chunk_vectors"):
        try:
            cursor.execute(f"DELETE FROM {table} WHERE chunk_id = ?", (chunk_id,))
        except Exception:
            pass
    cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
    clear_hybrid_search_cache(getattr(store, "db_path", None))


def _is_retryable_probe_writer_conflict(reason: str) -> bool:
    lowered = reason.lower()
    return (
        "another writer is using" in lowered
        or "database is locked" in lowered
        or "database table is locked" in lowered
        or "sqlite_busy" in lowered
    )


def _health_updated_recently(health: dict[str, Any], now: datetime, max_age_seconds: float) -> bool:
    raw_updated_at = health.get("updated_at")
    if not isinstance(raw_updated_at, str) or not raw_updated_at:
        return False
    try:
        updated_at = datetime.fromisoformat(raw_updated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    age_seconds = max(0.0, now.timestamp() - updated_at.astimezone(UTC).timestamp())
    return age_seconds < max(0.0, max_age_seconds)


def _roundtrip_probe(db_path: Path, timeout_seconds: float) -> tuple[bool, float, str]:
    started = time.monotonic()
    chunk_id = f"doctor-probe-{uuid.uuid4().hex}"
    content = f"doctor vector roundtrip probe {chunk_id}"
    setup_deadline = started + timeout_seconds
    last_retryable_reason = ""
    while True:
        store: VectorStore | None = None
        try:
            store = VectorStore(db_path)
            cursor = store.conn.cursor()
            cursor.execute(
                """
                INSERT INTO chunks (
                    id, content, metadata, source_file, project, content_type,
                    value_type, char_count, source, tags, summary, created_at,
                    enriched_at, enrich_status, chunk_origin, seen_count, last_seen_at,
                    content_class
                ) VALUES (?, ?, ?, 'doctor-probe', ?, 'doctor_probe',
                    'HIGH', ?, 'doctor', ?, ?, ?, NULL, NULL, 'raw', 1, ?,
                    'operational')
                """,
                (
                    chunk_id,
                    content,
                    json.dumps({"doctor_probe": True}),
                    DOCTOR_PROBE_PROJECT,
                    len(content),
                    json.dumps(["doctor-probe"]),
                    content,
                    datetime.now(UTC).isoformat(),
                    datetime.now(UTC).isoformat(),
                ),
            )
            store._upsert_chunk_vector(cursor, chunk_id, _probe_embedding(content))
            clear_hybrid_search_cache(getattr(store, "db_path", None))

            deadline = time.monotonic() + timeout_seconds
            while True:
                results = store.hybrid_search(
                    query_embedding=_probe_embedding(content),
                    query_text="no_keyword_match_for_brainlayer_doctor_probe",
                    n_results=1,
                    project_filter=DOCTOR_PROBE_PROJECT,
                    content_type_filter="doctor_probe",
                    source_filter="doctor",
                    include_operational=True,
                )
                ids = results.get("ids") or [[]]
                if ids and ids[0] and ids[0][0] == chunk_id:
                    return True, time.monotonic() - started, "vector_retrieved"
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            return False, time.monotonic() - started, "probe_not_vector_retrievable"
        except Exception as exc:
            reason = str(exc)
            if _is_retryable_probe_writer_conflict(reason) and time.monotonic() < setup_deadline:
                last_retryable_reason = reason
                time.sleep(0.05)
                continue
            return False, time.monotonic() - started, reason or last_retryable_reason
        finally:
            if store is not None:
                try:
                    _cleanup_probe(store, chunk_id)
                finally:
                    store.close()


def _check_launchd(
    result: DoctorResult,
    *,
    label: str,
    issue_code: str,
    message: str,
    command_runner: CommandRunner,
) -> bool | None:
    try:
        return verify_launchd_label_loaded(label, command_runner=command_runner)
    except LaunchdLabelNotLoadedError as exc:
        result.issues.append(DoctorIssue(issue_code, "fatal", message, exc.issue_details()))
        return False
    except LaunchdVerificationError as exc:
        result.issues.append(
            DoctorIssue(
                f"{issue_code}_unknown",
                "warning",
                f"could not determine whether {label} is loaded",
                exc.issue_details(),
            )
        )
        return None


def _counter_increased(before: Any, after: Any) -> bool:
    return isinstance(before, int) and isinstance(after, int) and after > before


def _check_deploy_drift(
    *,
    labels: tuple[str, ...],
    provenance_dir: Path,
    command_runner: CommandRunner,
) -> None:
    for label in labels:
        loaded = is_launchd_label_loaded(label, command_runner=command_runner)
        if loaded is not True:
            continue
        finding = detect_deploy_drift(label, provenance_dir)
        if finding is None:
            continue
        raise_alarm(
            "deploy_drift",
            f"daemon {label} running stale code, redeploy needed",
            finding.to_context(),
        )


def run_doctor(
    config: DoctorConfig,
    *,
    ps_output_fn: Callable[[], str] = _default_ps_output,
    command_runner: CommandRunner = _default_command_runner,
    now_fn: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DoctorResult:
    now = now_fn()
    result = DoctorResult(checked_at=now.isoformat(), ok=True, exit_code=0)

    def fatal(code: str, message: str, **details: Any) -> None:
        result.issues.append(DoctorIssue(code, "fatal", message, details))

    def warning(code: str, message: str, **details: Any) -> None:
        result.issues.append(DoctorIssue(code, "warning", message, details))

    store: VectorStore | None = None
    try:
        store = VectorStore(config.db_path, readonly=True)
        result.chunk_count = int(store.count())
        if result.chunk_count <= 0:
            fatal("zero_chunks", "BrainLayer DB has zero chunks")

        fts_health = store.check_fts5_health(cache_ttl_seconds=0, auto_rebuild=False)
        result.fts5_health = fts_health
        if not bool(fts_health.get("synced")):
            fatal(
                "fts5_desync",
                "FTS5 index is out of sync; doctor will not rebuild it",
                chunk_count=fts_health.get("chunk_count"),
                fts_count=fts_health.get("fts_count"),
                desync_pct=fts_health.get("desync_pct"),
            )
    except Exception as exc:
        fatal("db_open_failed", f"could not inspect DB: {exc}")
    finally:
        if store is not None:
            store.close()

    if result.chunk_count and result.chunk_count > 0:
        try:
            result.recent_unvectored_chunks = _recent_unvectored_chunks(
                config.db_path,
                now,
                config.recent_window_hours,
            )
            if result.recent_unvectored_chunks > 0:
                fatal(
                    "recent_unvectored_chunks",
                    "recent active chunks are missing chunk_vectors_rowids rows",
                    count=result.recent_unvectored_chunks,
                    window_hours=config.recent_window_hours,
                )
        except Exception as exc:
            fatal("recent_vector_coverage_failed", f"could not check recent vector coverage: {exc}")

        try:
            result.missing_vectors = count_missing_embeddings(config.db_path)
            if result.missing_vectors and result.missing_vectors > 0:
                fatal(
                    "vector_parity_gap",
                    "active chunks are missing chunk_vectors_rowids rows",
                    count=result.missing_vectors,
                )
        except Exception as exc:
            fatal("vector_parity_failed", f"could not check vector parity: {exc}")

        ok, latency, reason = _roundtrip_probe(config.db_path, config.roundtrip_timeout_seconds)
        result.roundtrip_latency_seconds = round(latency, 4)
        if not ok:
            fatal(
                "roundtrip_vector_probe_failed",
                "store-to-hybrid vector round-trip did not return the probe",
                latency_seconds=result.roundtrip_latency_seconds,
                reason=reason,
            )

    try:
        result.enrichment_backlog = _enrichment_backlog(config.db_path)
    except Exception as exc:
        fatal("enrichment_backlog_failed", f"could not count enrichment backlog: {exc}")

    drain_loaded: bool | None = None
    for label, code, message in (
        (config.enrichment_label, "enrichment_unloaded", "enrichment launchd label is not loaded"),
        (config.hotlane_label, "hotlane_unloaded", "hot-lane launchd label is not loaded"),
        (config.watch_label, "watch_unloaded", "watch launchd label is not loaded"),
        (config.drain_label, "drain_unloaded", "drain launchd label is not loaded"),
    ):
        if label:
            loaded = _check_launchd(
                result,
                label=label,
                issue_code=code,
                message=message,
                command_runner=command_runner,
            )
            if code == "drain_unloaded":
                drain_loaded = loaded

    if config.deploy_drift_enabled:
        _check_deploy_drift(
            labels=config.deploy_drift_labels,
            provenance_dir=config.deploy_provenance_dir,
            command_runner=command_runner,
        )

    hotlane_processes = parse_hotlane_processes(ps_output_fn())
    result.hotlane_running = bool(hotlane_processes)
    if not hotlane_processes:
        fatal("hotlane_dead", "hot-lane embedding process is not running")

    if result.enrichment_backlog and result.enrichment_backlog > 0:
        warning(
            "enrichment_idle_with_backlog",
            "enrichment backlog is present; loaded-but-idle is warning-only for quota/throttle reality",
            backlog=result.enrichment_backlog,
        )

    queue_count, queue_bytes, _queue_oldest_age = _queue_stats(config.queue_dir, now)
    result.queue_count = queue_count
    result.queue_bytes = queue_bytes
    drain_health = _load_json(config.drain_health_path)
    watcher_health = _load_json(config.watcher_health_path)
    # Drain should publish heartbeat cycles even while the durable queue is empty.
    # Stale drain health plus enrichment backlog is the loaded-but-dead case that
    # the generic loaded-but-idle warning cannot distinguish from quota throttling.
    pending_drain_liveness_issue = check_drain_liveness(
        drain_label=config.drain_label,
        drain_loaded=drain_loaded,
        queue_count=queue_count,
        enrichment_backlog=result.enrichment_backlog,
        drain_health=drain_health,
        now=now,
        stale_seconds=config.drain_liveness_stale_seconds,
        enrich_cost_counter_path=config.db_path.expanduser().parent / ENRICH_DAILY_COST_COUNTER_FILENAME,
    )
    drain_total = drain_health.get("drained_total")
    drain_cycles = drain_health.get("drain_cycles")
    watcher_poll_count = watcher_health.get("poll_count")
    has_drain_backlog = queue_count > 0 or bool(result.enrichment_backlog and result.enrichment_backlog > 0)
    drain_moving = queue_count == 0
    drain_liveness_moving = not has_drain_backlog
    watcher_moving = queue_count == 0
    sampled_drain_liveness_issue = pending_drain_liveness_issue
    if has_drain_backlog:
        time.sleep(max(0.0, config.queue_movement_sample_seconds))
        next_drain_health = _load_json(config.drain_health_path)
        next_drain_total = next_drain_health.get("drained_total")
        next_drain_cycles = next_drain_health.get("drain_cycles")
        drain_total_moving = _counter_increased(drain_total, next_drain_total)
        sample_now = now_fn()
        recent_drain_heartbeat = _health_updated_recently(
            next_drain_health,
            sample_now,
            config.drain_liveness_stale_seconds,
        )
        drain_moving = queue_count == 0 or drain_total_moving
        drain_liveness_moving = (
            drain_total_moving or _counter_increased(drain_cycles, next_drain_cycles) or recent_drain_heartbeat
        )
        drain_total = next_drain_health.get("drained_total", drain_total)
        drain_cycles = next_drain_health.get("drain_cycles", drain_cycles)
        if queue_count > 0:
            next_watcher_health = _load_json(config.watcher_health_path)
            watcher_moving = _counter_increased(watcher_poll_count, next_watcher_health.get("poll_count"))
            watcher_poll_count = next_watcher_health.get("poll_count", watcher_poll_count)
        sampled_drain_liveness_issue = check_drain_liveness(
            drain_label=config.drain_label,
            drain_loaded=drain_loaded,
            queue_count=queue_count,
            enrichment_backlog=result.enrichment_backlog,
            drain_health=next_drain_health,
            now=sample_now,
            stale_seconds=config.drain_liveness_stale_seconds,
            enrich_cost_counter_path=config.db_path.expanduser().parent / ENRICH_DAILY_COST_COUNTER_FILENAME,
        )
    queue_moving = drain_moving or watcher_moving
    active_drain_liveness_issue = sampled_drain_liveness_issue if has_drain_backlog else pending_drain_liveness_issue
    if active_drain_liveness_issue is not None:
        suppress_stale_drain = (
            active_drain_liveness_issue.code == STALLED_CODE and has_drain_backlog and drain_liveness_moving
        )
        if not suppress_stale_drain:
            if isinstance(active_drain_liveness_issue, BrainLayerAlarm):
                emit_alarm(active_drain_liveness_issue)
            result.issues.append(
                DoctorIssue(
                    active_drain_liveness_issue.code,
                    active_drain_liveness_issue.severity,
                    active_drain_liveness_issue.message,
                    active_drain_liveness_issue.details,
                )
            )
    if queue_count > 0 and not queue_moving:
        fatal(
            "queue_not_moving_with_backlog",
            "durable queue has backlog and watcher/drain movement counters did not advance",
            queue_count=queue_count,
            drain_total=drain_total,
            watcher_poll_count=watcher_poll_count,
        )
    elif queue_count >= config.queue_warning_count:
        warning(
            "queue_backed_up_but_moving",
            "durable queue is backed up but movement counters are present",
            queue_count=queue_count,
            queue_bytes=queue_bytes,
            drain_total=drain_total,
            watcher_poll_count=watcher_poll_count,
        )

    result.ok = not any(issue.severity == "fatal" for issue in result.issues)
    result.exit_code = 0 if result.ok else 1
    return result
