"""Single-writer drain loop for BrainLayer's durable JSONL queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import apsw
import sqlite_vec

from ._helpers import _is_sqlite_busy_error, serialize_f32
from .agent_provenance import normalize_source_class
from .chunk_origin import detect_chunk_origin
from .chunk_write import canonical_content_hash, insert_canonical_chunk
from .content_class import classify_content_class
from .dedupe import (
    ensure_dedupe_schema,
    find_duplicate,
    merge_duplicate_chunk,
    merge_existing_chunk_content,
    merge_existing_chunk_seen,
)
from .ingest_guard import recursive_mcp_output_reason
from .paths import get_db_path
from .pause import DEFAULT_PAUSE_SENTINEL_PATH, pause_applies_to_label, pause_sentinel_state
from .provenance_integration import enqueue_provenance_resolution_for_entities
from .vector_store import _configure_writer_pragmas
from .wal_checkpoint import checkpoint_guard
from .wal_checkpoint import get_wal_size as _wal_size_bytes
from .writer_telemetry import start_writer_span

logger = logging.getLogger(__name__)
_sleep = time.sleep

_DEFAULT_DRAIN_BUSY_TIMEOUT_MS = 30000
_MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647
_DEFAULT_DRAIN_OPEN_MAX_RETRIES = 12
_DEFAULT_DRAIN_OPEN_RETRY_BASE_DELAY_MS = 250.0
_DEFAULT_DRAIN_OPEN_RETRY_MAX_DELAY_MS = 5000.0
_DEFAULT_MAX_EVENTS_PER_TRANSACTION = 5
_DEFAULT_BURN_MAX_EVENTS_PER_TRANSACTION = 100
_DEFAULT_POST_COMMIT_YIELD_MS = 10.0
_DEFAULT_MAX_ENRICHMENT_FILES_PER_CYCLE = 16
# How many pending fallback files one daemon start may put on the queue. Bounded so a
# backlog that outgrows the drain cannot be re-enqueued in full on every restart.
DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT = 50
FALLBACK_REPLAY_ON_START_ENV = "BRAINLAYER_FALLBACK_REPLAY_ON_START"
FALLBACK_REPLAY_ON_START_LIMIT_ENV = "BRAINLAYER_FALLBACK_REPLAY_ON_START_LIMIT"
_ENRICHMENT_LABEL = "com.brainlayer.enrichment"
# The post-commit WAL checkpoint is best-effort and must stay non-blocking. A
# truncating checkpoint can wedge the live writer behind long-lived readers on a
# multi-GB WAL; the scheduled wal-checkpoint job owns truncation.
_DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS = 1000
_DEFAULT_WAL_RESTART_HIGH_WATER_BYTES = 192_000_000
_DEFAULT_WAL_RESTART_MIN_INTERVAL_SECONDS = 60.0
_LAST_RESTART_ATTEMPT: dict[Path, float] = {}


def _drain_busy_timeout_ms() -> int:
    try:
        timeout_ms = int(os.environ.get("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", str(_DEFAULT_DRAIN_BUSY_TIMEOUT_MS)))
    except (TypeError, ValueError):
        return _DEFAULT_DRAIN_BUSY_TIMEOUT_MS
    if timeout_ms <= 0 or timeout_ms > _MAX_APSW_BUSY_TIMEOUT_MS:
        return _DEFAULT_DRAIN_BUSY_TIMEOUT_MS
    return timeout_ms


def _set_drain_busy_timeout_hook(conn: apsw.Connection) -> None:
    """Install drain busy_timeout before APSW connection best-practice hooks run."""
    conn.setbusytimeout(_drain_busy_timeout_ms())


def _install_drain_busy_timeout_hook() -> None:
    try:
        apsw.connection_hooks.remove(_set_drain_busy_timeout_hook)
    except ValueError:
        pass
    apsw.connection_hooks.insert(0, _set_drain_busy_timeout_hook)


_install_drain_busy_timeout_hook()


def _checkpoint_busy_timeout_ms() -> int:
    try:
        timeout_ms = int(
            os.environ.get("BRAINLAYER_DRAIN_CHECKPOINT_BUSY_TIMEOUT_MS", str(_DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS))
        )
    except (TypeError, ValueError):
        return _DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS
    if timeout_ms < 0 or timeout_ms > _MAX_APSW_BUSY_TIMEOUT_MS:
        return _DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS
    return timeout_ms


def _wal_restart_high_water_bytes() -> int:
    return _positive_int_env(
        "BRAINLAYER_WAL_RESTART_HIGH_WATER_BYTES",
        _DEFAULT_WAL_RESTART_HIGH_WATER_BYTES,
    )


def _wal_restart_min_interval_seconds() -> float:
    return _nonnegative_float_env(
        "BRAINLAYER_WAL_RESTART_MIN_INTERVAL_SECONDS",
        _DEFAULT_WAL_RESTART_MIN_INTERVAL_SECONDS,
    )


def _post_commit_checkpoint(conn: apsw.Connection, db_path: Path, log_path: Path | None = None) -> int:
    """Backfill every commit and make bounded, rate-limited WAL reset attempts."""
    checkpoints = 0
    timeout_changed = False
    try:
        with checkpoint_guard(db_path, blocking=False) as acquired:
            if not acquired:
                return 0
            conn.setbusytimeout(_checkpoint_busy_timeout_ms())
            timeout_changed = True
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            checkpoints += 1
            resolved_path = db_path.resolve()
            now = time.monotonic()
            last_attempt = _LAST_RESTART_ATTEMPT.get(resolved_path)
            restart_due = last_attempt is None or now - last_attempt >= _wal_restart_min_interval_seconds()
            if _wal_size_bytes(str(db_path)) >= _wal_restart_high_water_bytes() and restart_due:
                _LAST_RESTART_ATTEMPT[resolved_path] = now
                conn.execute("PRAGMA wal_checkpoint(RESTART)")
                checkpoints += 1
    except Exception as exc:
        if log_path is not None:
            _log(log_path, f"drain checkpoint skipped: {exc}")
        else:
            logger.debug("Drain checkpoint skipped: %s", exc)
    finally:
        if timeout_changed:
            try:
                conn.setbusytimeout(_drain_busy_timeout_ms())
            except Exception:
                logger.debug("Failed to restore drain busy timeout after checkpoint", exc_info=True)
    return checkpoints


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _nonnegative_float_env(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _max_events_per_transaction() -> int:
    return _positive_int_env("BRAINLAYER_DRAIN_MAX_EVENTS_PER_TRANSACTION", _DEFAULT_MAX_EVENTS_PER_TRANSACTION)


def _max_enrichment_files_per_cycle() -> int:
    return _positive_int_env(
        "BRAINLAYER_DRAIN_MAX_ENRICHMENT_FILES_PER_CYCLE",
        _DEFAULT_MAX_ENRICHMENT_FILES_PER_CYCLE,
    )


def _post_commit_yield_seconds() -> float:
    return _nonnegative_float_env("BRAINLAYER_DRAIN_POST_COMMIT_YIELD_MS", _DEFAULT_POST_COMMIT_YIELD_MS) / 1000.0


def _drain_open_max_retries() -> int:
    return _positive_int_env("BRAINLAYER_DRAIN_OPEN_MAX_RETRIES", _DEFAULT_DRAIN_OPEN_MAX_RETRIES)


def _drain_open_retry_delay_seconds(attempt: int) -> float:
    base_ms = _nonnegative_float_env(
        "BRAINLAYER_DRAIN_OPEN_RETRY_BASE_DELAY_MS",
        _DEFAULT_DRAIN_OPEN_RETRY_BASE_DELAY_MS,
    )
    max_ms = _nonnegative_float_env(
        "BRAINLAYER_DRAIN_OPEN_RETRY_MAX_DELAY_MS",
        _DEFAULT_DRAIN_OPEN_RETRY_MAX_DELAY_MS,
    )
    return min(base_ms * (2**attempt), max_ms) / 1000.0


@dataclass
class ApplyResult:
    chunk_id: str | None = None
    collision_chunk_id: str | None = None
    fallback_markers: tuple[FallbackReplayMarker, ...] = ()


@dataclass
class BurnDrainResult:
    scanned_files: int = 0
    applied_events: int = 0
    skipped_verified_stale: int = 0
    files_deleted: int = 0
    failed_files: int = 0
    checkpoints: int = 0


@dataclass(frozen=True)
class FallbackReplayMarker:
    path: Path
    chunk_id: str
    project: str | None = None
    origin_repo_path: Path | None = None


def _default_db_path() -> Path:
    return get_db_path()


def _default_queue_dir() -> Path:
    from .queue_io import get_queue_dir

    return get_queue_dir()


def _default_log_path() -> Path:
    from .paths import _guard_test_runtime_path

    configured = os.environ.get("BRAINLAYER_DRAIN_LOG_PATH")
    path = Path(configured).expanduser() if configured else Path.home() / ".brainlayer" / "logs" / "drain.log"
    return _guard_test_runtime_path(path, source="drain log path")


def _default_drain_health_path() -> Path:
    from .paths import _guard_test_runtime_path

    configured = os.environ.get("BRAINLAYER_DRAIN_HEALTH_PATH")
    path = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".local" / "share" / "brainlayer" / "drain-health.json"
    )
    return _guard_test_runtime_path(path, source="drain health path")


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def _open_connection(db_path: Path) -> apsw.Connection:
    max_retries = _drain_open_max_retries()
    for attempt in range(max_retries + 1):
        try:
            conn = apsw.Connection(str(db_path))
            break
        except Exception as exc:
            if not _is_busy_error(exc) or attempt >= max_retries:
                raise
            delay = _drain_open_retry_delay_seconds(attempt)
            logger.warning(
                "Drain DB open hit SQLITE_BUSY (attempt %d/%d); retrying in %.2fs",
                attempt + 1,
                max_retries + 1,
                delay,
            )
            _sleep(delay)
    else:
        raise RuntimeError("unreachable drain open retry state")
    conn.setbusytimeout(_drain_busy_timeout_ms())
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    _configure_writer_pragmas(conn)
    return conn


def _ensure_drain_db_schema(db_path: Path) -> None:
    from .vector_store import VectorStore

    store = VectorStore(db_path)
    store.close()


def _db_needs_initial_schema(db_path: Path) -> bool:
    try:
        return not db_path.exists() or db_path.stat().st_size == 0
    except OSError:
        return False


def _ensure_drain_db_schema_preserving_queue(
    db_path: Path,
    log_path: Path,
    *,
    context: str,
    force: bool = False,
) -> bool:
    if not force and not _db_needs_initial_schema(db_path):
        return True
    try:
        _ensure_drain_db_schema(db_path)
        return True
    except Exception as exc:
        if not _is_busy_error(exc):
            raise
        _log(log_path, f"{context} schema init busy; batch preserved: {exc}")
        return False


def _acquire_queue_lock(queue_dir: Path) -> int:
    fd = os.open(queue_dir, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
    except Exception:
        os.close(fd)
        raise
    return fd


def _columns(conn: apsw.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_enrichment_update_schema(conn: apsw.Connection) -> None:
    """Ensure APSW-only drain writers can persist queued enrichment fields.

    Writable VectorStore instances run the full chunks migration, but queued
    enrichment updates may be produced by a read-only supervisor and applied by
    this plain APSW drain before any VectorStore writer opens the DB.
    """
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks'").fetchone():
        return
    cols = _columns(conn, "chunks")
    for col, typ in (
        ("raw_entities_json", "TEXT"),
        ("provenance_class", "TEXT"),
        ("enrichment_model", "TEXT"),
        ("enrichment_backend", "TEXT"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")
            cols.add(col)


def _ensure_rewind_archive_schema(conn: apsw.Connection) -> None:
    """Ensure watcher source-position columns before taking the writer lock."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks'").fetchone():
        return
    cols = _columns(conn, "chunks")
    for col, typ in (
        ("source_end_offset", "INTEGER"),
        ("source_last_queued_at", "REAL"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")
            cols.add(col)


# One implementation only -- see chunk_write.canonical_content_hash. drain writes
# content_hash on its INSERT payloads, so it must not carry its own copy.
_content_hash = canonical_content_hash


def _insert_chunk(conn: apsw.Connection, values: dict[str, Any]) -> None:
    insert_canonical_chunk(conn, values, on_conflict="ignore")


def _insert_or_merge_chunk(conn: apsw.Connection, values: dict[str, Any]) -> str:
    if "content" in values and "content_class" not in values:
        values = {
            **values,
            "content_class": classify_content_class(
                str(values["content"]),
                content_type=values.get("content_type"),
                tags=values.get("tags"),
                source=values.get("source"),
                source_file=values.get("source_file"),
                project=values.get("project"),
            ),
        }
    chunk_id = values["id"]
    duplicate, _ = find_duplicate(
        conn,
        chunk_id=chunk_id,
        content=str(values["content"]),
        created_at=values.get("created_at"),
        project=values.get("project"),
        content_type=values.get("content_type"),
    )
    if duplicate is not None:
        _merge_watcher_source_position(conn, chunk_id=duplicate.canonical_chunk_id, incoming=values)
        merge_duplicate_chunk(
            conn,
            canonical_id=duplicate.canonical_chunk_id,
            duplicate_id=chunk_id,
            incoming=values,
            mechanism=duplicate.mechanism,
            hamming_distance_value=duplicate.hamming_distance,
            ensure_schema=False,
        )
        return duplicate.canonical_chunk_id
    _merge_watcher_source_position(conn, chunk_id=chunk_id, incoming=values, reactivate=True)
    if merge_existing_chunk_seen(conn, chunk_id=chunk_id, incoming=values, ensure_schema=False):
        return chunk_id
    existing = conn.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if existing:
        merge_existing_chunk_content(conn, chunk_id=chunk_id, incoming=values, ensure_schema=False)
        return chunk_id
    _insert_chunk(conn, values)
    return chunk_id


def _merge_watcher_source_position(
    conn: apsw.Connection,
    *,
    chunk_id: str,
    incoming: dict[str, Any],
    reactivate: bool = False,
) -> None:
    """Merge same-file watcher provenance without mutating cross-source canonicals."""
    if incoming.get("source") != "realtime_watcher" or incoming.get("source_end_offset") is None:
        return
    cols = _columns(conn, "chunks")
    required = {"source", "source_file", "source_end_offset", "source_last_queued_at"}
    if not required.issubset(cols):
        return
    select_cols = ["source", "source_file", "source_end_offset", "source_last_queued_at"]
    has_archived_at = "archived_at" in cols
    if has_archived_at:
        select_cols.append("archived_at")
    row = conn.execute(
        f"SELECT {', '.join(select_cols)} FROM chunks WHERE id = ?",
        (chunk_id,),
    ).fetchone()
    if not row or row[0] != "realtime_watcher" or row[1] != incoming.get("source_file"):
        return
    existing_offset = row[2]
    incoming_offset = int(incoming["source_end_offset"])
    existing_archived_at = row[4] if has_archived_at else None
    reactivating = reactivate and existing_archived_at is not None
    if reactivating:
        merged_offset = incoming_offset
    else:
        merged_offset = incoming_offset if existing_offset is None else min(int(existing_offset), incoming_offset)
    existing_queued_at = row[3]
    incoming_queued_at = incoming.get("source_last_queued_at")
    if reactivating and incoming_queued_at is not None:
        merged_queued_at = float(incoming_queued_at)
    elif incoming_queued_at is None:
        merged_queued_at = existing_queued_at
    elif existing_queued_at is None:
        merged_queued_at = float(incoming_queued_at)
    else:
        merged_queued_at = max(float(existing_queued_at), float(incoming_queued_at))
    updates: dict[str, Any] = {
        "source_end_offset": merged_offset,
        "source_last_queued_at": merged_queued_at,
    }
    if reactivating:
        updates["archived_at"] = None
        if "archived" in cols:
            updates["archived"] = 0
        if "status" in cols:
            updates["status"] = "active"
        if "value_type" in cols:
            updates["value_type"] = incoming.get("value_type") or "high"
    assignments = ", ".join(f"{column} = ?" for column in updates)
    conn.execute(f"UPDATE chunks SET {assignments} WHERE id = ?", [*updates.values(), chunk_id])


def _refresh_realtime_watcher_ingested_at(conn: apsw.Connection, chunk_id: str, ingested_at: int) -> None:
    cols = _columns(conn, "chunks")
    if "ingested_at" not in cols or "source" not in cols:
        return
    conn.execute(
        """
        UPDATE chunks
        SET ingested_at = ?
        WHERE id = ?
          AND source = 'realtime_watcher'
        """,
        (ingested_at, chunk_id),
    )


def _ensure_watcher_liveness_schema(conn: apsw.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watcher_liveness_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            ingested_at INTEGER NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watcher_liveness_ingested_at ON watcher_liveness_events(ingested_at)")


def _record_watcher_liveness(conn: apsw.Connection, chunk_id: str, ingested_at: int) -> None:
    conn.execute(
        """
        INSERT INTO watcher_liveness_events (chunk_id, ingested_at)
        VALUES (?, ?)
        """,
        (chunk_id, ingested_at),
    )
    conn.execute(
        "DELETE FROM watcher_liveness_events WHERE ingested_at < ?",
        (ingested_at - 86_400,),
    )


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    if "kind" in event:
        return event
    if "content" in event and "memory_type" in event:
        return {"kind": "store_memory", **event}
    if "session_id" in event and "content" in event:
        return {"kind": "hook_chunk", **event}
    return {"kind": "unknown", **event}


def _fallback_replay_marker(event: dict[str, Any], chunk_id: str) -> FallbackReplayMarker | None:
    raw_metadata = event.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    raw_path = event.get("fallback_source_path") or metadata.get("fallback_source_path")
    if not raw_path:
        return None
    raw_origin = event.get("origin_repo_path") or metadata.get("origin_repo_path")
    origin_repo_path = Path(str(raw_origin)).expanduser() if raw_origin else None
    marker_path = Path(str(raw_path)).expanduser()
    if origin_repo_path is not None and not marker_path.is_absolute():
        marker_path = origin_repo_path / marker_path
    project = event.get("project")
    return FallbackReplayMarker(
        path=marker_path,
        chunk_id=chunk_id,
        project=str(project) if project else None,
        origin_repo_path=origin_repo_path,
    )


def _mark_fallback_replays(markers: list[FallbackReplayMarker], log_path: Path) -> None:
    if not markers:
        return
    from .fallback_replay import mark_fallback_stored

    seen: set[tuple[Path, str]] = set()
    for marker in markers:
        key = (marker.path, marker.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            mark_fallback_stored(
                marker.path,
                chunk_id=marker.chunk_id,
                project=marker.project,
                origin_repo_path=marker.origin_repo_path,
            )
        except Exception as exc:
            _log(log_path, f"WARN: stored fallback chunk {marker.chunk_id} but could not mark {marker.path}: {exc}")


def _mark_fallback_replays_best_effort(markers: list[FallbackReplayMarker], log_path: Path) -> None:
    try:
        _mark_fallback_replays(markers, log_path)
    except Exception as exc:
        message = f"WARN: post-commit fallback replay marker update failed: {exc}"
        try:
            _log(log_path, message)
        except Exception:
            logger.warning(message)


def _events_include_store(events: list[dict[str, Any]]) -> bool:
    return any(_event_payload(event).get("kind") == "store_memory" for event in events)


def _is_missing_chunks_error(exc: BaseException) -> bool:
    return "no such table: chunks" in str(exc).lower()


def _event_created_at(event: dict[str, Any]) -> str:
    raw_created_at = event.get("created_at")
    if raw_created_at:
        return str(raw_created_at)
    raw_queued_at = event.get("queued_at")
    if isinstance(raw_queued_at, int | float):
        return datetime.fromtimestamp(float(raw_queued_at), timezone.utc).isoformat()
    if raw_queued_at:
        return str(raw_queued_at)
    return datetime.now(timezone.utc).isoformat()


def _metadata_with_enrichment_provenance(raw_metadata: Any, event: dict[str, Any]) -> str | None:
    provenance: dict[str, str] = {}
    for key in ("enrichment_model", "enrichment_backend"):
        value = str(event.get(key) or "").strip()
        if value:
            provenance[key] = value
    enriched_by = str(event.get("enriched_by") or provenance.get("enrichment_model") or "").strip()
    if enriched_by:
        provenance["enriched_by"] = enriched_by
    if not provenance:
        return None

    if isinstance(raw_metadata, dict):
        metadata = dict(raw_metadata)
    elif raw_metadata:
        try:
            parsed = json.loads(str(raw_metadata))
            metadata = parsed if isinstance(parsed, dict) else {"_previous_metadata_raw": raw_metadata}
        except (TypeError, json.JSONDecodeError):
            metadata = {"_previous_metadata_raw": raw_metadata}
    else:
        metadata = {}

    metadata.update(provenance)
    return json.dumps(metadata)


def _auto_supersede_dry_run() -> bool | None:
    raw = str(os.environ.get("BRAINLAYER_AUTO_SUPERSEDE") or "").strip().lower()
    if raw in {"", "0", "false", "off", "no"}:
        return None
    return raw != "apply"


def _enrichment_hook_chunk(
    conn: apsw.Connection,
    chunk_id: str,
    *,
    entity: str,
    provenance_class: str,
) -> dict[str, Any] | None:
    cols = _columns(conn, "chunks")
    selected = [
        col for col in ("id", "content", "content_type", "sender", "created_at", "provenance_class") if col in cols
    ]
    if not selected:
        return None
    row = conn.execute(f"SELECT {', '.join(selected)} FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if not row:
        return None
    chunk = dict(zip(selected, row, strict=False))
    chunk["entity"] = entity
    if provenance_class:
        chunk["provenance_class"] = provenance_class
    return chunk


def _run_enrichment_provenance_hooks(
    conn: apsw.Connection,
    event: dict[str, Any],
    *,
    chunk_id: str,
    provenance_class: str,
) -> None:
    entities = event.get("entities")
    if not entities:
        return
    if not conn.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone():
        return

    try:
        enqueue_provenance_resolution_for_entities(conn, entities, chunk_id=chunk_id, commit=False)
    except Exception:
        logger.exception("Failed to enqueue provenance resolution for drained chunk_id=%s", chunk_id)

    dry_run = _auto_supersede_dry_run()
    if dry_run is None:
        return

    try:
        from .provenance_autosupersede import auto_supersede
        from .provenance_integration import _entity_name_from_payload

        for entity in entities:
            entity_name = _entity_name_from_payload(entity)
            if not entity_name:
                continue
            chunk = _enrichment_hook_chunk(
                conn,
                chunk_id,
                entity=entity_name,
                provenance_class=provenance_class,
            )
            if chunk is None:
                continue
            report = auto_supersede(conn, chunk, dry_run=dry_run, commit=False)
            if (
                report.candidate_count
                or report.contradiction_count
                or report.would_supersede_count
                or report.pending_confirm_count
                or report.skipped_count
            ):
                mode = "dry_run" if dry_run else "apply"
                logger.info(
                    "drain auto_supersede %s entity=%s candidates=%s contradictions=%s would_supersede=%s "
                    "superseded=%s pending_confirm=%s skipped=%s",
                    mode,
                    report.entity,
                    report.candidate_count,
                    report.contradiction_count,
                    report.would_supersede_count,
                    report.superseded_count,
                    report.pending_confirm_count,
                    report.skipped_reason or report.skipped_count,
                )
    except Exception:
        logger.exception("drain auto_supersede failed for chunk_id=%s", chunk_id)


def _apply_store_supersedes(conn: apsw.Connection, old_chunk_id: str | None, new_chunk_id: str) -> None:
    if not old_chunk_id:
        return
    cols = _columns(conn, "chunks")
    if "superseded_by" not in cols:
        return
    if "status" in cols:
        conn.execute(
            "UPDATE chunks SET superseded_by = ?, status = 'superseded' WHERE id = ?",
            (new_chunk_id, old_chunk_id),
        )
    else:
        conn.execute("UPDATE chunks SET superseded_by = ? WHERE id = ?", (new_chunk_id, old_chunk_id))
    for vector_table in ("chunk_vectors", "chunk_vectors_binary"):
        if conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (vector_table,)).fetchone():
            conn.execute(f"DELETE FROM {vector_table} WHERE chunk_id = ?", (old_chunk_id,))


def _apply_store(conn: apsw.Connection, event: dict[str, Any]) -> ApplyResult:
    raw_content = event.get("content")
    if raw_content is None:
        logger.warning("Skipping malformed store event without content")
        return ApplyResult()
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed store event with empty content")
        return ApplyResult()
    chunk_id = event.get("chunk_id") or f"manual-{uuid.uuid4().hex[:16]}"
    recursive_reason = recursive_mcp_output_reason(
        content,
        chunk_id=chunk_id,
        source_file=event.get("source_file"),
        reject_precompact=True,
    )
    if recursive_reason:
        logger.warning("Skipping recursive MCP store event: %s", recursive_reason)
        return ApplyResult()
    now = datetime.now(timezone.utc).isoformat()
    created_at = _event_created_at(event)
    metadata = {"memory_type": event.get("memory_type", "note")}
    raw_metadata = event.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)
    elif raw_metadata:
        logger.warning("Skipping non-object store metadata for chunk_id=%s", event.get("chunk_id"))
    supersedes = event.get("supersedes") or metadata.get("supersedes")
    tags = event.get("tags")
    explicit_chunk_origin = event.get("chunk_origin") or metadata.get("chunk_origin")
    conversation_id = event.get("conversation_id") or metadata.get("conversation_id")
    position = None
    if conversation_id:
        position_row = conn.execute(
            """
            SELECT COALESCE(MAX(position), -1) + 1
            FROM chunks
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        position = int(position_row[0]) if position_row else 0
    existing = conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if existing:
        if str(existing[0]).strip() == content:
            merge_existing_chunk_seen(
                conn,
                chunk_id=chunk_id,
                incoming={
                    "id": chunk_id,
                    "content": content,
                    "tags": json.dumps(tags) if tags else None,
                    "importance": float(event["importance"]) if event.get("importance") is not None else None,
                    "created_at": created_at,
                    "last_seen_at": now,
                },
            )
            _apply_store_supersedes(conn, supersedes, chunk_id)
            marker = _fallback_replay_marker(event, chunk_id)
            return ApplyResult(chunk_id=chunk_id, fallback_markers=(marker,) if marker else ())
        return ApplyResult(collision_chunk_id=chunk_id)
    stored_chunk_id = _insert_or_merge_chunk(
        conn,
        {
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps(metadata),
            "source_file": "brainlayer-queue",
            "project": event.get("project"),
            "content_type": event.get("memory_type", "note"),
            "value_type": "high",
            "char_count": len(content),
            "source": event.get("source") or "manual",
            "created_at": created_at,
            "enriched_at": now,
            "enrich_status": "success",
            "summary": content[:200],
            "tags": json.dumps(tags) if tags else None,
            "importance": float(event["importance"]) if event.get("importance") is not None else None,
            "content_hash": _content_hash(content),
            "conversation_id": conversation_id,
            "position": position,
            "chunk_origin": detect_chunk_origin(content, explicit_chunk_origin),
        },
    )
    _apply_store_supersedes(conn, supersedes, stored_chunk_id)
    entity_id = event.get("entity_id") or metadata.get("entity_id")
    if entity_id and {"kg_entities", "kg_entity_chunks"}.issubset(_table_names(conn)):
        if conn.execute("SELECT id FROM kg_entities WHERE id = ?", (entity_id,)).fetchone():
            conn.execute(
                """
                INSERT OR REPLACE INTO kg_entity_chunks(entity_id, chunk_id, relevance, context)
                VALUES (?, ?, ?, ?)
                """,
                (entity_id, stored_chunk_id, 1.0, f"Stored via brain_store: {event.get('memory_type', 'note')}"),
            )
        else:
            logger.warning("Skipping entity link for unknown entity_id=%s chunk_id=%s", entity_id, chunk_id)
    marker_chunk_id = str(event.get("chunk_id") or stored_chunk_id)
    marker = _fallback_replay_marker(event, marker_chunk_id)
    return ApplyResult(chunk_id=stored_chunk_id, fallback_markers=(marker,) if marker else ())


def _apply_watcher(conn: apsw.Connection, event: dict[str, Any]) -> ApplyResult:
    chunk_id = event.get("chunk_id")
    raw_content = event.get("content")
    if not chunk_id or raw_content is None:
        logger.warning("Skipping malformed watcher event without chunk_id/content")
        return ApplyResult()
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed watcher event with empty content")
        return ApplyResult()
    source_file = event.get("source_file") or "realtime-watcher"
    recursive_reason = recursive_mcp_output_reason(
        content,
        chunk_id=chunk_id,
        source_file=source_file,
        reject_precompact=True,
    )
    if recursive_reason:
        logger.warning("Skipping recursive MCP watcher event: %s", recursive_reason)
        return ApplyResult()
    tags = event.get("tags")
    ingested_at = int(time.time())
    values = {
        "id": chunk_id,
        "content": content,
        "metadata": json.dumps(event.get("metadata") or {}),
        "source_file": source_file,
        "project": event.get("project"),
        "content_type": event.get("content_type") or "assistant_text",
        "value_type": event.get("value_type") or "high",
        "char_count": len(content),
        "source": "realtime_watcher",
        "created_at": event.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "ingested_at": ingested_at,
        "conversation_id": event.get("conversation_id"),
        "sender": event.get("sender"),
        "tags": json.dumps(tags) if tags else None,
        "content_hash": _content_hash(content),
        "chunk_origin": detect_chunk_origin(content, event.get("chunk_origin")),
    }
    if event.get("source_end_offset") is not None:
        values["source_end_offset"] = int(event["source_end_offset"])
    if event.get("queued_at") is not None:
        values["source_last_queued_at"] = float(event["queued_at"])
    if event.get("content_class"):
        values["content_class"] = event.get("content_class")
    if event.get("provenance_class"):
        values["provenance_class"] = event.get("provenance_class")
    source_class = normalize_source_class(event.get("source_class"))
    if source_class is not None and "source_class" in _columns(conn, "chunks"):
        values["source_class"] = source_class
    stored_chunk_id = _insert_or_merge_chunk(
        conn,
        values,
    )
    _refresh_realtime_watcher_ingested_at(conn, stored_chunk_id, ingested_at)
    _record_watcher_liveness(conn, stored_chunk_id, ingested_at)
    return ApplyResult(chunk_id=stored_chunk_id)


def _apply_rewind_archive(conn: apsw.Connection, event: dict[str, Any]) -> ApplyResult:
    source_file = str(event.get("source_file") or "").strip()
    conversation_id = str(event.get("conversation_id") or "").strip()
    try:
        old_offset = int(event.get("old_offset"))
        new_offset = int(event.get("new_offset"))
        detected_at = float(event.get("rewind_detected_at"))
    except (TypeError, ValueError):
        logger.warning("Skipping malformed rewind archive event")
        return ApplyResult()
    if not source_file or not conversation_id or old_offset <= new_offset or new_offset < 0:
        logger.warning("Skipping malformed rewind archive bounds for %s", source_file or "unknown")
        return ApplyResult()
    cols = _columns(conn, "chunks")
    if not {"source_end_offset", "source_last_queued_at", "archived_at"}.issubset(cols):
        raise RuntimeError("rewind archival schema is incomplete")
    assignments = ["archived_at = ?"]
    params: list[Any] = [datetime.now(timezone.utc).isoformat()]
    params.extend([source_file, conversation_id, new_offset, old_offset, detected_at])
    conn.execute(
        f"""
        UPDATE chunks
        SET {", ".join(assignments)}
        WHERE source = 'realtime_watcher'
          AND source_file = ?
          AND conversation_id = ?
          AND archived_at IS NULL
          AND source_end_offset IS NOT NULL
          AND source_end_offset > ?
          AND source_end_offset <= ?
          AND (source_last_queued_at IS NULL OR source_last_queued_at <= ?)
        """,
        params,
    )
    return ApplyResult()


def _apply_hook(conn: apsw.Connection, event: dict[str, Any]) -> ApplyResult:
    raw_content = event.get("content")
    if raw_content is None:
        logger.warning("Skipping malformed hook event without content")
        return ApplyResult()
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed hook event with empty content")
        return ApplyResult()
    # Truncated digest that NAMES the chunk (id suffix + queue key). It is not a
    # content hash: the content_hash column is filled by the canonical contract.
    id_digest = event.get("content_hash") or hashlib.sha256(content.encode()).hexdigest()[:16]
    session_id = event.get("session_id") or "unknown"
    chunk_id = event.get("chunk_id") or f"rt-{str(session_id)[:8]}-{id_digest}"
    source_file = event.get("source_file") or "realtime-hook"
    recursive_reason = recursive_mcp_output_reason(
        content,
        chunk_id=chunk_id,
        source_file=source_file,
        reject_precompact=True,
    )
    if recursive_reason:
        logger.warning("Skipping recursive MCP hook event: %s", recursive_reason)
        return ApplyResult()
    ts_raw = event.get("timestamp")
    try:
        timestamp = float(ts_raw) if ts_raw is not None else time.time()
    except (TypeError, ValueError):
        logger.warning("Invalid hook timestamp %r; using current time", ts_raw)
        timestamp = time.time()
    created_at = (
        str(event["created_at"])
        if event.get("created_at")
        else datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    )
    stored_chunk_id = _insert_or_merge_chunk(
        conn,
        {
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps({"session_id": session_id, "id_digest": id_digest}),
            "source_file": source_file,
            "project": event.get("project"),
            "content_type": "assistant_text",
            "value_type": "high",
            "char_count": len(content),
            "source": "realtime",
            "created_at": created_at,
            "conversation_id": session_id,
            "importance": 5,
            "content_hash": _content_hash(content),
            "chunk_origin": detect_chunk_origin(content, event.get("chunk_origin")),
        },
    )
    return ApplyResult(chunk_id=stored_chunk_id)


def _apply_enrichment(conn: apsw.Connection, event: dict[str, Any]) -> None:
    chunk_id = event.get("chunk_id")
    if not chunk_id:
        logger.warning("Skipping malformed enrichment event without chunk_id")
        return
    enrichment = event.get("enrichment") or {}
    cols = _columns(conn, "chunks")
    if "content_hash" in cols and event.get("content_hash"):
        row = conn.execute("SELECT content_hash, content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not row:
            return
        current_hash = row[0] or _content_hash(str(row[1] or ""))
        if current_hash and current_hash != event["content_hash"]:
            logger.warning("Skipping stale enrichment for chunk_id=%s content_hash mismatch", chunk_id)
            return
    updates: dict[str, Any] = {}
    mappings = {
        "summary": "summary",
        "importance": "importance",
        "intent": "intent",
        "epistemic_level": "epistemic_level",
        "version_scope": "version_scope",
        "debt_impact": "debt_impact",
        "sentiment_label": "sentiment_label",
        "sentiment_score": "sentiment_score",
    }
    for key, col in mappings.items():
        if col in cols and key in enrichment:
            updates[col] = enrichment[key]
    for key in ("tags", "primary_symbols", "external_deps", "key_facts", "resolved_queries", "sentiment_signals"):
        if key in cols and enrichment.get(key) is not None:
            updates[key] = json.dumps(enrichment[key])
    if "resolved_query" in cols:
        resolved_query = enrichment.get("resolved_query")
        if (
            not resolved_query
            and isinstance(enrichment.get("resolved_queries"), list)
            and enrichment["resolved_queries"]
        ):
            resolved_query = enrichment["resolved_queries"][0]
        if resolved_query:
            updates["resolved_query"] = resolved_query
    if "raw_entities_json" in cols and event.get("entities") is not None:
        updates["raw_entities_json"] = json.dumps(event["entities"])
    if "content_hash" in cols and event.get("content_hash"):
        updates["content_hash"] = event["content_hash"]
    provenance_class = str(event.get("provenance_class") or "").strip()
    if "provenance_class" in cols and provenance_class:
        updates["provenance_class"] = provenance_class
    enrichment_model = str(event.get("enrichment_model") or "").strip()
    if "enrichment_model" in cols and enrichment_model:
        updates["enrichment_model"] = enrichment_model
    enrichment_backend = str(event.get("enrichment_backend") or "").strip()
    if "enrichment_backend" in cols and enrichment_backend:
        updates["enrichment_backend"] = enrichment_backend
    enrichment_version = str(event.get("enrichment_version") or "").strip()
    if "enrichment_version" in cols and enrichment_version:
        updates["enrichment_version"] = enrichment_version
    if "metadata" in cols:
        row = conn.execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not row:
            return
        metadata_json = _metadata_with_enrichment_provenance(row[0], event)
        if metadata_json is not None:
            updates["metadata"] = metadata_json
    requested_status = str(enrichment.get("enrich_status") or "success").strip()
    if requested_status not in {"success", "duplicate"}:
        requested_status = "success"
    if "enriched_at" in cols:
        updates["enriched_at"] = datetime.now(timezone.utc).isoformat()
    if "enrich_status" in cols:
        updates["enrich_status"] = requested_status
    if updates:
        assignments = ", ".join(f"{col} = ?" for col in updates)
        conn.execute(f"UPDATE chunks SET {assignments} WHERE id = ?", [*updates.values(), chunk_id])
    _run_enrichment_provenance_hooks(conn, event, chunk_id=chunk_id, provenance_class=provenance_class)


def _apply_event(conn: apsw.Connection, event: dict[str, Any]) -> ApplyResult:
    event = _event_payload(event)
    kind = event.get("kind")
    if kind == "store_memory":
        return _apply_store(conn, event)
    elif kind == "watcher_chunk":
        return _apply_watcher(conn, event)
    elif kind == "hook_chunk":
        return _apply_hook(conn, event)
    elif kind == "enrichment_update":
        _apply_enrichment(conn, event)
    elif kind == "rewind_archive":
        return _apply_rewind_archive(conn, event)
    return ApplyResult()


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON in %s:%d: %s", path, lineno, exc)
                continue
            if isinstance(event, dict):
                events.append(event)
            else:
                logger.warning("Skipping non-object JSON in %s:%d", path, lineno)
    return events


def _table_names(conn: apsw.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')")}


def _is_busy_error(exc: BaseException) -> bool:
    from .vector_store import WriterInUseError

    return isinstance(exc, WriterInUseError) or _is_sqlite_busy_error(exc)


def _default_embed_fn() -> Callable[[str], list[float]]:
    from .embeddings import get_embedding_model

    model = get_embedding_model()

    def embed_text(text: str) -> list[float]:
        embeddings = model.embed_texts([text], batch_size=1)
        if len(embeddings) != 1:
            raise RuntimeError(f"single text embedder returned {len(embeddings)} embeddings")
        return embeddings[0]

    return embed_text


def _embedding_enabled() -> bool:
    return os.environ.get("BRAINLAYER_DRAIN_EMBED", "1").lower() not in {"0", "false", "no"}


def _precompute_event_embeddings(
    events: list[dict[str, Any]],
    embed_fn: Callable[[str], list[float]] | None,
) -> dict[str, bytes]:
    if not _embedding_enabled():
        return {}
    contents = []
    for event in events:
        payload = _event_payload(event)
        if payload.get("kind") not in {"store_memory", "hook_chunk", "watcher_chunk"}:
            continue
        content = str(payload.get("content") or "").strip()
        if content:
            contents.append(content)
    if not contents:
        return {}
    resolved_embed_fn = embed_fn or _default_embed_fn()
    precomputed: dict[str, bytes] = {}
    for content in dict.fromkeys(contents):
        try:
            precomputed[content] = serialize_f32(resolved_embed_fn(content))
        except Exception as exc:
            logger.warning("Failed to precompute drained embedding: %s", exc)
    return precomputed


def _insert_precomputed_embedding(conn: apsw.Connection, chunk_id: str, embedding_bytes: bytes) -> None:
    if "chunk_vectors" not in _table_names(conn):
        return
    conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
    conn.execute("INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)", (chunk_id, embedding_bytes))
    if "chunk_vectors_binary" in _table_names(conn):
        conn.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))
        conn.execute(
            "INSERT INTO chunk_vectors_binary (chunk_id, embedding) VALUES (?, vec_quantize_binary(?))",
            (chunk_id, embedding_bytes),
        )


def _quarantine_file(path: Path, log_path: Path, reason: BaseException) -> None:
    target = path.with_name(f"{path.name}.bad")
    if target.exists():
        target = path.with_name(f"{path.name}.bad-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}")
    try:
        path.replace(target)
        _log(log_path, f"skipped poison queue file {path.name}: {reason}; moved_to={target.name}")
    except OSError as exc:
        _log(log_path, f"skipped poison queue file {path.name}: {reason}; quarantine_failed={exc}")


def _unlink_processed_file(path: Path, log_path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError as exc:
        _log(log_path, f"drain committed but could not unlink {path}: {exc}")
        return False


def _is_enrichment_queue_file(path: Path) -> bool:
    return path.name.startswith("enrichment-")


def _queue_telemetry(paths: list[Path], events: list[dict[str, Any]]) -> dict[str, Any]:
    sources = sorted({str(event.get("source") or "").strip() for event in events} - {""})
    kinds = {str(event.get("kind") or "").strip() for event in events}
    if "store_memory" in kinds:
        lane = "interactive"
    elif kinds & {"watcher_chunk", "hook_chunk", "rewind_archive"}:
        lane = "realtime"
    elif kinds == {"enrichment_update"}:
        lane = "enrichment"
    else:
        lane = "background"
    mtimes: list[float] = []
    for path in paths:
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return {
        "lane": lane,
        "queue_source": ",".join(sources) if sources else "unknown",
        "queue_wait_ms": max(0.0, (time.time() - min(mtimes)) * 1000.0) if mtimes else None,
    }


def _drain_operation(events: list[dict[str, Any]], default: str) -> str:
    kinds = {str(_event_payload(event).get("kind") or "").strip() for event in events}
    return "rewind_archive" if kinds == {"rewind_archive"} else default


def _has_high_priority_queue_files(queue_dir: Path) -> bool:
    try:
        return any(not _is_enrichment_queue_file(path) for path in queue_dir.glob("*.jsonl"))
    except OSError:
        return False


def _queue_file_priority(path: Path) -> tuple[int, str]:
    return (2 if _is_enrichment_queue_file(path) else 0, path.name)


def _select_priority_queue_files(files: list[Path], batch_size: int, *, cap_enrichment: bool = True) -> list[Path]:
    ordered = sorted(files, key=_queue_file_priority)
    high_priority = [path for path in ordered if not _is_enrichment_queue_file(path)]
    if high_priority:
        return high_priority[:batch_size]
    enrichment_limit = min(batch_size, _max_enrichment_files_per_cycle()) if cap_enrichment else batch_size
    return ordered[:enrichment_limit]


def _rewrite_events_file(path: Path, events: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        "".join(json.dumps(event, ensure_ascii=True) + "\n" for event in events),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _preserve_remaining_events(path: Path, events: list[dict[str, Any]]) -> None:
    if events and all(_event_payload(event).get("kind") == "enrichment_update" for event in events):
        if not _is_enrichment_queue_file(path):
            enrichment_path = path.with_name(f"enrichment-paused-{uuid.uuid4().hex}-{path.name}")
            _rewrite_events_file(enrichment_path, events)
            path.unlink()
            return
    _rewrite_events_file(path, events)


def _select_burn_batch(
    files: list[Path], max_events_per_transaction: int
) -> tuple[list[tuple[Path, list[dict[str, Any]]]], int]:
    selected: list[tuple[Path, list[dict[str, Any]]]] = []
    scanned = 0
    event_count = 0
    for path in files:
        scanned += 1
        try:
            events = _read_events(path)
        except (UnicodeDecodeError, OSError) as exc:
            raise RuntimeError(f"failed to read queue file {path.name}: {exc}") from exc
        if not events:
            selected.append((path, []))
            continue
        if selected and event_count + len(events) > max_events_per_transaction:
            break
        selected.append((path, events))
        event_count += len(events)
        if event_count >= max_events_per_transaction:
            break
    return selected, scanned


def _prefetch_enrichment_state(conn: apsw.Connection, events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    cols = _columns(conn, "chunks")
    if not {"content_hash", "enrich_status", "enriched_at"}.issubset(cols):
        return {}
    chunk_ids = sorted(
        {
            str(event.get("chunk_id"))
            for event in events
            if _event_payload(event).get("kind") == "enrichment_update" and event.get("chunk_id")
        }
    )
    if not chunk_ids:
        return {}
    placeholders = ", ".join("?" for _ in chunk_ids)
    selected_cols = ["id", "content_hash", "enrich_status", "enriched_at"]
    if "provenance_class" in cols:
        selected_cols.append("provenance_class")
    if "enrichment_model" in cols:
        selected_cols.append("enrichment_model")
    if "enrichment_backend" in cols:
        selected_cols.append("enrichment_backend")
    if "enrichment_version" in cols:
        selected_cols.append("enrichment_version")
    if "raw_entities_json" in cols:
        selected_cols.append("raw_entities_json")
    rows = conn.execute(
        f"SELECT {', '.join(selected_cols)} FROM chunks WHERE id IN ({placeholders})",
        chunk_ids,
    )
    return {str(row[0]): dict(zip(selected_cols[1:], row[1:], strict=False)) for row in rows}


def _already_enriched(enrich_status: str | None, enriched_at: str | None) -> bool:
    return enrich_status == "success" or bool(enriched_at)


def _is_verified_redundant_enrichment(
    event: dict[str, Any],
    prefetched_state: dict[str, dict[str, Any]],
) -> bool:
    payload = _event_payload(event)
    if payload.get("kind") != "enrichment_update":
        return False
    chunk_id = payload.get("chunk_id")
    expected_hash = payload.get("content_hash")
    if not chunk_id or not expected_hash:
        return False
    state = prefetched_state.get(str(chunk_id))
    if state is None:
        return False
    if state.get("content_hash") != expected_hash or not _already_enriched(
        state.get("enrich_status"), state.get("enriched_at")
    ):
        return False

    provenance_class = str(payload.get("provenance_class") or "").strip()
    if provenance_class and "provenance_class" in state:
        current_provenance_class = str(state.get("provenance_class") or "").strip()
        if current_provenance_class != provenance_class:
            return False

    enrichment_model = str(payload.get("enrichment_model") or "").strip()
    if enrichment_model and "enrichment_model" in state:
        current_enrichment_model = str(state.get("enrichment_model") or "").strip()
        if current_enrichment_model != enrichment_model:
            return False

    enrichment_backend = str(payload.get("enrichment_backend") or "").strip()
    if enrichment_backend and "enrichment_backend" in state:
        current_enrichment_backend = str(state.get("enrichment_backend") or "").strip()
        if current_enrichment_backend != enrichment_backend:
            return False

    enrichment_version = str(payload.get("enrichment_version") or "").strip()
    if enrichment_version and "enrichment_version" in state:
        current_enrichment_version = str(state.get("enrichment_version") or "").strip()
        if current_enrichment_version != enrichment_version:
            return False

    if payload.get("entities") is not None and "raw_entities_json" in state:
        current_entities = str(state.get("raw_entities_json") or "").strip()
        queued_entities = json.dumps(payload["entities"])
        if current_entities != queued_entities:
            return False

    return True


def burn_drain_once(
    *,
    db_path: Path | None = None,
    queue_dir: Path | None = None,
    batch_size: int = 5000,
    max_events_per_transaction: int = _DEFAULT_BURN_MAX_EVENTS_PER_TRANSACTION,
    log_path: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    pause_sentinel_path: Path | None = None,
) -> BurnDrainResult:
    """Drain a large queue backlog with one writer and one commit per large batch.

    Queue files are unlinked only after the transaction commits. Verified redundant
    enrichment updates are skipped inside the same committed batch so old queue
    files can be safely removed without reapplying stale work.
    """
    db_path = db_path or _default_db_path()
    queue_dir = queue_dir or _default_queue_dir()
    log_path = log_path or _default_log_path()
    pause_sentinel_path = pause_sentinel_path or DEFAULT_PAUSE_SENTINEL_PATH
    queue_dir.mkdir(parents=True, exist_ok=True)
    max_events_per_transaction = max(1, max_events_per_transaction)
    pause_payload, pause_active, _pause_stale = pause_sentinel_state(pause_sentinel_path)
    pause_enrichment = pause_active and pause_applies_to_label(pause_payload, _ENRICHMENT_LABEL)

    result = BurnDrainResult()
    lock_fd = _acquire_queue_lock(queue_dir)
    try:
        files = _select_priority_queue_files(list(queue_dir.glob("*.jsonl")), batch_size, cap_enrichment=False)
        if not files:
            return result
        try:
            batch, scanned = _select_burn_batch(files, max_events_per_transaction)
        except RuntimeError as exc:
            _log(log_path, f"burn drain failed before transaction: {exc}")
            result.failed_files = 1
            return result
        result.scanned_files = scanned
        if not batch:
            return result

        paused_events_by_path: dict[Path, list[dict[str, Any]]] = {}
        if pause_enrichment:
            apply_batch: list[tuple[Path, list[dict[str, Any]]]] = []
            for path, events in batch:
                paused_events = [event for event in events if _event_payload(event).get("kind") == "enrichment_update"]
                apply_events = [event for event in events if _event_payload(event).get("kind") != "enrichment_update"]
                if paused_events:
                    paused_events_by_path[path] = paused_events
                apply_batch.append((path, apply_events))
            batch = apply_batch

        all_events = [event for _, events in batch for event in events]
        if not all_events:
            for path, _events in batch:
                paused_events = paused_events_by_path.get(path)
                if paused_events and not _is_enrichment_queue_file(path):
                    try:
                        _preserve_remaining_events(path, paused_events)
                    except OSError as exc:
                        _log(log_path, f"burn drain could not preserve paused enrichment in {path}: {exc}")
            return result
        queue_telemetry = _queue_telemetry([path for path, _events in batch], all_events)
        batch_includes_store = _events_include_store(all_events)
        if batch_includes_store and not _ensure_drain_db_schema_preserving_queue(
            db_path,
            log_path,
            context="burn drain failed before transaction",
        ):
            result.failed_files = len(batch)
            return result
        precomputed_embeddings = _precompute_event_embeddings(all_events, embed_fn)
        for schema_attempt in range(2):
            conn = _open_connection(db_path)
            telemetry_span = start_writer_span(
                conn,
                db_path=db_path,
                producer="drain",
                lane=queue_telemetry["lane"],
                operation=_drain_operation(all_events, "burn_apply"),
                rows_planned=len(all_events),
                queue_wait_ms=queue_telemetry["queue_wait_ms"],
                queue_source=queue_telemetry["queue_source"],
            )
            try:
                _ensure_enrichment_update_schema(conn)
                _ensure_watcher_liveness_schema(conn)
                _ensure_rewind_archive_schema(conn)
                ensure_dedupe_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                prefetched_state = _prefetch_enrichment_state(conn, all_events)
                fallback_markers_by_path: dict[Path, list[FallbackReplayMarker]] = {}
                attempt_applied_events = 0
                attempt_skipped_verified_stale = 0
                for path, events in batch:
                    path_fallback_markers: list[FallbackReplayMarker] = []
                    for event in events:
                        if _is_verified_redundant_enrichment(event, prefetched_state):
                            payload = _event_payload(event)
                            enqueue_provenance_resolution_for_entities(
                                conn,
                                payload.get("entities"),
                                chunk_id=payload.get("chunk_id"),
                                commit=False,
                            )
                            attempt_skipped_verified_stale += 1
                            continue
                        applied = _apply_event(conn, event)
                        attempt_applied_events += 1
                        if applied.chunk_id:
                            content = str(_event_payload(event).get("content") or "").strip()
                            embedding_bytes = precomputed_embeddings.get(content)
                            if embedding_bytes:
                                _insert_precomputed_embedding(conn, applied.chunk_id, embedding_bytes)
                        path_fallback_markers.extend(applied.fallback_markers)
                    if path_fallback_markers:
                        fallback_markers_by_path[path] = path_fallback_markers
                conn.execute("COMMIT")
                telemetry_span.finish("commit", rows_touched=attempt_applied_events)
                result.applied_events += attempt_applied_events
                result.skipped_verified_stale += attempt_skipped_verified_stale
                result.checkpoints += _post_commit_checkpoint(conn, db_path, log_path)
                break
            except Exception as exc:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                telemetry_span.finish("rollback", error=f"{type(exc).__name__}: {exc}")
                if batch_includes_store and _is_missing_chunks_error(exc) and schema_attempt == 0:
                    conn.close()
                    conn = None
                    if not _ensure_drain_db_schema_preserving_queue(
                        db_path,
                        log_path,
                        context="burn drain failed before transaction",
                        force=True,
                    ):
                        result.failed_files = len(batch)
                        return result
                    continue
                result.failed_files = len(batch)
                _log(log_path, f"burn drain failed; batch preserved: {exc}")
                return result
            finally:
                if conn is not None:
                    conn.close()

        fallback_markers_to_apply: list[FallbackReplayMarker] = []
        for path, _events in batch:
            paused_events = paused_events_by_path.get(path)
            if paused_events:
                try:
                    _preserve_remaining_events(path, paused_events)
                except OSError as exc:
                    _log(log_path, f"burn drain committed but could not preserve paused enrichment in {path}: {exc}")
                continue
            try:
                path.unlink()
                result.files_deleted += 1
                fallback_markers_to_apply.extend(fallback_markers_by_path.get(path, []))
            except FileNotFoundError:
                result.files_deleted += 1
                fallback_markers_to_apply.extend(fallback_markers_by_path.get(path, []))
            except OSError as exc:
                _log(log_path, f"burn drain committed but could not unlink {path}: {exc}")
        _mark_fallback_replays_best_effort(fallback_markers_to_apply, log_path)
        if result.applied_events or result.skipped_verified_stale:
            _log(
                log_path,
                "burn drained="
                f"{result.applied_events} skipped_verified_stale={result.skipped_verified_stale} "
                f"files_deleted={result.files_deleted}",
            )
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def drain_once(
    *,
    db_path: Path | None = None,
    queue_dir: Path | None = None,
    batch_size: int = 250,
    log_path: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
    pause_sentinel_path: Path | None = None,
) -> int:
    db_path = db_path or _default_db_path()
    queue_dir = queue_dir or _default_queue_dir()
    log_path = log_path or _default_log_path()
    pause_sentinel_path = pause_sentinel_path or DEFAULT_PAUSE_SENTINEL_PATH
    queue_dir.mkdir(parents=True, exist_ok=True)
    pause_payload, pause_active, _pause_stale = pause_sentinel_state(pause_sentinel_path)
    pause_enrichment = pause_active and pause_applies_to_label(pause_payload, _ENRICHMENT_LABEL)

    lock_fd = _acquire_queue_lock(queue_dir)
    try:
        files = _select_priority_queue_files(list(queue_dir.glob("*.jsonl")), batch_size)
        if not files:
            return 0
        _log(log_path, f"queue_depth={len(files)}")

        drained = 0
        collisions_dropped = 0
        for path in files:
            try:
                events = _read_events(path)
            except (UnicodeDecodeError, OSError) as exc:
                _quarantine_file(path, log_path, exc)
                continue
            if not events:
                _unlink_processed_file(path, log_path)
                continue
            events_to_apply: list[dict[str, Any]] = []
            remaining_events: list[dict[str, Any]] = []
            for event in events:
                if pause_enrichment and _event_payload(event).get("kind") == "enrichment_update":
                    remaining_events.append(event)
                elif len(events_to_apply) < _max_events_per_transaction():
                    events_to_apply.append(event)
                else:
                    remaining_events.append(event)
            if not events_to_apply:
                if not _is_enrichment_queue_file(path):
                    try:
                        _preserve_remaining_events(path, remaining_events)
                    except OSError as exc:
                        _log(log_path, f"drain could not preserve paused enrichment in {path}: {exc}")
                continue
            queue_telemetry = _queue_telemetry([path], events_to_apply)
            events_include_store = _events_include_store(events_to_apply)
            if events_include_store and not _ensure_drain_db_schema_preserving_queue(
                db_path,
                log_path,
                context=f"drain failed for {path.name}",
            ):
                break

            stop_draining = False
            precomputed_embeddings = _precompute_event_embeddings(events_to_apply, embed_fn)
            for attempt in range(5):
                conn: apsw.Connection | None = None
                telemetry_span = None
                attempt_drained = 0
                collision_ids: list[str] = []
                store_chunk_ids: list[str] = []
                fallback_markers: list[FallbackReplayMarker] = []
                try:
                    conn = _open_connection(db_path)
                    telemetry_span = start_writer_span(
                        conn,
                        db_path=db_path,
                        producer="drain",
                        lane=queue_telemetry["lane"],
                        operation=_drain_operation(events_to_apply, "apply_file"),
                        rows_planned=len(events_to_apply),
                        queue_wait_ms=queue_telemetry["queue_wait_ms"],
                        queue_source=queue_telemetry["queue_source"],
                    )
                    _ensure_enrichment_update_schema(conn)
                    _ensure_watcher_liveness_schema(conn)
                    _ensure_rewind_archive_schema(conn)
                    ensure_dedupe_schema(conn)
                    conn.execute("BEGIN IMMEDIATE")
                    for event in events_to_apply:
                        result = _apply_event(conn, event)
                        if result.chunk_id:
                            store_chunk_ids.append(result.chunk_id)
                            fallback_markers.extend(result.fallback_markers)
                            content = str(_event_payload(event).get("content") or "").strip()
                            embedding_bytes = precomputed_embeddings.get(content)
                            if embedding_bytes:
                                _insert_precomputed_embedding(conn, result.chunk_id, embedding_bytes)
                        if result.collision_chunk_id:
                            collision_ids.append(result.collision_chunk_id)
                        attempt_drained += 1
                    conn.execute("COMMIT")
                    telemetry_span.finish("commit", rows_touched=attempt_drained)
                    # Best-effort WAL checkpoint. Keep the live writer path PASSIVE:
                    # TRUNCATE can block behind long-lived readers on the live multi-GB
                    # WAL, which stalls queue drain before it can publish health.
                    # journal_size_limit and the scheduled wal-checkpoint job reclaim
                    # disk later. The except keeps a busy checkpoint non-fatal.
                    _post_commit_checkpoint(conn, db_path, log_path)
                    drained += attempt_drained
                    collisions_dropped += len(collision_ids)
                    for chunk_id in collision_ids:
                        _log(log_path, f"WARN: queued chunk_id {chunk_id} collided with existing row, dropped")
                    queue_cleanup_succeeded = True
                    if remaining_events:
                        try:
                            _preserve_remaining_events(path, remaining_events)
                        except OSError as exc:
                            queue_cleanup_succeeded = False
                            _log(
                                log_path,
                                f"drain committed but could not preserve remaining events in {path}: {exc}",
                            )
                    else:
                        queue_cleanup_succeeded = _unlink_processed_file(path, log_path)
                    if queue_cleanup_succeeded:
                        _mark_fallback_replays_best_effort(fallback_markers, log_path)
                    yield_seconds = _post_commit_yield_seconds()
                    if yield_seconds > 0:
                        _sleep(yield_seconds)
                    if _is_enrichment_queue_file(path) and _has_high_priority_queue_files(queue_dir):
                        stop_draining = True
                    break
                except Exception as exc:
                    if conn is not None:
                        try:
                            conn.execute("ROLLBACK")
                        except Exception:
                            pass
                    if telemetry_span is not None:
                        telemetry_span.finish("rollback", error=f"{type(exc).__name__}: {exc}")
                    if _is_busy_error(exc) and attempt < 4:
                        delay = 0.05 * (2**attempt)
                        _log(log_path, f"drain busy; retrying in {delay:.2f}s")
                        _sleep(delay)
                        continue
                    if events_include_store and _is_missing_chunks_error(exc) and attempt < 4:
                        if conn is not None:
                            conn.close()
                            conn = None
                        if not _ensure_drain_db_schema_preserving_queue(
                            db_path,
                            log_path,
                            context=f"drain failed for {path.name}",
                            force=True,
                        ):
                            stop_draining = True
                            break
                        continue
                    _log(log_path, f"drain failed for {path.name}: {exc}")
                    break
                finally:
                    if conn is not None:
                        conn.close()
            if stop_draining:
                break
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    if drained:
        _log(log_path, f"drained={drained} collisions_dropped={collisions_dropped}")
    return drained


def _write_drain_health(path: Path, *, drain_cycles: int, drained_total: int) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "drain_cycles": drain_cycles,
        "drained_total": drained_total,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _fallback_replay_on_start_enabled() -> bool:
    """Default ON. The whole defect was a replay half that never ran by itself."""
    raw = os.environ.get(FALLBACK_REPLAY_ON_START_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _fallback_replay_on_start_limit() -> int:
    raw = os.environ.get(FALLBACK_REPLAY_ON_START_LIMIT_ENV)
    if raw is None:
        return DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", FALLBACK_REPLAY_ON_START_LIMIT_ENV, raw)
        return DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT
    if value <= 0:
        logger.warning("Ignoring non-positive %s=%r", FALLBACK_REPLAY_ON_START_LIMIT_ENV, raw)
        return DEFAULT_FALLBACK_REPLAY_ON_START_LIMIT
    return value


def replay_fallbacks_on_start(*, log_path: Path | None = None) -> dict[str, Any] | None:
    """Sweep the brain_store fallback queue onto the durable queue, once, at startup.

    Returns the receipt, or None when the sweep is switched off. Enqueue-only: the
    drain loop that runs next is what commits these and marks the files stored.
    """
    if not _fallback_replay_on_start_enabled():
        return None

    from . import fallback_replay as fallback_replay_module
    from .queue_io import enqueue_store

    log_path = log_path or _default_log_path()
    limit = _fallback_replay_on_start_limit()
    summary = fallback_replay_module.enqueue_pending_fallbacks(
        gits_root=Path(os.environ.get("BRAINLAYER_FALLBACK_GITS_ROOT") or Path.home() / "Gits"),
        scope_map=fallback_replay_module.load_scope_map(),
        enqueue_func=enqueue_store,
        replayed_by="brainlayer-drain-start",
        limit=limit,
    )
    try:
        _log(log_path, f"fallback replay on start: {json.dumps(summary, sort_keys=True)}")
    except OSError:
        logger.debug("Failed to log fallback replay receipt", exc_info=True)
    return summary


def run_daemon(
    interval: float,
    batch_size: int,
    *,
    health_path: Path | None = None,
    drain_once_fn: Callable[..., int] = drain_once,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_cycles: int | None = None,
    replay_fallbacks_fn: Callable[[], Any] | None = None,
) -> None:
    health_path = health_path or Path(os.environ.get("BRAINLAYER_DRAIN_HEALTH_PATH", str(_default_drain_health_path())))
    # Once, before the first cycle: whatever the fallback wrote while the DB was
    # unreachable gets queued, so the replay half no longer waits on a human. A
    # sweep that fails must never take the drain down with it.
    try:
        (replay_fallbacks_fn or replay_fallbacks_on_start)()
    except Exception:
        logger.warning("Fallback replay sweep failed at drain start; continuing to drain", exc_info=True)
    drain_cycles = 0
    drained_total = 0
    while True:
        if max_cycles is not None and drain_cycles >= max_cycles:
            return
        drained_total += int(drain_once_fn(batch_size=batch_size) or 0)
        drain_cycles += 1
        try:
            _write_drain_health(health_path, drain_cycles=drain_cycles, drained_total=drained_total)
        except OSError:
            logger.debug("Failed to write drain health snapshot", exc_info=True)
        sleep_fn(interval)


def main() -> int:
    from brainlayer.parent_death import install_parent_death_watcher

    install_parent_death_watcher()

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--burn", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-events-per-transaction", type=int, default=_DEFAULT_BURN_MAX_EVENTS_PER_TRANSACTION)
    args = parser.parse_args()
    if args.burn:
        print(
            json.dumps(
                asdict(
                    burn_drain_once(
                        batch_size=args.batch_size,
                        max_events_per_transaction=args.max_events_per_transaction,
                    )
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.once:
        print(drain_once(batch_size=args.batch_size))
        return 0
    try:
        from brainlayer.deploy_drift import record_launch_from_environment

        record_launch_from_environment()
    except Exception:
        logger.debug("Failed to record drain launch provenance", exc_info=True)
    run_daemon(args.interval, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
