"""Fail-open local telemetry for BrainLayer writer transactions."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import apsw

_MONOTONIC = time.monotonic
_FALSE_VALUES = {"0", "false", "no", "off"}
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUPS = 3
_DEFAULT_HEARTBEAT_INTERVAL_MS = 250
_DEFAULT_FTS_SAMPLE_TTL_SECONDS = 60.0
_TRACE_MASK = apsw.SQLITE_TRACE_STMT | apsw.SQLITE_TRACE_PROFILE
_SINK_LOCK = threading.Lock()
_ACTIVE_LOCK = threading.Lock()
_ACTIVE_SPANS: dict[Path, dict[str, _WriterSpan]] = {}
_HEARTBEAT_PATH_LOCKS: dict[Path, threading.Lock] = {}
_HEARTBEAT_THREAD: threading.Thread | None = None
_FTS_CACHE_LOCK = threading.Lock()
_FTS_CACHE: dict[Path, tuple[float, dict[str, int]]] = {}
_QUOTED_LITERAL_RE = re.compile(r"(?:[xX])?'(?:''|[^'])*'")
_NUMBER_LITERAL_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")


@dataclass(frozen=True)
class SqlFingerprint:
    digest: str
    normalized: str


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def telemetry_enabled() -> bool:
    return os.environ.get("BRAINLAYER_WRITER_TELEMETRY", "1").strip().lower() not in _FALSE_VALUES


def telemetry_path() -> Path:
    configured = os.environ.get("BRAINLAYER_WRITER_TELEMETRY_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "brainlayer" / "logs" / "writer-telemetry.jsonl"


def heartbeat_dir() -> Path:
    configured = os.environ.get("BRAINLAYER_WRITER_HEARTBEAT_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path("/tmp")


def fingerprint_sql(sql: str) -> SqlFingerprint:
    """Return a stable SQL shape without literal values."""
    stripped = _LINE_COMMENT_RE.sub(" ", _BLOCK_COMMENT_RE.sub(" ", str(sql)))
    without_strings = _QUOTED_LITERAL_RE.sub("?", stripped)
    without_numbers = _NUMBER_LITERAL_RE.sub("?", without_strings)
    normalized = " ".join(without_numbers.split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return SqlFingerprint(digest=digest, normalized=normalized)


def _rotate(path: Path, backups: int) -> None:
    if backups <= 0:
        path.unlink(missing_ok=True)
        return
    path.with_name(f"{path.name}.{backups}").unlink(missing_ok=True)
    for index in range(backups - 1, 0, -1):
        source = path.with_name(f"{path.name}.{index}")
        if source.exists():
            source.replace(path.with_name(f"{path.name}.{index + 1}"))
    if path.exists():
        path.replace(path.with_name(f"{path.name}.1"))


def append_event(event: dict[str, Any]) -> bool:
    """Append one JSON event, returning False rather than affecting a writer."""
    if not telemetry_enabled():
        return False
    try:
        path = telemetry_path()
        payload = {"schema_version": 1, **event}
        encoded = (json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        max_bytes = _int_env("BRAINLAYER_WRITER_TELEMETRY_MAX_BYTES", _DEFAULT_MAX_BYTES, minimum=1)
        backups = _int_env("BRAINLAYER_WRITER_TELEMETRY_BACKUPS", _DEFAULT_BACKUPS)
        with _SINK_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.stat().st_size + len(encoded) > max_bytes:
                _rotate(path, backups)
            with path.open("ab") as handle:
                handle.write(encoded)
        return True
    except Exception:
        return False


def tail_event_lines(path: Path | None = None, *, lines: int = 100) -> list[str]:
    """Read the newest telemetry lines without creating or modifying the sink."""
    resolved = Path(path) if path is not None else telemetry_path()
    limit = max(0, int(lines))
    if limit == 0:
        return []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            buffered: list[str] = []
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                if line:
                    buffered.append(line)
                    if len(buffered) > limit:
                        buffered.pop(0)
            return buffered
    except (OSError, UnicodeError):
        return []


def summarize_event_lines(raw_lines: list[str]) -> dict[str, Any]:
    """Summarize completed transaction events from JSONL lines."""
    counts_by_producer: dict[str, int] = {}
    counts_by_lane: dict[str, int] = {}
    counts_by_outcome: dict[str, int] = {}
    durations: list[float] = []
    finished = 0
    for raw_line in raw_lines:
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("event") != "txn_finished":
            continue
        finished += 1
        for field, target in (
            ("producer", counts_by_producer),
            ("lane", counts_by_lane),
            ("outcome", counts_by_outcome),
        ):
            value = str(event.get(field) or "unknown")
            target[value] = target.get(value, 0) + 1
        try:
            durations.append(max(0.0, float(event.get("duration_ms") or 0.0)))
        except (TypeError, ValueError):
            durations.append(0.0)
    durations.sort()
    p95_index = max(0, math.ceil(len(durations) * 0.95) - 1) if durations else 0
    return {
        "counts_by_lane": dict(sorted(counts_by_lane.items())),
        "counts_by_outcome": dict(sorted(counts_by_outcome.items())),
        "counts_by_producer": dict(sorted(counts_by_producer.items())),
        "duration_ms": {
            "max": durations[-1] if durations else 0.0,
            "p95": durations[p95_index] if durations else 0.0,
        },
        "finished_transactions": finished,
        "lines_read": len(raw_lines),
    }


def summarize_events(path: Path | None = None, *, lines: int = 1000) -> dict[str, Any]:
    return summarize_event_lines(tail_event_lines(path, lines=lines))


def _db_page_size(db_path: Path) -> int:
    try:
        with db_path.open("rb") as handle:
            header = handle.read(100)
        if len(header) < 18 or header[:16] != b"SQLite format 3\x00":
            return 4096
        value = int.from_bytes(header[16:18], "big")
        return 65536 if value == 1 else max(value, 512)
    except Exception:
        return 4096


def _wal_metrics(db_path: Path) -> tuple[int, int]:
    try:
        size = Path(f"{db_path}-wal").stat().st_size
        if size < 32:
            return size, 0
        return size, max(0, (size - 32) // (_db_page_size(db_path) + 24))
    except OSError:
        return 0, 0


def _fts_segment_counts(conn: apsw.Connection, db_path: Path) -> dict[str, int]:
    now = _MONOTONIC()
    ttl = _float_env(
        "BRAINLAYER_WRITER_TELEMETRY_FTS_SAMPLE_TTL_SECONDS",
        _DEFAULT_FTS_SAMPLE_TTL_SECONDS,
    )
    resolved = db_path.resolve()
    with _FTS_CACHE_LOCK:
        cached = _FTS_CACHE.get(resolved)
        if cached is not None and now - cached[0] < ttl:
            return dict(cached[1])
    counts: dict[str, int] = {}
    fts_tables = ("chunks_fts", "chunks_fts_operational", "chunks_fts_trigram")
    shadow_names = {f"{table}_idx" for table in fts_tables}
    try:
        placeholders = ",".join("?" for _ in shadow_names)
        existing = {
            str(row[0])
            for row in conn.execute(
                f"SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ({placeholders})",
                sorted(shadow_names),
            )
        }
    except Exception:
        existing = set()
    for table in fts_tables:
        if f"{table}_idx" not in existing:
            continue
        try:
            row = conn.execute(f'SELECT COUNT(DISTINCT segid) FROM "{table}_idx"').fetchone()
            counts[table] = int(row[0] or 0) if row else 0
        except Exception:
            continue
    with _FTS_CACHE_LOCK:
        _FTS_CACHE[resolved] = (now, dict(counts))
    return counts


def _heartbeat_path(db_path: Path) -> Path:
    digest = hashlib.sha256(str(db_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return heartbeat_dir() / f"writer-txn-{digest}-{os.getpid()}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return True
    except Exception:
        return False


def _heartbeat_path_lock(path: Path) -> threading.Lock:
    with _ACTIVE_LOCK:
        return _HEARTBEAT_PATH_LOCKS.setdefault(path, threading.Lock())


def _flush_heartbeat(path: Path) -> None:
    try:
        with _heartbeat_path_lock(path):
            with _ACTIVE_LOCK:
                spans = list(_ACTIVE_SPANS.get(path, {}).values())
            if not spans:
                path.unlink(missing_ok=True)
                return
            _atomic_write_json(
                path,
                {
                    "updated_at": _utc_now(),
                    "executor_pid": os.getpid(),
                    "active_transactions": [span.heartbeat_payload() for span in spans],
                },
            )
    except Exception:
        return


def _heartbeat_loop() -> None:
    while True:
        interval = _int_env(
            "BRAINLAYER_WRITER_TELEMETRY_HEARTBEAT_INTERVAL_MS",
            _DEFAULT_HEARTBEAT_INTERVAL_MS,
            minimum=10,
        )
        time.sleep(interval / 1000.0)
        try:
            with _ACTIVE_LOCK:
                paths = list(_ACTIVE_SPANS)
            for path in paths:
                _flush_heartbeat(path)
        except Exception:
            continue


def _ensure_heartbeat_thread() -> None:
    global _HEARTBEAT_THREAD
    try:
        with _ACTIVE_LOCK:
            if _HEARTBEAT_THREAD is not None and _HEARTBEAT_THREAD.is_alive():
                return
            _HEARTBEAT_THREAD = threading.Thread(
                target=_heartbeat_loop,
                name="brainlayer-writer-heartbeat",
                daemon=True,
            )
            _HEARTBEAT_THREAD.start()
    except Exception:
        return


class _NoopWriterSpan:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        return False

    def commit(self, *, rows_touched: int | None = None) -> None:
        return

    def rollback(self, *, error: str | None = None) -> None:
        return

    def complete(self, *, rows_touched: int | None = None) -> None:
        return

    def finish(
        self,
        outcome: str,
        *,
        rows_touched: int | None = None,
        error: str | None = None,
    ) -> None:
        return


class _WriterSpan:
    def __init__(
        self,
        conn: apsw.Connection,
        *,
        db_path: Path,
        producer: str,
        lane: str,
        operation: str,
        rows_planned: int | None,
        queue_wait_ms: float | None,
        queue_source: str | None,
        span_kind: str,
        transaction_mode: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        self.conn = conn
        self.db_path = Path(db_path)
        self.producer = producer
        self.lane = lane
        self.operation = operation
        self.rows_planned = rows_planned
        self.queue_wait_ms = queue_wait_ms
        self.queue_source = queue_source
        self.span_kind = span_kind
        self.transaction_mode = transaction_mode
        self.metadata = dict(metadata or {})
        self.txn_id = uuid.uuid4().hex
        self.started_at = _utc_now()
        self.started_monotonic = _MONOTONIC()
        self._total_changes_before = self._total_changes()
        self.wal_bytes_before, self.wal_frames_before = _wal_metrics(self.db_path)
        self.fts_segments_before: dict[str, int] = {}
        self._statements: dict[str, dict[str, Any]] = {}
        self._statement_starts: dict[int, dict[str, Any]] = {}
        self._current_statement: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._trace_id = ("brainlayer-writer-telemetry", self.txn_id)
        self._heartbeat_path = _heartbeat_path(self.db_path)
        self._outcome: str | None = None
        self._error: str | None = None
        self._rows_touched: int | None = None
        self._entered = False
        self._finished = False
        self._trace_installed = False

    def _total_changes(self) -> int:
        try:
            return int(self.conn.totalchanges())
        except Exception:
            return 0

    def __enter__(self):
        try:
            self.fts_segments_before = _fts_segment_counts(self.conn, self.db_path)
        except Exception:
            self.fts_segments_before = {}
        try:
            self.conn.trace_v2(_TRACE_MASK, self._trace, id=self._trace_id)
            self._trace_installed = True
        except Exception:
            self._trace_installed = False
        self._entered = True
        append_event(self._base_event("txn_started"))
        try:
            with _ACTIVE_LOCK:
                _ACTIVE_SPANS.setdefault(self._heartbeat_path, {})[self.txn_id] = self
            _flush_heartbeat(self._heartbeat_path)
            _ensure_heartbeat_thread()
        except Exception:
            pass
        return self

    def __exit__(self, exc_type, exc, _traceback) -> bool:
        if exc is not None:
            self._outcome = "rollback" if self._outcome == "rollback" else "error"
            self._error = self._error or f"{type(exc).__name__}: {exc}"
        self._finish()
        return False

    def _trace(self, event: dict[str, Any]) -> None:
        try:
            code = event.get("code")
            sql = str(event.get("sql") or "")
            fingerprint = fingerprint_sql(sql)
            if code == apsw.SQLITE_TRACE_STMT:
                if bool(event.get("trigger")):
                    return
                statement_start = {
                    "fingerprint": fingerprint.digest,
                    "normalized_sql": fingerprint.normalized[:500],
                    "started_at": _utc_now(),
                    "started_monotonic": _MONOTONIC(),
                    "trigger": False,
                }
                with self._lock:
                    self._current_statement = statement_start
                    statement_id = event.get("id")
                    if isinstance(statement_id, int):
                        self._statement_starts[statement_id] = statement_start
                return
            if code != apsw.SQLITE_TRACE_PROFILE:
                return
            duration_ms = max(0.0, float(event.get("nanoseconds") or 0) / 1_000_000.0)
            status = event.get("stmt_status") if isinstance(event.get("stmt_status"), dict) else {}
            with self._lock:
                statement_id = event.get("id")
                statement_start = self._statement_starts.get(statement_id, {}) if isinstance(statement_id, int) else {}
                aggregate = self._statements.setdefault(
                    fingerprint.digest,
                    {
                        "fingerprint": fingerprint.digest,
                        "normalized_sql": fingerprint.normalized[:500],
                        "count": 0,
                        "total_duration_ms": 0.0,
                        "max_duration_ms": 0.0,
                        "vm_steps": 0,
                        "fullscan_steps": 0,
                        "sorts": 0,
                        "started_at": statement_start.get("started_at"),
                    },
                )
                if aggregate.get("started_at") is None and statement_start.get("started_at") is not None:
                    aggregate["started_at"] = statement_start["started_at"]
                aggregate["count"] += 1
                aggregate["total_duration_ms"] += duration_ms
                aggregate["max_duration_ms"] = max(aggregate["max_duration_ms"], duration_ms)
                aggregate["vm_steps"] += int(status.get("SQLITE_STMTSTATUS_VM_STEP", 0) or 0)
                aggregate["fullscan_steps"] += int(status.get("SQLITE_STMTSTATUS_FULLSCAN_STEP", 0) or 0)
                aggregate["sorts"] += int(status.get("SQLITE_STMTSTATUS_SORT", 0) or 0)
                if statement_start.get("fingerprint") == fingerprint.digest and isinstance(statement_id, int):
                    self._statement_starts.pop(statement_id, None)
        except Exception:
            return

    def _base_event(self, event_name: str) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event": event_name,
            "txn_id": self.txn_id,
            "producer": self.producer,
            "lane": self.lane,
            "operation": self.operation,
            "span_kind": self.span_kind,
            "transaction_mode": self.transaction_mode,
            "executor_pid": os.getpid(),
            "db_path": str(self.db_path),
            "txn_started_at": self.started_at,
            "txn_started_monotonic": self.started_monotonic,
            "rows_planned": self.rows_planned,
            "queue_wait_ms": self.queue_wait_ms,
            "queue_source": self.queue_source,
            **self.metadata,
        }
        return event

    def heartbeat_payload(self) -> dict[str, Any]:
        now = _MONOTONIC()
        with self._lock:
            current = dict(self._current_statement or {})
        started = current.get("started_monotonic")
        return {
            **self._base_event("txn_active"),
            "open_ms": max(0.0, (now - self.started_monotonic) * 1000.0),
            "current_statement_fingerprint": current.get("fingerprint"),
            "current_statement": current.get("normalized_sql"),
            "current_statement_started_at": current.get("started_at"),
            "current_statement_open_ms": max(0.0, (now - started) * 1000.0) if isinstance(started, float) else None,
            "current_statement_trigger": current.get("trigger"),
        }

    def commit(self, *, rows_touched: int | None = None) -> None:
        self._outcome = "commit"
        self._rows_touched = rows_touched

    def rollback(self, *, error: str | None = None) -> None:
        self._outcome = "rollback"
        self._error = error

    def complete(self, *, rows_touched: int | None = None) -> None:
        self._outcome = "completed"
        self._rows_touched = rows_touched

    def finish(
        self,
        outcome: str,
        *,
        rows_touched: int | None = None,
        error: str | None = None,
    ) -> None:
        self._outcome = outcome
        self._rows_touched = rows_touched
        self._error = error
        self._finish()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._trace_installed:
            try:
                self.conn.trace_v2(_TRACE_MASK, None, id=self._trace_id)
            except Exception:
                pass
        try:
            fts_segments = _fts_segment_counts(self.conn, self.db_path)
        except Exception:
            fts_segments = dict(self.fts_segments_before)
        wal_bytes_after, wal_frames_after = _wal_metrics(self.db_path)
        changed_rows = max(0, self._total_changes() - self._total_changes_before)
        with self._lock:
            statements = sorted(self._statements.values(), key=lambda item: item["max_duration_ms"], reverse=True)[:64]
        finished = {
            **self._base_event("txn_finished"),
            "finished_at": _utc_now(),
            "duration_ms": max(0.0, (_MONOTONIC() - self.started_monotonic) * 1000.0),
            "outcome": self._outcome or "completed",
            "error": self._error,
            "rows_touched": self._rows_touched if self._rows_touched is not None else changed_rows,
            "wal_bytes_before": self.wal_bytes_before,
            "wal_bytes_after": wal_bytes_after,
            "wal_frames_before": self.wal_frames_before,
            "wal_frames_after": wal_frames_after,
            "fts_segments_before": self.fts_segments_before,
            "fts_segments": fts_segments,
            "statements": statements,
        }
        append_event(finished)
        try:
            with _ACTIVE_LOCK:
                active = _ACTIVE_SPANS.get(self._heartbeat_path)
                if active is not None:
                    active.pop(self.txn_id, None)
                    if not active:
                        _ACTIVE_SPANS.pop(self._heartbeat_path, None)
            _flush_heartbeat(self._heartbeat_path)
        except Exception:
            pass


def writer_span(
    conn: apsw.Connection,
    *,
    db_path: Path,
    producer: str,
    lane: str,
    operation: str,
    rows_planned: int | None = None,
    queue_wait_ms: float | None = None,
    queue_source: str | None = None,
    span_kind: str = "transaction",
    transaction_mode: str = "explicit",
    metadata: dict[str, Any] | None = None,
):
    """Create a fail-open observer; the caller retains transaction ownership."""
    if not telemetry_enabled():
        return _NoopWriterSpan()
    try:
        return _WriterSpan(
            conn,
            db_path=Path(db_path),
            producer=producer,
            lane=lane,
            operation=operation,
            rows_planned=rows_planned,
            queue_wait_ms=queue_wait_ms,
            queue_source=queue_source,
            span_kind=span_kind,
            transaction_mode=transaction_mode,
            metadata=metadata,
        )
    except Exception:
        return _NoopWriterSpan()


def start_writer_span(*args, **kwargs):
    """Start a span for call sites that cannot use a lexical context manager."""
    span = writer_span(*args, **kwargs)
    try:
        return span.__enter__()
    except Exception:
        return _NoopWriterSpan()
