#!/usr/bin/env python3
"""Locally embed recent BrainBar MCP writes and pending embedding backlog.

BrainBar owns the live MCP socket and stores ``brain_store`` rows directly in
SQLite. This adapter owns the single BrainLayer writer slot and must therefore
perform local hot embedding and optional embedding backlog work. Legacy
Python enrichment arguments are accepted for compatibility but cannot invoke a model.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import inspect
import json
import logging
import math
import os
import signal
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, NamedTuple

import apsw
import sqlite_vec

from brainlayer._helpers import serialize_f32
from brainlayer.embeddings import get_embedding_model
from brainlayer.hotlane_budget import HotlaneCpuSampler
from brainlayer.paths import get_db_path
from brainlayer.store import embed_hot_chunk, embed_pending_chunks
from brainlayer.vector_store import VectorStore, skip_connection_hooks, temporary_write_busy_timeout_ms
from brainlayer.writer_telemetry import IngestMetrics, start_writer_span

LOGGER = logging.getLogger("brainlayer.hotlane_brainbar")
STOP = False
DEFAULT_HOTLANE_ENRICH_LIMIT = 0
DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS = 87_600  # Legacy Python compatibility only.
DEFAULT_BACKLOG_BATCH = 4
DEFAULT_HOTLANE_EMBED_DEVICE = "cpu"
MAX_BACKLOG_BATCH = 16
DEFAULT_HOTLANE_WRITE_BUSY_TIMEOUT_MS = 1000
MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647
VECTOR_WRITE_YIELD_SECONDS = 0.005
# S17 (2026-11-02): a busy writer slot must never cost an embedded batch. The write retries
# inside the cycle (each try waits up to the hotlane busy timeout at BEGIN IMMEDIATE), then
# whatever is still unwritten is carried to the next cycle instead of being discarded.
HOTLANE_WRITE_ATTEMPTS = 3
MAX_CHUNK_FAILURE_ATTEMPTS = 3
DEFAULT_QUARANTINE_TTL_SECONDS = 3600.0
MAX_CHUNK_FAILURE_ENTRIES = 4096
# This many consecutive distinct chunks failing with one error class in one loop is the
# environment failing, not the chunks. Above the 2-row poison batch the backlog lane sees.
SYSTEMIC_FAILURE_STREAK = 4
MAX_SYSTEMIC_BACKOFF_SECONDS = 60.0
# 2**16 times any base of 1 ms or more already exceeds the 60 s cap; never exponentiate past it.
SYSTEMIC_BACKOFF_MAX_EXPONENT = 16
_ENVIRONMENTAL_ERRNOS = frozenset({errno.ENOSPC, errno.EIO, errno.EROFS, errno.EDQUOT})
_ENVIRONMENTAL_SQLITE_ERRORS = (apsw.FullError, apsw.IOError, apsw.ReadOnlyError)
_ENVIRONMENTAL_SQLITE_PRIMARY_CODES = frozenset({8, 10, 13})  # READONLY, IOERR, FULL
HOTLANE_WRITE_RETRY_BACKOFF_SECONDS = 0.25
MAX_CARRIED_VECTORS = 64
# Hooks the hotlane writer never runs in its Connection() constructor. `connection_optimize`
# runs PRAGMA optimize (a write lock) on every open; `connection_busy_timeout` would reset the
# constructor's busy timeout to APSW's 100 ms. connection_wal and the rest still run.
HOTLANE_SKIPPED_CONNECTION_HOOKS = ("connection_optimize", "connection_busy_timeout")

# Incident 2026-08-24: the loop slept a fixed `interval` (1.0s) whether or not a cycle
# did any work, so an idle hotlane kept probing once per second forever. Idle cycles now
# back off geometrically; any real work snaps the cadence straight back to `interval`.
#
# The cap is deliberately SMALL, and it is not a tuning knob to raise casually (PR #735
# review, codex P2): a normal `brain_store` writes straight to SQLite and does not touch
# the durable queue, so nothing here wakes on arrival — this cap IS the worst-case time a
# freshly stored chunk stays unembedded, during which a related search can miss it.
# Note the per-cycle idle probe is already bounded by rowid lanes (HotCandidateScanner),
# so backing off further would increase local embedding latency.
IDLE_BACKOFF_FACTOR = 2.0
MAX_IDLE_INTERVAL_SECONDS = 5.0
MAX_PENDING_CANDIDATE_SCAN_PAGES = 16
HOT_CANDIDATE_SCAN_LIMIT = 256
HOT_CANDIDATE_HEAD_LIMIT = HOT_CANDIDATE_SCAN_LIMIT // 2
MAX_HOT_CANDIDATE_RETRIES = 256
_sleep = time.sleep

HOT_CANDIDATE_SCAN_SQL = """
    SELECT
        c.id,
        c.content,
        c.source_file,
        c.source,
        c.archived_at,
        c.superseded_by,
        c.aggregated_into,
        COALESCE(c.archived, 0),
        COALESCE(c.status, 'active'),
        r.id
    FROM chunks c
    LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
    ORDER BY c.created_at DESC
    LIMIT ?
"""

HOT_CANDIDATE_ROWID_SCAN_SQL = """
    SELECT
        c.rowid,
        c.id,
        c.content,
        c.source_file,
        c.source,
        c.archived_at,
        c.superseded_by,
        c.aggregated_into,
        COALESCE(c.archived, 0),
        COALESCE(c.status, 'active'),
        r.id
    FROM chunks c
    LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
    ORDER BY c.rowid DESC
    LIMIT ?
"""

HOT_CANDIDATE_ROWID_PAGE_SQL = """
    SELECT
        c.rowid,
        c.id,
        c.content,
        c.source_file,
        c.source,
        c.archived_at,
        c.superseded_by,
        c.aggregated_into,
        COALESCE(c.archived, 0),
        COALESCE(c.status, 'active'),
        r.id
    FROM chunks c
    LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
    WHERE c.rowid < ?
    ORDER BY c.rowid DESC
    LIMIT ?
"""

HOT_CANDIDATE_ROWID_FORWARD_SQL = """
    SELECT
        c.rowid,
        c.id,
        c.content,
        c.source_file,
        c.source,
        c.archived_at,
        c.superseded_by,
        c.aggregated_into,
        COALESCE(c.archived, 0),
        COALESCE(c.status, 'active'),
        r.id
    FROM chunks c
    LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
    WHERE c.rowid > ?
    ORDER BY c.rowid ASC
    LIMIT ?
"""


class CycleResult(NamedTuple):
    embedded: int = 0
    enrich_attempted: int = 0
    enriched: int = 0
    enrich_skipped: int = 0
    enrich_failed: int = 0
    enrich_daily_cap_reached: bool = False
    deferred: int = 0
    backlog_encoded: int = 0


class EmbedCandidate(NamedTuple):
    chunk_id: str
    content: str


def _systemic_backoff_seconds(base: float, failures: int) -> float:
    """Geometric backoff for the ``failures``-th consecutive systemic cycle, capped.

    The exponent is clamped BEFORE the power: ``base * 2 ** 1024`` is a float overflow,
    which would escape ``run()``'s cycle handler and stop the daemon at the 1,025th
    consecutive systemic cycle (~17 h of a full disk at the 60 s cap).
    """
    exponent = min(max(failures - 1, 0), SYSTEMIC_BACKOFF_MAX_EXPONENT)
    return min(base * 2**exponent, MAX_SYSTEMIC_BACKOFF_SECONDS)


def _quarantine_ttl_seconds() -> float:
    raw = os.environ.get("BRAINLAYER_HOTLANE_QUARANTINE_TTL_S")
    if raw is None:
        return DEFAULT_QUARANTINE_TTL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        LOGGER.warning(
            "ignoring invalid BRAINLAYER_HOTLANE_QUARANTINE_TTL_S=%r; using %.0fs",
            raw,
            DEFAULT_QUARANTINE_TTL_SECONDS,
        )
        return DEFAULT_QUARANTINE_TTL_SECONDS
    return value


def _is_environmental_error(exc: BaseException | None) -> bool:
    """True when the failure belongs to the machine (disk, I/O, read-only), not to one chunk."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, _ENVIRONMENTAL_SQLITE_ERRORS):
            return True
        if isinstance(exc, OSError) and exc.errno in _ENVIRONMENTAL_ERRNOS:
            return True
        code = getattr(exc, "sqlite_errorcode", None)
        if isinstance(code, int) and code & 0xFF in _ENVIRONMENTAL_SQLITE_PRIMARY_CODES:
            return True
        exc = exc.__cause__ if exc.__cause__ is not None else (None if exc.__suppress_context__ else exc.__context__)
    return False


class HotlaneSystemicFailure(RuntimeError):
    """A cycle-level failure: the environment, not individual chunks, is failing."""

    def __init__(self, stage: str, error_type: str, chunk_ids: list[str]) -> None:
        super().__init__(
            f"hotlane systemic {stage} failure error_type={error_type} chunks={len(chunk_ids)}; "
            "not counted toward per-chunk quarantine"
        )
        self.stage = stage
        self.error_type = error_type
        self.chunk_ids = chunk_ids


class ChunkFailureAttempts:
    """Process-local quarantine; source rows stay intact and restart permits a new try.

    Key by content digest as well as id so an edited chunk can recover. Only failures
    attributable to an individual chunk count; writer connection errors do not. An entry
    expires ``ttl_s`` after its last failure, so a quarantined chunk is retried (loudly)
    without a restart, and the table never holds more than ``max_entries`` keys.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_s: float | None = None,
        max_entries: int = MAX_CHUNK_FAILURE_ENTRIES,
    ) -> None:
        # key -> (attempts, monotonic time of the last failure); oldest failure first.
        self.attempts: OrderedDict[tuple[str, str], tuple[int, float]] = OrderedDict()
        self.quarantined_total = 0
        self.clock = clock
        self.ttl_s = _quarantine_ttl_seconds() if ttl_s is None else ttl_s
        self.max_entries = max(1, max_entries)

    def _key(self, candidate: EmbedCandidate) -> tuple[str, str]:
        return candidate.chunk_id, hashlib.sha256(candidate.content.encode()).hexdigest()

    def _expire(self, key: tuple[str, str], entry: tuple[int, float], now: float) -> bool:
        if now - entry[1] < self.ttl_s:
            return False
        del self.attempts[key]
        if entry[0] >= MAX_CHUNK_FAILURE_ATTEMPTS:
            LOGGER.warning(
                "hotlane chunk re-admitted after quarantine ttl chunk_id=%s attempts=%d ttl_s=%.0f",
                key[0],
                entry[0],
                self.ttl_s,
            )
        return True

    def count(self, candidate: EmbedCandidate) -> int:
        key = self._key(candidate)
        entry = self.attempts.get(key)
        if entry is None or self._expire(key, entry, self.clock()):
            return 0
        return entry[0]

    def quarantined(self, candidate: EmbedCandidate) -> bool:
        return self.count(candidate) >= MAX_CHUNK_FAILURE_ATTEMPTS

    def _make_room(self, now: float) -> None:
        for key, entry in list(self.attempts.items()):
            if not self._expire(key, entry, now):
                break  # Ordered by last failure: the rest are younger.
        while len(self.attempts) >= self.max_entries:
            key, (attempts, _) = self.attempts.popitem(last=False)
            LOGGER.warning(
                "hotlane chunk failure table full; evicting oldest chunk_id=%s attempts=%d max_entries=%d",
                key[0],
                attempts,
                self.max_entries,
            )

    def failed(self, candidate: EmbedCandidate, exc: Exception) -> None:
        if isinstance(exc, (apsw.BusyError, apsw.LockedError)) or self.quarantined(candidate):
            return
        key = self._key(candidate)
        now = self.clock()
        attempts = self.count(candidate) + 1
        self.attempts.pop(key, None)
        self._make_room(now)
        self.attempts[key] = (attempts, now)
        if attempts == MAX_CHUNK_FAILURE_ATTEMPTS:
            self.quarantined_total += 1
        LOGGER.error(
            "hotlane chunk %s chunk_id=%s error_type=%s attempts=%d quarantined_total=%d",
            "quarantined" if attempts >= MAX_CHUNK_FAILURE_ATTEMPTS else "retrying",
            candidate.chunk_id,
            type(exc).__name__,
            attempts,
            self.quarantined_total,
        )

    def succeeded(self, candidate: EmbedCandidate) -> None:
        self.attempts.pop(self._key(candidate), None)


CHUNK_FAILURE_ATTEMPTS = ChunkFailureAttempts()


class _FailureStreak:
    """Hold one loop's per-chunk failures until they prove to be about the chunk.

    An environmental error, or ``SYSTEMIC_FAILURE_STREAK`` consecutive distinct chunks
    failing with one error class, raises ``HotlaneSystemicFailure`` and counts nothing.
    Anything else is charged to its chunk when the streak breaks or the loop ends.
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.error_type: type[BaseException] | None = None
        self.members: list[tuple[EmbedCandidate, Exception]] = []

    def _systemic(self, error_type: str, chunk_ids: list[str]) -> HotlaneSystemicFailure:
        self.members = []
        self.error_type = None
        failure = HotlaneSystemicFailure(self.stage, error_type, chunk_ids)
        LOGGER.error("%s; backing off", failure)
        return failure

    def failed(self, candidate: EmbedCandidate, exc: Exception) -> None:
        if _is_environmental_error(exc):
            ids = [member.chunk_id for member, _ in self.members] + [candidate.chunk_id]
            raise self._systemic(type(exc).__name__, ids) from exc
        if type(exc) is not self.error_type:
            self.flush()
        self.error_type = type(exc)
        if all(member.chunk_id != candidate.chunk_id for member, _ in self.members):
            self.members.append((candidate, exc))
        if len(self.members) >= SYSTEMIC_FAILURE_STREAK:
            raise self._systemic(type(exc).__name__, [member.chunk_id for member, _ in self.members]) from exc

    def flush(self) -> None:
        members, self.members, self.error_type = self.members, [], None
        for candidate, exc in members:
            CHUNK_FAILURE_ATTEMPTS.failed(candidate, exc)


def _sysctl(name: str) -> str:
    return subprocess.check_output(["/usr/sbin/sysctl", "-n", name], text=True, timeout=5).strip()


def _performance_cores() -> int:
    if sys.platform == "darwin":
        return max(1, int(_sysctl("hw.perflevel0.physicalcpu")))
    return os.cpu_count() or 1


def _backlog_max_load() -> float:
    value = float(os.environ.get("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD", "") or _performance_cores())
    if not math.isfinite(value) or value < 0:
        raise ValueError("BRAINLAYER_HOTLANE_BACKLOG_MAX_LOAD must be finite and nonnegative")
    return value


def setup_cpu_budget(db_path: Path) -> dict:
    """CPU-only priority: never request Darwin background or a disk I/O policy."""
    policy = {"policy": "not applicable", "platform": sys.platform}
    if sys.platform == "darwin":
        os.setpriority(os.PRIO_PROCESS, 0, 20)
        policy.update(policy="nice", nice=os.getpriority(os.PRIO_PROCESS, 0))
    default_threads = 1 if sys.platform == "darwin" and "Apple M1" in _sysctl("machdep.cpu.brand_string") else 2
    threads = int(os.environ.get("BRAINLAYER_HOTLANE_THREADS", str(default_threads)))
    if threads < 1:
        raise ValueError("BRAINLAYER_HOTLANE_THREADS must be positive")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    # embeddings.py imports sentence_transformers/torch only when the model first encodes.
    import torch

    torch.set_num_threads(threads)
    policy.update(threads=torch.get_num_threads(), backlog_max_load=_backlog_max_load(), pid=os.getpid())
    receipt = db_path.parent / "hotlane-budget.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(policy, sort_keys=True) + "\n")
    temporary.replace(receipt)
    LOGGER.info("hotlane CPU budget %s receipt=%s", json.dumps(policy, sort_keys=True), receipt)
    return policy


def _backlog_yield_reason(db_path: Path, max_load: float) -> str:
    if (db_path.parent / "hotlane.pause").exists():
        return "pause marker"
    try:
        load = os.getloadavg()[0]
    except OSError:
        return "load unavailable"
    return "high load" if load > max_load else ""


class PendingCandidateScanState:
    def __init__(self) -> None:
        self.active = False
        self.after_created_at: str | None = None
        self.after_rowid = 0

    def reset(self) -> None:
        self.active = False
        self.after_created_at = None
        self.after_rowid = 0


class EmbeddedVector(NamedTuple):
    chunk_id: str
    content: str
    embedding: list[float]


class HotlaneVectorWriteDeferred(apsw.BusyError):
    """A busy write with progress for cycle carry-over.

    Remains a BusyError so callers using the writer's existing lock-error contract
    can retry after contention; the split cycle also retains ``remaining``.
    """

    def __init__(self, *, written: int, remaining: list[EmbeddedVector], attempts: int, cause: BaseException):
        super().__init__(f"{type(cause).__name__}: {cause} (after {attempts} attempts, {len(remaining)} unwritten)")
        self.written = written
        self.remaining = remaining
        self.attempts = attempts
        self.cause = cause


class VectorCarryover:
    """Computed vectors whose write was deferred by a busy writer slot (S17).

    Held in memory for the next cycle so a lock never costs the embedding work. The
    writer re-checks each chunk's content and vector state at write time, so a stale
    entry (content edited, or vectorized meanwhile) is skipped, never written wrongly.
    Bounded: past ``capacity`` the oldest entries are dropped, loudly and counted. A
    dropped vector loses only compute; its chunk stays unvectored and is re-embedded.
    """

    def __init__(self, capacity: int = MAX_CARRIED_VECTORS) -> None:
        self.capacity = capacity
        self._vectors: OrderedDict[str, EmbeddedVector] = OrderedDict()
        self.carried_total = 0
        self.dropped_total = 0

    def __len__(self) -> int:
        return len(self._vectors)

    def put(self, vectors: list[EmbeddedVector]) -> int:
        for vector in vectors:
            self._vectors.pop(vector.chunk_id, None)
            self._vectors[vector.chunk_id] = vector
        self.carried_total += len(vectors)
        dropped = 0
        while len(self._vectors) > self.capacity:
            self._vectors.popitem(last=False)
            dropped += 1
        if dropped:
            self.dropped_total += dropped
            LOGGER.warning(
                "hotlane carry-over full; dropped %d oldest computed vectors (chunks stay unvectored and "
                "will be re-embedded) capacity=%d dropped_total=%d",
                dropped,
                self.capacity,
                self.dropped_total,
            )
        return dropped

    def peek(self) -> list[EmbeddedVector]:
        return list(self._vectors.values())

    def clear(self) -> None:
        self._vectors.clear()


VECTOR_CARRYOVER = VectorCarryover()


def _candidates_from_scanned_rows(
    rows: list[tuple], *, limit: int, exclude_ids: set[str] | None = None
) -> tuple[list[EmbedCandidate], int | None, int | None, bool]:
    if limit <= 0:
        return [], None, None, False
    candidates: list[EmbedCandidate] = []
    exclude_ids = exclude_ids or set()
    first_candidate_rowid: int | None = None
    last_inspected_rowid: int | None = None
    for index, row in enumerate(rows):
        (
            rowid,
            chunk_id,
            content,
            source_file,
            source,
            archived_at,
            superseded_by,
            aggregated_into,
            archived,
            status,
            vector_id,
        ) = row
        last_inspected_rowid = int(rowid)
        eligible = (
            vector_id is None
            and source_file == "brainbar-store"
            and source == "mcp"
            and content
            and archived_at is None
            and superseded_by is None
            and aggregated_into is None
        )
        if eligible:
            if first_candidate_rowid is None:
                first_candidate_rowid = int(rowid)
            if str(chunk_id) in exclude_ids:
                continue
            candidate = EmbedCandidate(str(chunk_id), str(content))
            if CHUNK_FAILURE_ATTEMPTS.quarantined(candidate):
                continue
            candidates.append(candidate)
            if len(candidates) >= limit:
                return candidates, first_candidate_rowid, last_inspected_rowid, index == len(rows) - 1
    return candidates, first_candidate_rowid, last_inspected_rowid, True


class HotCandidateScanner:
    """Scan bounded rowid lanes while retaining returned candidates for retry."""

    def __init__(self) -> None:
        self._before_rowid: int | None = None
        self._newest_seen_rowid: int | None = None
        self._retries: OrderedDict[str, EmbedCandidate] = OrderedDict()
        self._retry_turn = True
        self._retry_buffer_full_logged = False

    def _refresh_retries(self, cursor: apsw.Cursor) -> None:
        if not self._retries:
            return
        placeholders = ",".join("?" for _ in self._retries)
        rows = cursor.execute(
            f"""
            SELECT c.id, c.content
            FROM chunks c
            LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
            WHERE c.id IN ({placeholders})
              AND r.id IS NULL
              AND c.source_file = 'brainbar-store'
              AND c.source = 'mcp'
              AND c.content IS NOT NULL
              AND c.content != ''
              AND c.archived_at IS NULL
              AND c.superseded_by IS NULL
              AND c.aggregated_into IS NULL
            """,
            tuple(self._retries),
        )
        active = {str(chunk_id): EmbedCandidate(str(chunk_id), str(content)) for chunk_id, content in rows}
        for chunk_id in list(self._retries):
            if chunk_id not in active or CHUNK_FAILURE_ATTEMPTS.quarantined(active[chunk_id]):
                del self._retries[chunk_id]
            else:
                self._retries[chunk_id] = active[chunk_id]
        if len(self._retries) < MAX_HOT_CANDIDATE_RETRIES:
            self._retry_buffer_full_logged = False

    def _take_retries(self, *, limit: int) -> list[EmbedCandidate]:
        selected: list[EmbedCandidate] = []
        for _ in range(min(limit, len(self._retries))):
            chunk_id, candidate = self._retries.popitem(last=False)
            self._retries[chunk_id] = candidate
            selected.append(candidate)
        return selected

    def _retain(self, candidates: list[EmbedCandidate]) -> None:
        for candidate in candidates:
            if candidate.chunk_id in self._retries:
                continue
            if len(self._retries) >= MAX_HOT_CANDIDATE_RETRIES:
                if not self._retry_buffer_full_logged:
                    LOGGER.warning("hot candidate retry buffer full; pausing new retry retention")
                    self._retry_buffer_full_logged = True
                break
            self._retries[candidate.chunk_id] = candidate

    def __call__(self, store: VectorStore, *, limit: int) -> list[EmbedCandidate]:
        cursor = store.conn.cursor()
        self._refresh_retries(cursor)
        if limit <= 0:
            return []
        if limit == 1 and self._retries:
            retry_limit = 1 if self._retry_turn else 0
            self._retry_turn = not self._retry_turn
        else:
            retry_limit = min(len(self._retries), max(1, limit // 2))
        candidates = self._take_retries(limit=retry_limit)
        retention_slots = MAX_HOT_CANDIDATE_RETRIES - len(self._retries)
        scan_limit = min(limit - len(candidates), retention_slots)
        if scan_limit <= 0:
            return candidates
        retry_ids = set(self._retries)
        head_rows = list(cursor.execute(HOT_CANDIDATE_ROWID_SCAN_SQL, (HOT_CANDIDATE_HEAD_LIMIT,)))
        head_candidates, _, last_inspected_rowid, _ = _candidates_from_scanned_rows(
            head_rows,
            limit=scan_limit,
            exclude_ids=retry_ids,
        )
        candidates.extend(head_candidates)
        retention_slots -= len(head_candidates)
        if not head_rows:
            self._retain(candidates)
            return candidates
        if self._newest_seen_rowid is None:
            self._newest_seen_rowid = last_inspected_rowid
        if self._before_rowid is None and last_inspected_rowid is not None:
            self._before_rowid = last_inspected_rowid
        if len(candidates) >= limit or retention_slots <= 0:
            self._retain(candidates)
            return candidates

        page_limit = HOT_CANDIDATE_SCAN_LIMIT - HOT_CANDIDATE_HEAD_LIMIT
        forward_rows = list(
            cursor.execute(
                HOT_CANDIDATE_ROWID_FORWARD_SQL,
                (self._newest_seen_rowid, page_limit),
            )
        )
        if forward_rows:
            forward_candidates, _, last_inspected_rowid, _ = _candidates_from_scanned_rows(
                forward_rows,
                limit=min(limit - len(candidates), retention_slots),
                exclude_ids=retry_ids | {candidate.chunk_id for candidate in candidates},
            )
            candidates.extend(forward_candidates)
            retention_slots -= len(forward_candidates)
            self._newest_seen_rowid = last_inspected_rowid
        if len(candidates) >= limit or retention_slots <= 0 or self._before_rowid is None:
            self._retain(candidates)
            return candidates[:limit]

        page_rows = list(cursor.execute(HOT_CANDIDATE_ROWID_PAGE_SQL, (self._before_rowid, page_limit)))
        if page_rows:
            page_candidates, _, last_inspected_rowid, fully_scanned = _candidates_from_scanned_rows(
                page_rows,
                limit=min(limit - len(candidates), retention_slots),
                exclude_ids=retry_ids | {candidate.chunk_id for candidate in candidates},
            )
            candidates.extend(page_candidates)
            self._before_rowid = last_inspected_rowid
            if fully_scanned and len(page_rows) < page_limit:
                self._before_rowid = None
        self._retain(candidates)
        return candidates[:limit]


def _stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def _candidate_chunk_ids(store: VectorStore, *, limit: int) -> list[str]:
    rows = store.conn.cursor().execute(
        """
        SELECT c.id
        FROM chunks c
        LEFT JOIN chunk_vectors_rowids r ON r.id = c.id
        WHERE r.id IS NULL
          AND c.source_file = 'brainbar-store'
          AND c.source = 'mcp'
          AND c.content IS NOT NULL
          AND c.content != ''
          AND c.archived_at IS NULL
          AND c.superseded_by IS NULL
          AND c.aggregated_into IS NULL
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [str(row[0]) for row in rows]


def _candidate_chunk_rows(store: VectorStore, *, limit: int) -> list[EmbedCandidate]:
    rows = store.conn.cursor().execute(
        HOT_CANDIDATE_SCAN_SQL,
        (HOT_CANDIDATE_SCAN_LIMIT,),
    )
    candidates: list[EmbedCandidate] = []
    for row in rows:
        (
            chunk_id,
            content,
            source_file,
            source,
            archived_at,
            superseded_by,
            aggregated_into,
            archived,
            status,
            vector_id,
        ) = row
        if (
            vector_id is None
            and source_file == "brainbar-store"
            and source == "mcp"
            and content
            and archived_at is None
            and superseded_by is None
            and aggregated_into is None
        ):
            candidate = EmbedCandidate(str(chunk_id), str(content))
            if CHUNK_FAILURE_ATTEMPTS.quarantined(candidate):
                continue
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
    return candidates


def _pending_chunk_rows(
    store: VectorStore,
    *,
    limit: int,
    scan_state: PendingCandidateScanState | None = None,
) -> list[EmbedCandidate]:
    if limit <= 0:
        return []

    scan_limit = limit
    candidates: list[EmbedCandidate] = []
    after_created_at = scan_state.after_created_at if scan_state and scan_state.active else None
    after_rowid = scan_state.after_rowid if scan_state and scan_state.active else 0
    first_page = not scan_state or not scan_state.active
    exhausted = False

    for _page in range(MAX_PENDING_CANDIDATE_SCAN_PAGES):
        if first_page:
            page_filter = ""
            bindings: tuple[object, ...] = (scan_limit,)
        elif after_created_at is None:
            page_filter = "AND ((c.created_at IS NULL AND c.rowid > ?) OR c.created_at IS NOT NULL)"
            bindings = (after_rowid, scan_limit)
        else:
            page_filter = """
              AND (c.created_at, c.rowid) > (?, ?)
            """
            bindings = (after_created_at, after_rowid, scan_limit)

        id_rows = list(
            store.conn.cursor().execute(
                f"""
                SELECT c.id, c.created_at, c.rowid
                FROM chunks c
                LEFT JOIN chunk_vectors_rowids r ON c.id = r.id
                WHERE r.id IS NULL
                  {page_filter}
                ORDER BY c.created_at ASC
                LIMIT ?
                """,
                bindings,
            )
        )
        if not id_rows:
            exhausted = True
            break

        candidate_ids = [str(row[0]) for row in id_rows]
        placeholders = ", ".join("?" for _chunk_id in candidate_ids)
        content_rows = store.conn.cursor().execute(
            f"""
            SELECT
                id,
                content,
                archived_at,
                superseded_by,
                aggregated_into,
                COALESCE(archived, 0),
                COALESCE(status, 'active')
            FROM chunks
            WHERE id IN ({placeholders})
            """,
            tuple(candidate_ids),
        )
        content_by_id = {
            str(row[0]): str(row[1])
            for row in content_rows
            if row[1] and row[2] is None and row[3] is None and row[4] is None
        }
        for chunk_id in candidate_ids:
            if chunk_id not in content_by_id:
                continue
            candidate = EmbedCandidate(chunk_id, content_by_id[chunk_id])
            if not CHUNK_FAILURE_ATTEMPTS.quarantined(candidate):
                candidates.append(candidate)

        after_created_at = id_rows[-1][1]
        after_rowid = int(id_rows[-1][2])
        first_page = False
        if len(candidates) >= limit or len(id_rows) < scan_limit:
            exhausted = len(id_rows) < scan_limit
            break

    if scan_state is not None:
        if exhausted:
            scan_state.reset()
        else:
            scan_state.active = True
            scan_state.after_created_at = after_created_at
            scan_state.after_rowid = after_rowid
    return candidates[:limit]


PENDING_CANDIDATE_SCAN_STATE = PendingCandidateScanState()


def _pending_chunk_rows_with_resume(store: VectorStore, *, limit: int) -> list[EmbedCandidate]:
    return _pending_chunk_rows(store, limit=limit, scan_state=PENDING_CANDIDATE_SCAN_STATE)


def _embed_candidates(
    candidates: list[EmbedCandidate],
    *,
    embed_fn: Callable[[str], list[float]],
    embed_batch_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[EmbeddedVector]:
    if not candidates:
        return []

    if embed_batch_fn is not None and len(candidates) > 1:
        try:
            embeddings = embed_batch_fn([candidate.content for candidate in candidates])
            if len(embeddings) != len(candidates):
                raise ValueError(f"batch embedder returned {len(embeddings)} embeddings for {len(candidates)} chunks")
            return [
                EmbeddedVector(candidate.chunk_id, candidate.content, embedding)
                for candidate, embedding in zip(candidates, embeddings)
            ]
        except Exception as exc:
            LOGGER.warning("Failed to embed hotlane batch: %s", exc)

    embedded: list[EmbeddedVector] = []
    streak = _FailureStreak("encode")
    for candidate in candidates:
        try:
            embedded.append(EmbeddedVector(candidate.chunk_id, candidate.content, embed_fn(candidate.content)))
        except Exception as exc:
            LOGGER.warning("Failed to embed chunk %s: %s", candidate.chunk_id, exc)
            streak.failed(candidate, exc)
        else:
            streak.flush()
    streak.flush()
    return embedded


def _hotlane_write_busy_timeout_ms() -> int:
    raw_value = os.environ.get("BRAINLAYER_HOTLANE_WRITE_BUSY_TIMEOUT_MS")
    if raw_value is None:
        return DEFAULT_HOTLANE_WRITE_BUSY_TIMEOUT_MS
    try:
        return max(1, min(int(raw_value), MAX_APSW_BUSY_TIMEOUT_MS))
    except ValueError:
        LOGGER.warning(
            "Invalid BRAINLAYER_HOTLANE_WRITE_BUSY_TIMEOUT_MS=%r; using %dms",
            raw_value,
            DEFAULT_HOTLANE_WRITE_BUSY_TIMEOUT_MS,
        )
        return DEFAULT_HOTLANE_WRITE_BUSY_TIMEOUT_MS


def _open_hotlane_write_connection(db_path: Path) -> apsw.Connection:
    timeout_ms = _hotlane_write_busy_timeout_ms()
    # S17: the constructor runs the connection hooks. Without this, PRAGMA optimize took the
    # write lock on every cycle at APSW's 100 ms timeout and threw away the embedded batch
    # (1,264 of 1,430 logged cycle failures). The hotlane timeout covers the hooks that do run.
    with skip_connection_hooks(*HOTLANE_SKIPPED_CONNECTION_HOOKS), temporary_write_busy_timeout_ms(timeout_ms):
        conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READWRITE)
    try:
        conn.setbusytimeout(timeout_ms)
        conn.enableloadextension(True)
        conn.loadextension(sqlite_vec.loadable_path())
        conn.enableloadextension(False)
    except BaseException:
        conn.close()
        raise
    return conn


def _table_exists(conn: apsw.Connection, table_name: str) -> bool:
    return bool(
        conn.cursor()
        .execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (table_name,),
        )
        .fetchone()
    )


def _upsert_chunk_vector_raw(
    cursor,
    chunk_id: str,
    embedding: list[float],
    *,
    binary_index_available: bool,
) -> None:
    embedding_bytes = serialize_f32(embedding)
    cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
    cursor.execute(
        """
        INSERT INTO chunk_vectors (chunk_id, embedding)
        VALUES (?, ?)
        """,
        (chunk_id, embedding_bytes),
    )
    if binary_index_available:
        cursor.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))
        cursor.execute(
            """
            INSERT INTO chunk_vectors_binary (chunk_id, embedding)
            VALUES (?, vec_quantize_binary(?))
            """,
            (chunk_id, embedding_bytes),
        )


class _WriteProgress:
    def __init__(self) -> None:
        self.written = 0
        self.processed = 0


def _write_embedded_vectors(store_or_path: VectorStore | Path | str, vectors: list[EmbeddedVector]) -> int:
    """Write vectors one transaction each; retry busy/locked failures, never drop the batch.

    Each vector is its own transaction, so a lock error part-way leaves earlier vectors
    committed; the retry resumes at the first unprocessed vector. After
    ``HOTLANE_WRITE_ATTEMPTS`` busy failures it raises ``HotlaneVectorWriteDeferred``
    carrying what is still unwritten, for the caller to keep for the next cycle.
    """
    remaining = list(vectors)
    written = 0
    for attempt in range(1, HOTLANE_WRITE_ATTEMPTS + 1):
        if not remaining:
            break
        progress = _WriteProgress()
        try:
            _write_embedded_vectors_once(store_or_path, remaining, progress)
        except (apsw.BusyError, apsw.LockedError) as exc:
            written += progress.written
            remaining = remaining[progress.processed :]
            if attempt == HOTLANE_WRITE_ATTEMPTS:
                raise HotlaneVectorWriteDeferred(
                    written=written, remaining=remaining, attempts=attempt, cause=exc
                ) from exc
            LOGGER.info(
                "hotlane vector write busy; retrying attempt=%d/%d unwritten=%d error_type=%s",
                attempt + 1,
                HOTLANE_WRITE_ATTEMPTS,
                len(remaining),
                type(exc).__name__,
            )
            _sleep(HOTLANE_WRITE_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            continue
        written += progress.written
        remaining = []
    return written


def _write_embedded_vectors_once(
    store_or_path: VectorStore | Path | str,
    vectors: list[EmbeddedVector],
    progress: _WriteProgress,
) -> None:
    if not vectors:
        return

    conn: apsw.Connection | None = None
    db_path: Path | None
    if hasattr(store_or_path, "conn"):
        store = store_or_path
        cursor = store.conn.cursor()
        db_path = getattr(store, "db_path", None)
        upsert_chunk_vector = store._upsert_chunk_vector
    else:
        db_path = Path(store_or_path)
        conn = _open_hotlane_write_connection(db_path)
        cursor = conn.cursor()
        binary_index_available = _table_exists(conn, "chunk_vectors_binary")

        def upsert_chunk_vector(cursor, chunk_id, embedding):
            return _upsert_chunk_vector_raw(
                cursor,
                chunk_id,
                embedding,
                binary_index_available=binary_index_available,
            )

    count = 0
    streak = _FailureStreak("write")
    try:
        for vector_index, (chunk_id, content, embedding) in enumerate(vectors):
            metrics = IngestMetrics("hotlane", "vector")
            transaction_started = False
            telemetry_span = start_writer_span(
                cursor.connection if hasattr(cursor, "connection") else (conn or store.conn),
                db_path=db_path,
                producer="hotlane",
                lane="hotlane",
                operation="write_vectors",
                rows_planned=1,
            )
            try:
                metrics.run("begin", cursor.execute, "BEGIN IMMEDIATE")
                transaction_started = True
                still_eligible = metrics.run(
                    "apply",
                    cursor.execute,
                    """
                    SELECT 1, c.metadata
                    FROM chunks c
                    LEFT JOIN chunk_vectors_rowids r ON c.id = r.id
                    WHERE c.id = ?
                      AND c.content = ?
                      AND r.id IS NULL
                      AND c.content IS NOT NULL
                      AND c.content != ''
                      AND c.archived_at IS NULL
                      AND c.superseded_by IS NULL
                      AND c.aggregated_into IS NULL
                    """,
                    (chunk_id, content),
                ).fetchone()
                rows_touched = 0
                if still_eligible:
                    metrics.run("apply", upsert_chunk_vector, cursor, chunk_id, embedding)
                    rows_touched = 1
                metrics.run("commit", cursor.execute, "COMMIT")
                transaction_started = False
                if rows_touched:
                    try:
                        metadata = json.loads(still_eligible[1]) if len(still_eligible) > 1 else {}
                    except (TypeError, ValueError):
                        metadata = {}
                    metadata = metadata if isinstance(metadata, dict) else {}
                    metrics.committed([{"queued_at": metadata.get("ingest_queued_at"), "metadata": metadata}])
                metrics.finish("commit")
                streak.flush()
                CHUNK_FAILURE_ATTEMPTS.succeeded(EmbedCandidate(chunk_id, content))
                count += rows_touched
                progress.written = count
                progress.processed = vector_index + 1
                telemetry_span.finish("commit", rows_touched=rows_touched)
                if vector_index < len(vectors) - 1:
                    _sleep(VECTOR_WRITE_YIELD_SECONDS)
            except Exception as exc:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                metrics.finish("error")
                telemetry_span.finish("rollback", error=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, (apsw.BusyError, apsw.LockedError)):
                    streak.flush()
                    raise
                progress.processed = vector_index + 1
                LOGGER.error(
                    "hotlane vector write skipped chunk_id=%s error_type=%s error=%s",
                    chunk_id,
                    type(exc).__name__,
                    exc,
                )
                streak.failed(EmbedCandidate(chunk_id, content), exc)
                if vector_index < len(vectors) - 1:
                    _sleep(VECTOR_WRITE_YIELD_SECONDS)
        streak.flush()
    finally:
        if conn is not None:
            conn.close()
        if count:
            from brainlayer.search_repo import clear_hybrid_search_cache

            clear_hybrid_search_cache(db_path)


def _open_store(
    vector_store_cls: Callable[..., VectorStore],
    db_path: Path,
    *,
    readonly: bool,
) -> VectorStore:
    if not _callable_accepts_keyword(vector_store_cls, "readonly"):
        return vector_store_cls(db_path)
    try:
        return vector_store_cls(db_path, readonly=readonly)
    except TypeError as exc:
        if not _type_error_rejects_keyword(exc, "readonly"):
            raise
        return vector_store_cls(db_path)


def _callable_accepts_keyword(func: Callable[..., object], keyword: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    parameter = signature.parameters.get(keyword)
    return (
        parameter is not None
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ) or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def _type_error_rejects_keyword(exc: TypeError, keyword: str) -> bool:
    message = str(exc)
    return keyword in message and (
        "unexpected keyword argument" in message
        or "got an unexpected keyword" in message
        or "invalid keyword argument" in message
    )


def _create_embedding_model(model_factory: Callable[..., object], *, device: str) -> object:
    if _callable_accepts_keyword(model_factory, "device"):
        return model_factory(device=device)
    return model_factory()


def _run_split_cycle(
    *,
    db_path: Path,
    vector_store_cls: Callable[..., VectorStore],
    embed_fn: Callable[[str], list[float]],
    recent_limit: int,
    backlog_batch: int,
    enrich_limit: int,
    enrich_since_hours: int,
    embed_batch_fn: Callable[[list[str]], list[list[float]]] | None = None,
    enrich_fn: Callable[..., object] | None = None,
    candidate_rows_fn: Callable[..., list[EmbedCandidate]] = _candidate_chunk_rows,
    pending_rows_fn: Callable[..., list[EmbedCandidate]] = _pending_chunk_rows_with_resume,
    write_vectors_fn: Callable[..., int] = _write_embedded_vectors,
    carryover: VectorCarryover | None = None,
) -> CycleResult:
    carryover = VECTOR_CARRYOVER if carryover is None else carryover
    metrics = IngestMetrics("hotlane", "hotlane")
    embedded = 0

    def finish(outcome: str = "completed", exc: BaseException | None = None) -> None:
        metrics.fields["quarantined_total"] = CHUNK_FAILURE_ATTEMPTS.quarantined_total
        if isinstance(exc, HotlaneSystemicFailure):
            metrics.fields["systemic_failure"] = f"{exc.stage}:{exc.error_type}"
        metrics.finish(outcome)

    # Peek, not take: a scan or encode failure later in this cycle must not lose them.
    carried = [
        vector
        for vector in carryover.peek()
        if not CHUNK_FAILURE_ATTEMPTS.quarantined(EmbedCandidate(vector.chunk_id, vector.content))
    ]
    carried_by_id = {vector.chunk_id: vector for vector in carried}
    hot_rows: list[EmbedCandidate] = []
    pending_rows: list[EmbedCandidate] = []
    pending_scan_state = PENDING_CANDIDATE_SCAN_STATE if pending_rows_fn is _pending_chunk_rows_with_resume else None
    pending_scan_checkpoint = (
        (pending_scan_state.active, pending_scan_state.after_created_at, pending_scan_state.after_rowid)
        if pending_scan_state is not None and backlog_batch > 0
        else None
    )

    def restore_pending_scan() -> None:
        if pending_scan_checkpoint is not None and pending_scan_state is not None:
            # Committed rows are filtered by the next scan; unresolved rows retry immediately.
            pending_scan_state.active, pending_scan_state.after_created_at, pending_scan_state.after_rowid = (
                pending_scan_checkpoint
            )

    if recent_limit > 0 or backlog_batch > 0:
        if not db_path.exists():
            bootstrap_store = _open_store(vector_store_cls, db_path, readonly=False)
            bootstrap_store.close()
        read_store = _open_store(vector_store_cls, db_path, readonly=True)
        try:
            if recent_limit > 0:
                hot_rows = metrics.run("candidate", candidate_rows_fn, read_store, limit=recent_limit)
            if backlog_batch > 0:
                pending_rows = metrics.run("candidate", pending_rows_fn, read_store, limit=backlog_batch)
        finally:
            read_store.close()

    hot_rows = [candidate for candidate in hot_rows if not CHUNK_FAILURE_ATTEMPTS.quarantined(candidate)]
    pending_rows = [candidate for candidate in pending_rows if not CHUNK_FAILURE_ATTEMPTS.quarantined(candidate)]
    pending_candidates = pending_rows
    seen_hot_ids = {candidate.chunk_id for candidate in hot_rows}
    pending_rows = [candidate for candidate in pending_rows if candidate.chunk_id not in seen_hot_ids]

    def needs_embedding(candidate: EmbedCandidate) -> bool:
        # A carried vector for the same content is reused, not recomputed.
        carried_vector = carried_by_id.get(candidate.chunk_id)
        return carried_vector is None or carried_vector.content != candidate.content

    hot_rows = [candidate for candidate in hot_rows if needs_embedding(candidate)]
    pending_rows = [candidate for candidate in pending_rows if needs_embedding(candidate)]
    try:
        vectors = metrics.run("encode", _embed_candidates, hot_rows, embed_fn=embed_fn, embed_batch_fn=embed_batch_fn)
        backlog_vectors = metrics.run(
            "encode", _embed_candidates, pending_rows, embed_fn=embed_fn, embed_batch_fn=embed_batch_fn
        )
    except Exception as exc:
        restore_pending_scan()
        finish("error", exc)
        raise
    vectors.extend(backlog_vectors)
    fresh_ids = {vector.chunk_id for vector in vectors}
    vectors = [vector for vector in carried if vector.chunk_id not in fresh_ids] + vectors
    encoded_content = {vector.chunk_id: vector.content for vector in vectors}
    if any(
        encoded_content.get(candidate.chunk_id) != candidate.content
        and not CHUNK_FAILURE_ATTEMPTS.quarantined(candidate)
        for candidate in pending_candidates
    ):
        # Individual encode failures are logged and omitted, including hot/backlog overlaps.
        restore_pending_scan()
    if vectors:
        try:
            embedded += metrics.run("write", write_vectors_fn, db_path, vectors)
            carryover.clear()
            if any(
                0 < CHUNK_FAILURE_ATTEMPTS.count(candidate) < MAX_CHUNK_FAILURE_ATTEMPTS
                for candidate in pending_candidates
            ):
                restore_pending_scan()
        except Exception as exc:
            deferred = exc if isinstance(exc, HotlaneVectorWriteDeferred) else None
            is_lock_error = deferred is not None or isinstance(exc, (apsw.BusyError, apsw.LockedError))
            restore_pending_scan()
            if not is_lock_error:
                LOGGER.error(
                    "hotlane vector write failed; retrying chunk_ids=%s error_type=%s",
                    [candidate.chunk_id for candidate in pending_rows],
                    type(exc).__name__,
                )
            if not is_lock_error:
                finish("error", exc)
                raise
            # S17: a busy writer slot never costs the embedded batch. A writer that does not
            # report progress gets all of its vectors carried; its writes are idempotent
            # because each one re-checks that the chunk still lacks a vector.
            unwritten = deferred.remaining if deferred is not None else vectors
            embedded += deferred.written if deferred is not None else 0
            carryover.clear()
            carryover.put(unwritten)
            LOGGER.warning(
                "hotlane vector write deferred: writer slot busy; carrying %d computed vectors to the next "
                "cycle written=%d carried_now=%d carried_total=%d dropped_total=%d error=%s",
                len(unwritten),
                embedded,
                len(carryover),
                carryover.carried_total,
                carryover.dropped_total,
                exc,
            )
            finish("deferred")
            _warn_retired_enrichment(enrich_limit, enrich_since_hours, enrich_fn)
            return CycleResult(embedded=embedded, deferred=len(unwritten), backlog_encoded=len(backlog_vectors))

    _warn_retired_enrichment(enrich_limit, enrich_since_hours, enrich_fn)
    finish()
    return CycleResult(embedded=embedded, backlog_encoded=len(backlog_vectors))


def _warn_retired_enrichment(limit: int, since_hours: int, callback: object) -> None:
    if limit > 0 or callback is not None:
        LOGGER.warning(
            "Hotlane enrichment has been retired; ignoring limit=%d since_hours=%d and any callback",
            limit,
            since_hours,
        )


def _default_queue_dir() -> Path:
    import os

    return Path(os.environ.get("BRAINLAYER_QUEUE_DIR", str(Path.home() / ".brainlayer" / "queue"))).expanduser()


def _queue_depth(queue_dir: Path) -> int:
    try:
        return sum(1 for _path in queue_dir.glob("*.jsonl"))
    except OSError:
        return 0


def _is_enrichment_queue_file(path: Path) -> bool:
    return path.name.startswith("enrichment-") or path.name.startswith("queue-enrichment")


def _high_priority_queue_depth(queue_dir: Path) -> int:
    try:
        return sum(1 for path in queue_dir.glob("*.jsonl") if not _is_enrichment_queue_file(path))
    except OSError:
        return 0


def run_cycle(
    *,
    store: VectorStore,
    embed_fn: Callable[[str], list[float]],
    recent_limit: int,
    backlog_batch: int,
    enrich_limit: int,
    enrich_since_hours: int,
    candidate_chunk_ids_fn: Callable[..., list[str]] = _candidate_chunk_ids,
    hot_embed_fn: Callable[..., bool] = embed_hot_chunk,
    pending_embed_fn: Callable[..., int] = embed_pending_chunks,
    embed_batch_fn: Callable[[list[str]], list[list[float]]] | None = None,
    enrich_fn: Callable[..., object] | None = None,
) -> CycleResult:
    embedded = 0
    for chunk_id in candidate_chunk_ids_fn(store, limit=recent_limit):
        if hot_embed_fn(store=store, embed_fn=embed_fn, chunk_id=chunk_id):
            embedded += 1
            break

    if backlog_batch > 0:
        embedded += pending_embed_fn(
            store=store,
            embed_fn=embed_fn,
            batch_size=backlog_batch,
            embed_batch_fn=embed_batch_fn,
        )

    _warn_retired_enrichment(enrich_limit, enrich_since_hours, enrich_fn)
    return CycleResult(embedded=embedded)


def run(
    *,
    db_path: Path,
    interval: float,
    recent_limit: int,
    backlog_interval: float,
    backlog_batch: int,
    enrich_interval: float,
    enrich_limit: int,
    enrich_since_hours: int,
    vector_store_cls: Callable[[Path], VectorStore] = VectorStore,
    model_factory: Callable[..., object] = get_embedding_model,
    embedding_device: str = DEFAULT_HOTLANE_EMBED_DEVICE,
    cycle_fn: Callable[..., CycleResult] = run_cycle,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_cycles: int | None = None,
    queue_dir: Path | None = None,
    queue_depth_fn: Callable[[Path], int] = _queue_depth,
    high_priority_queue_depth_fn: Callable[[Path], int] = _high_priority_queue_depth,
    cpu_sampler: HotlaneCpuSampler | None = None,
) -> None:
    model = _create_embedding_model(model_factory, device=embedding_device)
    embed_batch_fn = getattr(model, "embed_texts", None)
    if embed_batch_fn is not None:

        def embed_fn(text: str) -> list[float]:
            embeddings = embed_batch_fn([text])
            if len(embeddings) != 1:
                raise RuntimeError(f"single text embedder returned {len(embeddings)} embeddings")
            return embeddings[0]

    else:
        embed_fn = model.embed_query
    queue_dir_was_explicit = queue_dir is not None
    queue_dir = queue_dir or _default_queue_dir()
    last_backlog = time_fn() - backlog_interval
    last_backlog_duty = last_backlog + backlog_interval
    max_backlog_wait = float(os.environ.get("BRAINLAYER_HOTLANE_BACKLOG_MAX_WAIT_SECONDS", "300"))
    if not math.isfinite(max_backlog_wait) or max_backlog_wait <= 0:
        raise ValueError("BRAINLAYER_HOTLANE_BACKLOG_MAX_WAIT_SECONDS must be finite and positive")
    _warn_retired_enrichment(enrich_limit, enrich_since_hours, None)
    if enrich_limit > 0:
        LOGGER.warning("Ignoring retired hotlane enrichment interval=%s", enrich_interval)
    queue_backpressure_active = False
    backlog_max_load = _backlog_max_load()
    backlog_yield_reason = ""
    backlog_slice_logged = False
    hot_candidate_scanner = HotCandidateScanner()
    cycles = 0
    backlog_batch = min(max(backlog_batch, 0), MAX_BACKLOG_BATCH)
    LOGGER.info("hotlane adapter started db=%s", db_path)
    idle_sleep = interval
    systemic_failures = 0
    while not STOP:
        if max_cycles is not None and cycles >= max_cycles:
            break
        did_work = False
        try:
            now = time_fn()
            queue_has_backlog = queue_depth_fn(queue_dir) > 0
            queue_has_high_priority_backlog = queue_has_backlog and high_priority_queue_depth_fn(queue_dir) > 0
            cycle_backlog_batch = backlog_batch if backlog_batch > 0 and now - last_backlog >= backlog_interval else 0
            cycle_recent_limit = recent_limit
            reason = _backlog_yield_reason(db_path, backlog_max_load) if backlog_batch else ""
            if reason == "high load" and now - last_backlog_duty >= max_backlog_wait:
                # Reserve the existing backlog lane; never lift store-queue backpressure.
                reason = ""
                cycle_backlog_batch = backlog_batch
            if reason != backlog_yield_reason:
                LOGGER.info("backlog embedding %s", f"yielding: {reason}" if reason else "resumed")
                backlog_yield_reason = reason
            if reason:
                cycle_backlog_batch = 0
            if queue_has_high_priority_backlog:
                if not queue_backpressure_active:
                    LOGGER.info("durable high-priority queue has backlog; suppressing hot embedding")
                queue_backpressure_active = True
                if cycle_backlog_batch <= 0:
                    if cpu_sampler is not None:
                        cpu_sampler.observe("store queue")
                    cycles += 1
                    # Backpressure is not idleness: hold the configured cadence so the
                    # reserved backlog slice resumes on time.
                    idle_sleep = interval
                    sleep_fn(interval)
                    continue
                cycle_recent_limit = 0
                if not backlog_slice_logged:
                    LOGGER.info(
                        "durable high-priority queue has backlog; reserving backlog embedding slice batch=%d",
                        cycle_backlog_batch,
                    )
                    backlog_slice_logged = True
            else:
                queue_backpressure_active = False
                backlog_slice_logged = False
            if cycle_backlog_batch > 0:
                last_backlog = now
                last_backlog_duty = now
            cycle_start = cpu_sampler.begin_cycle() if cpu_sampler is not None else None
            if cycle_fn is run_cycle:
                result = _run_split_cycle(
                    db_path=db_path,
                    vector_store_cls=vector_store_cls,
                    embed_fn=embed_fn,
                    recent_limit=cycle_recent_limit,
                    backlog_batch=cycle_backlog_batch,
                    embed_batch_fn=embed_batch_fn,
                    enrich_limit=0,
                    enrich_since_hours=enrich_since_hours,
                    candidate_rows_fn=hot_candidate_scanner,
                )
            else:
                store = vector_store_cls(db_path)
                try:
                    result = cycle_fn(
                        store=store,
                        embed_fn=embed_fn,
                        recent_limit=cycle_recent_limit,
                        backlog_batch=cycle_backlog_batch,
                        embed_batch_fn=embed_batch_fn,
                        enrich_limit=0,
                        enrich_since_hours=enrich_since_hours,
                    )
                finally:
                    store.close()
            # A queued backlog means the system has work even when THIS cycle embedded
            # nothing (the hot slice is deliberately suppressed to yield to the drain),
            # so it must not be treated as idle -- backing off there would slow recovery.
            did_work = bool(result.embedded) or queue_has_backlog
            if cpu_sampler is not None:
                state = reason or ("running" if result.backlog_encoded else "waiting")
                cpu_sampler.observe(state, result.backlog_encoded, cycle_start=cycle_start)
            if result.embedded:
                LOGGER.info("embedded=%d", result.embedded)
        except Exception as exc:
            LOGGER.exception("hotlane adapter cycle failed")
            if cpu_sampler is not None:
                cpu_sampler.observe("error")
            cycles += 1
            error_sleep = min(interval * 2, 5.0)
            if isinstance(exc, HotlaneSystemicFailure):
                # The environment is failing for every chunk: back off geometrically so a
                # full disk is not re-encoded every few seconds. Never counted per chunk.
                systemic_failures += 1
                error_sleep = _systemic_backoff_seconds(error_sleep, systemic_failures)
                LOGGER.warning("hotlane systemic failure streak=%d; backing off %.1fs", systemic_failures, error_sleep)
            else:
                systemic_failures = 0
            # A raised cycle is a RETRY, not an idle cycle (PR #735 review, macroscope):
            # it already slept its own backoff here, and letting it fall through to the
            # geometric idle path would compound repeated transient failures into a much
            # longer retry gap. Reset the idle cadence so recovery stays prompt.
            idle_sleep = interval
            sleep_fn(error_sleep)
            continue
        systemic_failures = 0
        cycles += 1
        # Work resets the cadence so latency stays at `interval`; an idle cycle backs off
        # geometrically up to MAX_IDLE_INTERVAL_SECONDS instead of probing every second.
        if did_work:
            idle_sleep = interval
        sleep_fn(idle_sleep)
        if not did_work:
            idle_sleep = min(idle_sleep * IDLE_BACKOFF_FACTOR, MAX_IDLE_INTERVAL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_db_path())
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--recent-limit", type=int, default=5)
    parser.add_argument("--backlog-interval", type=float, default=10.0)
    parser.add_argument("--backlog-batch", type=int, default=DEFAULT_BACKLOG_BATCH)
    # Old installed plists outlive a keg upgrade. Remove these inert shims only
    # after the lead receipts hotlane plist re-rendering on BOTH Macs.
    parser.add_argument("--enrich-interval", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--enrich-limit", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--enrich-since-hours", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if any(value != 0 for value in (args.enrich_interval, args.enrich_limit, args.enrich_since_hours)):
        LOGGER.warning("Hotlane enrichment options are retired and ignored; continuing local embedding")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    setup_cpu_budget(args.db)
    sampler = HotlaneCpuSampler(args.db)
    sampler.start()
    try:
        run(
            db_path=args.db,
            interval=max(args.interval, 0.25),
            recent_limit=max(args.recent_limit, 1),
            backlog_interval=max(args.backlog_interval, 1.0),
            backlog_batch=min(max(args.backlog_batch, 0), MAX_BACKLOG_BATCH),
            enrich_interval=0.0,
            enrich_limit=0,
            enrich_since_hours=0,
            cpu_sampler=sampler,
        )
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
