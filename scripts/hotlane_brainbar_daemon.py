#!/usr/bin/env python3
"""Hot-embed recent BrainBar MCP writes and interleave realtime enrichment.

BrainBar owns the live MCP socket and stores ``brain_store`` rows directly in
SQLite. This adapter owns the single BrainLayer writer slot and must therefore
perform all write-side background work that needs steady progress: hot embedding,
optional embedding backlog, and enrichment backlog.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import signal
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, NamedTuple

import apsw
import sqlite_vec

from brainlayer._helpers import serialize_f32
from brainlayer.embeddings import get_embedding_model
from brainlayer.enrichment_controller import (
    DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS,
    _result_hit_daily_cap,
    enrich_realtime,
)
from brainlayer.paths import get_db_path
from brainlayer.store import embed_hot_chunk, embed_pending_chunks
from brainlayer.vector_store import VectorStore
from brainlayer.writer_telemetry import start_writer_span

LOGGER = logging.getLogger("brainlayer.hotlane_brainbar")
STOP = False
DEFAULT_HOTLANE_ENRICH_LIMIT = 5
DEFAULT_BACKLOG_BATCH = 4
DEFAULT_HOTLANE_EMBED_DEVICE = "cpu"
MAX_BACKLOG_BATCH = 16
DEFAULT_HOTLANE_WRITE_BUSY_TIMEOUT_MS = 1000
MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647
VECTOR_WRITE_YIELD_SECONDS = 0.005
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


class EmbedCandidate(NamedTuple):
    chunk_id: str
    content: str


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
            and not archived
            and status == "active"
        )
        if eligible:
            if first_candidate_rowid is None:
                first_candidate_rowid = int(rowid)
            if str(chunk_id) in exclude_ids:
                continue
            candidates.append(EmbedCandidate(str(chunk_id), str(content)))
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
              AND COALESCE(c.archived, 0) = 0
              AND COALESCE(c.status, 'active') = 'active'
            """,
            tuple(self._retries),
        )
        active = {str(chunk_id): EmbedCandidate(str(chunk_id), str(content)) for chunk_id, content in rows}
        for chunk_id in list(self._retries):
            if chunk_id not in active:
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
          AND COALESCE(c.archived, 0) = 0
          AND COALESCE(c.status, 'active') = 'active'
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
            and not archived
            and status == "active"
        ):
            candidates.append(EmbedCandidate(str(chunk_id), str(content)))
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
            if row[1] and row[2] is None and row[3] is None and row[4] is None and not row[5] and row[6] == "active"
        }
        candidates.extend(
            EmbedCandidate(chunk_id, content_by_id[chunk_id]) for chunk_id in candidate_ids if chunk_id in content_by_id
        )

        after_created_at = id_rows[-1][1]
        after_rowid = int(id_rows[-1][2])
        first_page = False
        if len(candidates) >= limit or len(id_rows) < scan_limit:
            exhausted = len(id_rows) < scan_limit
            break

    if scan_state is not None:
        if candidates or exhausted:
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
    for candidate in candidates:
        try:
            embedded.append(EmbeddedVector(candidate.chunk_id, candidate.content, embed_fn(candidate.content)))
        except Exception as exc:
            LOGGER.warning("Failed to embed chunk %s: %s", candidate.chunk_id, exc)
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
    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READWRITE)
    conn.setbusytimeout(_hotlane_write_busy_timeout_ms())
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
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


def _write_embedded_vectors(store_or_path: VectorStore | Path | str, vectors: list[EmbeddedVector]) -> int:
    if not vectors:
        return 0

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
    try:
        for vector_index, (chunk_id, content, embedding) in enumerate(vectors):
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
                cursor.execute("BEGIN IMMEDIATE")
                transaction_started = True
                still_eligible = cursor.execute(
                    """
                    SELECT 1
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
                      AND COALESCE(c.archived, 0) = 0
                      AND COALESCE(c.status, 'active') = 'active'
                    """,
                    (chunk_id, content),
                ).fetchone()
                rows_touched = 0
                if still_eligible:
                    upsert_chunk_vector(cursor, chunk_id, embedding)
                    rows_touched = 1
                cursor.execute("COMMIT")
                transaction_started = False
                count += rows_touched
                telemetry_span.finish("commit", rows_touched=rows_touched)
                if vector_index < len(vectors) - 1:
                    _sleep(VECTOR_WRITE_YIELD_SECONDS)
            except Exception as exc:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                telemetry_span.finish("rollback", error=f"{type(exc).__name__}: {exc}")
                raise
    finally:
        if conn is not None:
            conn.close()
        if count:
            from brainlayer.search_repo import clear_hybrid_search_cache

            clear_hybrid_search_cache(db_path)
    return count


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
    enrich_fn: Callable[..., object] = enrich_realtime,
    candidate_rows_fn: Callable[..., list[EmbedCandidate]] = _candidate_chunk_rows,
    pending_rows_fn: Callable[..., list[EmbedCandidate]] = _pending_chunk_rows_with_resume,
    write_vectors_fn: Callable[..., int] = _write_embedded_vectors,
) -> CycleResult:
    embedded = 0
    hot_rows: list[EmbedCandidate] = []
    pending_rows: list[EmbedCandidate] = []

    if recent_limit > 0 or backlog_batch > 0:
        if not db_path.exists():
            bootstrap_store = _open_store(vector_store_cls, db_path, readonly=False)
            bootstrap_store.close()
        read_store = _open_store(vector_store_cls, db_path, readonly=True)
        try:
            if recent_limit > 0:
                hot_rows = candidate_rows_fn(read_store, limit=recent_limit)
            if backlog_batch > 0:
                pending_rows = pending_rows_fn(read_store, limit=backlog_batch)
        finally:
            read_store.close()

    seen_hot_ids = {candidate.chunk_id for candidate in hot_rows}
    pending_rows = [candidate for candidate in pending_rows if candidate.chunk_id not in seen_hot_ids]
    vectors = _embed_candidates(hot_rows, embed_fn=embed_fn, embed_batch_fn=embed_batch_fn)
    vectors.extend(_embed_candidates(pending_rows, embed_fn=embed_fn, embed_batch_fn=embed_batch_fn))
    if vectors:
        embedded += write_vectors_fn(db_path, vectors)

    if enrich_limit <= 0:
        return CycleResult(embedded=embedded)

    enrich_store = _open_store(vector_store_cls, db_path, readonly=False)
    try:
        enrich_result = enrich_fn(store=enrich_store, limit=enrich_limit, since_hours=enrich_since_hours)
    finally:
        enrich_store.close()
    return CycleResult(
        embedded=embedded,
        enrich_attempted=int(getattr(enrich_result, "attempted", 0) or 0),
        enriched=int(getattr(enrich_result, "enriched", 0) or 0),
        enrich_skipped=int(getattr(enrich_result, "skipped", 0) or 0),
        enrich_failed=int(getattr(enrich_result, "failed", 0) or 0),
        enrich_daily_cap_reached=hasattr(enrich_result, "errors") and _result_hit_daily_cap(enrich_result),
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
    enrich_fn: Callable[..., object] = enrich_realtime,
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

    if enrich_limit <= 0:
        return CycleResult(embedded=embedded)

    enrich_result = enrich_fn(store, limit=enrich_limit, since_hours=enrich_since_hours)
    return CycleResult(
        embedded=embedded,
        enrich_attempted=int(getattr(enrich_result, "attempted", 0) or 0),
        enriched=int(getattr(enrich_result, "enriched", 0) or 0),
        enrich_skipped=int(getattr(enrich_result, "skipped", 0) or 0),
        enrich_failed=int(getattr(enrich_result, "failed", 0) or 0),
        enrich_daily_cap_reached=hasattr(enrich_result, "errors") and _result_hit_daily_cap(enrich_result),
    )


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
    last_enrich = 0.0
    enrich_disabled = False
    queue_backpressure_active = False
    backlog_slice_logged = False
    hot_candidate_scanner = HotCandidateScanner()
    cycles = 0
    backlog_batch = min(max(backlog_batch, 0), MAX_BACKLOG_BATCH)
    LOGGER.info("hotlane adapter started db=%s", db_path)
    while not STOP:
        if max_cycles is not None and cycles >= max_cycles:
            break
        try:
            now = time_fn()
            queue_has_backlog = queue_depth_fn(queue_dir) > 0
            queue_has_high_priority_backlog = queue_has_backlog and high_priority_queue_depth_fn(queue_dir) > 0
            cycle_backlog_batch = backlog_batch if backlog_batch > 0 and now - last_backlog >= backlog_interval else 0
            cycle_recent_limit = recent_limit
            if queue_has_high_priority_backlog:
                if not queue_backpressure_active:
                    LOGGER.info("durable high-priority queue has backlog; suppressing hot embedding and enrichment")
                queue_backpressure_active = True
                if cycle_backlog_batch <= 0:
                    cycles += 1
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
            cycle_enrich_limit = (
                enrich_limit
                if not queue_has_backlog
                and not enrich_disabled
                and enrich_limit > 0
                and now - last_enrich >= enrich_interval
                else 0
            )
            if cycle_backlog_batch > 0:
                last_backlog = now
            if cycle_enrich_limit > 0:
                last_enrich = now
            if cycle_fn is run_cycle:
                result = _run_split_cycle(
                    db_path=db_path,
                    vector_store_cls=vector_store_cls,
                    embed_fn=embed_fn,
                    recent_limit=cycle_recent_limit,
                    backlog_batch=cycle_backlog_batch,
                    embed_batch_fn=embed_batch_fn,
                    enrich_limit=cycle_enrich_limit,
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
                        enrich_limit=cycle_enrich_limit,
                        enrich_since_hours=enrich_since_hours,
                    )
                finally:
                    store.close()
            if result.enrich_daily_cap_reached:
                enrich_disabled = True
                LOGGER.warning("enrichment daily cap reached; disabling hotlane enrichment until restart")
            if result.embedded or result.enrich_attempted:
                LOGGER.info(
                    "embedded=%d enrich_attempted=%d enriched=%d skipped=%d failed=%d",
                    result.embedded,
                    result.enrich_attempted,
                    result.enriched,
                    result.enrich_skipped,
                    result.enrich_failed,
                )
        except Exception:
            LOGGER.exception("hotlane adapter cycle failed")
            sleep_fn(min(interval * 2, 5.0))
        cycles += 1
        sleep_fn(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_db_path())
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--recent-limit", type=int, default=5)
    parser.add_argument("--backlog-interval", type=float, default=10.0)
    parser.add_argument("--backlog-batch", type=int, default=DEFAULT_BACKLOG_BATCH)
    parser.add_argument("--enrich-interval", type=float, default=10.0)
    parser.add_argument("--enrich-limit", type=int, default=DEFAULT_HOTLANE_ENRICH_LIMIT)
    parser.add_argument("--enrich-since-hours", type=int, default=DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    run(
        db_path=args.db,
        interval=max(args.interval, 0.25),
        recent_limit=max(args.recent_limit, 1),
        backlog_interval=max(args.backlog_interval, 1.0),
        backlog_batch=min(max(args.backlog_batch, 0), MAX_BACKLOG_BATCH),
        enrich_interval=max(args.enrich_interval, 1.0),
        enrich_limit=max(args.enrich_limit, 0),
        enrich_since_hours=max(args.enrich_since_hours, 0),
    )


if __name__ == "__main__":
    main()
