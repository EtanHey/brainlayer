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
import signal
import time
from pathlib import Path
from typing import Callable, NamedTuple

from brainlayer.embeddings import get_embedding_model
from brainlayer.enrichment_controller import (
    DEFAULT_ENRICH_SUPERVISOR_SINCE_HOURS,
    _result_hit_daily_cap,
    enrich_realtime,
)
from brainlayer.paths import get_db_path
from brainlayer.store import embed_hot_chunk, embed_pending_chunks
from brainlayer.vector_store import VectorStore

LOGGER = logging.getLogger("brainlayer.hotlane_brainbar")
STOP = False
DEFAULT_HOTLANE_ENRICH_LIMIT = 5
DEFAULT_BACKLOG_BATCH = 4
MAX_BACKLOG_BATCH = 16


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


class EmbeddedVector(NamedTuple):
    chunk_id: str
    content: str
    embedding: list[float]


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
        """
        SELECT c.id, c.content
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
    return [EmbedCandidate(str(row[0]), str(row[1])) for row in rows]


def _pending_chunk_rows(store: VectorStore, *, limit: int) -> list[EmbedCandidate]:
    rows = store.conn.cursor().execute(
        """
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN chunk_vectors_rowids r ON c.id = r.id
        WHERE r.id IS NULL
          AND c.content IS NOT NULL
          AND c.content != ''
          AND c.archived_at IS NULL
          AND c.superseded_by IS NULL
          AND c.aggregated_into IS NULL
          AND COALESCE(c.archived, 0) = 0
          AND COALESCE(c.status, 'active') = 'active'
        ORDER BY c.created_at ASC
        LIMIT ?
        """,
        (limit,),
    )
    return [EmbedCandidate(str(row[0]), str(row[1])) for row in rows]


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


def _write_embedded_vectors(store: VectorStore, vectors: list[EmbeddedVector]) -> int:
    if not vectors:
        return 0

    cursor = store.conn.cursor()
    transaction_started = False
    count = 0
    try:
        cursor.execute("BEGIN IMMEDIATE")
        transaction_started = True
        for chunk_id, content, embedding in vectors:
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
            if not still_eligible:
                continue
            store._upsert_chunk_vector(cursor, chunk_id, embedding)
            count += 1
        cursor.execute("COMMIT")
        transaction_started = False
    except Exception:
        if transaction_started:
            cursor.execute("ROLLBACK")
        raise

    if count:
        from brainlayer.search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(store, "db_path", None))
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
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def _type_error_rejects_keyword(exc: TypeError, keyword: str) -> bool:
    message = str(exc)
    return keyword in message and (
        "unexpected keyword argument" in message
        or "got an unexpected keyword" in message
        or "invalid keyword argument" in message
    )


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
    pending_rows_fn: Callable[..., list[EmbedCandidate]] = _pending_chunk_rows,
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
        write_store = _open_store(vector_store_cls, db_path, readonly=False)
        try:
            embedded += write_vectors_fn(write_store, vectors)
        finally:
            write_store.close()

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


def _high_priority_queue_depth(queue_dir: Path) -> int:
    try:
        return sum(1 for path in queue_dir.glob("*.jsonl") if not path.name.startswith("enrichment-"))
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
    model_factory: Callable[[], object] = get_embedding_model,
    cycle_fn: Callable[..., CycleResult] = run_cycle,
    time_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_cycles: int | None = None,
    queue_dir: Path | None = None,
    queue_depth_fn: Callable[[Path], int] = _queue_depth,
    high_priority_queue_depth_fn: Callable[[Path], int] = _high_priority_queue_depth,
) -> None:
    model = model_factory()
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
            if queue_has_backlog:
                if queue_has_high_priority_backlog:
                    LOGGER.info("durable high-priority queue has backlog; yielding backlog/enrichment writer work")
                else:
                    LOGGER.info("durable queue has enrichment backlog; yielding backlog/enrichment writer work")
                if cycle_fn is not run_cycle or recent_limit <= 0:
                    cycles += 1
                    sleep_fn(interval)
                    continue
                cycle_backlog_batch = 0
                cycle_enrich_limit = 0
            else:
                cycle_backlog_batch = (
                    backlog_batch if backlog_batch > 0 and now - last_backlog >= backlog_interval else 0
                )
                cycle_enrich_limit = (
                    enrich_limit
                    if not enrich_disabled and enrich_limit > 0 and now - last_enrich >= enrich_interval
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
                    recent_limit=recent_limit,
                    backlog_batch=cycle_backlog_batch,
                    embed_batch_fn=embed_batch_fn,
                    enrich_limit=cycle_enrich_limit,
                    enrich_since_hours=enrich_since_hours,
                )
            else:
                store = vector_store_cls(db_path)
                try:
                    result = cycle_fn(
                        store=store,
                        embed_fn=embed_fn,
                        recent_limit=recent_limit,
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
