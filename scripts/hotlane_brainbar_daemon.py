#!/usr/bin/env python3
"""Hot-embed recent BrainBar MCP writes and interleave realtime enrichment.

BrainBar owns the live MCP socket and stores ``brain_store`` rows directly in
SQLite. This adapter owns the single BrainLayer writer slot and must therefore
perform all write-side background work that needs steady progress: hot embedding,
optional embedding backlog, and enrichment backlog.
"""

from __future__ import annotations

import argparse
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
DEFAULT_HOTLANE_ENRICH_LIMIT = 25


class CycleResult(NamedTuple):
    embedded: int = 0
    enrich_attempted: int = 0
    enriched: int = 0
    enrich_skipped: int = 0
    enrich_failed: int = 0
    enrich_daily_cap_reached: bool = False


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
    enrich_fn: Callable[..., object] = enrich_realtime,
) -> CycleResult:
    embedded = 0
    for chunk_id in candidate_chunk_ids_fn(store, limit=recent_limit):
        if hot_embed_fn(store=store, embed_fn=embed_fn, chunk_id=chunk_id):
            embedded += 1
            break

    if backlog_batch > 0:
        embedded += pending_embed_fn(store=store, embed_fn=embed_fn, batch_size=backlog_batch)

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
) -> None:
    store = vector_store_cls(db_path)
    try:
        model = model_factory()
        embed_fn = model.embed_query
        last_backlog = time_fn()
        last_enrich = 0.0
        enrich_disabled = False
        cycles = 0
        LOGGER.info("hotlane adapter started db=%s", db_path)
        while not STOP:
            if max_cycles is not None and cycles >= max_cycles:
                break
            try:
                now = time_fn()
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
                result = cycle_fn(
                    store=store,
                    embed_fn=embed_fn,
                    recent_limit=recent_limit,
                    backlog_batch=cycle_backlog_batch,
                    enrich_limit=cycle_enrich_limit,
                    enrich_since_hours=enrich_since_hours,
                )
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
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=get_db_path())
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--recent-limit", type=int, default=5)
    parser.add_argument("--backlog-interval", type=float, default=10.0)
    parser.add_argument("--backlog-batch", type=int, default=0)
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
        backlog_batch=max(args.backlog_batch, 0),
        enrich_interval=max(args.enrich_interval, 1.0),
        enrich_limit=max(args.enrich_limit, 0),
        enrich_since_hours=max(args.enrich_since_hours, 0),
    )


if __name__ == "__main__":
    main()
