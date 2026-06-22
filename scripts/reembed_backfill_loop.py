#!/usr/bin/env python3
"""Continuous, polite embedding backfill for active chunks missing vectors."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import apsw
import sqlite_vec

from brainlayer._helpers import serialize_f32
from brainlayer.embeddings import get_embedding_model
from brainlayer.paths import get_db_path
from brainlayer.reembed_backfill import _pending_where_sql

DEFAULT_COLLAB = Path("/Users/etanheyman/Gits/orchestrator/collab/2026-06-21-brainlayer-health.md")


class _ConnStore:
    def __init__(self, conn: apsw.Connection) -> None:
        self.conn = conn


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_enrichment_queue_file(path: Path) -> bool:
    return path.name.startswith("enrichment-") or path.name.startswith("queue-enrichment")


def _has_high_priority_queue_file(queue_dir: Path) -> bool:
    try:
        return any(not _is_enrichment_queue_file(path) for path in queue_dir.glob("*.jsonl"))
    except OSError:
        return False


def _append_backfill_receipt(collab: Path | None, remaining: int, *, extra: str = "") -> None:
    if collab is None:
        return
    suffix = f" {extra}" if extra else ""
    try:
        with collab.open("a", encoding="utf-8") as handle:
            handle.write(f"\n### implementation-codex ({_utc_ts()}) backfill UNEMBEDDED={remaining}{suffix}\n")
    except OSError as exc:
        logging.warning("failed to append collab backfill receipt: %s", exc)


def _is_busy_error(exc: BaseException) -> bool:
    if isinstance(exc, (apsw.BusyError, apsw.LockedError)):
        return True
    message = str(exc).casefold()
    return "database is locked" in message or "database is busy" in message or "database table is locked" in message


def _open_backfill_conn(db_path: Path, *, attempts: int = 60) -> apsw.Connection:
    last_busy: BaseException | None = None
    for attempt in range(1, attempts + 1):
        conn: apsw.Connection | None = None
        try:
            conn = apsw.Connection(str(db_path))
            conn.setbusytimeout(1000)
            conn.enableloadextension(True)
            conn.loadextension(sqlite_vec.loadable_path())
            conn.enableloadextension(False)
            conn.cursor().execute("PRAGMA journal_mode=WAL")
            return conn
        except Exception as exc:
            if not _is_busy_error(exc):
                raise
            last_busy = exc
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
            sleep_seconds = min(5.0, 0.25 * attempt)
            logging.warning(
                "backfill DB open busy attempt=%d/%d sleep=%.2fs: %s",
                attempt,
                attempts,
                sleep_seconds,
                exc,
            )
            time.sleep(sleep_seconds)
    assert last_busy is not None
    raise last_busy


def _binary_index_available(conn: apsw.Connection) -> bool:
    return bool(
        conn.cursor()
        .execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunk_vectors_binary'")
        .fetchone()
    )


def _count_unvectored_min_chars(store: _ConnStore, *, min_chars: int) -> int:
    where_sql = _pending_where_sql(store, include_inactive=False)
    return int(
        store.conn.cursor()
        .execute(
            f"""
            SELECT COUNT(*)
            FROM chunks c
            LEFT JOIN chunk_vectors_rowids r ON c.id = r.id
            WHERE {where_sql}
              AND COALESCE(c.char_count, LENGTH(c.content)) >= ?
            """,
            (min_chars,),
        )
        .fetchone()[0]
    )


def _fetch_unvectored_min_chars(store: _ConnStore, *, batch_size: int, min_chars: int):
    from brainlayer.reembed_backfill import PendingChunk

    where_sql = _pending_where_sql(store, include_inactive=False)
    rows = store.conn.cursor().execute(
        f"""
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN chunk_vectors_rowids r ON c.id = r.id
        WHERE {where_sql}
          AND COALESCE(c.char_count, LENGTH(c.content)) >= ?
        ORDER BY c.created_at ASC, c.id ASC
        LIMIT ?
        """,
        (min_chars, batch_size),
    )
    return [PendingChunk(chunk_id=str(chunk_id), content=str(content)) for chunk_id, content in rows]


def _write_chunk_embeddings_direct(
    conn: apsw.Connection,
    chunks: list[Any],
    embeddings: list[list[float]],
    *,
    binary_index: bool,
) -> int:
    cursor = conn.cursor()
    written = 0
    started = False
    try:
        cursor.execute("BEGIN IMMEDIATE")
        started = True
        for chunk, embedding in zip(chunks, embeddings):
            embedding_bytes = serialize_f32([float(value) for value in embedding])
            cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk.chunk_id,))
            cursor.execute(
                "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
                (chunk.chunk_id, embedding_bytes),
            )
            if binary_index:
                cursor.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk.chunk_id,))
                cursor.execute(
                    "INSERT INTO chunk_vectors_binary (chunk_id, embedding) VALUES (?, vec_quantize_binary(?))",
                    (chunk.chunk_id, embedding_bytes),
                )
            written += 1
        cursor.execute("COMMIT")
        started = False
        return written
    except Exception:
        if started:
            cursor.execute("ROLLBACK")
        raise


def _model_device(model) -> str:
    loaded = model._load_model()
    device = getattr(loaded, "device", None)
    return str(device) if device is not None else "unknown"


def run_loop(
    *,
    db_path: Path,
    queue_dir: Path,
    fetch_page_size: int,
    embed_batch_size: int,
    write_batch_size: int,
    min_chars: int,
    sleep_seconds: float,
    yield_seconds: float,
    busy_sleep_seconds: float,
    report_seconds: float,
    collab: Path | None,
    max_pages: int | None = None,
) -> int:
    conn = _open_backfill_conn(db_path)
    store = _ConnStore(conn)
    binary_index = _binary_index_available(conn)
    processed = 0
    failed = 0
    start = time.monotonic()
    last_report = 0.0
    pages = 0

    try:
        remaining = _count_unvectored_min_chars(store, min_chars=min_chars)
        logging.info(
            "embedding backfill starting remaining=%d fetch_page_size=%d embed_batch_size=%d write_batch_size=%d",
            remaining,
            fetch_page_size,
            embed_batch_size,
            write_batch_size,
        )
        if remaining == 0:
            _append_backfill_receipt(collab, remaining, extra="already-complete")
            return 0

        model = get_embedding_model()
        device = _model_device(model)
        logging.info("embedding backfill device=%s", device)
        _append_backfill_receipt(
            collab,
            remaining,
            extra=f"started device={device} fetch_page={fetch_page_size} embed_batch={embed_batch_size} write_batch={write_batch_size}",
        )
        while remaining > 0:
            chunks = _fetch_unvectored_min_chars(store, batch_size=fetch_page_size, min_chars=min_chars)
            if not chunks:
                remaining = _count_unvectored_min_chars(store, min_chars=min_chars)
                break
            pages += 1

            embeddings: list[list[float]] = []
            embed_start = time.monotonic()
            for offset in range(0, len(chunks), embed_batch_size):
                embed_chunks = chunks[offset : offset + embed_batch_size]
                embeddings.extend(
                    model.embed_texts(
                        [chunk.content for chunk in embed_chunks],
                        batch_size=embed_batch_size,
                    )
                )
            embed_seconds = time.monotonic() - embed_start
            embed_rate = len(embeddings) / embed_seconds if embed_seconds > 0 else 0.0

            page_written = 0
            page_failed = 0
            page_write_seconds = 0.0
            for offset in range(0, len(chunks), write_batch_size):
                if _has_high_priority_queue_file(queue_dir):
                    logging.info("interactive queue appeared before vector write; yielding %.2fs", yield_seconds)
                    time.sleep(yield_seconds)
                write_chunks = chunks[offset : offset + write_batch_size]
                write_embeddings = embeddings[offset : offset + write_batch_size]
                while True:
                    write_start = time.monotonic()
                    try:
                        written = _write_chunk_embeddings_direct(
                            conn,
                            write_chunks,
                            write_embeddings,
                            binary_index=binary_index,
                        )
                        break
                    except Exception as exc:
                        if not _is_busy_error(exc):
                            raise
                        logging.warning("vector write batch busy; yielding before retry: %s", exc)
                        time.sleep(busy_sleep_seconds)
                page_write_seconds += time.monotonic() - write_start
                page_written += written
                page_failed += len(write_chunks) - written

            processed += page_written
            failed += page_failed
            remaining = _count_unvectored_min_chars(store, min_chars=min_chars)
            elapsed = time.monotonic() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            write_rate = page_written / page_write_seconds if page_write_seconds > 0 else 0.0
            logging.info(
                "embedding backfill page size=%d embedded=%d written=%d remaining=%d device=%s "
                "embed_sec=%.2f embed_rate=%.2f/s write_sec=%.2f write_rate=%.2f/s "
                "processed=%d failed=%d end_to_end_rate=%.2f/s",
                len(chunks),
                len(embeddings),
                page_written,
                remaining,
                device,
                embed_seconds,
                embed_rate,
                page_write_seconds,
                write_rate,
                processed,
                failed,
                rate,
            )

            now = time.monotonic()
            if now - last_report >= report_seconds or remaining == 0:
                _append_backfill_receipt(
                    collab,
                    remaining,
                    extra=(
                        f"device={device} embed_rate={embed_rate:.2f}/s "
                        f"write_rate={write_rate:.2f}/s end_to_end={rate:.2f}/s processed={processed}"
                    ),
                )
                last_report = now

            time.sleep(sleep_seconds)
            if max_pages is not None and pages >= max_pages:
                _append_backfill_receipt(
                    collab,
                    remaining,
                    extra=f"stopped-after-max-pages pages={pages} processed={processed}",
                )
                return 0

        _append_backfill_receipt(collab, remaining, extra=f"complete processed={processed} failed={failed}")
        logging.info("embedding backfill complete remaining=%d processed=%d failed=%d", remaining, processed, failed)
        return 0 if remaining == 0 else 1
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuously backfill missing chunk embeddings.")
    parser.add_argument("--db", type=Path, default=get_db_path())
    parser.add_argument("--queue-dir", type=Path, default=Path.home() / ".brainlayer" / "queue")
    parser.add_argument("--fetch-page-size", type=int, default=512)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--write-batch-size", type=int, default=32)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=0.1)
    parser.add_argument("--yield-seconds", type=float, default=0.75)
    parser.add_argument("--busy-sleep-seconds", type=float, default=2.0)
    parser.add_argument("--report-seconds", type=float, default=900.0)
    parser.add_argument("--collab", type=Path, default=DEFAULT_COLLAB)
    parser.add_argument("--no-collab", action="store_true")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return run_loop(
        db_path=args.db.expanduser(),
        queue_dir=args.queue_dir.expanduser(),
        fetch_page_size=max(1, args.fetch_page_size),
        embed_batch_size=max(1, args.embed_batch_size),
        write_batch_size=max(1, min(args.write_batch_size, 32)),
        min_chars=max(1, args.min_chars),
        sleep_seconds=max(0.0, args.sleep_seconds),
        yield_seconds=max(0.0, args.yield_seconds),
        busy_sleep_seconds=max(0.1, args.busy_sleep_seconds),
        report_seconds=max(60.0, args.report_seconds),
        collab=None if args.no_collab else args.collab.expanduser(),
        max_pages=args.max_pages if args.max_pages is None else max(1, args.max_pages),
    )


if __name__ == "__main__":
    raise SystemExit(main())
