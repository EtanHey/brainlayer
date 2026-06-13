#!/usr/bin/env python3
"""Measure BrainLayer hot store-to-search currentness.

This script intentionally separates durable row visibility, lexical FTS/trigram
visibility, embedding availability, and hybrid/RRF visibility. It can run
against the canonical DB or a caller-supplied test DB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# isort: off
from brainlayer._helpers import _escape_fts5_query
from brainlayer.drain import drain_once
from brainlayer.fallback_replay import load_scope_map, parse_fallback_file, replay_entry
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.queue_io import enqueue_store, get_queue_dir
from brainlayer.store import embed_pending_chunks, store_memory
from brainlayer.vector_store import VectorStore
# isort: on


EmbedFn = Callable[[str], list[float]]


def _now() -> float:
    return time.monotonic()


def _latency_ms(start: float) -> float:
    return round((_now() - start) * 1000.0, 3)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 3)


def summarize_latencies(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": round(max(values), 3) if values else None,
    }


def drain_throughput(*, events: int, elapsed_s: float) -> float | None:
    if elapsed_s <= 0:
        return None
    return round(events / elapsed_s, 3)


def _ensure_db_initialized(db_path: Path) -> None:
    store = VectorStore(db_path)
    store.close()


def _set_drain_embed_env(enabled: bool) -> str | None:
    previous = os.environ.get("BRAINLAYER_DRAIN_EMBED")
    os.environ["BRAINLAYER_DRAIN_EMBED"] = "1" if enabled else "0"
    return previous


def _restore_drain_embed_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("BRAINLAYER_DRAIN_EMBED", None)
    else:
        os.environ["BRAINLAYER_DRAIN_EMBED"] = previous


def queue_metrics(*, queue_dir: Path, now: float | None = None) -> dict[str, float | int | None]:
    now = time.time() if now is None else now
    files = sorted(queue_dir.glob("*.jsonl")) if queue_dir.exists() else []
    event_count = 0
    oldest_queued_at: float | None = None
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_count += 1
            queued_at = event.get("queued_at")
            if isinstance(queued_at, int | float):
                oldest_queued_at = (
                    float(queued_at) if oldest_queued_at is None else min(oldest_queued_at, float(queued_at))
                )
    oldest_age = round(now - oldest_queued_at, 3) if oldest_queued_at is not None else None
    return {
        "queue_depth_files": len(files),
        "queue_depth_events": event_count,
        "oldest_queued_age_s": oldest_age,
    }


def embedding_backlog_metrics(*, db_path: Path) -> dict[str, Any]:
    store = VectorStore(db_path, readonly=True)
    try:
        row = (
            store.conn.cursor()
            .execute(
                """
            SELECT COUNT(*), MIN(c.created_at)
            FROM chunks c
            LEFT JOIN chunk_vectors v ON c.id = v.chunk_id
            WHERE v.chunk_id IS NULL
              AND c.source IN ('manual', 'mcp')
            """
            )
            .fetchone()
        )
        return {
            "pending_manual_mcp_embeddings": int(row[0] or 0),
            "oldest_pending_manual_mcp_created_at": row[1],
        }
    finally:
        store.close()


def _poll(timeout_s: float, poll_interval_s: float, predicate: Callable[[], bool]) -> float | None:
    start = _now()
    deadline = start + timeout_s
    while _now() <= deadline:
        if predicate():
            return _latency_ms(start)
        time.sleep(poll_interval_s)
    return None


def _poll_from(
    start: float,
    timeout_s: float,
    poll_interval_s: float,
    predicate: Callable[[], bool],
) -> float | None:
    deadline = start + timeout_s
    while _now() <= deadline:
        if predicate():
            return _latency_ms(start)
        time.sleep(poll_interval_s)
    return None


def _row_exists(store: VectorStore, sql: str, params: tuple[Any, ...]) -> bool:
    return bool(store.conn.cursor().execute(sql, params).fetchone())


def _fts_visible(store: VectorStore, table_name: str, chunk_id: str, query_text: str) -> bool:
    query = _escape_fts5_query(query_text)
    if not query:
        return False
    sql = f"SELECT 1 FROM {table_name} WHERE {table_name} MATCH ? AND chunk_id = ? LIMIT 1"
    return _row_exists(store, sql, (query, chunk_id))


def _embedding_visible(store: VectorStore, chunk_id: str) -> bool:
    return _row_exists(store, "SELECT 1 FROM chunk_vectors WHERE chunk_id = ? LIMIT 1", (chunk_id,))


def _hybrid_visible(
    store: VectorStore,
    *,
    chunk_id: str,
    query_text: str,
    query_embedding: list[float],
    project: str | None,
) -> bool:
    result = store.hybrid_search(
        query_embedding=query_embedding,
        query_text=query_text,
        n_results=10,
        project_filter=project,
        include_audit=True,
        include_operational=True,
        filter_meta_noise=False,
    )
    ids = result.get("ids") or []
    return bool(ids and ids[0] and chunk_id in ids[0])


def run_store_probe(
    *,
    db_path: Path,
    content: str,
    project: str | None,
    embed_fn: EmbedFn | None,
    embed_after_store: bool,
    timeout_s: float = 30.0,
    poll_interval_s: float = 0.05,
) -> dict[str, Any]:
    store = VectorStore(db_path)
    try:
        call_start = _now()
        stored = store_memory(
            store=store,
            embed_fn=None,
            content=content,
            memory_type="note",
            project=project,
            tags=["hot-currentness-benchmark"],
            importance=5,
        )
        call_latency_ms = _latency_ms(call_start)
        chunk_id = stored["id"]

        durable_latency = _poll(
            timeout_s,
            poll_interval_s,
            lambda: _row_exists(store, "SELECT 1 FROM chunks WHERE id = ? LIMIT 1", (chunk_id,)),
        )
        fts_latency = _poll(timeout_s, poll_interval_s, lambda: _fts_visible(store, "chunks_fts", chunk_id, content))
        trigram_latency = _poll(
            timeout_s,
            poll_interval_s,
            lambda: _fts_visible(store, "chunks_fts_trigram", chunk_id, content),
        )

        embedding_latency = None
        hybrid_latency = None
        hybrid_visible_with_embedding = False
        query_embedding = embed_fn(content) if embed_fn is not None else None
        if embed_after_store and embed_fn is not None:
            embed_start = _now()
            embed_pending_chunks(store=store, embed_fn=embed_fn, batch_size=1)
            embedding_latency = _poll(
                timeout_s,
                poll_interval_s,
                lambda: _embedding_visible(store, chunk_id),
            )
            if embedding_latency is not None:
                embedding_latency = round((_now() - embed_start) * 1000.0, 3)

        if query_embedding is not None:
            hybrid_latency = _poll_from(
                call_start,
                timeout_s,
                poll_interval_s,
                lambda: (
                    _embedding_visible(store, chunk_id)
                    and _hybrid_visible(
                        store,
                        chunk_id=chunk_id,
                        query_text=content,
                        query_embedding=query_embedding,
                        project=project,
                    )
                ),
            )
            hybrid_visible_with_embedding = hybrid_latency is not None and _embedding_visible(store, chunk_id)

        return {
            "chunk_id": chunk_id,
            "brain_store_call_latency_ms": call_latency_ms,
            "durable_row_latency_ms": durable_latency,
            "fts_visibility_latency_ms": fts_latency,
            "trigram_visibility_latency_ms": trigram_latency,
            "embedding_availability_latency_ms": embedding_latency,
            "hybrid_rrf_visibility_latency_ms": hybrid_latency,
            "hybrid_rrf_visible_with_embedding": hybrid_visible_with_embedding,
        }
    finally:
        store.close()


def run_store_scenario(
    *,
    db_path: Path,
    project: str,
    count: int,
    embed_fn: EmbedFn | None,
    embed_after_store: bool,
    label: str,
    timeout_s: float,
) -> dict[str, Any]:
    probes = []
    for index in range(count):
        marker = f"hcphase1b-{label}-{uuid.uuid4().hex[:12]}-{index}"
        probes.append(
            run_store_probe(
                db_path=db_path,
                content=f"{marker} hot hybrid currentness benchmark marker",
                project=project,
                embed_fn=embed_fn,
                embed_after_store=embed_after_store,
                timeout_s=timeout_s,
            )
        )
    return _summarize_probe_set(label, probes)


def run_fake_load_scenario(
    *,
    db_path: Path,
    project: str,
    count: int,
    workers: int,
    label: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Issue concurrent stores; SQLite still serializes writes through BEGIN IMMEDIATE."""
    probes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for index in range(count):
            marker = f"hcphase1b-{label}-{uuid.uuid4().hex[:12]}-{index}"
            futures.append(
                executor.submit(
                    run_store_probe,
                    db_path=db_path,
                    content=f"{marker} fake concurrent Codex load currentness marker",
                    project=project,
                    embed_fn=None,
                    embed_after_store=False,
                    timeout_s=timeout_s,
                )
            )
        for future in as_completed(futures):
            probes.append(future.result())
    result = _summarize_probe_set(label, probes)
    result["workers"] = workers
    return result


def run_queue_drain_scenario(
    *,
    db_path: Path,
    queue_dir: Path,
    project: str,
    count: int,
    embed_fn: EmbedFn | None,
    embed_after_drain: bool = True,
    label: str,
) -> dict[str, Any]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    _ensure_db_initialized(db_path)
    benchmark_queue_dir = queue_dir / f"{label}-{uuid.uuid4().hex[:12]}"
    benchmark_queue_dir.mkdir(parents=True, exist_ok=True)
    before = queue_metrics(queue_dir=benchmark_queue_dir)
    expected_ids: list[str] = []
    for index in range(count):
        chunk_id = f"manual-{uuid.uuid4().hex[:16]}"
        expected_ids.append(chunk_id)
        enqueue_store(
            content=f"hcphase1b-{label}-{uuid.uuid4().hex[:12]} queued drain hot currentness marker {index}",
            memory_type="note",
            project=project,
            tags=["hot-currentness-benchmark", "queue-drain"],
            importance=5,
            chunk_id=chunk_id,
            benchmark_label=label,
            source="mcp",
            queue_dir=benchmark_queue_dir,
        )
    queued = queue_metrics(queue_dir=benchmark_queue_dir)
    start = _now()
    previous_drain_embed = _set_drain_embed_env(embed_after_drain)
    try:
        drained = drain_once(db_path=db_path, queue_dir=benchmark_queue_dir, batch_size=count, embed_fn=embed_fn)
    finally:
        _restore_drain_embed_env(previous_drain_embed)
    elapsed = _now() - start
    after = queue_metrics(queue_dir=benchmark_queue_dir)
    store = VectorStore(db_path)
    try:
        durable = sum(
            1
            for chunk_id in expected_ids
            if _row_exists(store, "SELECT 1 FROM chunks WHERE id = ? LIMIT 1", (chunk_id,))
        )
        embedded = sum(1 for chunk_id in expected_ids if _embedding_visible(store, chunk_id))
    finally:
        store.close()
    return {
        "label": label,
        "benchmark_queue_dir": str(benchmark_queue_dir),
        "queued_ids": expected_ids,
        "queue_before": before,
        "queue_after_enqueue": queued,
        "queue_after_drain": after,
        "drained_events": drained,
        "drain_elapsed_s": round(elapsed, 3),
        "drain_throughput_events_s": drain_throughput(events=drained, elapsed_s=elapsed),
        "durable_rows": durable,
        "embedded_rows": embedded,
    }


def run_fallback_replay_scenario(
    *,
    db_path: Path,
    project: str,
    count: int,
    label: str,
) -> dict[str, Any]:
    replayed: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="brainlayer-hc-fallback-") as tmp:
        root = Path(tmp) / "sample-repo"
        decisions = root / "docs.local" / "decisions"
        decisions.mkdir(parents=True)
        scope_map = load_scope_map()
        store = VectorStore(db_path)
        try:
            for index in range(count):
                path = decisions / f"hcphase1b-{index}.md"
                path.write_text(
                    "---\n"
                    "intended_brain_store: true\n"
                    "memory_type: note\n"
                    f"project: {project}\n"
                    "tags:\n"
                    "  - hot-currentness-benchmark\n"
                    "  - fallback-replay\n"
                    "importance: 5\n"
                    "timestamp: '2026-06-13T00:00:00+00:00'\n"
                    "chunk_id:\n"
                    "---\n"
                    f"hcphase1b-{label}-{uuid.uuid4().hex[:12]} fallback replay marker {index}\n",
                    encoding="utf-8",
                )
                entry = parse_fallback_file(path, scope_map=scope_map)
                start = _now()
                result = replay_entry(
                    entry,
                    store_func=lambda **kwargs: store_memory(store=store, embed_fn=None, **kwargs),
                    replayed_by="hot-currentness-benchmark",
                )
                replayed.append(
                    {
                        "path": str(path),
                        "chunk_id": result.chunk_id,
                        "error": result.error,
                        "replay_latency_ms": _latency_ms(start),
                    }
                )
        finally:
            store.close()
    return {"label": label, "replayed": replayed}


def _summarize_probe_set(label: str, probes: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "brain_store_call_latency_ms",
        "durable_row_latency_ms",
        "fts_visibility_latency_ms",
        "trigram_visibility_latency_ms",
        "embedding_availability_latency_ms",
        "hybrid_rrf_visibility_latency_ms",
    ]
    return {
        "label": label,
        "probes": probes,
        "summary": {
            field: summarize_latencies([probe[field] for probe in probes if probe.get(field) is not None])
            for field in fields
        },
    }


def _load_default_embed_fn() -> EmbedFn:
    from brainlayer.embeddings import get_embedding_model

    model = get_embedding_model()
    return model.embed_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure BrainLayer hot hybrid currentness.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--queue-dir", type=Path, default=get_queue_dir())
    parser.add_argument("--project", default="brainlayer")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--scenario",
        choices=["store-no-embed", "store-with-embed", "fake-load", "queue-drain", "fallback-replay", "all"],
        default="store-with-embed",
    )
    parser.add_argument("--no-real-embed", action="store_true", help="Skip loading the real embedding model.")
    args = parser.parse_args()

    embed_fn = None if args.no_real_embed else _load_default_embed_fn()
    _ensure_db_initialized(args.db)
    output: dict[str, Any] = {
        "db_path": str(args.db),
        "queue_dir": str(args.queue_dir),
        "project": args.project,
        "count": args.count,
        "started_at_epoch": time.time(),
        "embedding_backlog_before": embedding_backlog_metrics(db_path=args.db),
        "queue_before": queue_metrics(queue_dir=args.queue_dir),
        "scenarios": [],
    }

    if args.scenario in {"store-no-embed", "all"}:
        output["scenarios"].append(
            run_store_scenario(
                db_path=args.db,
                project=args.project,
                count=args.count,
                embed_fn=embed_fn,
                embed_after_store=False,
                label="store-no-embed",
                timeout_s=args.timeout_s,
            )
        )
    if args.scenario in {"store-with-embed", "all"}:
        output["scenarios"].append(
            run_store_scenario(
                db_path=args.db,
                project=args.project,
                count=args.count,
                embed_fn=embed_fn,
                embed_after_store=embed_fn is not None,
                label="store-with-embed",
                timeout_s=args.timeout_s,
            )
        )
    if args.scenario in {"queue-drain", "all"}:
        output["scenarios"].append(
            run_queue_drain_scenario(
                db_path=args.db,
                queue_dir=args.queue_dir,
                project=args.project,
                count=args.count,
                embed_fn=embed_fn,
                embed_after_drain=embed_fn is not None,
                label="queue-drain",
            )
        )
    if args.scenario in {"fallback-replay", "all"}:
        output["scenarios"].append(
            run_fallback_replay_scenario(
                db_path=args.db,
                project=args.project,
                count=args.count,
                label="fallback-replay",
            )
        )

    if args.scenario in {"fake-load", "all"}:
        output["scenarios"].append(
            run_fake_load_scenario(
                db_path=args.db,
                project=args.project,
                count=args.count,
                workers=args.workers,
                label="fake-load",
                timeout_s=args.timeout_s,
            )
        )

    output["embedding_backlog_after"] = embedding_backlog_metrics(db_path=args.db)
    output["queue_after"] = queue_metrics(queue_dir=args.queue_dir)
    output["finished_at_epoch"] = time.time()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
