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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import apsw
import sqlite_vec

from ._helpers import serialize_f32
from .chunk_origin import detect_chunk_origin
from .dedupe import (
    compute_dedupe_fields,
    ensure_dedupe_schema,
    find_duplicate,
    merge_duplicate_chunk,
    merge_existing_chunk_content,
    merge_existing_chunk_seen,
)
from .ingest_guard import recursive_mcp_output_reason
from .paths import get_db_path

logger = logging.getLogger(__name__)


@dataclass
class ApplyResult:
    chunk_id: str | None = None
    collision_chunk_id: str | None = None


def _default_db_path() -> Path:
    return get_db_path()


def _default_queue_dir() -> Path:
    from .queue_io import get_queue_dir

    return get_queue_dir()


def _default_log_path() -> Path:
    return Path.home() / ".brainlayer" / "logs" / "drain.log"


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def _open_connection(db_path: Path) -> apsw.Connection:
    conn = apsw.Connection(str(db_path))
    conn.setbusytimeout(200)
    conn.enableloadextension(True)
    conn.loadextension(sqlite_vec.loadable_path())
    conn.enableloadextension(False)
    return conn


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


def _insert_chunk(conn: apsw.Connection, values: dict[str, Any]) -> None:
    cols = _columns(conn, "chunks")
    if "content" in values:
        fields = compute_dedupe_fields(str(values["content"]), values.get("created_at"))
        values = {
            **values,
            "seen_count": values.get("seen_count") or 1,
            "last_seen_at": values.get("last_seen_at") or values.get("created_at"),
            "dedupe_hash": fields.dedupe_hash,
            "simhash": fields.simhash,
            "simhash_band_0": fields.bands[0],
            "simhash_band_1": fields.bands[1],
            "simhash_band_2": fields.bands[2],
            "simhash_band_3": fields.bands[3],
        }
    row = {key: value for key, value in values.items() if key in cols}
    if "id" not in row and "chunk_id" in cols:
        row["chunk_id"] = values["id"]
    if "chunk_id" not in row and "id" in cols:
        row["id"] = values["id"]
    names = list(row)
    placeholders = ", ".join("?" for _ in names)
    sql = f"INSERT OR IGNORE INTO chunks ({', '.join(names)}) VALUES ({placeholders})"
    conn.execute(sql, [row[name] for name in names])


def _insert_or_merge_chunk(conn: apsw.Connection, values: dict[str, Any]) -> str:
    ensure_dedupe_schema(conn)
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
        merge_duplicate_chunk(
            conn,
            canonical_id=duplicate.canonical_chunk_id,
            duplicate_id=chunk_id,
            incoming=values,
            mechanism=duplicate.mechanism,
            hamming_distance_value=duplicate.hamming_distance,
        )
        return duplicate.canonical_chunk_id
    if merge_existing_chunk_seen(conn, chunk_id=chunk_id, incoming=values):
        return chunk_id
    existing = conn.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if existing:
        merge_existing_chunk_content(conn, chunk_id=chunk_id, incoming=values)
        return chunk_id
    _insert_chunk(conn, values)
    return chunk_id


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    if "kind" in event:
        return event
    if "content" in event and "memory_type" in event:
        return {"kind": "store_memory", **event}
    if "session_id" in event and "content" in event:
        return {"kind": "hook_chunk", **event}
    return {"kind": "unknown", **event}


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
    )
    if recursive_reason:
        logger.warning("Skipping recursive MCP store event: %s", recursive_reason)
        return ApplyResult()
    now = datetime.now(timezone.utc).isoformat()
    metadata = {"memory_type": event.get("memory_type", "note")}
    raw_metadata = event.get("metadata")
    if isinstance(raw_metadata, dict):
        metadata.update(raw_metadata)
    elif raw_metadata:
        logger.warning("Skipping non-object store metadata for chunk_id=%s", event.get("chunk_id"))
    tags = event.get("tags")
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
                    "created_at": now,
                    "last_seen_at": now,
                },
            )
            return ApplyResult(chunk_id=chunk_id)
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
            "value_type": "HIGH",
            "char_count": len(content),
            "source": event.get("source") or "manual",
            "created_at": now,
            "enriched_at": now,
            "summary": content[:200],
            "tags": json.dumps(tags) if tags else None,
            "importance": float(event["importance"]) if event.get("importance") is not None else None,
            "chunk_origin": detect_chunk_origin(content, event.get("chunk_origin")),
        },
    )
    supersedes = event.get("supersedes") or metadata.get("supersedes")
    cols = _columns(conn, "chunks")
    if supersedes and "superseded_by" in cols:
        conn.execute("UPDATE chunks SET superseded_by = ? WHERE id = ?", (stored_chunk_id, supersedes))
        for vector_table in ("chunk_vectors", "chunk_vectors_binary"):
            if conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (vector_table,)
            ).fetchone():
                conn.execute(f"DELETE FROM {vector_table} WHERE chunk_id = ?", (supersedes,))
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
    return ApplyResult(chunk_id=stored_chunk_id)


def _apply_watcher(conn: apsw.Connection, event: dict[str, Any]) -> None:
    chunk_id = event.get("chunk_id")
    raw_content = event.get("content")
    if not chunk_id or raw_content is None:
        logger.warning("Skipping malformed watcher event without chunk_id/content")
        return
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed watcher event with empty content")
        return
    source_file = event.get("source_file") or "realtime-watcher"
    recursive_reason = recursive_mcp_output_reason(content, chunk_id=chunk_id, source_file=source_file)
    if recursive_reason:
        logger.warning("Skipping recursive MCP watcher event: %s", recursive_reason)
        return
    tags = event.get("tags")
    _insert_or_merge_chunk(
        conn,
        {
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps(event.get("metadata") or {}),
            "source_file": source_file,
            "project": event.get("project"),
            "content_type": event.get("content_type") or "assistant_text",
            "value_type": event.get("value_type") or "HIGH",
            "char_count": len(content),
            "source": "realtime_watcher",
            "created_at": event.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "conversation_id": event.get("conversation_id"),
            "sender": event.get("sender"),
            "tags": json.dumps(tags) if tags else None,
            "chunk_origin": detect_chunk_origin(content, event.get("chunk_origin")),
        },
    )


def _apply_hook(conn: apsw.Connection, event: dict[str, Any]) -> None:
    raw_content = event.get("content")
    if raw_content is None:
        logger.warning("Skipping malformed hook event without content")
        return
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed hook event with empty content")
        return
    content_hash = event.get("content_hash") or hashlib.sha256(content.encode()).hexdigest()[:16]
    session_id = event.get("session_id") or "unknown"
    chunk_id = event.get("chunk_id") or f"rt-{str(session_id)[:8]}-{content_hash}"
    source_file = event.get("source_file") or "realtime-hook"
    recursive_reason = recursive_mcp_output_reason(content, chunk_id=chunk_id, source_file=source_file)
    if recursive_reason:
        logger.warning("Skipping recursive MCP hook event: %s", recursive_reason)
        return
    ts_raw = event.get("timestamp")
    try:
        timestamp = float(ts_raw) if ts_raw is not None else time.time()
    except (TypeError, ValueError):
        logger.warning("Invalid hook timestamp %r; using current time", ts_raw)
        timestamp = time.time()
    _insert_or_merge_chunk(
        conn,
        {
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps({"session_id": session_id, "content_hash": content_hash}),
            "source_file": source_file,
            "project": event.get("project"),
            "content_type": "assistant_text",
            "value_type": "HIGH",
            "char_count": len(content),
            "source": "realtime",
            "created_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "conversation_id": session_id,
            "importance": 5,
            "chunk_origin": detect_chunk_origin(content, event.get("chunk_origin")),
        },
    )


def _apply_enrichment(conn: apsw.Connection, event: dict[str, Any]) -> None:
    chunk_id = event.get("chunk_id")
    if not chunk_id:
        logger.warning("Skipping malformed enrichment event without chunk_id")
        return
    enrichment = event.get("enrichment") or {}
    cols = _columns(conn, "chunks")
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
    if "enriched_at" in cols:
        updates["enriched_at"] = datetime.now(timezone.utc).isoformat()
    if not updates:
        return
    assignments = ", ".join(f"{col} = ?" for col in updates)
    conn.execute(f"UPDATE chunks SET {assignments} WHERE id = ?", [*updates.values(), chunk_id])


def _apply_event(conn: apsw.Connection, event: dict[str, Any]) -> ApplyResult:
    event = _event_payload(event)
    kind = event.get("kind")
    if kind == "store_memory":
        return _apply_store(conn, event)
    elif kind == "watcher_chunk":
        _apply_watcher(conn, event)
    elif kind == "hook_chunk":
        _apply_hook(conn, event)
    elif kind == "enrichment_update":
        _apply_enrichment(conn, event)
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
    return isinstance(exc, apsw.BusyError)


def _default_embed_fn() -> Callable[[str], list[float]]:
    from .embeddings import get_embedding_model

    return get_embedding_model().embed_query


def _embedding_enabled() -> bool:
    return os.environ.get("BRAINLAYER_DRAIN_EMBED", "1").lower() not in {"0", "false", "no"}


def _embed_store_chunks(
    conn: apsw.Connection,
    chunk_ids: list[str],
    embed_fn: Callable[[str], list[float]] | None,
) -> None:
    if not chunk_ids or "chunk_vectors" not in _table_names(conn):
        return
    resolved_embed_fn = embed_fn or _default_embed_fn()
    unique_chunk_ids = list(dict.fromkeys(chunk_ids))
    for chunk_id in unique_chunk_ids:
        try:
            row = conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            if not row:
                continue
            embedding_bytes = serialize_f32(resolved_embed_fn(row[0]))
            conn.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
            conn.execute("INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)", (chunk_id, embedding_bytes))
            if "chunk_vectors_binary" in _table_names(conn):
                conn.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))
                conn.execute(
                    "INSERT INTO chunk_vectors_binary (chunk_id, embedding) VALUES (?, vec_quantize_binary(?))",
                    (chunk_id, embedding_bytes),
                )
        except Exception as exc:
            logger.warning("Failed to embed drained chunk %s: %s", chunk_id, exc)


def _quarantine_file(path: Path, log_path: Path, reason: BaseException) -> None:
    target = path.with_name(f"{path.name}.bad")
    if target.exists():
        target = path.with_name(f"{path.name}.bad-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}")
    try:
        path.replace(target)
        _log(log_path, f"skipped poison queue file {path.name}: {reason}; moved_to={target.name}")
    except OSError as exc:
        _log(log_path, f"skipped poison queue file {path.name}: {reason}; quarantine_failed={exc}")


def _unlink_processed_file(path: Path, log_path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        _log(log_path, f"drain committed but could not unlink {path}: {exc}")


def drain_once(
    *,
    db_path: Path | None = None,
    queue_dir: Path | None = None,
    batch_size: int = 250,
    log_path: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> int:
    db_path = db_path or _default_db_path()
    queue_dir = queue_dir or _default_queue_dir()
    log_path = log_path or _default_log_path()
    queue_dir.mkdir(parents=True, exist_ok=True)

    lock_fd = _acquire_queue_lock(queue_dir)
    try:
        files = sorted(queue_dir.glob("*.jsonl"))[:batch_size]
        if not files:
            return 0
        _log(log_path, f"queue_depth={len(files)}")

        drained = 0
        collisions_dropped = 0
        should_embed = _embedding_enabled()
        for path in files:
            try:
                events = _read_events(path)
            except (UnicodeDecodeError, OSError) as exc:
                _quarantine_file(path, log_path, exc)
                continue
            if not events:
                _unlink_processed_file(path, log_path)
                continue

            for attempt in range(5):
                conn = _open_connection(db_path)
                attempt_drained = 0
                collision_ids: list[str] = []
                store_chunk_ids: list[str] = []
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    ensure_dedupe_schema(conn)
                    for event in events:
                        result = _apply_event(conn, event)
                        if result.chunk_id:
                            store_chunk_ids.append(result.chunk_id)
                        if result.collision_chunk_id:
                            collision_ids.append(result.collision_chunk_id)
                        attempt_drained += 1
                    if should_embed:
                        _embed_store_chunks(conn, store_chunk_ids, embed_fn)
                    conn.execute("COMMIT")
                    drained += attempt_drained
                    collisions_dropped += len(collision_ids)
                    for chunk_id in collision_ids:
                        _log(log_path, f"WARN: queued chunk_id {chunk_id} collided with existing row, dropped")
                    _unlink_processed_file(path, log_path)
                    break
                except Exception as exc:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    if _is_busy_error(exc) and attempt < 4:
                        delay = 0.05 * (2**attempt)
                        _log(log_path, f"drain busy; retrying in {delay:.2f}s")
                        time.sleep(delay)
                        continue
                    _log(log_path, f"drain failed for {path.name}: {exc}")
                    break
                finally:
                    conn.close()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    if drained:
        _log(log_path, f"drained={drained} collisions_dropped={collisions_dropped}")
    return drained


def run_daemon(interval: float, batch_size: int) -> None:
    while True:
        drain_once(batch_size=batch_size)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()
    if args.once:
        print(drain_once(batch_size=args.batch_size))
        return 0
    run_daemon(args.interval, args.batch_size)
    return 0
