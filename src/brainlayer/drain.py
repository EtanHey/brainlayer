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
from .chunk_origin import CHUNK_ORIGIN_UNKNOWN, detect_chunk_origin
from .content_class import classify_content_class, normalize_content_class
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

_DEFAULT_DRAIN_BUSY_TIMEOUT_MS = 30000
_MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647
_DEFAULT_DRAIN_OPEN_MAX_RETRIES = 12
_DEFAULT_DRAIN_OPEN_RETRY_BASE_DELAY_MS = 250.0
_DEFAULT_DRAIN_OPEN_RETRY_MAX_DELAY_MS = 5000.0
_DEFAULT_MAX_EVENTS_PER_TRANSACTION = 5
_DEFAULT_BURN_MAX_EVENTS_PER_TRANSACTION = 100
_DEFAULT_POST_COMMIT_YIELD_MS = 10.0
# The post-commit WAL checkpoint is best-effort. Bound how long TRUNCATE may wait
# on readers so a pinned reader can't stall every other writer for the full drain
# busy_timeout (30s). Short window → reclaim when possible, skip cheaply otherwise.
_DEFAULT_CHECKPOINT_BUSY_TIMEOUT_MS = 1000


def _drain_busy_timeout_ms() -> int:
    try:
        timeout_ms = int(os.environ.get("BRAINLAYER_DRAIN_BUSY_TIMEOUT_MS", str(_DEFAULT_DRAIN_BUSY_TIMEOUT_MS)))
    except (TypeError, ValueError):
        return _DEFAULT_DRAIN_BUSY_TIMEOUT_MS
    if timeout_ms <= 0 or timeout_ms > _MAX_APSW_BUSY_TIMEOUT_MS:
        return _DEFAULT_DRAIN_BUSY_TIMEOUT_MS
    return timeout_ms


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


@dataclass
class BurnDrainResult:
    scanned_files: int = 0
    applied_events: int = 0
    skipped_verified_stale: int = 0
    files_deleted: int = 0
    failed_files: int = 0
    checkpoints: int = 0


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
            time.sleep(delay)
    else:
        raise RuntimeError("unreachable drain open retry state")
    conn.setbusytimeout(_drain_busy_timeout_ms())
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


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _preview_text(values: dict[str, Any]) -> str:
    summary = str(values.get("summary") or "").strip()
    content = str(values.get("content") or "").strip()
    source = summary or content
    return source.replace("\n", " ").replace("\r", " ").replace("\t", " ")[:220]


def _insert_chunk(conn: apsw.Connection, values: dict[str, Any]) -> None:
    cols = _columns(conn, "chunks")
    if "preview_text" in cols and not str(values.get("preview_text") or "").strip():
        values = {**values, "preview_text": _preview_text(values)}
    if "content" in values:
        fields = compute_dedupe_fields(str(values["content"]), values.get("created_at"))
        content_class = normalize_content_class(values.get("content_class"))
        if "content_class" not in values:
            content_class = classify_content_class(
                str(values["content"]),
                content_type=values.get("content_type"),
                tags=values.get("tags"),
                source=values.get("source"),
                source_file=values.get("source_file"),
                project=values.get("project"),
            )
        values = {
            **values,
            "seen_count": values.get("seen_count") or 1,
            "last_seen_at": values.get("last_seen_at") or values.get("created_at"),
            "content_class": content_class,
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
        reject_precompact=True,
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
            "enrich_status": "success",
            "summary": content[:200],
            "tags": json.dumps(tags) if tags else None,
            "importance": float(event["importance"]) if event.get("importance") is not None else None,
            "content_hash": _content_hash(content),
            "chunk_origin": detect_chunk_origin(content, event.get("chunk_origin")),
        },
    )
    supersedes = event.get("supersedes") or metadata.get("supersedes")
    cols = _columns(conn, "chunks")
    if supersedes and "superseded_by" in cols:
        if "status" in cols:
            conn.execute(
                "UPDATE chunks SET superseded_by = ?, status = 'superseded' WHERE id = ?",
                (stored_chunk_id, supersedes),
            )
        else:
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
    recursive_reason = recursive_mcp_output_reason(
        content,
        chunk_id=chunk_id,
        source_file=source_file,
        reject_precompact=True,
    )
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
            "content_hash": _content_hash(content),
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
    recursive_reason = recursive_mcp_output_reason(
        content,
        chunk_id=chunk_id,
        source_file=source_file,
        reject_precompact=True,
    )
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
            "content_hash": _content_hash(content),
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
    chunk_origin = str(event.get("chunk_origin") or "").strip()
    if "chunk_origin" in cols and chunk_origin:
        row = conn.execute("SELECT chunk_origin FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not row:
            return
        current_origin = str(row[0] or "").strip()
        if current_origin in {"", CHUNK_ORIGIN_UNKNOWN}:
            updates["chunk_origin"] = chunk_origin
    if "enriched_at" in cols:
        updates["enriched_at"] = datetime.now(timezone.utc).isoformat()
    if "enrich_status" in cols:
        updates["enrich_status"] = "success"
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
    return _is_sqlite_busy_error(exc)


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


def _rewrite_events_file(path: Path, events: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(
        "".join(json.dumps(event, ensure_ascii=True) + "\n" for event in events),
        encoding="utf-8",
    )
    tmp_path.replace(path)


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


def _prefetch_enrichment_state(
    conn: apsw.Connection, events: list[dict[str, Any]]
) -> dict[str, tuple[str | None, str | None, str | None]]:
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
    rows = conn.execute(
        f"SELECT id, content_hash, enrich_status, enriched_at FROM chunks WHERE id IN ({placeholders})",
        chunk_ids,
    )
    return {str(row[0]): (row[1], row[2], row[3]) for row in rows}


def _already_enriched(enrich_status: str | None, enriched_at: str | None) -> bool:
    return enrich_status == "success" or bool(enriched_at)


def _is_verified_redundant_enrichment(
    event: dict[str, Any],
    prefetched_state: dict[str, tuple[str | None, str | None, str | None]],
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
    content_hash, enrich_status, enriched_at = state
    return content_hash == expected_hash and _already_enriched(enrich_status, enriched_at)


def burn_drain_once(
    *,
    db_path: Path | None = None,
    queue_dir: Path | None = None,
    batch_size: int = 5000,
    max_events_per_transaction: int = _DEFAULT_BURN_MAX_EVENTS_PER_TRANSACTION,
    log_path: Path | None = None,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> BurnDrainResult:
    """Drain a large queue backlog with one writer and one commit per large batch.

    Queue files are unlinked only after the transaction commits. Verified redundant
    enrichment updates are skipped inside the same committed batch so old queue
    files can be safely removed without reapplying stale work.
    """
    db_path = db_path or _default_db_path()
    queue_dir = queue_dir or _default_queue_dir()
    log_path = log_path or _default_log_path()
    queue_dir.mkdir(parents=True, exist_ok=True)
    max_events_per_transaction = max(1, max_events_per_transaction)

    result = BurnDrainResult()
    lock_fd = _acquire_queue_lock(queue_dir)
    try:
        files = sorted(queue_dir.glob("*.jsonl"), key=lambda path: (path.name.startswith("enrichment-"), path.name))[
            :batch_size
        ]
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

        all_events = [event for _, events in batch for event in events]
        conn = _open_connection(db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            prefetched_state = _prefetch_enrichment_state(conn, all_events)
            store_chunk_ids: list[str] = []
            for _, events in batch:
                for event in events:
                    if _is_verified_redundant_enrichment(event, prefetched_state):
                        result.skipped_verified_stale += 1
                        continue
                    applied = _apply_event(conn, event)
                    result.applied_events += 1
                    if applied.chunk_id:
                        store_chunk_ids.append(applied.chunk_id)
            if _embedding_enabled():
                _embed_store_chunks(conn, store_chunk_ids, embed_fn)
            conn.execute("COMMIT")
            conn.setbusytimeout(_checkpoint_busy_timeout_ms())
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                result.checkpoints += 1
            except apsw.Error as exc:
                _log(log_path, f"burn drain checkpoint skipped: {exc}")
            finally:
                conn.setbusytimeout(_drain_busy_timeout_ms())
        except Exception as exc:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            result.failed_files = len(batch)
            _log(log_path, f"burn drain failed; batch preserved: {exc}")
            return result
        finally:
            conn.close()

        for path, _events in batch:
            try:
                path.unlink()
                result.files_deleted += 1
            except FileNotFoundError:
                result.files_deleted += 1
            except OSError as exc:
                _log(log_path, f"burn drain committed but could not unlink {path}: {exc}")
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
) -> int:
    db_path = db_path or _default_db_path()
    queue_dir = queue_dir or _default_queue_dir()
    log_path = log_path or _default_log_path()
    queue_dir.mkdir(parents=True, exist_ok=True)

    lock_fd = _acquire_queue_lock(queue_dir)
    try:
        files = sorted(queue_dir.glob("*.jsonl"), key=lambda path: (path.name.startswith("enrichment-"), path.name))[
            :batch_size
        ]
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
            events_to_apply = events[: _max_events_per_transaction()]
            remaining_events = events[len(events_to_apply) :]

            for attempt in range(5):
                conn: apsw.Connection | None = None
                attempt_drained = 0
                collision_ids: list[str] = []
                store_chunk_ids: list[str] = []
                try:
                    conn = _open_connection(db_path)
                    conn.execute("BEGIN IMMEDIATE")
                    ensure_dedupe_schema(conn)
                    for event in events_to_apply:
                        result = _apply_event(conn, event)
                        if result.chunk_id:
                            store_chunk_ids.append(result.chunk_id)
                        if result.collision_chunk_id:
                            collision_ids.append(result.collision_chunk_id)
                        attempt_drained += 1
                    if should_embed:
                        _embed_store_chunks(conn, store_chunk_ids, embed_fn)
                    conn.execute("COMMIT")
                    # Best-effort WAL reclaim. The truncating checkpoint (not PASSIVE)
                    # shrinks the WAL file after each drained batch — PASSIVE leaves
                    # frames in place when a reader pins a page, letting the WAL grow
                    # unbounded (observed multi-GB) and starve brain_store writes. But it
                    # blocks other writers while waiting for readers, so bound that wait
                    # to a short window (default 1s): if a reader pins the WAL we skip
                    # this round rather than stall every writer for the full drain
                    # busy_timeout. journal_size_limit and the out-of-band wal-checkpoint
                    # job reclaim later. The except keeps a busy checkpoint non-fatal.
                    conn.setbusytimeout(_checkpoint_busy_timeout_ms())
                    try:
                        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    except apsw.Error:
                        pass
                    finally:
                        conn.setbusytimeout(_drain_busy_timeout_ms())
                    drained += attempt_drained
                    collisions_dropped += len(collision_ids)
                    for chunk_id in collision_ids:
                        _log(log_path, f"WARN: queued chunk_id {chunk_id} collided with existing row, dropped")
                    if remaining_events:
                        _rewrite_events_file(path, remaining_events)
                    else:
                        _unlink_processed_file(path, log_path)
                    yield_seconds = _post_commit_yield_seconds()
                    if yield_seconds > 0:
                        time.sleep(yield_seconds)
                    break
                except Exception as exc:
                    if conn is not None:
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
                    if conn is not None:
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
    run_daemon(args.interval, args.batch_size)
    return 0
