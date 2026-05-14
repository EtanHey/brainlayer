"""Single-writer drain loop for BrainLayer's durable JSONL queue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import get_db_path

logger = logging.getLogger(__name__)


def _default_db_path() -> Path:
    return get_db_path()


def _default_queue_dir() -> Path:
    return Path(os.environ.get("BRAINLAYER_QUEUE_DIR", Path.home() / ".brainlayer/queue"))


def _default_log_path() -> Path:
    return Path.home() / ".brainlayer" / "logs" / "drain.log"


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _insert_chunk(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
    cols = _columns(conn, "chunks")
    row = {key: value for key, value in values.items() if key in cols}
    if "id" not in row and "chunk_id" in cols:
        row["chunk_id"] = values["id"]
    if "chunk_id" not in row and "id" in cols:
        row["id"] = values["id"]
    names = list(row)
    placeholders = ", ".join("?" for _ in names)
    sql = f"INSERT OR IGNORE INTO chunks ({', '.join(names)}) VALUES ({placeholders})"
    conn.execute(sql, [row[name] for name in names])


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    if "kind" in event:
        return event
    if "content" in event and "memory_type" in event:
        return {"kind": "store_memory", **event}
    if "session_id" in event and "content" in event:
        return {"kind": "hook_chunk", **event}
    return {"kind": "unknown", **event}


def _apply_store(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
    raw_content = event.get("content")
    if raw_content is None:
        logger.warning("Skipping malformed store event without content")
        return
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed store event with empty content")
        return
    now = datetime.now(timezone.utc).isoformat()
    metadata = {"memory_type": event.get("memory_type", "note")}
    metadata.update(event.get("metadata") or {})
    chunk_id = event.get("chunk_id") or f"manual-{uuid.uuid4().hex[:16]}"
    tags = event.get("tags")
    _insert_chunk(
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
        },
    )
    supersedes = event.get("supersedes") or metadata.get("supersedes")
    cols = _columns(conn, "chunks")
    if supersedes and "superseded_by" in cols:
        conn.execute("UPDATE chunks SET superseded_by = ? WHERE id = ?", (chunk_id, supersedes))
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
                (entity_id, chunk_id, 1.0, f"Stored via brain_store: {event.get('memory_type', 'note')}"),
            )
        else:
            logger.warning("Skipping entity link for unknown entity_id=%s chunk_id=%s", entity_id, chunk_id)


def _apply_watcher(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
    chunk_id = event.get("chunk_id")
    raw_content = event.get("content")
    if not chunk_id or raw_content is None:
        logger.warning("Skipping malformed watcher event without chunk_id/content")
        return
    content = str(raw_content).strip()
    if not content:
        logger.warning("Skipping malformed watcher event with empty content")
        return
    tags = event.get("tags")
    _insert_chunk(
        conn,
        {
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps(event.get("metadata") or {}),
            "source_file": event.get("source_file") or "realtime-watcher",
            "project": event.get("project"),
            "content_type": event.get("content_type") or "assistant_text",
            "value_type": event.get("value_type") or "HIGH",
            "char_count": len(content),
            "source": "realtime_watcher",
            "created_at": event.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "conversation_id": event.get("conversation_id"),
            "sender": event.get("sender"),
            "tags": json.dumps(tags) if tags else None,
        },
    )


def _apply_hook(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
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
    ts_raw = event.get("timestamp")
    try:
        timestamp = float(ts_raw) if ts_raw is not None else time.time()
    except (TypeError, ValueError):
        logger.warning("Invalid hook timestamp %r; using current time", ts_raw)
        timestamp = time.time()
    _insert_chunk(
        conn,
        {
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps({"session_id": session_id, "content_hash": content_hash}),
            "source_file": "realtime-hook",
            "project": event.get("project"),
            "content_type": "assistant_text",
            "value_type": "HIGH",
            "char_count": len(content),
            "source": "realtime",
            "created_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
            "conversation_id": session_id,
            "importance": 5,
        },
    )


def _apply_enrichment(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
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


def _apply_event(conn: sqlite3.Connection, event: dict[str, Any]) -> None:
    event = _event_payload(event)
    kind = event.get("kind")
    if kind == "store_memory":
        _apply_store(conn, event)
    elif kind == "watcher_chunk":
        _apply_watcher(conn, event)
    elif kind == "hook_chunk":
        _apply_hook(conn, event)
    elif kind == "enrichment_update":
        _apply_enrichment(conn, event)


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


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual')")}


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()


def drain_once(
    *,
    db_path: Path | None = None,
    queue_dir: Path | None = None,
    batch_size: int = 250,
    log_path: Path | None = None,
) -> int:
    db_path = db_path or _default_db_path()
    queue_dir = queue_dir or _default_queue_dir()
    log_path = log_path or _default_log_path()
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_path = queue_dir / ".drain.lock"

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        files = sorted(queue_dir.glob("*.jsonl"))[:batch_size]
        if not files:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            return 0

        drained = 0
        for attempt in range(5):
            conn = sqlite3.connect(db_path, timeout=0.2)
            attempt_drained = 0
            try:
                conn.execute("PRAGMA busy_timeout=200")
                conn.execute("BEGIN IMMEDIATE")
                for path in files:
                    for event in _read_events(path):
                        _apply_event(conn, event)
                        attempt_drained += 1
                conn.commit()
                drained = attempt_drained
                for path in files:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        _log(log_path, f"drain committed but could not unlink {path}: {exc}")
                break
            except Exception as exc:
                conn.rollback()
                if _is_busy_error(exc) and attempt < 4:
                    delay = 0.05 * (2**attempt)
                    _log(log_path, f"drain busy; retrying in {delay:.2f}s")
                    time.sleep(delay)
                    continue
                _log(log_path, f"drain failed: {exc}")
                return 0
            finally:
                conn.close()

    if drained:
        _log(log_path, f"drained={drained}")
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
