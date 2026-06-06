"""Durable JSONL queue for BrainLayer write arbitration."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _safe_source_name(source: str) -> str:
    safe = "".join(char if char.isalnum() or char in {".", "_", "-"} else "_" for char in source)
    return safe.strip("._-") or "queue"


def get_queue_dir() -> Path:
    env = os.environ.get("BRAINLAYER_QUEUE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".brainlayer" / "queue"


def enqueue_jsonl_batch(events: list[dict[str, Any]], *, source: str, queue_dir: Path | None = None) -> Path:
    """Atomically append one or more write intents as a JSONL file."""
    if not events:
        raise ValueError("enqueue_jsonl_batch requires at least one event")
    resolved_dir = queue_dir or get_queue_dir()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    queued_at = time.time()
    safe_source = _safe_source_name(source)
    lines = [
        json.dumps(
            {
                **event,
                "source": source,
                "queued_at": queued_at,
            },
            ensure_ascii=True,
        )
        for event in events
    ]
    final_path = resolved_dir / f"{safe_source}-{now_ms}-{uuid.uuid4().hex}.jsonl"
    tmp_path = final_path.with_suffix(".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(final_path)
    return final_path


def enqueue_jsonl(event: dict[str, Any], *, source: str, queue_dir: Path | None = None) -> Path:
    """Atomically append a write intent as a one-line JSONL file."""
    return enqueue_jsonl_batch([event], source=source, queue_dir=queue_dir)


def enqueue_store(
    *,
    content: str,
    memory_type: str = "note",
    project: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    created_at: str | None = None,
    source: str = "mcp",
    queue_dir: Path | None = None,
    **metadata: Any,
) -> Path:
    supersedes = metadata.pop("supersedes", None)
    chunk_id = metadata.pop("chunk_id", None) or f"manual-{uuid.uuid4().hex[:16]}"
    created_at = created_at or datetime.now(timezone.utc).isoformat()
    return enqueue_jsonl(
        {
            "kind": "store_memory",
            "chunk_id": chunk_id,
            "content": content,
            "memory_type": memory_type,
            "project": project,
            "tags": tags,
            "importance": importance,
            "created_at": created_at,
            "supersedes": supersedes,
            "metadata": {key: value for key, value in metadata.items() if value is not None},
        },
        source=source,
        queue_dir=queue_dir,
    )


def enqueue_watcher_chunk(
    *,
    chunk_id: str,
    content: str,
    metadata: dict[str, Any],
    source_file: str,
    project: str | None,
    content_type: str,
    value_type: str,
    created_at: str,
    conversation_id: str,
    sender: str | None = None,
    tags: list[str] | None = None,
    chunk_origin: str | None = None,
    queue_dir: Path | None = None,
) -> Path:
    return enqueue_jsonl(
        {
            "kind": "watcher_chunk",
            "chunk_id": chunk_id,
            "content": content,
            "metadata": metadata,
            "source_file": source_file,
            "project": project,
            "content_type": content_type,
            "value_type": value_type,
            "created_at": created_at,
            "conversation_id": conversation_id,
            "sender": sender,
            "tags": tags,
            "chunk_origin": chunk_origin,
        },
        source="watcher",
        queue_dir=queue_dir,
    )


def enqueue_hook_chunk(
    *,
    session_id: str,
    content: str,
    chunk_id: str | None = None,
    content_hash: str | None = None,
    project: str | None = None,
    source_file: str | None = None,
    timestamp: float | None = None,
    queue_dir: Path | None = None,
) -> Path:
    return enqueue_jsonl(
        {
            "kind": "hook_chunk",
            "session_id": session_id,
            "chunk_id": chunk_id,
            "content": content,
            "content_hash": content_hash,
            "project": project,
            "source_file": source_file,
            "timestamp": timestamp if timestamp is not None else time.time(),
        },
        source="hook",
        queue_dir=queue_dir,
    )


def enqueue_enrichment_update(
    *,
    chunk_id: str,
    enrichment: dict[str, Any],
    content_hash: str | None = None,
    entities: list[Any] | None = None,
    chunk_origin: str | None = None,
    queue_dir: Path | None = None,
) -> Path:
    return enqueue_enrichment_updates(
        [
            {
                "chunk_id": chunk_id,
                "enrichment": enrichment,
                "content_hash": content_hash,
                "entities": entities,
                "chunk_origin": chunk_origin,
            }
        ],
        queue_dir=queue_dir,
    )


def enqueue_enrichment_updates(
    updates: list[dict[str, Any]],
    *,
    queue_dir: Path | None = None,
) -> Path:
    events = [
        {
            "kind": "enrichment_update",
            "chunk_id": update["chunk_id"],
            "enrichment": update["enrichment"],
            "content_hash": update.get("content_hash"),
            "entities": update.get("entities"),
            "chunk_origin": update.get("chunk_origin"),
        }
        for update in updates
    ]
    return enqueue_jsonl_batch(events, source="enrichment", queue_dir=queue_dir)
