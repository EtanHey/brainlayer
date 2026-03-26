"""Axiom telemetry for BrainLayer components.

Emits structured events to Axiom for observability. Gracefully degrades
when AXIOM_TOKEN is not set or Axiom is unreachable — never blocks the
main pipeline.

Usage:
    from brainlayer.telemetry import emit, emit_many

    emit("brainlayer-watcher", {"_type": "flush", "chunks_indexed": 5})
"""

import logging
import os
import threading
import traceback
from typing import Any

logger = logging.getLogger(__name__)

_DATASET_WATCHER = "brainlayer-watcher"

# Lazy-init client — avoids import-time side effects
_client = None
_client_lock = threading.Lock()
_client_failed = False


def _get_client():
    """Lazy-init the Axiom client. Returns None if unavailable."""
    global _client, _client_failed
    if _client_failed:
        return None
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client
        if _client_failed:
            return None

        token = os.environ.get("AXIOM_TOKEN")
        if not token:
            logger.debug("AXIOM_TOKEN not set — telemetry disabled")
            _client_failed = True
            return None

        try:
            import axiom_py

            _client = axiom_py.Client(token=token)
            return _client
        except Exception as e:
            logger.warning("Failed to init Axiom client: %s", e)
            _client_failed = True
            return None


def emit(dataset: str, event: dict[str, Any]) -> bool:
    """Emit a single event to Axiom. Returns True on success."""
    client = _get_client()
    if not client:
        return False
    try:
        client.ingest_events(dataset=dataset, events=[event])
        return True
    except Exception as e:
        logger.debug("Axiom emit failed: %s", e)
        return False


def emit_many(dataset: str, events: list[dict[str, Any]]) -> bool:
    """Emit multiple events to Axiom. Returns True on success."""
    if not events:
        return True
    client = _get_client()
    if not client:
        return False
    try:
        client.ingest_events(dataset=dataset, events=events)
        return True
    except Exception as e:
        logger.debug("Axiom emit_many failed (%d events): %s", len(events), e)
        return False


# ── Watcher-specific helpers ─────────────────────────────────────────────────


def emit_watcher_startup(sessions_watched: int, watcher_pid: int) -> bool:
    """Emit watcher startup event."""
    return emit(
        _DATASET_WATCHER,
        {
            "_type": "startup",
            "sessions_watched": sessions_watched,
            "watcher_pid": watcher_pid,
            "hostname": os.uname().nodename,
        },
    )


def emit_watcher_flush(
    chunks_indexed: int,
    chunks_skipped: int,
    latency_ms: float,
    source_files: list[str] | None = None,
) -> bool:
    """Emit batch flush metrics."""
    event = {
        "_type": "flush",
        "chunks_indexed": chunks_indexed,
        "chunks_skipped": chunks_skipped,
        "latency_ms": round(latency_ms, 2),
    }
    if source_files:
        event["source_files"] = source_files[:5]  # Cap to avoid bloat
    return emit(_DATASET_WATCHER, event)


def emit_watcher_error(
    error_type: str,
    message: str,
    file_path: str | None = None,
) -> bool:
    """Emit error event with traceback snippet."""
    tb = traceback.format_exc()
    # Keep last 500 chars of traceback
    tb_snippet = tb[-500:] if len(tb) > 500 else tb
    event = {
        "_type": "error",
        "error_type": error_type,
        "message": message[:300],
        "traceback_snippet": tb_snippet if tb_snippet.strip() != "NoneType: None" else None,
    }
    if file_path:
        event["file_path"] = file_path
    return emit(_DATASET_WATCHER, event)


def emit_watcher_heartbeat(
    sessions_tracked: int,
    chunks_indexed_total: int,
    uptime_seconds: float,
) -> bool:
    """Emit periodic heartbeat (health check)."""
    return emit(
        _DATASET_WATCHER,
        {
            "_type": "heartbeat",
            "sessions_tracked": sessions_tracked,
            "chunks_indexed_total": chunks_indexed_total,
            "uptime_seconds": round(uptime_seconds, 1),
            "watcher_pid": os.getpid(),
        },
    )
