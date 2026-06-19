"""Store, update, and digest MCP handlers."""

import asyncio
import fcntl
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import apsw
from mcp.types import CallToolResult, TextContent

from ._format import format_digest_result, format_store_result
from ._shared import (
    _auto_importance,
    _detect_memory_type,
    _error_result,
    _get_embedding_model,
    _get_vector_store,
    _normalize_project_name,
    logger,
)

# Retry settings for DB lock resilience
_RETRY_MAX_ATTEMPTS = 4
_retry_delay = 0.15  # base delay in seconds (exposed for test patching)
_QUEUE_MAX_SIZE = 100
_DEFAULT_STORE_BUSY_BUDGET_MS = 3_000
_MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647
_STORE_BUSY_TIMEOUT_LOCK = threading.Lock()


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value <= 0 or value > _MAX_APSW_BUSY_TIMEOUT_MS:
        return default
    return value


def _store_busy_budget_ms() -> int:
    return _positive_int_env("BRAINLAYER_STORE_BUSY_BUDGET_MS", _DEFAULT_STORE_BUSY_BUDGET_MS)


def _store_busy_deadline() -> float:
    return time.monotonic() + (_store_busy_budget_ms() / 1000)


def _is_lock_error(exc: BaseException) -> bool:
    from ..vector_store import WriterInUseError

    text = str(exc).lower()
    return (
        isinstance(exc, apsw.BusyError)
        or isinstance(exc, WriterInUseError)
        or "locked" in text
        or "busy" in text
        or "sqlite prepare failed" in text
    )


def _new_manual_chunk_id() -> str:
    return f"manual-{uuid.uuid4().hex[:16]}"


def _reservation_created_at(item: dict) -> str | None:
    raw_created_at = item.get("created_at")
    if raw_created_at:
        return str(raw_created_at)
    raw_queued_at = item.get("queued_at")
    if raw_queued_at is None:
        return None
    if isinstance(raw_queued_at, int | float):
        return datetime.fromtimestamp(float(raw_queued_at), timezone.utc).isoformat()
    return str(raw_queued_at)


async def _brain_digest(
    content: str | None = None,
    title: str | None = None,
    project: str | None = None,
    participants: list[str] | None = None,
    mode: str = "digest",
    limit: int = 25,
) -> CallToolResult:
    """Handle brain_digest tool call."""
    # Validate inputs before initializing DB connection
    if mode not in ("digest", "enrich", "connect"):
        return _error_result(f"Unknown brain_digest mode: {mode}")
    if mode in ("digest", "connect") and (not content or not content.strip()):
        return _error_result(f"content is required for brain_digest mode='{mode}'")

    store = _get_vector_store()

    try:
        if mode == "enrich":
            from ..enrichment_controller import enrich_realtime

            loop = asyncio.get_event_loop()
            enrich_result = await loop.run_in_executor(None, lambda: enrich_realtime(store=store, limit=limit))
            result = {
                "mode": enrich_result.mode,
                "attempted": enrich_result.attempted,
                "enriched": enrich_result.enriched,
                "skipped": enrich_result.skipped,
                "failed": enrich_result.failed,
                "errors": enrich_result.errors,
            }
            formatted = format_digest_result(result)
            return CallToolResult(content=[TextContent(type="text", text=formatted)])

        from ..pipeline.digest import digest_connect, digest_content

        model = _get_embedding_model()
        loop = asyncio.get_event_loop()
        norm_project = _normalize_project_name(project) if project else None

        if mode == "connect":
            result = await loop.run_in_executor(
                None,
                lambda: digest_connect(
                    content=content,
                    store=store,
                    embed_fn=model.embed_query,
                    title=title,
                    project=norm_project,
                    participants=participants,
                ),
            )
        else:
            result = await loop.run_in_executor(
                None,
                lambda: digest_content(
                    content=content,
                    store=store,
                    embed_fn=model.embed_query,
                    title=title,
                    project=norm_project,
                    participants=participants,
                ),
            )
        formatted = format_digest_result(result)
        return CallToolResult(content=[TextContent(type="text", text=formatted)])
    except ValueError as e:
        return _error_result(str(e))
    except Exception as e:
        return _error_result(f"Digest failed: {e}")


async def _store_new(
    content: str,
    memory_type: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    confidence_score: float | None = None,
    outcome: str | None = None,
    reversibility: str | None = None,
    files_changed: list[str] | None = None,
    entity_id: str | None = None,
    status: str | None = None,
    severity: str | None = None,
    file_path: str | None = None,
    function_name: str | None = None,
    line_number: int | None = None,
    supersedes: str | None = None,
    agent_id: str | None = None,
):
    """Wrapper for _store with auto-type detection and auto-importance."""
    resolved_type = memory_type or _detect_memory_type(content)
    resolved_importance = importance if importance is not None else _auto_importance(content)
    # Tag chunk with agent identity for per-agent scoping
    if agent_id:
        tags = list(tags or [])
        agent_tag = f"agent:{agent_id}"
        if agent_tag not in tags:
            tags.append(agent_tag)
    if resolved_type == "issue" and status is None:
        status = "open"
    return await _store(
        content=content,
        memory_type=resolved_type,
        project=project,
        tags=tags,
        importance=resolved_importance,
        confidence_score=confidence_score,
        outcome=outcome,
        reversibility=reversibility,
        files_changed=files_changed,
        entity_id=entity_id,
        status=status,
        severity=severity,
        file_path=file_path,
        function_name=function_name,
        line_number=line_number,
        supersedes=supersedes,
    )


async def _brain_update(
    action: str,
    chunk_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    merge_chunk_ids: list[str] | None = None,
):
    """Update, archive, or merge memories. Retries on BusyError."""
    last_err = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            store = _get_vector_store()

            if action == "archive":
                ok = store.archive_chunk(chunk_id)
                if not ok:
                    return _error_result(f"Chunk not found: {chunk_id}")
                return [TextContent(type="text", text=json.dumps({"action": "archived", "chunk_id": chunk_id}))]

            elif action == "update":
                existing = store.get_chunk(chunk_id)
                if not existing:
                    return _error_result(f"Chunk not found: {chunk_id}")

                embedding = None
                if content is not None:
                    loop = asyncio.get_running_loop()
                    model = _get_embedding_model()
                    embedding = await loop.run_in_executor(None, model.embed_query, content)

                ok = store.update_chunk(
                    chunk_id=chunk_id,
                    content=content,
                    tags=tags,
                    importance=float(importance) if importance is not None else None,
                    embedding=embedding,
                )
                if not ok:
                    return _error_result(f"Update failed for: {chunk_id}")

                result = {"action": "updated", "chunk_id": chunk_id, "fields": []}
                if content is not None:
                    result["fields"].append("content")
                if tags is not None:
                    result["fields"].append("tags")
                if importance is not None:
                    result["fields"].append("importance")
                return [TextContent(type="text", text=json.dumps(result))]

            elif action == "merge":
                if not merge_chunk_ids:
                    return _error_result("merge requires merge_chunk_ids (the duplicates to archive)")
                keeper = store.get_chunk(chunk_id)
                if not keeper:
                    return _error_result(f"Keeper chunk not found: {chunk_id}")
                archived = []
                failed = []
                for dup_id in merge_chunk_ids:
                    ok = store.archive_chunk(dup_id)
                    if ok:
                        archived.append(dup_id)
                    else:
                        failed.append(dup_id)
                result = {"action": "merged", "kept": chunk_id, "archived": archived, "failed": failed}
                return [TextContent(type="text", text=json.dumps(result))]

            else:
                return _error_result(f"Unknown action: {action}. Use update, archive, or merge.")

        except Exception as e:
            is_lock_error = isinstance(e, apsw.BusyError) or "locked" in str(e).lower() or "busy" in str(e).lower()
            if is_lock_error and attempt < _RETRY_MAX_ATTEMPTS - 1:
                delay = _retry_delay * (2**attempt)
                logger.warning(
                    "brain_update BusyError (attempt %d/%d), retrying in %.2fs", attempt + 1, _RETRY_MAX_ATTEMPTS, delay
                )
                await asyncio.sleep(delay)
                last_err = e
                continue
            logger.error("brain_update failed: %s", e)
            return _error_result(f"brain_update error: {e}")


_PERSONAL_TYPES = frozenset({"journal", "note", "bookmark"})
_PERSONAL_KEYWORDS = ("health", "family", "relationship", "finance", "personal", "therapy", "medical")


def _is_personal_content(chunk: dict) -> bool:
    """Heuristic: return True if chunk likely contains personal data."""
    content_type = chunk.get("content_type", "")
    if content_type in _PERSONAL_TYPES:
        return True
    content_lower = (chunk.get("content") or "").lower()
    return any(kw in content_lower for kw in _PERSONAL_KEYWORDS)


async def _brain_supersede(
    old_chunk_id: str,
    new_chunk_id: str,
    safety_check: str = "auto",
    confirm: bool = False,
):
    """Mark old chunk as superseded by new chunk. Retries on BusyError."""
    if safety_check not in ("auto", "confirm"):
        return _error_result("safety_check must be 'auto' or 'confirm'")

    last_err = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            store = _get_vector_store()

            old_chunk = store.get_chunk(old_chunk_id)
            if not old_chunk:
                return _error_result(f"Old chunk not found: {old_chunk_id}")
            new_chunk = store.get_chunk(new_chunk_id)
            if not new_chunk:
                return _error_result(f"New chunk not found: {new_chunk_id}")

            # Safety gate: personal content requires explicit confirmation
            if safety_check == "auto" and _is_personal_content(old_chunk):
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "action": "confirm_required",
                                "reason": "Old chunk contains personal data — requires safety_check='confirm' and confirm=true",
                                "old_chunk_id": old_chunk_id,
                                "old_preview": (old_chunk.get("content") or "")[:200],
                                "new_chunk_id": new_chunk_id,
                            }
                        ),
                    )
                ]

            if safety_check == "confirm" and not confirm:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "action": "confirm_required",
                                "old_chunk_id": old_chunk_id,
                                "old_preview": (old_chunk.get("content") or "")[:200],
                                "new_chunk_id": new_chunk_id,
                                "new_preview": (new_chunk.get("content") or "")[:200],
                                "instruction": "Re-call with confirm=true to proceed",
                            }
                        ),
                    )
                ]

            ok = store.supersede_chunk(old_chunk_id, new_chunk_id)
            if not ok:
                return _error_result(f"Supersede failed for: {old_chunk_id}")

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "action": "superseded",
                            "old_chunk_id": old_chunk_id,
                            "new_chunk_id": new_chunk_id,
                        }
                    ),
                )
            ]

        except Exception as e:
            is_lock_error = isinstance(e, apsw.BusyError) or "locked" in str(e).lower() or "busy" in str(e).lower()
            if is_lock_error and attempt < _RETRY_MAX_ATTEMPTS - 1:
                delay = _retry_delay * (2**attempt)
                logger.warning(
                    "brain_supersede BusyError (attempt %d/%d), retrying in %.2fs",
                    attempt + 1,
                    _RETRY_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
                last_err = e
                continue
            logger.error("brain_supersede failed: %s", e)
            return _error_result(f"brain_supersede error: {e}")
    return _error_result(f"brain_supersede failed after {_RETRY_MAX_ATTEMPTS} retries: {last_err}")


async def _brain_archive(
    chunk_id: str,
    reason: str | None = None,
):
    """Archive a chunk (soft-delete with timestamp). Retries on BusyError."""
    last_err = None
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            store = _get_vector_store()

            chunk = store.get_chunk(chunk_id)
            if not chunk:
                return _error_result(f"Chunk not found: {chunk_id}")

            ok = store.archive_chunk(chunk_id)
            if not ok:
                return _error_result(f"Archive failed for: {chunk_id}")

            result = {"action": "archived", "chunk_id": chunk_id}
            if reason:
                result["reason"] = reason
            return [TextContent(type="text", text=json.dumps(result))]

        except Exception as e:
            is_lock_error = isinstance(e, apsw.BusyError) or "locked" in str(e).lower() or "busy" in str(e).lower()
            if is_lock_error and attempt < _RETRY_MAX_ATTEMPTS - 1:
                delay = _retry_delay * (2**attempt)
                logger.warning(
                    "brain_archive BusyError (attempt %d/%d), retrying in %.2fs",
                    attempt + 1,
                    _RETRY_MAX_ATTEMPTS,
                    delay,
                )
                await asyncio.sleep(delay)
                last_err = e
                continue
            logger.error("brain_archive failed: %s", e)
            return _error_result(f"brain_archive error: {e}")
    return _error_result(f"brain_archive failed after {_RETRY_MAX_ATTEMPTS} retries: {last_err}")


def _get_pending_store_path():
    """Path for the store queue buffer file."""
    from ..paths import get_db_path

    return get_db_path().parent / "pending-stores.jsonl"


def _atomic_rewrite_pending_store(path, lines: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _pending_store_file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _pending_store_lock_path(path)
    with open(lock_path, "a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _pending_store_lock_path(path):
    return path.parent / f".{path.name}.lock"


def _queue_store(item: dict):
    """Buffer a store request to JSONL when DB is locked.

    Enforces _QUEUE_MAX_SIZE: if the file exceeds the limit, oldest lines
    are dropped to make room.
    """
    item = dict(item)
    item["created_at"] = _reservation_created_at(item) or datetime.now(timezone.utc).isoformat()
    try:
        from ..queue_io import enqueue_store

        return enqueue_store(**item, source="mcp")
    except Exception:
        logger.debug("Unified queue write failed; falling back to pending-stores.jsonl", exc_info=True)

    path = _get_pending_store_path()
    with _pending_store_file_lock(path):
        # Legacy fallback file shares this lock with the flusher to avoid append/rewrite loss.
        with open(path, "a") as f:
            f.write(json.dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Enforce max size — read, trim oldest, atomic rewrite via tempfile
        try:
            lines = path.read_text().strip().splitlines()
            if len(lines) > _QUEUE_MAX_SIZE:
                trimmed = lines[-_QUEUE_MAX_SIZE:]
                _atomic_rewrite_pending_store(path, trimmed)
                logger.warning(
                    "Pending store queue trimmed: %d -> %d (dropped %d oldest)",
                    len(lines),
                    _QUEUE_MAX_SIZE,
                    len(lines) - _QUEUE_MAX_SIZE,
                )
        except Exception:
            logger.debug("Queue trim failed (non-critical)", exc_info=True)
    return path


def _deferred_store_receipt(chunk_id: str, queue_path, *, reason: str = "DB_BUSY") -> dict:
    action = "queued_for_replay" if str(queue_path).endswith("pending-stores.jsonl") else "queued_for_drain"
    return {
        "chunk_id": chunk_id,
        "queued": True,
        "status": "DEFERRED",
        "related": [],
        "deferred": {
            "status": "DEFERRED",
            "reason": reason,
            "chunk_id": chunk_id,
            "queue_path": str(queue_path),
            "action": action,
        },
    }


def _connection_busy_timeout_ms(conn) -> int | None:
    if conn is None:
        return None
    try:
        row = conn.cursor().execute("PRAGMA busy_timeout").fetchone()
        return int(row[0])
    except Exception:
        return None


def _set_connection_busy_timeout(conn, timeout_ms: int | None) -> None:
    if conn is None or timeout_ms is None:
        return
    try:
        conn.setbusytimeout(max(1, min(timeout_ms, _MAX_APSW_BUSY_TIMEOUT_MS)))
    except Exception:
        logger.debug("Failed to set store busy timeout", exc_info=True)


def _remaining_store_busy_budget_ms(deadline: float) -> int:
    remaining_ms = int((deadline - time.monotonic()) * 1000)
    if remaining_ms <= 0:
        raise apsw.BusyError("brain_store busy budget exceeded")
    return max(1, min(remaining_ms, _MAX_APSW_BUSY_TIMEOUT_MS))


@contextmanager
def _temporary_store_busy_timeout(conn, deadline: float):
    if conn is None:
        yield
        return

    timeout_ms = _remaining_store_busy_budget_ms(deadline)
    acquired = _STORE_BUSY_TIMEOUT_LOCK.acquire(timeout=max(timeout_ms, 1) / 1000)
    if not acquired:
        raise apsw.BusyError("brain_store busy_timeout lock wait exceeded")

    original_busy_timeout_ms = _connection_busy_timeout_ms(conn)
    try:
        timeout_ms = _remaining_store_busy_budget_ms(deadline)
        _set_connection_busy_timeout(conn, timeout_ms)
        yield
    finally:
        _set_connection_busy_timeout(conn, original_busy_timeout_ms)
        _STORE_BUSY_TIMEOUT_LOCK.release()


def _flush_pending_stores(store, embed_fn) -> int:
    """Flush pending-stores.jsonl (FIFO). Returns count flushed."""
    from ..store import store_memory

    path = _get_pending_store_path()
    with _pending_store_file_lock(path):
        if not path.exists():
            return 0

        try:
            lines = path.read_text().strip().splitlines()
        except Exception:
            logger.warning("Failed to read pending stores file: %s", path)
            return 0

        if not lines:
            return 0

        flushed = 0
        remaining = []
        for line in lines:
            try:
                item = json.loads(line)
                item_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                result = store_memory(
                    store=store,
                    embed_fn=embed_fn,
                    content=item["content"],
                    memory_type=item.get("memory_type", "note"),
                    project=item.get("project"),
                    tags=item.get("tags"),
                    importance=item.get("importance"),
                    confidence_score=item.get("confidence_score"),
                    outcome=item.get("outcome"),
                    reversibility=item.get("reversibility"),
                    files_changed=item.get("files_changed"),
                    entity_id=item.get("entity_id"),
                    status=item.get("status"),
                    severity=item.get("severity"),
                    file_path=item.get("file_path"),
                    function_name=item.get("function_name"),
                    line_number=item.get("line_number"),
                    chunk_id=item.get("chunk_id"),
                    created_at=_reservation_created_at(item),
                    chunk_origin=item.get("chunk_origin") or item_metadata.get("chunk_origin"),
                )
                if item.get("supersedes"):
                    if not store.supersede_chunk(item["supersedes"], result["id"]):
                        logger.error(
                            "Failed to supersede queued chunk %s with durable replacement %s",
                            item["supersedes"],
                            result["id"],
                        )
                flushed += 1
            except Exception as e:
                logger.warning("Failed to flush pending store item: %s", e)
                remaining.append(line)

        # Legacy fallback file shares this lock with appends to avoid dropping late arrivals.
        if remaining:
            _atomic_rewrite_pending_store(path, remaining)
        else:
            path.unlink(missing_ok=True)

        return flushed


async def _store_memory_with_retries(store_memory, *, deadline: float | None = None, **kwargs):
    last_err = None
    budget_ms = _store_busy_budget_ms()
    if deadline is None:
        deadline = _store_busy_deadline()
    conn = getattr(kwargs.get("store"), "conn", None)
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        remaining_ms = int((deadline - time.monotonic()) * 1000)
        if remaining_ms <= 0 and last_err is not None:
            raise last_err
        try:
            with _temporary_store_busy_timeout(conn, deadline):
                return store_memory(**kwargs, busy_deadline=deadline, retry_on_busy=False)
        except Exception as exc:
            if not _is_lock_error(exc) or attempt >= _RETRY_MAX_ATTEMPTS - 1:
                raise
            last_err = exc
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                logger.warning(
                    "brain_store BusyError exceeded %dms busy budget after attempt %d/%d; deferring",
                    budget_ms,
                    attempt + 1,
                    _RETRY_MAX_ATTEMPTS,
                )
                raise
            delay = _retry_delay * (2**attempt)
            if int(delay * 1000) >= remaining_ms:
                logger.warning(
                    "brain_store BusyError has %dms budget left before retry delay; deferring",
                    remaining_ms,
                )
                raise
            logger.warning(
                "brain_store BusyError (attempt %d/%d), retrying in %.2fs",
                attempt + 1,
                _RETRY_MAX_ATTEMPTS,
                delay,
            )
            await asyncio.sleep(delay)
    raise last_err  # type: ignore[misc]


def _get_store_vector_store(deadline: float):
    timeout = max(0.0, deadline - time.monotonic())
    try:
        return _get_vector_store(timeout=timeout)
    except TypeError as exc:
        if "timeout" not in str(exc):
            raise
        return _get_vector_store()


async def _store(
    content: str,
    memory_type: str,
    project: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    confidence_score: float | None = None,
    outcome: str | None = None,
    reversibility: str | None = None,
    files_changed: list[str] | None = None,
    entity_id: str | None = None,
    status: str | None = None,
    severity: str | None = None,
    file_path: str | None = None,
    function_name: str | None = None,
    line_number: int | None = None,
    supersedes: str | None = None,
):
    """Store a memory into BrainLayer with deferred embedding.

    The chunk is stored immediately without waiting for embedding generation.
    A background task embeds pending chunks after the response is sent.
    """
    promised_chunk_id = _new_manual_chunk_id()
    reservation_created_at = datetime.now(timezone.utc).isoformat()
    try:
        if os.environ.get("BRAINLAYER_ARBITRATED") == "1":
            from ..ingest_guard import reject_recursive_mcp_output
            from ..pipeline.classify import looks_like_system_prompt
            from ..search_repo import clear_hybrid_search_cache
            from ..store import VALID_MEMORY_TYPES

            if not content or not content.strip():
                raise ValueError("content must be non-empty")
            content = content.strip()
            if memory_type not in VALID_MEMORY_TYPES:
                raise ValueError(f"type must be one of: {', '.join(VALID_MEMORY_TYPES)}")
            reject_recursive_mcp_output(content)
            if looks_like_system_prompt(content):
                raise ValueError("system prompt content is not stored in BrainLayer")
            queue_path = _queue_store(
                {
                    "chunk_id": promised_chunk_id,
                    "content": content,
                    "memory_type": memory_type,
                    "project": _normalize_project_name(project),
                    "tags": tags,
                    "importance": importance,
                    "confidence_score": confidence_score,
                    "outcome": outcome,
                    "reversibility": reversibility,
                    "files_changed": files_changed,
                    "entity_id": entity_id,
                    "status": status,
                    "severity": severity,
                    "file_path": file_path,
                    "function_name": function_name,
                    "line_number": line_number,
                    "supersedes": supersedes,
                    "created_at": reservation_created_at,
                }
            )
            clear_hybrid_search_cache()
            structured = _deferred_store_receipt(promised_chunk_id, queue_path, reason="ARBITRATED")
            return ([TextContent(type="text", text=format_store_result(promised_chunk_id, queued=True))], structured)

        from ..store import embed_hot_chunk, embed_pending_chunks, store_memory
        from ..vector_store import temporary_write_busy_timeout_ms

        deadline = _store_busy_deadline()
        with temporary_write_busy_timeout_ms(_remaining_store_busy_budget_ms(deadline), deadline=deadline):
            store = _get_store_vector_store(deadline)
        normalized_project = _normalize_project_name(project)

        # Store WITHOUT embedding — returns immediately (no executor needed)
        result = await _store_memory_with_retries(
            store_memory,
            deadline=deadline,
            store=store,
            embed_fn=None,
            content=content,
            memory_type=memory_type,
            project=normalized_project,
            tags=tags,
            importance=importance,
            confidence_score=confidence_score,
            outcome=outcome,
            reversibility=reversibility,
            files_changed=files_changed,
            entity_id=entity_id,
            status=status,
            severity=severity,
            file_path=file_path,
            function_name=function_name,
            line_number=line_number,
            chunk_id=promised_chunk_id,
            created_at=reservation_created_at,
        )

        chunk_id = result["id"]

        # If supersedes is set, mark the old chunk as superseded by the new one
        superseded_ok = None
        if supersedes:
            with _temporary_store_busy_timeout(getattr(store, "conn", None), deadline):
                superseded_ok = store.supersede_chunk(supersedes, chunk_id)

        # Schedule background embedding + flush in a single daemon thread.
        # CRITICAL: must use a separate VectorStore connection — APSW enforces
        # same-thread usage. The main thread's `store.conn` cannot be shared.
        db_path = store.db_path

        def _background_embed_and_flush():
            from ..vector_store import VectorStore as _VS

            bg_store = None
            try:
                bg_store = _VS(db_path)
                model = _get_embedding_model()
                embed_fn = model.embed_query
                if embed_hot_chunk(store=bg_store, embed_fn=embed_fn, chunk_id=chunk_id):
                    logger.info("Hot-embedded chunk %s", chunk_id)
                count = embed_pending_chunks(
                    store=bg_store,
                    embed_fn=embed_fn,
                    batch_size=64,
                    embed_batch_fn=lambda texts: model.embed_texts(texts, batch_size=64),
                )
                if count > 0:
                    logger.info("Embedded %d pending chunks", count)
                _flush_pending_stores(bg_store, embed_fn)

                # Pass 2: async Gemini enrichment (R47 two-pass pattern)
                # Fire-and-forget — failure here never affects the store result
                try:
                    from ..enrichment_controller import enrich_single

                    enrich_single(bg_store, chunk_id)
                except Exception as enrich_err:
                    logger.debug("Auto-enrichment skipped for %s: %s", chunk_id, enrich_err)
            except Exception as e:
                logger.warning("Background embedding failed: %s", e)
                # Model loading failed — still try to flush queued stores
                # in deferred mode (without embeddings) so they aren't stranded
                if bg_store:
                    try:
                        _flush_pending_stores(bg_store, None)
                    except Exception as flush_err:
                        logger.warning("Fallback flush also failed: %s", flush_err)
            finally:
                if bg_store:
                    bg_store.close()

        t = threading.Thread(target=_background_embed_and_flush, daemon=True)
        t.start()

        superseded_id = supersedes if supersedes and superseded_ok else None
        formatted = format_store_result(chunk_id, superseded=superseded_id)
        if supersedes and not superseded_ok:
            formatted += f"\n  warn: could not supersede {supersedes} (not found)"
        structured = {"chunk_id": chunk_id, "related": result["related"]}
        if supersedes:
            structured["superseded"] = supersedes if superseded_ok else None
        return ([TextContent(type="text", text=formatted)], structured)

    except ValueError as e:
        return _error_result(f"Validation error: {str(e)}")
    except Exception as e:
        if _is_lock_error(e):
            queue_path = _queue_store(
                {
                    "chunk_id": promised_chunk_id,
                    "content": content,
                    "memory_type": memory_type,
                    "project": _normalize_project_name(project),
                    "tags": tags,
                    "importance": importance,
                    "confidence_score": confidence_score,
                    "outcome": outcome,
                    "reversibility": reversibility,
                    "files_changed": files_changed,
                    "entity_id": entity_id,
                    "status": status,
                    "severity": severity,
                    "file_path": file_path,
                    "function_name": function_name,
                    "line_number": line_number,
                    "supersedes": supersedes,
                    "created_at": reservation_created_at,
                }
            )
            structured = _deferred_store_receipt(promised_chunk_id, queue_path)
            formatted = format_store_result(promised_chunk_id, queued=True)
            return (
                [TextContent(type="text", text=formatted)],
                structured,
            )
        return _error_result(f"Store error: {str(e)}")
