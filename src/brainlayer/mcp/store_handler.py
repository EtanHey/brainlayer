"""Store, update, and digest MCP handlers."""

import asyncio
import json
import threading

import apsw
from mcp.types import CallToolResult, TextContent

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


async def _brain_digest(
    content: str | None = None,
    title: str | None = None,
    project: str | None = None,
    participants: list[str] | None = None,
    mode: str = "digest",
    limit: int = 25,
) -> CallToolResult:
    """Handle brain_digest tool call."""
    store = _get_vector_store()

    try:
        if mode == "enrich":
            from ..enrichment_controller import enrich_realtime

            result = enrich_realtime(store=store, limit=limit)
            result = {
                "mode": result.mode,
                "attempted": result.attempted,
                "enriched": result.enriched,
                "skipped": result.skipped,
                "failed": result.failed,
                "errors": result.errors,
            }
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])

        if mode != "digest":
            return _error_result(f"Unknown brain_digest mode: {mode}")

        if not content or not content.strip():
            return _error_result("content is required for brain_digest mode='digest'")

        from ..pipeline.digest import digest_content

        model = _get_embedding_model()
        loop = asyncio.get_event_loop()
        norm_project = _normalize_project_name(project) if project else None

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
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])
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
):
    """Wrapper for _store with auto-type detection and auto-importance."""
    resolved_type = memory_type or _detect_memory_type(content)
    resolved_importance = importance if importance is not None else _auto_importance(content)
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


def _get_pending_store_path():
    """Path for the store queue buffer file."""
    from ..paths import DEFAULT_DB_PATH

    return DEFAULT_DB_PATH.parent / "pending-stores.jsonl"


def _queue_store(item: dict) -> None:
    """Buffer a store request to JSONL when DB is locked.

    Enforces _QUEUE_MAX_SIZE: if the file exceeds the limit, oldest lines
    are dropped to make room.
    """
    path = _get_pending_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Append the new item
    with open(path, "a") as f:
        f.write(json.dumps(item) + "\n")

    # Enforce max size — read, trim oldest, atomic rewrite via tempfile
    try:
        lines = path.read_text().strip().splitlines()
        if len(lines) > _QUEUE_MAX_SIZE:
            trimmed = lines[-_QUEUE_MAX_SIZE:]
            tmp = path.with_suffix(".tmp")
            tmp.write_text("\n".join(trimmed) + "\n")
            tmp.rename(path)  # atomic on POSIX
            logger.warning(
                "Pending store queue trimmed: %d -> %d (dropped %d oldest)",
                len(lines),
                _QUEUE_MAX_SIZE,
                len(lines) - _QUEUE_MAX_SIZE,
            )
    except Exception:
        pass  # Non-critical — queue still works, just unbounded


def _flush_pending_stores(store, embed_fn) -> int:
    """Flush pending-stores.jsonl (FIFO). Returns count flushed."""
    from ..store import store_memory

    path = _get_pending_store_path()
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
            store_memory(
                store=store,
                embed_fn=embed_fn,
                content=item["content"],
                memory_type=item["memory_type"],
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
            )
            flushed += 1
        except Exception as e:
            logger.warning("Failed to flush pending store item: %s", e)
            remaining.append(line)

    if remaining:
        path.write_text("\n".join(remaining) + "\n")
    else:
        path.unlink(missing_ok=True)

    return flushed


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
):
    """Store a memory into BrainLayer with deferred embedding.

    The chunk is stored immediately without waiting for embedding generation.
    A background task embeds pending chunks after the response is sent.
    """
    try:
        from ..store import embed_pending_chunks, store_memory

        store = _get_vector_store()
        normalized_project = _normalize_project_name(project)

        # Store WITHOUT embedding — returns immediately (no executor needed)
        result = store_memory(
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
        )

        chunk_id = result["id"]

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
                count = embed_pending_chunks(store=bg_store, embed_fn=embed_fn)
                if count > 0:
                    logger.info("Embedded %d pending chunks", count)
                _flush_pending_stores(bg_store, embed_fn)
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

        parts = [f"Stored memory `{chunk_id}`"]
        structured = {"chunk_id": chunk_id, "related": result["related"]}
        return ([TextContent(type="text", text="\n".join(parts))], structured)

    except ValueError as e:
        return _error_result(f"Validation error: {str(e)}")
    except Exception as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            _queue_store(
                {
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
                }
            )
            structured = {"chunk_id": "queued", "related": []}
            return (
                [TextContent(type="text", text="Memory queued (DB busy). Will flush on next successful store.")],
                structured,
            )
        return _error_result(f"Store error: {str(e)}")
