"""Store, update, and digest MCP handlers."""

import asyncio
import json

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


async def _brain_digest(
    content: str,
    title: str | None = None,
    project: str | None = None,
    participants: list[str] | None = None,
) -> CallToolResult:
    """Handle brain_digest tool call."""
    from ..pipeline.digest import digest_content

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_event_loop()
    norm_project = _normalize_project_name(project) if project else None

    try:
        result = await loop.run_in_executor(
            None,
            lambda: digest_content(
                content=content, store=store, embed_fn=model.embed_query,
                title=title, project=norm_project, participants=participants,
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
        content=content, memory_type=resolved_type, project=project, tags=tags,
        importance=resolved_importance, confidence_score=confidence_score,
        outcome=outcome, reversibility=reversibility, files_changed=files_changed,
        entity_id=entity_id, status=status, severity=severity,
        file_path=file_path, function_name=function_name, line_number=line_number,
    )


async def _brain_update(
    action: str,
    chunk_id: str,
    content: str | None = None,
    tags: list[str] | None = None,
    importance: int | None = None,
    merge_chunk_ids: list[str] | None = None,
):
    """Update, archive, or merge memories."""
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
                chunk_id=chunk_id, content=content, tags=tags,
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
        logger.error("brain_update failed: %s", e)
        return _error_result(f"brain_update error: {e}")


def _get_pending_store_path():
    """Path for the store queue buffer file."""
    from ..paths import DEFAULT_DB_PATH
    return DEFAULT_DB_PATH.parent / "pending-stores.jsonl"


def _queue_store(item: dict) -> None:
    """Buffer a store request to JSONL when DB is locked."""
    path = _get_pending_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(item) + "\n")


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
                store=store, embed_fn=embed_fn, content=item["content"],
                memory_type=item["memory_type"], project=item.get("project"),
                tags=item.get("tags"), importance=item.get("importance"),
                confidence_score=item.get("confidence_score"),
                outcome=item.get("outcome"), reversibility=item.get("reversibility"),
                files_changed=item.get("files_changed"), entity_id=item.get("entity_id"),
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
    """Store a memory into BrainLayer. Buffers to JSONL on DB lock."""
    try:
        from ..store import store_memory

        store = _get_vector_store()
        model = _get_embedding_model()
        normalized_project = _normalize_project_name(project)

        loop = asyncio.get_running_loop()

        def _embed(text: str) -> list[float]:
            return model.embed_query(text)

        result = await loop.run_in_executor(
            None,
            lambda: store_memory(
                store=store, embed_fn=_embed, content=content, memory_type=memory_type,
                project=normalized_project, tags=tags, importance=importance,
                confidence_score=confidence_score, outcome=outcome,
                reversibility=reversibility, files_changed=files_changed,
                entity_id=entity_id, status=status, severity=severity,
                file_path=file_path, function_name=function_name, line_number=line_number,
            ),
        )

        try:
            flushed = await loop.run_in_executor(None, lambda: _flush_pending_stores(store, _embed))
        except Exception as e:
            logger.debug("Pending store flush failed: %s", e)
            flushed = 0

        chunk_id = result["id"]
        parts = [f"Stored memory `{chunk_id}`"]
        if flushed > 0:
            parts.append(f"(also flushed {flushed} queued items)")
        if result["related"]:
            parts.append(f"\n**Related memories ({len(result['related'])}):**")
            for r in result["related"]:
                summary = r.get("summary") or r.get("content", "")[:100]
                parts.append(f"- {summary}")

        structured = {"chunk_id": chunk_id, "related": result["related"]}
        return ([TextContent(type="text", text="\n".join(parts))], structured)

    except ValueError as e:
        return _error_result(f"Validation error: {str(e)}")
    except Exception as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            _queue_store({
                "content": content, "memory_type": memory_type,
                "project": _normalize_project_name(project), "tags": tags,
                "importance": importance, "confidence_score": confidence_score,
                "outcome": outcome, "reversibility": reversibility,
                "files_changed": files_changed, "entity_id": entity_id,
                "status": status, "severity": severity, "file_path": file_path,
                "function_name": function_name, "line_number": line_number,
            })
            structured = {"chunk_id": "queued", "related": []}
            return (
                [TextContent(type="text", text="Memory queued (DB busy). Will flush on next successful store.")],
                structured,
            )
        return _error_result(f"Store error: {str(e)}")
