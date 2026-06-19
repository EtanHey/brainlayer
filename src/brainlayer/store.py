"""brainlayer_store — Write-side memory for BrainLayer.

Store ideas, mistakes, decisions, learnings, todos, and bookmarks
into the BrainLayer knowledge base.

When embed_fn is provided, items are embedded synchronously at write time.
When embed_fn is None, items are stored without embedding (deferred mode)
and are still searchable via FTS5. Embeddings are backfilled later by
embed_pending_chunks().

Usage:
    from brainlayer.store import store_memory

    # Deferred (fast) — no embedding, returns immediately
    result = store_memory(
        store=vector_store,
        embed_fn=None,
        content="Always use exponential backoff for retries",
        memory_type="learning",
        project="my-project",
        tags=["reliability", "api"],
    )

    # Synchronous (slow) — embeds at write time
    result = store_memory(
        store=vector_store,
        embed_fn=model.embed_query,
        content="...",
        memory_type="learning",
    )
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import apsw

from .chunk_origin import CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT, detect_chunk_origin
from .content_class import classify_content_class
from .dedupe import find_duplicate, merge_duplicate_chunk, merge_existing_chunk_content, merge_existing_chunk_seen
from .ingest_guard import reject_recursive_mcp_output
from .pipeline.classify import looks_like_system_prompt
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

VALID_MEMORY_TYPES = ["idea", "mistake", "decision", "learning", "todo", "bookmark", "note", "journal", "issue"]
_MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647


def _busy_deadline_timeout_ms(busy_deadline: Optional[float]) -> int | None:
    if busy_deadline is None:
        return None
    remaining_ms = int((busy_deadline - time.monotonic()) * 1000)
    if remaining_ms <= 0:
        raise apsw.BusyError("store busy budget exceeded")
    return max(1, min(remaining_ms, _MAX_APSW_BUSY_TIMEOUT_MS))


def _set_busy_timeout_for_deadline(conn, busy_deadline: Optional[float]) -> None:
    timeout_ms = _busy_deadline_timeout_ms(busy_deadline)
    if timeout_ms is None:
        return
    conn.setbusytimeout(timeout_ms)


def _sleep_before_busy_retry(delay: float, busy_deadline: Optional[float]) -> None:
    if busy_deadline is not None:
        remaining = busy_deadline - time.monotonic()
        if remaining <= 0 or delay >= remaining:
            raise apsw.BusyError("store busy budget exceeded")
        delay = min(delay, remaining)
    time.sleep(delay)


def store_memory(
    store: VectorStore,
    embed_fn: Optional[Callable[[str], List[float]]],
    content: str,
    memory_type: str,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    importance: Optional[int] = None,
    confidence_score: Optional[float] = None,
    outcome: Optional[str] = None,
    reversibility: Optional[str] = None,
    files_changed: Optional[List[str]] = None,
    entity_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    file_path: Optional[str] = None,
    function_name: Optional[str] = None,
    line_number: Optional[int] = None,
    chunk_id: Optional[str] = None,
    created_at: Optional[str] = None,
    fallback_source_path: Optional[str] = None,
    origin_repo_path: Optional[str] = None,
    replayed_by: Optional[str] = None,
    chunk_origin: Optional[str] = None,
    busy_deadline: Optional[float] = None,
    retry_on_busy: bool = True,
) -> Dict[str, Any]:
    """Persistently store a memory into BrainLayer.

    Args:
        store: VectorStore instance.
        embed_fn: Function that takes text and returns a 1024-dim embedding vector.
                  If None, the chunk is stored without embedding (deferred mode).
                  Un-embedded chunks are searchable via FTS5 and will be embedded
                  later by embed_pending_chunks().
        content: The text content to store.
        memory_type: One of VALID_MEMORY_TYPES.
        project: Optional project name to scope the memory.
        tags: Optional list of tags for categorization.
        importance: Optional importance score (1-10, clamped).
        confidence_score: Optional decision confidence (0-1).
        outcome: Optional decision outcome (pending/validated/reversed).
        reversibility: Optional reversibility (easy/hard/destructive).
        files_changed: Optional list of affected file paths.
        entity_id: Optional entity ID to link this memory to via kg_entity_chunks.
                   Used for per-person memory tagging.
        status: Optional issue status (open/in_progress/done/archived). Only for type=issue.
        severity: Optional issue severity (critical/high/medium/low). Only for type=issue.
        file_path: Optional code file reference. Only for type=issue.
        function_name: Optional function reference. Only for type=issue.
        line_number: Optional line number reference. Only for type=issue.
        chunk_id: Optional caller-supplied chunk ID for durable queued writes.
        created_at: Optional caller-supplied reservation timestamp for queued writes.
        fallback_source_path: Optional path to a local fallback file being replayed.
        origin_repo_path: Optional git root for the fallback's originating repo.
        replayed_by: Optional replay worker identifier.
        chunk_origin: Optional explicit origin classification preserved from queued fallback metadata.
        busy_deadline: Optional monotonic deadline for internal BusyError retries.
        retry_on_busy: Whether store_memory should run its own synchronous BusyError retry loop.

    Returns:
        Dict with 'id' (chunk ID) and 'related' (list of similar existing memories).

    Raises:
        ValueError: If content is empty or memory_type is invalid.
    """
    # Validate
    if not content or not content.strip():
        raise ValueError("content must be non-empty")
    if memory_type not in VALID_MEMORY_TYPES:
        raise ValueError(f"type must be one of: {', '.join(VALID_MEMORY_TYPES)}")

    content = content.strip()
    reject_recursive_mcp_output(content)
    if looks_like_system_prompt(content):
        raise ValueError("system prompt content is not stored in BrainLayer")

    # Clamp importance
    if importance is not None:
        importance = max(1, min(10, importance))

    # Generate chunk ID and timestamps
    chunk_id = chunk_id or f"manual-{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()
    effective_created_at = created_at or now
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Embed at write time (if embed_fn provided), otherwise defer
    embedding = None
    related: List[Dict[str, Any]] = []
    if embed_fn is not None:
        embedding = embed_fn(content)
        related = _find_related(store, embedding, project=project, limit=3)

    # Build metadata dict
    meta = {"memory_type": memory_type}
    if confidence_score is not None:
        meta["confidence_score"] = confidence_score
    if outcome is not None:
        meta["outcome"] = outcome
    if reversibility is not None:
        meta["reversibility"] = reversibility
    if files_changed is not None:
        meta["files_changed"] = files_changed
    # Issue-specific fields
    if status is not None:
        meta["status"] = status
    if severity is not None:
        meta["severity"] = severity
    if file_path is not None:
        meta["file_path"] = file_path
    if function_name is not None:
        meta["function_name"] = function_name
    if line_number is not None:
        meta["line_number"] = line_number
    if fallback_source_path is not None:
        meta["fallback_source_path"] = fallback_source_path
    if origin_repo_path is not None:
        meta["origin_repo_path"] = origin_repo_path
    if replayed_by is not None:
        meta["replayed_by"] = replayed_by

    resolved_chunk_origin = detect_chunk_origin(content, chunk_origin)
    content_class = classify_content_class(content, content_type=memory_type, tags=tags, source="manual")
    tags_json = json.dumps(tags) if tags else None
    incoming_chunk_id = chunk_id
    stored_chunk_id = chunk_id
    pending_reembed: tuple[str, str] | None = None
    incoming = {
        "id": incoming_chunk_id,
        "content": content,
        "tags": tags_json,
        "importance": float(importance) if importance is not None else None,
        "content_class": content_class,
        "created_at": effective_created_at,
        "last_seen_at": now,
        "content_hash": content_hash,
    }
    for attempt in range(5):
        cursor = store.conn.cursor()
        transaction_started = False
        try:
            _set_busy_timeout_for_deadline(store.conn, busy_deadline)
            cursor.execute("BEGIN IMMEDIATE")
            transaction_started = True
            pending_reembed = None
            if cursor.execute("SELECT 1 FROM chunks WHERE id = ?", (incoming_chunk_id,)).fetchone():
                stored_chunk_id = incoming_chunk_id
                content_changed = False
                if not merge_existing_chunk_seen(store.conn, chunk_id=incoming_chunk_id, incoming=incoming):
                    content_changed = merge_existing_chunk_content(
                        store.conn,
                        chunk_id=incoming_chunk_id,
                        incoming=incoming,
                    )
                if embedding is not None:
                    if content_changed:
                        merged_row = cursor.execute(
                            "SELECT content FROM chunks WHERE id = ?",
                            (stored_chunk_id,),
                        ).fetchone()
                        if merged_row:
                            pending_reembed = (stored_chunk_id, str(merged_row[0]))
                    elif not store._chunk_vector_exists(cursor, stored_chunk_id):
                        store._upsert_chunk_vector(cursor, stored_chunk_id, embedding)
            else:
                duplicate, dedupe_fields = find_duplicate(
                    store.conn,
                    chunk_id=incoming_chunk_id,
                    content=content,
                    created_at=effective_created_at,
                    project=project,
                    content_type=memory_type,
                )
                if duplicate is not None:
                    content_changed = merge_duplicate_chunk(
                        store.conn,
                        canonical_id=duplicate.canonical_chunk_id,
                        duplicate_id=incoming_chunk_id,
                        incoming=incoming,
                        mechanism=duplicate.mechanism,
                        hamming_distance_value=duplicate.hamming_distance,
                    )
                    stored_chunk_id = duplicate.canonical_chunk_id
                    if embedding is not None:
                        if content_changed:
                            merged_row = cursor.execute(
                                "SELECT content FROM chunks WHERE id = ?",
                                (stored_chunk_id,),
                            ).fetchone()
                            if merged_row:
                                pending_reembed = (stored_chunk_id, str(merged_row[0]))
                        elif not store._chunk_vector_exists(cursor, stored_chunk_id):
                            store._upsert_chunk_vector(cursor, stored_chunk_id, embedding)
                else:
                    stored_chunk_id = incoming_chunk_id
                    chunk_columns = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
                    insert_columns = [
                        "id",
                        "content",
                        "metadata",
                        "source_file",
                        "project",
                        "content_type",
                        "value_type",
                        "char_count",
                        "source",
                        "created_at",
                        "enriched_at",
                        "enrich_status",
                        "summary",
                        "tags",
                        "importance",
                        "chunk_origin",
                        "seen_count",
                        "last_seen_at",
                        "dedupe_hash",
                        "simhash",
                        "simhash_band_0",
                        "simhash_band_1",
                        "simhash_band_2",
                        "simhash_band_3",
                        "content_class",
                    ]
                    insert_values = [
                        incoming_chunk_id,
                        content,
                        json.dumps(meta),
                        "brainlayer-store",
                        project,
                        memory_type,
                        "high",
                        len(content),
                        "manual",
                        effective_created_at,
                        now,
                        "success",
                        content[:200],
                        tags_json,
                        float(importance) if importance is not None else None,
                        resolved_chunk_origin,
                        1,
                        now,
                        dedupe_fields.dedupe_hash,
                        dedupe_fields.simhash,
                        dedupe_fields.bands[0],
                        dedupe_fields.bands[1],
                        dedupe_fields.bands[2],
                        dedupe_fields.bands[3],
                        content_class,
                    ]
                    optional_values = {
                        "content_hash": content_hash,
                        "valid_from": effective_created_at,
                        "invalid_at": None,
                        "sys_period_start": now,
                        "sys_period_end": "9999-12-31T23:59:59.999999Z",
                    }
                    for column, value in optional_values.items():
                        if column in chunk_columns:
                            insert_columns.append(column)
                            insert_values.append(value)
                    column_sql = ", ".join(insert_columns)
                    placeholders = ", ".join("?" for _ in insert_values)
                    cursor.execute(
                        f"""
                        INSERT INTO chunks ({column_sql}, ingested_at)
                        VALUES ({placeholders}, CAST(strftime('%s', 'now') AS INTEGER))
                        """,
                        insert_values,
                    )
                    if embedding is not None:
                        store._upsert_chunk_vector(cursor, stored_chunk_id, embedding)

            if entity_id:
                entity = store.get_entity(entity_id)
                if entity is None:
                    raise ValueError(f"Unknown entity_id: {entity_id}")
                store.link_entity_chunk(
                    entity_id=entity_id,
                    chunk_id=stored_chunk_id,
                    relevance=1.0,
                    context=f"Stored via brain_store: {memory_type}",
                )
            cursor.execute("COMMIT")
            transaction_started = False
            break
        except apsw.BusyError:
            if transaction_started:
                cursor.execute("ROLLBACK")
            if not retry_on_busy or attempt == 4:
                raise
            _sleep_before_busy_retry(0.1 * (2**attempt), busy_deadline)
        except Exception:
            if transaction_started:
                cursor.execute("ROLLBACK")
            raise

    if pending_reembed is not None and embed_fn is not None:
        reembed_chunk_id, reembed_content = pending_reembed
        merged_embedding = embed_fn(reembed_content)
        for attempt in range(5):
            cursor = store.conn.cursor()
            transaction_started = False
            try:
                _set_busy_timeout_for_deadline(store.conn, busy_deadline)
                cursor.execute("BEGIN IMMEDIATE")
                transaction_started = True
                store._upsert_chunk_vector(cursor, reembed_chunk_id, merged_embedding)
                cursor.execute("COMMIT")
                transaction_started = False
                break
            except apsw.BusyError:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                if not retry_on_busy or attempt == 4:
                    raise
                _sleep_before_busy_retry(0.1 * (2**attempt), busy_deadline)
            except Exception:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                raise

    from .search_repo import clear_hybrid_search_cache

    clear_hybrid_search_cache(getattr(store, "db_path", None))
    store._invalidate_audit_recursion_count_cache()
    if resolved_chunk_origin == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT:
        store._invalidate_checkpoint_count_cache()

    return {
        "id": stored_chunk_id,
        "related": related,
    }


def embed_pending_chunks(
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    batch_size: int = 50,
    embed_batch_fn: Optional[Callable[[List[str]], List[List[float]]]] = None,
) -> int:
    """Backfill embeddings for chunks stored without them.

    Finds chunks in the chunks table that have no corresponding row in
    chunk_vectors and generates embeddings for them. Only active lifecycle
    rows are eligible; archived, superseded, and aggregated chunks are
    intentionally skipped because normal semantic search excludes them.

    Args:
        store: VectorStore instance.
        embed_fn: Function that takes text and returns a 1024-dim embedding vector.
        batch_size: Max chunks to process in one call.
        embed_batch_fn: Optional function that takes a list of texts and returns
            one 1024-dim embedding vector per text. When provided, the daemon
            embeds the selected rows in one model batch instead of one row at a time.

    Returns:
        Number of chunks embedded.
    """
    cursor = store.conn.cursor()
    rows = list(
        cursor.execute(
            """
            SELECT c.id, c.content FROM chunks c
            LEFT JOIN chunk_vectors_rowids r ON c.id = r.id
            WHERE r.id IS NULL
              AND c.content IS NOT NULL
              AND c.content != ''
              AND c.archived_at IS NULL
              AND c.superseded_by IS NULL
              AND c.aggregated_into IS NULL
              AND COALESCE(c.archived, 0) = 0
              AND COALESCE(c.status, 'active') = 'active'
            ORDER BY c.created_at ASC
            LIMIT ?
            """,
            (batch_size,),
        )
    )

    if not rows:
        return 0

    count = 0
    if embed_batch_fn is not None:
        try:
            embeddings = embed_batch_fn([content for _, content in rows])
            if len(embeddings) != len(rows):
                raise ValueError(f"batch embedder returned {len(embeddings)} embeddings for {len(rows)} chunks")
            for (chunk_id, _), embedding in zip(rows, embeddings):
                try:
                    store._upsert_chunk_vector(cursor, chunk_id, embedding)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to write pending chunk %s: %s", chunk_id, e)
        except Exception as e:
            logger.warning("Failed to embed pending chunk batch: %s", e)
            for chunk_id, content in rows:
                try:
                    embedding = embed_fn(content)
                    store._upsert_chunk_vector(cursor, chunk_id, embedding)
                    count += 1
                except Exception as row_error:
                    logger.warning("Failed to embed chunk %s: %s", chunk_id, row_error)
    else:
        for chunk_id, content in rows:
            try:
                embedding = embed_fn(content)
                store._upsert_chunk_vector(cursor, chunk_id, embedding)
                count += 1
            except Exception as e:
                logger.warning("Failed to embed chunk %s: %s", chunk_id, e)

    from .search_repo import clear_hybrid_search_cache

    clear_hybrid_search_cache(getattr(store, "db_path", None))
    return count


def embed_hot_chunk(
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    chunk_id: str,
) -> bool:
    """Embed one just-stored active chunk without disturbing the FIFO backlog drain.

    Returns True only when this call writes a vector. Missing, inactive, empty,
    already-embedded, or failed chunks return False so callers can safely fall
    through to deferred backlog processing.
    """
    cursor = store.conn.cursor()
    row = cursor.execute(
        """
        SELECT c.content
        FROM chunks c
        LEFT JOIN chunk_vectors v ON c.id = v.chunk_id
        WHERE c.id = ?
          AND v.chunk_id IS NULL
          AND c.content IS NOT NULL
          AND c.content != ''
          AND c.archived_at IS NULL
          AND c.superseded_by IS NULL
          AND c.aggregated_into IS NULL
          AND COALESCE(c.archived, 0) = 0
          AND COALESCE(c.status, 'active') = 'active'
        """,
        (chunk_id,),
    ).fetchone()
    if row is None:
        return False

    try:
        embedding = embed_fn(row[0])
        store._upsert_chunk_vector(cursor, chunk_id, embedding)
    except Exception as e:
        logger.warning("Failed to hot-embed chunk %s: %s", chunk_id, e)
        return False

    from .search_repo import clear_hybrid_search_cache

    clear_hybrid_search_cache(getattr(store, "db_path", None))
    return True


def _find_related(
    store: VectorStore,
    embedding: List[float],
    project: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Find related existing memories by vector similarity."""
    try:
        results = store.search(
            query_embedding=embedding,
            n_results=limit,
            project_filter=project,
        )
        related = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            item: Dict[str, Any] = {"content": doc[:300]}
            if meta.get("summary"):
                item["summary"] = meta["summary"]
            if meta.get("project"):
                item["project"] = meta["project"]
            if meta.get("content_type"):
                item["type"] = meta["content_type"]
            if meta.get("created_at"):
                item["date"] = meta["created_at"][:10]
            related.append(item)
        return related
    except Exception as e:
        # Don't let related search failure block the store — intentionally broad
        logger.warning("Related memory search failed: %s", e)
        return []
