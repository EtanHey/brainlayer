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

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import apsw

from .chunk_origin import CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT, detect_chunk_origin
from .dedupe import find_duplicate, merge_duplicate_chunk
from .ingest_guard import reject_recursive_mcp_output
from .pipeline.classify import looks_like_system_prompt
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

VALID_MEMORY_TYPES = ["idea", "mistake", "decision", "learning", "todo", "bookmark", "note", "journal", "issue"]


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
    chunk_id = f"manual-{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()

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

    chunk_origin = detect_chunk_origin(content)
    tags_json = json.dumps(tags) if tags else None
    incoming_chunk_id = chunk_id
    stored_chunk_id = chunk_id
    pending_reembed: tuple[str, str] | None = None
    for attempt in range(5):
        cursor = store.conn.cursor()
        transaction_started = False
        try:
            cursor.execute("BEGIN IMMEDIATE")
            transaction_started = True
            pending_reembed = None
            duplicate, dedupe_fields = find_duplicate(
                store.conn,
                chunk_id=incoming_chunk_id,
                content=content,
                created_at=now,
                project=project,
                content_type=memory_type,
            )
            if duplicate is not None:
                content_changed = merge_duplicate_chunk(
                    store.conn,
                    canonical_id=duplicate.canonical_chunk_id,
                    duplicate_id=incoming_chunk_id,
                    incoming={
                        "id": incoming_chunk_id,
                        "content": content,
                        "tags": tags_json,
                        "importance": float(importance) if importance is not None else None,
                        "created_at": now,
                        "last_seen_at": now,
                    },
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
                cursor.execute(
                    """
                    INSERT INTO chunks
                    (id, content, metadata, source_file, project, content_type,
                     value_type, char_count, source, created_at, enriched_at,
                     summary, tags, importance, chunk_origin, seen_count, last_seen_at,
                     dedupe_hash, simhash, simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        incoming_chunk_id,
                        content,
                        json.dumps(meta),
                        "brainlayer-store",
                        project,
                        memory_type,
                        "HIGH",
                        len(content),
                        "manual",
                        now,
                        now,
                        content[:200],
                        tags_json,
                        float(importance) if importance is not None else None,
                        chunk_origin,
                        1,
                        now,
                        dedupe_fields.dedupe_hash,
                        dedupe_fields.simhash,
                        dedupe_fields.bands[0],
                        dedupe_fields.bands[1],
                        dedupe_fields.bands[2],
                        dedupe_fields.bands[3],
                    ),
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
            if attempt == 4:
                raise
            time.sleep(0.1 * (2**attempt))
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
                cursor.execute("BEGIN IMMEDIATE")
                transaction_started = True
                store._upsert_chunk_vector(cursor, reembed_chunk_id, merged_embedding)
                cursor.execute("COMMIT")
                transaction_started = False
                break
            except apsw.BusyError:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                if attempt == 4:
                    raise
                time.sleep(0.1 * (2**attempt))
            except Exception:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                raise

    from .search_repo import clear_hybrid_search_cache

    clear_hybrid_search_cache(getattr(store, "db_path", None))
    if chunk_origin == CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT:
        store._invalidate_checkpoint_count_cache()

    return {
        "id": stored_chunk_id,
        "related": related,
    }


def embed_pending_chunks(
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    batch_size: int = 50,
) -> int:
    """Backfill embeddings for chunks stored without them.

    Finds chunks in the chunks table that have no corresponding row in
    chunk_vectors and generates embeddings for them.

    Args:
        store: VectorStore instance.
        embed_fn: Function that takes text and returns a 1024-dim embedding vector.
        batch_size: Max chunks to process in one call.

    Returns:
        Number of chunks embedded.
    """
    cursor = store.conn.cursor()
    rows = list(
        cursor.execute(
            """
            SELECT c.id, c.content FROM chunks c
            LEFT JOIN chunk_vectors v ON c.id = v.chunk_id
            WHERE v.chunk_id IS NULL AND c.source IN ('manual', 'mcp')
            ORDER BY c.created_at ASC
            LIMIT ?
            """,
            (batch_size,),
        )
    )

    if not rows:
        return 0

    count = 0
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
