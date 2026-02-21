"""brainlayer_store — Write-side memory for BrainLayer.

Store ideas, mistakes, decisions, learnings, todos, and bookmarks
into the BrainLayer knowledge base. Items are embedded at write time
and searchable immediately.

Usage:
    from brainlayer.store import store_memory

    result = store_memory(
        store=vector_store,
        embed_fn=model.embed_query,
        content="Always use exponential backoff for retries",
        memory_type="learning",
        project="my-project",
        tags=["reliability", "api"],
    )
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .vector_store import VectorStore, serialize_f32

logger = logging.getLogger(__name__)

VALID_MEMORY_TYPES = ["idea", "mistake", "decision", "learning", "todo", "bookmark", "note", "journal"]


def store_memory(
    store: VectorStore,
    embed_fn: Callable[[str], List[float]],
    content: str,
    memory_type: str,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    importance: Optional[int] = None,
) -> Dict[str, Any]:
    """Persistently store a memory into BrainLayer.

    Args:
        store: VectorStore instance.
        embed_fn: Function that takes text and returns a 1024-dim embedding vector.
        content: The text content to store.
        memory_type: One of VALID_MEMORY_TYPES.
        project: Optional project name to scope the memory.
        tags: Optional list of tags for categorization.
        importance: Optional importance score (1-10, clamped).

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

    # Clamp importance
    if importance is not None:
        importance = max(1, min(10, importance))

    # Generate chunk ID and timestamps
    chunk_id = f"manual-{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()

    # Embed at write time
    embedding = embed_fn(content)

    # Search for related existing memories BEFORE inserting
    related = _find_related(store, embedding, project=project, limit=3)

    # Insert into chunks table
    cursor = store.conn.cursor()
    cursor.execute(
        """
        INSERT INTO chunks
        (id, content, metadata, source_file, project, content_type,
         value_type, char_count, source, created_at, enriched_at,
         summary, tags, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            chunk_id,
            content,
            json.dumps({"memory_type": memory_type}),
            "brainlayer-store",
            project,
            memory_type,  # content_type = memory_type for easy filtering
            "HIGH",
            len(content),
            "manual",
            now,
            now,  # enriched_at = now (user-provided content is pre-enriched)
            content[:200],  # summary = first 200 chars
            json.dumps(tags) if tags else None,
            float(importance) if importance is not None else None,
        ),
    )

    # Insert embedding into chunk_vectors
    cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
    cursor.execute(
        "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, serialize_f32(embedding)),
    )

    # Also insert into FTS5 for keyword search
    # (The trigger handles this for INSERT INTO chunks, but since we bypass
    # the normal upsert_chunks flow, verify it's there)

    return {
        "id": chunk_id,
        "related": related,
    }


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
