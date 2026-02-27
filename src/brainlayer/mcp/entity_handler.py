"""Entity lookup MCP handlers."""

import asyncio
import json

from mcp.types import CallToolResult, TextContent

from ._shared import (
    _error_result,
    _get_embedding_model,
    _get_vector_store,
    logger,
)


async def _brain_entity(
    query: str,
    entity_type: str | None = None,
) -> CallToolResult:
    """Handle brain_entity tool call."""
    from ..pipeline.digest import entity_lookup

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_event_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: entity_lookup(
                query=query, store=store, embed_fn=model.embed_query, entity_type=entity_type,
            ),
        )
    except Exception as e:
        return _error_result(f"Entity lookup failed: {e}")

    if result is None:
        return CallToolResult(content=[TextContent(type="text", text=f"No entity found matching '{query}'.")])
    return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])


async def _brain_get_person(
    name: str,
    context: str | None = None,
    num_memories: int = 10,
) -> CallToolResult:
    """Composite tool: look up a person entity + retrieve their scoped memories."""
    from ..pipeline.digest import entity_lookup

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_event_loop()

    try:
        entity = await loop.run_in_executor(
            None,
            lambda: entity_lookup(query=name, store=store, embed_fn=model.embed_query, entity_type="person"),
        )
    except Exception as e:
        return _error_result(f"Person lookup failed: {e}")

    if entity is None:
        return CallToolResult(content=[TextContent(type="text", text=f"No person entity found matching '{name}'.")])

    entity_id = entity["id"]

    memories = []
    try:
        if context:
            query_embedding = await loop.run_in_executor(None, model.embed_query, context)
            results = await loop.run_in_executor(
                None,
                lambda: store.hybrid_search(
                    query_embedding=query_embedding, query_text=context,
                    n_results=num_memories, entity_id=entity_id,
                ),
            )
            if results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    memories.append({
                        "content": doc[:500],
                        "type": meta.get("content_type", "unknown"),
                        "date": meta.get("created_at", "")[:10] if meta.get("created_at") else None,
                        "summary": meta.get("summary"),
                    })
        else:
            entity_chunks = await loop.run_in_executor(
                None,
                lambda: store.get_entity_chunks(entity_id, limit=num_memories),
            )
            for chunk in entity_chunks:
                memories.append({
                    "content": chunk["content"][:500] if chunk.get("content") else "",
                    "type": chunk.get("content_type", "unknown"),
                    "date": chunk.get("created_at", "")[:10] if chunk.get("created_at") else None,
                    "relevance": chunk.get("relevance"),
                })
    except Exception as e:
        logger.warning("Memory retrieval for person '%s' failed: %s", name, e)

    metadata = entity.get("metadata", {})
    result = {
        "entity_id": entity_id,
        "name": entity["name"],
        "profile": metadata,
        "hard_constraints": metadata.get("hard_constraints", {}),
        "preferences": metadata.get("preferences", {}),
        "contact_info": metadata.get("contact_info", {}),
        "relations": entity.get("relations", []),
        "memories": memories,
        "memory_count": len(memories),
    }

    return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, indent=2))])
