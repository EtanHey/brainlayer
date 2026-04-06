"""Entity lookup MCP handlers."""

import asyncio

from mcp.types import CallToolResult, TextContent

from ._format import format_entity_card, format_entity_simple
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
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: entity_lookup(
                query=query,
                store=store,
                embed_fn=model.embed_query,
                entity_type=entity_type,
            ),
        )
    except Exception as e:
        return _error_result(f"Entity lookup failed: {e}")

    if result is None:
        return CallToolResult(content=[TextContent(type="text", text=f"No entity found matching '{query}'.")])

    # Map 'evidence' to 'chunks' for format_entity_simple
    if "evidence" in result and "chunks" not in result:
        result["chunks"] = result["evidence"]

    formatted = format_entity_simple(result)
    return CallToolResult(content=[TextContent(type="text", text=formatted)])


def _format_entity_list(result: dict) -> str:
    """Format list_entities result as a readable text block."""
    total = result.get("total", 0)
    entities = result.get("entities", [])
    lines = [f"Entities ({total} total):"]
    if not entities:
        lines.append("  (none)")
        return "\n".join(lines)
    for e in entities:
        desc = (e.get("description") or "")[:80].replace("\n", " ").strip()
        imp = e.get("importance", "")
        type_label = e.get("entity_type", "?")
        name = e.get("name", "?")
        suffix = f" -- {desc}" if desc else ""
        imp_str = f" (importance={imp})" if imp else ""
        lines.append(f"  [{type_label}] {name}{imp_str}{suffix}")
    return "\n".join(lines)


async def _brain_entity_list(
    entity_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> CallToolResult:
    """Handle brain_entity list action."""
    store = _get_vector_store()
    loop = asyncio.get_running_loop()

    try:
        result = await loop.run_in_executor(
            None,
            lambda: store.list_entities(entity_type=entity_type, limit=limit, offset=offset),
        )
    except Exception as e:
        return _error_result(f"Entity listing failed: {e}")

    formatted = _format_entity_list(result)
    return CallToolResult(content=[TextContent(type="text", text=formatted)])


async def _brain_get_person(
    name: str,
    context: str | None = None,
    num_memories: int = 10,
) -> CallToolResult:
    """Composite tool: look up a person entity + retrieve their scoped memories."""
    from ..pipeline.digest import entity_lookup

    store = _get_vector_store()
    model = _get_embedding_model()
    loop = asyncio.get_running_loop()

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
                    query_embedding=query_embedding,
                    query_text=context,
                    n_results=num_memories,
                    entity_id=entity_id,
                ),
            )
            if results["documents"][0]:
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    memories.append(
                        {
                            "content": doc[:500],
                            "type": meta.get("content_type", "unknown"),
                            "date": meta.get("created_at", "")[:10] if meta.get("created_at") else None,
                            "summary": meta.get("summary"),
                        }
                    )
        else:
            entity_chunks = await loop.run_in_executor(
                None,
                lambda: store.get_entity_chunks(entity_id, limit=num_memories),
            )
            for chunk in entity_chunks:
                memories.append(
                    {
                        "content": chunk["content"][:500] if chunk.get("content") else "",
                        "type": chunk.get("content_type", "unknown"),
                        "date": chunk.get("created_at", "")[:10] if chunk.get("created_at") else None,
                        "relevance": chunk.get("relevance"),
                    }
                )
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

    formatted = format_entity_card(result)
    return CallToolResult(content=[TextContent(type="text", text=formatted)])
