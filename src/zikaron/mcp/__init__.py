"""Zikaron MCP Server - Model Context Protocol interface for Claude Code."""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from ..vector_store import VectorStore
from ..embeddings import get_embedding_model

# Default paths
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"

# Create MCP server
server = Server("zikaron")

# Lazy-loaded globals
_vector_store = None
_embedding_model = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(DEFAULT_DB_PATH)
    return _vector_store


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="zikaron_search",
            description="""Search through past Claude Code conversations and learnings.

Use this to find:
- How you previously implemented something
- Past solutions to similar problems
- Code patterns and approaches used before
- Error solutions from previous debugging sessions

The knowledge base contains indexed conversations organized by:
- Project (which codebase the conversation was about)
- Content type (ai_code, stack_trace, user_message, etc.)
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'how did I implement authentication' or 'React useEffect cleanup')"
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional: filter by project name"
                    },
                    "content_type": {
                        "type": "string",
                        "enum": ["ai_code", "stack_trace", "user_message", "assistant_text", "file_read", "git_diff"],
                        "description": "Optional: filter by content type"
                    },
                    "num_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="zikaron_stats",
            description="Get statistics about the knowledge base (total chunks, projects, content types).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="zikaron_list_projects",
            description="List all projects in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls."""

    if name == "zikaron_search":
        return await _search(
            query=arguments["query"],
            project=arguments.get("project"),
            content_type=arguments.get("content_type"),
            num_results=arguments.get("num_results", 5)
        )

    elif name == "zikaron_stats":
        return await _stats()

    elif name == "zikaron_list_projects":
        return await _list_projects()

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search(
    query: str,
    project: str | None = None,
    content_type: str | None = None,
    num_results: int = 5
) -> list[TextContent]:
    """Execute a search query."""
    try:
        if num_results < 1:
            num_results = 5
        elif num_results > 100:
            num_results = 100

        store = _get_vector_store()

        if store.count() == 0:
            return [TextContent(
                type="text",
                text="Knowledge base is empty. Run 'zikaron index' to populate it."
            )]

        # Generate embedding (run in thread to not block)
        loop = asyncio.get_running_loop()
        model = _get_embedding_model()
        query_embedding = await loop.run_in_executor(None, model.embed_query, query)

        # Search
        results = store.search(
            query_embedding=query_embedding,
            n_results=num_results,
            project_filter=project,
            content_type_filter=content_type
        )

        if not results["documents"][0]:
            return [TextContent(type="text", text="No results found.")]

        # Format results
        output_parts = [f"## Search Results for: {query}\n"]

        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = 1 - dist if dist is not None else 0
            output_parts.append(f"\n### Result {i+1} (score: {score:.3f})")
            output_parts.append(f"**Project:** {meta.get('project', 'unknown')} | **Type:** {meta.get('content_type', 'unknown')}")
            output_parts.append(f"**Source:** `{meta.get('source_file', 'unknown')}`\n")
            output_parts.append(doc[:1000] + ("..." if len(doc) > 1000 else ""))
            output_parts.append("\n---")

        return [TextContent(type="text", text="\n".join(output_parts))]

    except Exception as e:
        return [TextContent(type="text", text=f"Search error (query='{query[:50]}...'): {str(e)}")]


async def _stats() -> list[TextContent]:
    """Get knowledge base statistics."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()

        output = f"""## Zikaron Knowledge Base Stats

- **Total Chunks:** {stats['total_chunks']}
- **Projects:** {', '.join(stats['projects'][:15])}{'...' if len(stats['projects']) > 15 else ''}
- **Content Types:** {', '.join(stats['content_types'])}
"""
        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Stats error: {str(e)}")]


async def _list_projects() -> list[TextContent]:
    """List all projects."""
    try:
        store = _get_vector_store()
        stats = store.get_stats()

        if not stats['projects']:
            return [TextContent(type="text", text="No projects indexed yet.")]

        output = "## Indexed Projects\n\n"
        for proj in sorted(stats['projects']):
            output += f"- {proj}\n"

        return [TextContent(type="text", text=output)]

    except Exception as e:
        return [TextContent(type="text", text=f"Error listing projects: {str(e)}")]


def serve():
    """Start the MCP server using stdio.

    Note: MCP uses stdin/stdout for communication, not network ports.
    This is designed for integration with Claude Code via mcpServers config.
    """
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())

    asyncio.run(main())


if __name__ == "__main__":
    serve()
