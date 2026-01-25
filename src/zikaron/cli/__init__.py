"""Zikaron CLI - Command line interface for the knowledge pipeline."""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

app = typer.Typer(
    name="zikaron",
    help="זיכרון - Local knowledge pipeline for Claude Code conversations",
    no_args_is_help=True
)
console = Console()


@app.command()
def index(
    source: Path = typer.Argument(
        Path.home() / ".claude" / "projects",
        help="Source directory containing JSONL conversations"
    ),
    project: str = typer.Option(
        None, "--project", "-p",
        help="Only index specific project (folder name)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-index all files (ignore cache)"
    )
):
    """Index Claude Code conversations into the knowledge base."""
    from ..pipeline import (
        extract_system_prompts,
        classify_content,
        chunk_content,
        embed_chunks,
        index_to_chromadb
    )
    from ..pipeline.embed import ensure_model
    from ..pipeline.index import get_client, get_or_create_collection
    from ..pipeline.extract import parse_jsonl

    rprint(f"[bold blue]זיכרון[/] - Indexing conversations from {source}")

    # Ensure embedding model is available
    with console.status("[bold green]Checking embedding model..."):
        ensure_model()

    # Get ChromaDB client
    client = get_client()
    collection = get_or_create_collection(client)

    # Find all JSONL files
    if project:
        jsonl_files = list((source / project).rglob("*.jsonl")) if (source / project).exists() else []
    else:
        jsonl_files = list(source.rglob("*.jsonl"))

    if not jsonl_files:
        rprint("[yellow]No JSONL files found[/]")
        raise typer.Exit(1)

    rprint(f"Found [bold]{len(jsonl_files)}[/] conversation files")

    # Extract system prompts first (for deduplication)
    with console.status("[bold green]Extracting system prompts..."):
        system_prompts = extract_system_prompts(source)
        rprint(f"  Found [bold]{len(system_prompts)}[/] unique system prompts")

    # Process each file
    total_chunks = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=len(jsonl_files))

        for jsonl_file in jsonl_files:
            progress.update(task, description=f"Processing {jsonl_file.name[:40]}...")

            # Determine project from path
            proj_name = jsonl_file.parent.name
            if proj_name == "subagents":
                proj_name = jsonl_file.parent.parent.name

            chunks_for_file = []

            for entry in parse_jsonl(jsonl_file):
                classified = classify_content(entry)
                if classified:
                    file_chunks = chunk_content(classified)
                    chunks_for_file.extend(file_chunks)

            if chunks_for_file:
                embedded = embed_chunks(chunks_for_file)
                indexed = index_to_chromadb(
                    embedded,
                    collection,
                    source_file=str(jsonl_file),
                    project=proj_name
                )
                total_chunks += indexed

            progress.advance(task)

    rprint(f"\n[bold green]✓[/] Indexed [bold]{total_chunks}[/] chunks from [bold]{len(jsonl_files)}[/] files")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n: int = typer.Option(5, "--num", "-n", help="Number of results"),
    project: str = typer.Option(None, "--project", "-p", help="Filter by project"),
    content_type: str = typer.Option(None, "--type", "-t", help="Filter by content type")
):
    """Search the knowledge base."""
    from ..pipeline.embed import embed_query
    from ..pipeline.index import get_client, get_or_create_collection, search as db_search

    rprint(f"[bold blue]זיכרון[/] - Searching: [italic]{query}[/]")

    client = get_client()
    collection = get_or_create_collection(client)

    # Check if collection has data
    if collection.count() == 0:
        rprint("[yellow]Knowledge base is empty. Run 'zikaron index' first.[/]")
        raise typer.Exit(1)

    # Generate query embedding
    with console.status("[bold green]Generating query embedding..."):
        query_embedding = embed_query(query)

    # Build filters
    where = {}
    if project:
        where["project"] = project
    if content_type:
        where["content_type"] = content_type

    # Search
    results = db_search(
        collection,
        query_embedding,
        n_results=n,
        where=where if where else None
    )

    # Display results
    if not results["documents"][0]:
        rprint("[yellow]No results found[/]")
        return

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        score = 1 - dist  # Convert distance to similarity
        rprint(f"\n[bold cyan]─── Result {i+1} ───[/] [dim](score: {score:.3f})[/]")
        rprint(f"[dim]Project:[/] {meta.get('project', 'unknown')} | [dim]Type:[/] {meta.get('content_type', 'unknown')}")
        rprint(f"[dim]Source:[/] {Path(meta.get('source_file', '')).name}")
        rprint()

        # Truncate long content
        content = doc[:500] + "..." if len(doc) > 500 else doc
        rprint(content)


@app.command()
def stats():
    """Show knowledge base statistics."""
    from ..pipeline.index import get_client, get_or_create_collection, get_stats

    client = get_client()
    collection = get_or_create_collection(client)

    stats = get_stats(collection)

    table = Table(title="זיכרון Knowledge Base")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(stats["total_chunks"]))
    table.add_row("Projects", ", ".join(stats["projects"][:10]) + ("..." if len(stats["projects"]) > 10 else ""))
    table.add_row("Content Types", ", ".join(stats["content_types"]))

    console.print(table)


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
):
    """Clear the entire knowledge base."""
    from ..pipeline.index import get_client

    if not confirm:
        confirm = typer.confirm("Are you sure you want to clear the knowledge base?")

    if confirm:
        client = get_client()
        client.reset()
        rprint("[bold green]✓[/] Knowledge base cleared")
    else:
        rprint("[yellow]Cancelled[/]")


@app.command()
def serve(
    port: int = typer.Option(8765, "--port", "-p", help="MCP server port")
):
    """Start the MCP server for Claude Code integration."""
    from ..mcp import serve as mcp_serve
    rprint(f"[bold blue]זיכרון[/] - Starting MCP server on port {port}")
    mcp_serve(port=port)


if __name__ == "__main__":
    app()
