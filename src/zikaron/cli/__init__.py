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
) -> None:
    """Index Claude Code conversations into the knowledge base."""
    try:
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

        # Validate source directory exists
        if not source.exists():
            rprint(f"[bold red]Error:[/] Source directory does not exist: {source}")
            raise typer.Exit(1)

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

                # Determine project from path and normalize
                proj_name = jsonl_file.parent.name
                if proj_name == "subagents":
                    proj_name = jsonl_file.parent.parent.name
                proj_name = _normalize_project_name(proj_name)

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

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command("index-md")
def index_md(
    source: Path = typer.Argument(
        ...,
        help="Source directory containing markdown files"
    ),
    patterns: list[str] = typer.Option(
        ["**/*.md"],
        "--pattern", "-p",
        help="Glob patterns to match (can specify multiple)"
    ),
    exclude: list[str] = typer.Option(
        ["node_modules", ".git", "dist", "__pycache__", ".venv", "venv"],
        "--exclude", "-e",
        help="Directory names to exclude"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-index all files (ignore cache)"
    )
) -> None:
    """Index markdown files into the knowledge base.

    Supports learnings, skills, CLAUDE.md files, research docs, and more.
    Files are classified by path and chunked by header sections.
    """
    try:
        from ..pipeline import (
            find_markdown_files,
            extract_markdown_content,
            chunk_content,
            embed_chunks,
            index_to_chromadb
        )
        from ..pipeline.embed import ensure_model
        from ..pipeline.index import get_client, get_or_create_collection

        rprint(f"[bold blue]זיכרון[/] - Indexing markdown files from {source}")

        # Validate source directory exists
        if not source.exists():
            rprint(f"[bold red]Error:[/] Source directory does not exist: {source}")
            raise typer.Exit(1)

        # Ensure embedding model is available
        with console.status("[bold green]Checking embedding model..."):
            ensure_model()

        # Get ChromaDB client
        client = get_client()
        collection = get_or_create_collection(client)

        # Find markdown files
        md_files = list(find_markdown_files(source, patterns, exclude))

        if not md_files:
            rprint(f"[yellow]No markdown files found matching patterns: {patterns}[/]")
            raise typer.Exit(1)

        rprint(f"Found [bold]{len(md_files)}[/] markdown files")

        # Process each file
        total_chunks = 0
        type_counts: dict[str, int] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(md_files))

            for md_file in md_files:
                progress.update(task, description=f"Processing {md_file.name[:40]}...")

                try:
                    # Extract and classify content sections
                    classified_sections = extract_markdown_content(md_file)

                    chunks_for_file = []
                    for classified in classified_sections:
                        file_chunks = chunk_content(classified)
                        chunks_for_file.extend(file_chunks)

                        # Track content types
                        ct = classified.content_type.value
                        type_counts[ct] = type_counts.get(ct, 0) + 1

                    if chunks_for_file:
                        embedded = embed_chunks(chunks_for_file)
                        indexed = index_to_chromadb(
                            embedded,
                            collection,
                            source_file=str(md_file),
                            project=source.name  # Use source dir name as project
                        )
                        total_chunks += indexed

                except Exception as e:
                    rprint(f"[yellow]Warning:[/] Failed to process {md_file.name}: {e}")

                progress.advance(task)

        # Summary
        rprint(f"\n[bold green]✓[/] Indexed [bold]{total_chunks}[/] chunks from [bold]{len(md_files)}[/] files")

        if type_counts:
            rprint("\n[bold]Content types indexed:[/]")
            for ct, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                rprint(f"  {ct}: {count}")

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n: int = typer.Option(5, "--num", "-n", help="Number of results", min=1, max=100),
    project: str = typer.Option(None, "--project", "-p", help="Filter by project"),
    content_type: str = typer.Option(None, "--type", "-t", help="Filter by content type")
) -> None:
    """Search the knowledge base."""
    try:
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
            proj = _clean_project_name(meta.get('project', 'unknown'))
            content_type = meta.get('content_type', 'unknown')

            rprint(f"\n[bold cyan]─── Result {i+1} ───[/] [dim](score: {score:.3f})[/]")
            rprint(f"[bold]{proj}[/] · [dim]{content_type}[/]")
            rprint()

            # Clean up content display - skip raw dict representations
            content = doc
            if content.startswith("{'") or content.startswith('{"'):
                # Try to extract meaningful text from dict-like content
                try:
                    import ast
                    parsed = ast.literal_eval(content)
                    if isinstance(parsed, dict):
                        # Extract command/description if present
                        if 'command' in parsed:
                            content = f"[dim]Command:[/] {parsed['command']}"
                            if 'description' in parsed:
                                content += f"\n[dim]Description:[/] {parsed['description']}"
                        elif 'text' in parsed:
                            content = parsed['text']
                        else:
                            content = str(parsed)
                except:
                    pass

            # Truncate and display
            content = content[:600] + "..." if len(content) > 600 else content
            rprint(content)

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


# Known project renames/aliases - map old names to canonical names
PROJECT_ALIASES = {
    "ralphtools": "claude-golem",
    "config-ralphtools": "claude-golem",
    "config-ralph": "claude-golem",
}


def _normalize_project_name(raw_name: str) -> str:
    """Normalize a raw project folder name to a clean canonical name.

    Used during indexing to store clean names in the database.
    """
    # First clean the path-style name
    cleaned = _clean_project_name(raw_name)

    # Then apply aliases for renamed projects
    return PROJECT_ALIASES.get(cleaned, cleaned)


def _clean_project_name(name: str) -> str:
    """Clean up project names by extracting the repo name from path-style names.

    Examples:
        -Users-etanheyman-Gits-rudy-monorepo -> rudy-monorepo
        -Users-etanheyman-Desktop-Gits-domica -> domica
        -Users-etanheyman-Gits-etanheyman-com -> etanheyman-com
        -Users-etanheyman--config-ralph -> ralph
    """
    if not name.startswith("-Users-") and not name.startswith("-home-"):
        return name

    parts = name.split("-")

    # Find index of "Gits" or last occurrence of path markers
    markers = {"Gits", "Desktop", "projects", "config"}
    last_marker_idx = -1
    for i, part in enumerate(parts):
        if part in markers:
            last_marker_idx = i

    if last_marker_idx >= 0 and last_marker_idx < len(parts) - 1:
        # Take everything after the last marker
        repo_parts = [p for p in parts[last_marker_idx + 1:] if p]
        if repo_parts:
            return "-".join(repo_parts)

    # Fallback: filter out common path parts and join remaining
    skip = {"Users", "home", "Desktop", "Gits", "projects", "config", "etanheyman"}
    meaningful = [p for p in parts if p and p not in skip]
    if meaningful:
        return "-".join(meaningful[-2:]) if len(meaningful) > 1 else meaningful[-1]

    return name


@app.command()
def stats() -> None:
    """Show knowledge base statistics."""
    try:
        from ..pipeline.index import get_client, get_or_create_collection, get_db_path
        import sqlite3

        client = get_client()
        collection = get_or_create_collection(client)

        count = collection.count()
        if count == 0:
            rprint("[yellow]Knowledge base is empty. Run 'zikaron index' first.[/]")
            return

        # Query SQLite directly for accurate stats (avoids ChromaDB peek limitations)
        db_path = get_db_path() / "chroma.sqlite3"

        projects: dict[str, int] = {}
        content_types: dict[str, int] = {}
        unique_files = 0

        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()

            # Get project counts
            cur.execute("""
                SELECT string_value, COUNT(*) as cnt
                FROM embedding_metadata
                WHERE key='project'
                GROUP BY string_value
                ORDER BY cnt DESC
            """)
            for proj, cnt in cur.fetchall():
                clean_name = _clean_project_name(proj)
                projects[clean_name] = projects.get(clean_name, 0) + cnt

            # Get content type counts
            cur.execute("""
                SELECT string_value, COUNT(*) as cnt
                FROM embedding_metadata
                WHERE key='content_type'
                GROUP BY string_value
                ORDER BY cnt DESC
            """)
            for ct, cnt in cur.fetchall():
                content_types[ct] = cnt

            # Get unique source files count
            cur.execute("""
                SELECT COUNT(DISTINCT string_value)
                FROM embedding_metadata
                WHERE key='source_file'
            """)
            unique_files = cur.fetchone()[0]

            conn.close()

        # Summary table
        rprint(f"\n[bold blue]זיכרון Knowledge Base[/]\n")
        rprint(f"[bold]Total Chunks:[/] {count:,}")
        rprint(f"[bold]Source Files:[/] {unique_files:,}")
        rprint(f"[bold]Projects:[/] {len(projects)}")
        rprint(f"[bold]Content Types:[/] {len(content_types)}\n")

        # Projects table (sorted by count)
        proj_table = Table(title="Projects (top 15)")
        proj_table.add_column("Project", style="cyan")
        proj_table.add_column("Chunks", style="green", justify="right")

        for proj, cnt in sorted(projects.items(), key=lambda x: -x[1])[:15]:
            proj_table.add_row(proj, f"{cnt:,}")

        console.print(proj_table)

        # Content types table
        rprint()
        type_table = Table(title="Content Types")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Chunks", style="green", justify="right")

        for ct, cnt in sorted(content_types.items(), key=lambda x: -x[1]):
            type_table.add_row(ct, f"{cnt:,}")

        console.print(type_table)

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
) -> None:
    """Clear the entire knowledge base."""
    try:
        from ..pipeline.index import get_client

        if not confirm:
            confirm = typer.confirm("Are you sure you want to clear the knowledge base?")

        if confirm:
            client = get_client()
            client.reset()
            rprint("[bold green]✓[/] Knowledge base cleared")
        else:
            rprint("[yellow]Cancelled[/]")

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def serve() -> None:
    """Start the MCP server for Claude Code integration.

    Note: MCP uses stdio (stdin/stdout), not network ports.
    Configure in ~/.claude/settings.json under mcpServers.
    """
    try:
        from ..mcp import serve as mcp_serve
        rprint("[bold blue]זיכרון[/] - Starting MCP server (stdio mode)")
        mcp_serve()

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
