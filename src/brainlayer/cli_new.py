"""Updated CLI commands using daemon client."""

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from pathlib import Path

from .client import get_client

console = Console()


def search_command(
    query: str = typer.Argument(..., help="Search query"),
    n: int = typer.Option(5, "--num", "-n", help="Number of results", min=1, max=100),
    project: str = typer.Option(None, "--project", "-p", help="Filter by project"),
    content_type: str = typer.Option(None, "--type", "-t", help="Filter by content type"),
    text: bool = typer.Option(False, "--text", help="Use text-based search instead of semantic search"),
    hybrid: bool = True
) -> None:
    """Search the knowledge base using fast daemon."""
    try:
        # Auto-detect domain-like queries and use text search
        if not text and ("." in query or query.startswith("http") or "/" in query):
            text = True
            rprint("[dim]Auto-detected domain/URL query, using text search[/]")

        search_type = "text" if text else ("hybrid" if hybrid else "semantic")
        rprint(f"[bold blue]זיכרון[/] - Searching ({search_type}): [italic]{query}[/]")

        # Search using daemon
        client = get_client()

        with console.status("[bold green]Searching..."):
            results = client.search(
                query=query,
                n_results=n,
                project_filter=project,
                content_type_filter=content_type,
                use_semantic=not text,
                hybrid=hybrid and not text
            )

        # Display results
        if not results["documents"]:
            rprint("[yellow]No results found[/]")
            return

        search_time = results["total_time_ms"]
        rprint(f"[dim]Found {len(results['documents'])} results in {search_time:.1f}ms[/]\n")

        result_ids = results.get("ids", [])
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        )):
            score = 1 - dist if dist is not None else None
            score_str = f"[dim](score: {score:.3f})[/]" if score is not None else "[dim](text match)[/]"
            proj = _clean_project_name(meta.get('project', 'unknown'))
            chunk_id = result_ids[i] if i < len(result_ids) else None

            # Truncate long content
            content = doc[:500] + "..." if len(doc) > 500 else doc

            rprint(f"[bold cyan]{i+1}.[/] {score_str} [dim]({proj})[/]")
            rprint(f"[white]{content}[/]")
            if chunk_id:
                rprint(f"[dim]ID: {chunk_id}[/]")
            rprint()

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def stats_command() -> None:
    """Show knowledge base statistics."""
    try:
        client = get_client()
        
        with console.status("[bold green]Getting stats..."):
            stats = client.get_stats()

        rprint(f"[bold blue]זיכרון[/] - Knowledge Base Statistics\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Chunks", f"{stats['total_chunks']:,}")
        table.add_row("Projects", str(len(stats['projects'])))
        table.add_row("Content Types", str(len(stats['content_types'])))
        
        console.print(table)
        
        if stats['projects']:
            rprint(f"\n[bold]Projects:[/] {', '.join(stats['projects'])}")
        
        if stats['content_types']:
            rprint(f"[bold]Content Types:[/] {', '.join(stats['content_types'])}")

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def migrate_command() -> None:
    """Migrate from ChromaDB to sqlite-vec."""
    try:
        from .migrate import migrate_from_chromadb
        
        rprint("[bold blue]זיכרון[/] - Migration Tool\n")
        
        sqlite_path = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"
        
        if sqlite_path.exists():
            response = typer.confirm(f"sqlite-vec database already exists. Overwrite?")
            if not response:
                rprint("Migration cancelled")
                return
            sqlite_path.unlink()
        
        with console.status("[bold green]Migrating data..."):
            success = migrate_from_chromadb()
        
        if success:
            rprint("[bold green]✓[/] Migration completed successfully!")
            rprint("You can now use the fast daemon service.")
        else:
            rprint("[bold red]✗[/] Migration failed or skipped")
            raise typer.Exit(1)

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def _clean_project_name(project: str) -> str:
    """Clean project name for display."""
    if not project or project == "unknown":
        return "unknown"
    
    # Remove common prefixes
    if project.startswith("/Users/"):
        parts = project.split("/")
        if len(parts) > 4:
            return "/".join(parts[-2:])  # Last two parts
    
    return project
