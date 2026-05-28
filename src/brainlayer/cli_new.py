"""CLI commands backed by direct readonly SQLite access."""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from .paths import get_db_path
from .vector_store import VectorStore

console = Console()


@contextmanager
def _readonly_store() -> Iterator[VectorStore]:
    db_path = get_db_path()
    _ensure_readonly_db_ready(db_path)
    store = VectorStore(db_path, readonly=True)
    try:
        yield store
    finally:
        store.close()


def _ensure_readonly_db_ready(db_path: Path) -> None:
    if _has_readonly_schema(db_path):
        return

    bootstrap_store = VectorStore(db_path)
    try:
        pass
    finally:
        bootstrap_store.close()


def _has_readonly_schema(db_path: Path) -> bool:
    if not db_path.exists():
        return False

    required_tables = {"chunks", "schema_migrations", "chunk_vectors", "chunks_fts"}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')").fetchall()
            existing_tables = {row[0] for row in rows}
            if not required_tables.issubset(existing_tables):
                return False
            chunk_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)").fetchall()}
            return {"id", "content", "metadata", "source_file"}.issubset(chunk_columns)
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def get_embedding_model():
    from .embeddings import get_embedding_model as _get_embedding_model

    return _get_embedding_model()


def _flatten_search_results(results: dict, total_time_ms: float) -> dict:
    return {
        "ids": results.get("ids", [[]])[0],
        "documents": results.get("documents", [[]])[0],
        "metadatas": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0],
        "total_time_ms": total_time_ms,
    }


def search_command(
    query: str = typer.Argument(..., help="Search query"),
    n: int = typer.Option(5, "--num", "-n", help="Number of results", min=1, max=100),
    project: str = typer.Option(None, "--project", "-p", help="Filter by project"),
    content_type: str = typer.Option(None, "--type", "-t", help="Filter by content type"),
    agent_id: str | None = None,
    text: bool = typer.Option(False, "--text", help="Use text-based search instead of semantic search"),
    hybrid: bool = True,
) -> None:
    """Search the knowledge base using direct readonly SQLite."""
    try:
        # Auto-detect domain-like queries and use text search
        if not text and ("." in query or query.startswith("http") or "/" in query):
            text = True
            rprint("[dim]Auto-detected domain/URL query, using text search[/]")

        search_type = "text" if text else ("hybrid" if hybrid else "semantic")
        rprint(f"[bold blue]זיכרון[/] - Searching ({search_type}): [italic]{query}[/]")

        with console.status("[bold green]Searching..."):
            start = time.time()
            with _readonly_store() as store:
                if text:
                    raw_results = store.search(
                        query_text=query,
                        n_results=n,
                        project_filter=project,
                        content_type_filter=content_type,
                    )
                elif hybrid:
                    query_embedding = get_embedding_model().embed_query(query)
                    raw_results = store.hybrid_search(
                        query_embedding=query_embedding,
                        query_text=query,
                        n_results=n,
                        project_filter=project,
                        content_type_filter=content_type,
                        agent_id=agent_id,
                    )
                else:
                    query_embedding = get_embedding_model().embed_query(query)
                    raw_results = store.search(
                        query_embedding=query_embedding,
                        n_results=n,
                        project_filter=project,
                        content_type_filter=content_type,
                    )
            results = _flatten_search_results(raw_results, (time.time() - start) * 1000)

        # Display results
        if not results["documents"]:
            rprint("[yellow]No results found[/]")
            return

        search_time = results["total_time_ms"]
        rprint(f"[dim]Found {len(results['documents'])} results in {search_time:.1f}ms[/]\n")

        result_ids = results.get("ids", [])
        for i, (doc, meta, dist) in enumerate(zip(results["documents"], results["metadatas"], results["distances"])):
            score = 1 - dist if dist is not None else None
            score_str = f"[dim](score: {score:.3f})[/]" if score is not None else "[dim](text match)[/]"
            proj = _clean_project_name(meta.get("project", "unknown"))
            # Show contact name for WhatsApp/messaging sources
            if proj == "unknown" and meta.get("contact_name"):
                proj = meta["contact_name"]
            chunk_id = result_ids[i] if i < len(result_ids) else None

            # Truncate long content
            content = doc[:500] + "..." if len(doc) > 500 else doc

            rprint(f"[bold cyan]{i + 1}.[/] {score_str} [dim]({proj})[/]")
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
        with console.status("[bold green]Getting stats..."):
            with _readonly_store() as store:
                stats = store.get_stats()

        rprint("[bold blue]זיכרון[/] - Knowledge Base Statistics\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Chunks", f"{stats['total_chunks']:,}")
        table.add_row("Projects", str(len(stats["projects"])))
        table.add_row("Content Types", str(len(stats["content_types"])))

        console.print(table)

        if stats["projects"]:
            rprint(f"\n[bold]Projects:[/] {', '.join(stats['projects'])}")

        if stats["content_types"]:
            rprint(f"[bold]Content Types:[/] {', '.join(stats['content_types'])}")

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def migrate_command() -> None:
    """Migrate from ChromaDB to sqlite-vec."""
    try:
        from .migrate import migrate_from_chromadb

        rprint("[bold blue]זיכרון[/] - Migration Tool\n")

        sqlite_path = get_db_path()

        if sqlite_path.exists():
            response = typer.confirm("sqlite-vec database already exists. Overwrite?")
            if not response:
                rprint("Migration cancelled")
                return
            sqlite_path.unlink()

        with console.status("[bold green]Migrating data..."):
            success = migrate_from_chromadb()

        if success:
            rprint("[bold green]✓[/] Migration completed successfully!")
            rprint("You can now use direct SQLite search.")
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
