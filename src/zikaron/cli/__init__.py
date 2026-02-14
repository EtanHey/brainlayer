"""Zikaron CLI - Command line interface for the knowledge pipeline."""

import os
import sys
import time

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
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
    """Index Claude Code conversations into the knowledge base (sqlite-vec)."""
    # Delegate to the sqlite-vec index implementation
    index_fast(source, project, force)


@app.command("index-md", hidden=True)
def index_md() -> None:
    """[Deprecated] Use 'zikaron index' instead."""
    rprint("[yellow]index-md is deprecated. Use 'zikaron index' instead.[/]")
    raise typer.Exit(1)


@app.command()
def dashboard() -> None:
    """Launch interactive dashboard for memory search and management."""
    try:
        from ..dashboard import DashboardApp
        
        app = DashboardApp()
        app.run()
        
    except ImportError as e:
        rprint(f"[red]Dashboard dependencies missing: {e}[/]")
        rprint("[yellow]Install with: pip install -e .[dev][/]")
    except Exception as e:
        rprint(f"[red]Dashboard error: {e}[/]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n: int = typer.Option(5, "--num", "-n", help="Number of results", min=1, max=100),
    project: str = typer.Option(None, "--project", "-p", help="Filter by project"),
    content_type: str = typer.Option(None, "--type", "-t", help="Filter by content type"),
    text: bool = typer.Option(False, "--text", help="Use text-based search instead of semantic search"),
    hybrid: bool = typer.Option(True, "--hybrid/--no-hybrid", help="Use hybrid search (semantic + keyword). Default: hybrid")
) -> None:
    """Search the knowledge base (sqlite-vec). Uses hybrid search by default."""
    from ..cli_new import search_command
    search_command(query, n, project, content_type, text, hybrid)


@app.command()
def context(
    chunk_id: str = typer.Argument(..., help="Chunk ID from a search result"),
    before: int = typer.Option(3, "--before", "-b", help="Chunks before target"),
    after: int = typer.Option(3, "--after", "-a", help="Chunks after target"),
) -> None:
    """Show surrounding conversation context for a search result."""
    try:
        from ..client import get_client

        rprint(f"[bold blue]זיכרון[/] - Context for chunk: [dim]{chunk_id[:40]}...[/]")

        client = get_client()

        with console.status("[bold green]Fetching context..."):
            result = client.get_context(chunk_id, before=before, after=after)

        if not result.get("context"):
            rprint("[yellow]No context available for this chunk.[/]")
            return

        for chunk in result["context"]:
            marker = " [bold green]<< TARGET >>[/]" if chunk.get("is_target") else ""
            ctype = chunk.get("content_type", "unknown")
            pos = chunk.get("position", "?")
            rprint(f"\n[bold cyan]Position {pos}[/] [dim]({ctype})[/]{marker}")
            content = chunk.get("content", "")
            rprint(content[:1500] + ("..." if len(content) > 1500 else ""))
            rprint("[dim]---[/]")

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


@app.command()
def review(
    limit: int = typer.Option(20, "--limit", "-n", help="Number of low-confidence chunks to review"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Confidence threshold"),
) -> None:
    """Review low-confidence auto-tagged chunks and correct labels."""
    try:
        from ..vector_store import VectorStore
        from pathlib import Path
        import json

        db_path = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
        store = VectorStore(db_path)
        cursor = store.conn.cursor()

        # Count tagged chunks
        total = store.count()
        tagged = list(cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND tags != '[]'"
        ))[0][0]
        low_conf = list(cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE tag_confidence IS NOT NULL AND tag_confidence < ?",
            (threshold,)
        ))[0][0]

        rprint(f"[bold blue]זיכרון[/] - Tag Review Queue")
        rprint(f"Total: {total:,} | Tagged: {tagged:,} | Low confidence (<{threshold}): {low_conf:,}")

        if tagged == 0:
            rprint("[yellow]No chunks tagged yet. Run: python scripts/classify-all.py[/]")
            return

        # Show worst chunks
        rows = list(cursor.execute("""
            SELECT id, content, tags, tag_confidence, project, content_type
            FROM chunks
            WHERE tag_confidence IS NOT NULL AND tag_confidence < ?
            ORDER BY tag_confidence ASC
            LIMIT ?
        """, (threshold, limit)))

        if not rows:
            rprint(f"[green]No chunks below {threshold} confidence![/]")
            return

        for i, row in enumerate(rows):
            chunk_id, content, tags_json, conf, proj, ct = row
            tags = json.loads(tags_json) if tags_json else []
            rprint(f"\n[bold cyan]{i+1}.[/] [dim]({proj}/{ct})[/] conf={conf:.2f}")
            rprint(f"  Tags: {', '.join(tags) if tags else '[none]'}")
            rprint(f"  Content: {content[:200]}...")
            rprint(f"  [dim]ID: {chunk_id}[/]")

        rprint(f"\n[dim]Use label-chunks.py for interactive correction, then retrain.[/]")
        store.close()

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


def _clean_project_name(name: str) -> str:
    """Clean up project names by extracting the repo name from path-style names.

    Examples:
        -Users-etanheyman-Gits-my-project -> my-project
        -Users-etanheyman-Desktop-Gits-another-project -> another-project
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
    """Show knowledge base statistics (sqlite-vec)."""
    from ..cli_new import stats_command
    stats_command()


@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")
) -> None:
    """Clear the entire knowledge base."""
    try:
        from pathlib import Path
        db_path = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"

        if not confirm:
            confirm = typer.confirm("Are you sure you want to clear the knowledge base?")

        if confirm:
            if db_path.exists():
                db_path.unlink()
                for suffix in ["-shm", "-wal"]:
                    p = db_path.parent / (db_path.name + suffix)
                    if p.exists():
                        p.unlink()
            rprint("[bold green]✓[/] Knowledge base cleared")
        else:
            rprint("[yellow]Cancelled[/]")

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command(hidden=True)
def fix_projects() -> None:
    """[Legacy/ChromaDB] Fix project names in the database."""
    try:
        from ..pipeline.index import get_client, get_or_create_collection
        import re
        from pathlib import Path

        rprint("[bold blue]זיכרון[/] - Fixing project names in database")

        client = get_client()
        collection = get_or_create_collection(client)

        if collection.count() == 0:
            rprint("[yellow]Knowledge base is empty.[/]")
            return

        # UUID pattern
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)

        # Get all chunks with UUID project names
        rprint("Scanning for chunks with UUID project names...")
        
        # Get ALL chunks - ChromaDB get() without ids returns all chunks
        total_count = collection.count()
        rprint(f"Total chunks in database: {total_count:,}")
        
        # Get all chunks (ChromaDB get() without parameters returns all chunks)
        # We'll process in batches to avoid memory issues
        batch_size = 10000
        all_ids = []
        all_metadatas = []
        
        # ChromaDB get() might have limits, so we'll use limit/offset if needed
        # But first try getting all at once
        try:
            all_results = collection.get(limit=total_count, include=["metadatas"])
            all_ids = all_results.get("ids", [])
            all_metadatas = all_results.get("metadatas", [])
        except Exception as e:
            rprint(f"[yellow]Warning:[/] Could not get all chunks at once: {e}")
            rprint("Trying to get chunks in batches...")
            # Fallback: get in batches
            for offset in range(0, total_count, batch_size):
                batch_results = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
                all_ids.extend(batch_results.get("ids", []))
                all_metadatas.extend(batch_results.get("metadatas", []))
        
        if not all_metadatas or not all_ids:
            rprint("[yellow]No metadata found.[/]")
            return
        
        rprint(f"Retrieved [bold]{len(all_ids):,}[/] chunks for scanning (expected {total_count:,})")
        
        if len(all_ids) != total_count:
            rprint(f"[yellow]Warning:[/] Retrieved {len(all_ids):,} chunks but database has {total_count:,}. Some chunks may be missed.")

        # Build mapping of UUID -> correct project name based on source_file paths
        uuid_to_project: dict[str, str] = {}
        chunks_to_update: list[tuple[str, dict]] = []  # (id, new_metadata)

        for i, metadata in enumerate(all_metadatas):
            if not metadata:
                continue
            
            project = metadata.get("project", "")
            source_file = metadata.get("source_file", "")
            
            # If project is a UUID, we need to fix it
            if uuid_pattern.match(project):
                # Extract correct project from source_file path
                try:
                    source_path = Path(source_file)
                    # Walk up from the file to find the project folder
                    current_path = source_path.parent
                    proj_name = current_path.name
                    
                    # Skip UUID directories
                    while uuid_pattern.match(proj_name) or proj_name == "subagents":
                        current_path = current_path.parent
                        if current_path == current_path.parent:  # Reached root
                            break
                        proj_name = current_path.name
                    
                    # Normalize the project name
                    correct_project = _normalize_project_name(proj_name)
                    
                    # Store mapping
                    if project not in uuid_to_project:
                        uuid_to_project[project] = correct_project
                    
                    # If this chunk needs updating
                    if correct_project != project:
                        chunk_id = all_ids[i]
                        new_metadata = metadata.copy()
                        new_metadata["project"] = correct_project
                        chunks_to_update.append((chunk_id, new_metadata))
                        
                except Exception as e:
                    rprint(f"[yellow]Warning:[/] Could not process {source_file}: {e}")
                    continue

        if not chunks_to_update:
            rprint("[green]No chunks need fixing![/]")
            return

        rprint(f"Found [bold]{len(chunks_to_update)}[/] chunks to update")
        rprint(f"UUID project mappings:")
        for uuid, proj in sorted(uuid_to_project.items()):
            rprint(f"  {uuid} -> {proj}")

        # Update chunks in batches using upsert (which updates existing records)
        # Close and reopen client to ensure we have write access
        batch_size = 1000
        updated = 0
        
        # Close the current client and collection
        del collection
        del client
        
        with console.status("[bold green]Updating project names..."):
            # Reopen client for write operations
            client = get_client()
            collection = get_or_create_collection(client)
            
            for i in range(0, len(chunks_to_update), batch_size):
                batch = chunks_to_update[i:i + batch_size]
                ids_to_update = [item[0] for item in batch]
                metadatas_to_update = [item[1] for item in batch]
                
                # ChromaDB upsert requires embeddings and documents too, so we need to get them
                existing = collection.get(ids=ids_to_update, include=["embeddings", "documents", "metadatas"])
                
                if not existing.get("ids"):
                    continue
                
                # Update with new metadata using update() method
                try:
                    collection.update(
                        ids=ids_to_update,
                        metadatas=metadatas_to_update
                    )
                except Exception as update_error:
                    # Fallback to upsert if update doesn't work
                    rprint(f"[yellow]Update failed, trying upsert:[/] {update_error}")
                    collection.upsert(
                        ids=ids_to_update,
                        embeddings=existing["embeddings"],
                        documents=existing["documents"],
                        metadatas=metadatas_to_update
                    )
                
                updated += len(batch)
                if (i + batch_size) % 5000 == 0:
                    rprint(f"  Updated {updated:,} chunks...")

        rprint(f"[bold green]✓[/] Updated [bold]{updated}[/] chunks")

    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def analyze_style(
    whatsapp_limit: int = typer.Option(
        1000, "--whatsapp-limit", "-w",
        help="Number of WhatsApp messages to analyze"
    ),
    claude_export: Path = typer.Option(
        None, "--claude-export", "-c",
        help="Path to Claude chat export JSON (optional)"
    ),
    output: Path = typer.Option(
        Path.home() / ".cursor/rules/communication-style.md",
        "--output", "-o",
        help="Output path for generated rules"
    ),
    export_analysis: bool = typer.Option(
        False, "--export-json",
        help="Also export full analysis as JSON"
    )
) -> None:
    """Analyze communication patterns from WhatsApp and Claude chats.
    
    Extracts your writing style and Claude's response patterns to generate
    personalized rules for desktop apps that help with text interactions.
    """
    try:
        from ..pipeline.extract_whatsapp import extract_whatsapp_messages, analyze_writing_style
        from ..pipeline.extract_claude_desktop import extract_claude_chats
        from ..pipeline.analyze_communication import CommunicationAnalyzer
        
        rprint("[bold blue]זיכרון[/] - Analyzing communication patterns\n")
        
        analyzer = CommunicationAnalyzer()
        
        # Extract WhatsApp messages
        with console.status("[bold green]Extracting WhatsApp messages..."):
            try:
                messages = list(extract_whatsapp_messages(
                    limit=whatsapp_limit,
                    only_from_me=False,  # Get both sides for context
                    exclude_groups=True
                ))
                analyzer.add_whatsapp_messages(messages)
                rprint(f"[green]✓[/] Extracted {len(messages)} WhatsApp messages")
            except FileNotFoundError as e:
                rprint("[yellow]Warning:[/] WhatsApp database not found")
                rprint(f"[dim]  {e}[/]")
            except Exception as e:
                rprint(f"[yellow]Warning:[/] Failed to extract WhatsApp: {e}")
        
        # Extract Claude chats if export provided
        if claude_export and claude_export.exists():
            with console.status("[bold green]Extracting Claude conversations..."):
                try:
                    conversations = list(extract_claude_chats(method="manual"))
                    analyzer.add_claude_conversations(conversations)
                    rprint(f"[green]✓[/] Extracted {len(conversations)} Claude conversations")
                except Exception as e:
                    rprint(f"[yellow]Warning:[/] Failed to extract Claude chats: {e}")
        else:
            rprint("[dim]No Claude export provided (use --claude-export)[/]")
            rprint("[dim]To export from Claude.ai:[/]")
            rprint("[dim]  1. Go to Settings > Data & Privacy[/]")
            rprint("[dim]  2. Click 'Export my data'[/]")
            rprint("[dim]  3. Pass the file with --claude-export PATH[/]\n")
        
        # Check if we have enough data
        if not analyzer.user_messages:
            rprint("[bold red]Error:[/] No messages found to analyze")
            rprint("Make sure WhatsApp database is accessible or provide Claude export")
            raise typer.Exit(1)
        
        # Generate analysis
        with console.status("[bold green]Analyzing communication patterns..."):
            writing_style = analyzer.analyze_writing_style()
            response_patterns = analyzer.analyze_claude_response_patterns()
            clarifications = analyzer.extract_common_clarifications()
        
        # Display summary
        rprint("\n[bold]Analysis Summary:[/]\n")
        rprint(f"Messages analyzed: [bold]{writing_style.get('total_messages_analyzed', 0)}[/]")
        rprint(f"Avg message length: [bold]{writing_style.get('avg_message_length', 0):.0f}[/] chars")
        rprint(f"Formality score: [bold]{writing_style.get('formality_score', 0):.2f}[/] (0=informal, 1=formal)")
        rprint(f"Emoji rate: [bold]{writing_style.get('emoji_rate', 0):.2f}[/] per message")
        
        if response_patterns.get('total_responses_analyzed', 0) > 0:
            rprint(f"\nClaude responses analyzed: [bold]{response_patterns['total_responses_analyzed']}[/]")
            rprint(f"Common clarifications found: [bold]{len(clarifications)}[/]")
        
        # Generate rules
        with console.status("[bold green]Generating rules..."):
            rules = analyzer.generate_rules(output)
        
        rprint(f"\n[bold green]✓[/] Rules generated: [cyan]{output}[/]")
        
        # Export full analysis if requested
        if export_analysis:
            json_path = output.with_suffix('.json')
            analyzer.export_analysis(json_path)
            rprint(f"[bold green]✓[/] Analysis exported: [cyan]{json_path}[/]")
        
        # Show preview
        rprint("\n[bold]Preview of generated rules:[/]\n")
        preview_lines = rules.split('\n')[:25]
        for line in preview_lines:
            rprint(f"[dim]{line}[/]")
        if len(rules.split('\n')) > 25:
            rprint(f"[dim]... ({len(rules.split('\n')) - 25} more lines)[/]")
        
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("list-chats")
def list_chats(
    output: Path = typer.Option(
        None, "--output", "-o",
        help="Save to CSV file (default: print only)"
    ),
    source: str = typer.Option(
        "whatsapp", "--source", "-s",
        help="Filter by source: whatsapp, claude, or all"
    ),
    claude_export: Path = typer.Option(
        None, "--claude-export", "-c",
        help="Path to Claude export (for source=claude or all)"
    ),
) -> None:
    """List chats with message counts for tagging (family, friends, co-workers)."""
    try:
        from ..pipeline.unified_timeline import UnifiedTimeline
        
        rprint("[bold blue]זיכרון[/] - Listing chats\n")
        
        timeline = UnifiedTimeline()
        src_filter = None if source == "all" else source
        
        if source in ("whatsapp", "all"):
            with console.status("[bold green]Loading WhatsApp..."):
                try:
                    timeline.load_whatsapp()
                except FileNotFoundError as e:
                    rprint(f"[yellow]WhatsApp:[/] {e}")
        
        if source in ("claude", "all"):
            claude_path = claude_export or Path("/tmp/claude-export/conversations.json")
            if claude_path.exists():
                with console.status("[bold green]Loading Claude..."):
                    timeline.load_claude(claude_path)
            else:
                rprint("[yellow]Claude:[/] No export at /tmp/claude-export/")
        
        chats = timeline.get_chat_list(source=src_filter)
        if not chats:
            rprint("[yellow]No chats found[/]")
            raise typer.Exit(0)
        
        # Print table
        table = Table(title=f"Chats ({len(chats)} total)")
        table.add_column("chat_id", style="dim")
        table.add_column("contact_name")
        table.add_column("messages", justify="right")
        for chat_id, contact_name, count in chats[:30]:
            table.add_row(chat_id[:40] + "..." if len(str(chat_id)) > 43 else str(chat_id), str(contact_name)[:30], str(count))
        if len(chats) > 30:
            table.add_row("...", f"... and {len(chats) - 30} more", "")
        console.print(table)
        
        # Save CSV if requested
        if output:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            import csv
            with open(output, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["chat_id", "contact_name", "message_count"])
                w.writerows(chats)
            rprint(f"\n[green]Saved to[/] {output}")
            rprint("[dim]Add a 'tag' column (family|friends|co-workers) and use for chat-tags.yaml[/]")
        
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command("analyze-evolution")
def analyze_evolution(
    claude_export: Path = typer.Option(
        None, "--claude-export", "-c",
        help="Path to Claude.ai export (conversations.json)"
    ),
    instagram_export: Path = typer.Option(
        None, "--instagram", "-i",
        help="Path to Instagram export directory"
    ),
    gemini_export: Path = typer.Option(
        None, "--gemini", "-g",
        help="Path to Gemini/Google Takeout export"
    ),
    output_dir: Path = typer.Option(
        Path("/tmp/style-analysis"),
        "--output", "-o",
        help="Output directory for analysis files"
    ),
    granularity: str = typer.Option(
        "half",
        "--granularity",
        help="Time period granularity: year, half, quarter, month"
    ),
    model: str = typer.Option(
        "qwen3-coder-64k",
        "--model", "-m",
        help="Ollama model for analysis"
    ),
    chat_tags: Path = typer.Option(
        None, "--chat-tags",
        help="Path to chat-tags.yaml (default: ~/.config/zikaron/chat-tags.yaml)"
    ),
    use_embeddings: bool = typer.Option(
        False, "--use-embeddings", "-e",
        help="Use StyleDistance for style-aware cluster sampling (best for style analysis)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip confirmation prompt (for background/scripted runs)"
    ),
) -> None:
    """Analyze communication style evolution over time.
    
    Loads messages from WhatsApp, Claude, Instagram, and Gemini,
    batches them by time period, analyzes each batch with LLM,
    and generates evolution analysis with weighted master guide.
    """
    try:
        from ..pipeline.unified_timeline import UnifiedTimeline
        from ..pipeline.time_batcher import create_time_batches, format_batches_summary
        from ..pipeline.longitudinal_analyzer import run_full_analysis
        
        rprint("[bold blue]זיכרון[/] - Longitudinal Style Analysis\n")
        
        # Create unified timeline
        timeline = UnifiedTimeline()
        
        # Load WhatsApp
        with console.status("[bold green]Loading WhatsApp messages..."):
            try:
                wa_count = timeline.load_whatsapp()
                rprint(f"[green]✓[/] WhatsApp: {wa_count:,} messages")
            except FileNotFoundError as e:
                rprint(f"[yellow]⚠[/] WhatsApp not found: {e}")
        
        # Load Claude if provided
        if claude_export and claude_export.exists():
            with console.status("[bold green]Loading Claude conversations..."):
                try:
                    cl_count = timeline.load_claude(claude_export)
                    rprint(f"[green]✓[/] Claude: {cl_count:,} messages")
                except Exception as e:
                    rprint(f"[yellow]⚠[/] Claude error: {e}")
        
        # Load Instagram if provided
        if instagram_export and instagram_export.exists():
            with console.status("[bold green]Loading Instagram data..."):
                try:
                    ig_count = timeline.load_instagram(instagram_export)
                    rprint(f"[green]✓[/] Instagram: {ig_count:,} messages")
                except Exception as e:
                    rprint(f"[yellow]⚠[/] Instagram error: {e}")
        
        # Load Gemini if provided or auto-detect in data/google-takeout
        gemini_path = gemini_export
        if not gemini_path or not gemini_path.exists():
            takeout_dir = Path(__file__).resolve().parents[3] / "data" / "google-takeout"
            if takeout_dir.exists():
                for z in sorted(takeout_dir.glob("*.zip")):
                    if "-11-" in z.name:
                        gemini_path = z
                        break
                if not gemini_path:
                    gemini_path = next(takeout_dir.glob("*.zip"), None)
        if gemini_path and gemini_path.exists():
            with console.status("[bold green]Loading Gemini data..."):
                try:
                    ge_count = timeline.load_gemini(gemini_path)
                    rprint(f"[green]✓[/] Gemini: {ge_count:,} messages")
                except Exception as e:
                    rprint(f"[yellow]⚠[/] Gemini error: {e}")
        
        # Check if we have data
        if not timeline.messages:
            rprint("[bold red]Error:[/] No messages loaded")
            raise typer.Exit(1)
        
        # Apply chat tags if config exists
        tagged = timeline.apply_chat_tags(chat_tags)
        if tagged:
            rprint(f"[green]✓[/] Applied relationship tags to {tagged:,} messages")
        
        # Sort by time
        timeline.sort_by_time()
        
        # Embed and index if requested
        style_collection = None
        if use_embeddings:
            from ..pipeline.style_embed import embed_messages, ensure_model
            from ..pipeline.style_index import (
                get_style_client,
                get_style_collection,
                index_style_messages,
                clear_style_collection,
            )
            try:
                ensure_model()
            except ImportError as e:
                rprint(f"[bold red]Error:[/] {e}")
                rprint("[dim]Install: pip install sentence-transformers[/]")
                raise typer.Exit(1)
            def embed_progress(n, t):
                if n > 0 and n % 5000 == 0:
                    rprint(f"  [dim]Embedded {n:,} / {t:,}[/]")

            with console.status("[bold green]Embedding messages with StyleDistance..."):
                msg_embs = embed_messages(
                    timeline.messages,
                    on_progress=embed_progress,
                )
            rprint(f"[green]✓[/] Embedded {len(msg_embs):,} messages")
            client = get_style_client()
            coll = get_style_collection(client)
            clear_style_collection(coll)
            indexed = index_style_messages(msg_embs, coll)
            rprint(f"[green]✓[/] Indexed {indexed:,} messages to vector store")
            style_collection = coll
        
        # Show stats
        stats = timeline.get_stats()
        rprint(f"\n[bold]Timeline Stats:[/]")
        rprint(f"  Total messages: {stats['total_messages']:,}")
        rprint(f"  Sources: {', '.join(stats['sources'])}")
        rprint(f"  Languages: {stats['by_language']}")
        if stats['date_range']:
            rprint(f"  Date range: {stats['date_range']['start'][:10]} to {stats['date_range']['end'][:10]}")
        
        # Create time batches
        rprint(f"\n[bold]Creating time batches (granularity: {granularity})...[/]")
        batches = create_time_batches(timeline, granularity=granularity)
        rprint(f"Created {len(batches)} batches")
        
        # Show batch summary
        rprint("\n" + format_batches_summary(batches))
        
        # Confirm before running LLM analysis
        rprint(f"\n[bold]Ready to analyze with {model}[/]")
        rprint(f"This will take approximately {len(batches) * 2} minutes.")
        
        if not yes and not typer.confirm("Continue with analysis?"):
            rprint("[yellow]Cancelled[/]")
            raise typer.Exit(0)
        
        # Run full analysis
        def progress_callback(msg, current, total):
            rprint(f"[dim][{current+1}/{total}] {msg}[/]")
        
        rprint(f"\n[bold]Running analysis...[/]\n")
        results = run_full_analysis(
            batches,
            output_dir,
            languages=["hebrew", "english"],
            model=model,
            progress_callback=progress_callback,
            style_collection=style_collection,
        )
        
        rprint(f"\n[bold green]✓ Analysis complete![/]")
        rprint(f"\nOutput files:")
        for name, path in results.items():
            rprint(f"  [cyan]{path}[/]")
        
        # Show human summary
        rprint(f"\n[bold]Human Summary:[/]\n")
        summary_path = Path(results['summary_file'])
        if summary_path.exists():
            with open(summary_path) as f:
                summary = f.read()
            # Show first 30 lines
            for line in summary.split('\n')[:30]:
                rprint(f"[dim]{line}[/]")
        
    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
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


@app.command("migrate")
def migrate() -> None:
    """Migrate from ChromaDB to sqlite-vec (one-time)."""
    from ..cli_new import migrate_command
    migrate_command()


@app.command("enrich")
def enrich(
    batch_size: int = typer.Option(50, "--batch-size", "-b", help="Chunks per batch"),
    max_chunks: int = typer.Option(0, "--max", "-m", help="Max chunks (0=unlimited)"),
    no_context: bool = typer.Option(False, "--no-context", help="Skip surrounding context"),
    stats_only: bool = typer.Option(False, "--stats", help="Show progress and exit"),
) -> None:
    """Enrich chunks with LLM-generated metadata (summary, tags, importance, intent)."""
    try:
        from ..pipeline.enrichment import run_enrichment, DEFAULT_DB_PATH
        from ..vector_store import VectorStore

        if stats_only:
            store = VectorStore(DEFAULT_DB_PATH)
            try:
                s = store.get_enrichment_stats()
                console.print(f"[bold]Total:[/] {s['total_chunks']}")
                console.print(f"[bold]Enriched:[/] {s['enriched']} ({s['percent']}%)")
                console.print(f"[bold]Remaining:[/] {s['remaining']}")
                if s['by_intent']:
                    console.print(f"[bold]Intent distribution:[/] {s['by_intent']}")
            finally:
                store.close()
        else:
            run_enrichment(
                batch_size=batch_size,
                max_chunks=max_chunks,
                with_context=not no_context,
            )
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command("git-overlay")
def git_overlay(
    project: str = typer.Option(
        None, "--project", "-p",
        help="Only process specific project slug"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-process sessions that already have context"
    ),
    max_sessions: int = typer.Option(
        0, "--max", "-m",
        help="Max sessions to process (0=all)"
    ),
    stats_only: bool = typer.Option(
        False, "--stats", help="Show stats and exit"
    ),
    file_timeline: str = typer.Option(
        None, "--file", help="Show timeline for a specific file"
    ),
) -> None:
    """Build git overlay: cross-reference sessions with git history (Phase 8b)."""
    from ..vector_store import VectorStore

    db_path = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
    store = VectorStore(db_path)

    try:
        if file_timeline:
            results = store.get_file_timeline(file_timeline, project=project)
            if not results:
                console.print(f"[dim]No interactions found for '{file_timeline}'[/]")
                return

            table = Table(title=f"File Timeline: {file_timeline}")
            table.add_column("Timestamp", style="cyan")
            table.add_column("Action", style="yellow")
            table.add_column("Project", style="green")
            table.add_column("Branch", style="magenta")
            table.add_column("PR", style="blue")
            table.add_column("Session", style="dim")

            for r in results:
                ts = (r["timestamp"] or "")[:19]
                table.add_row(
                    ts,
                    r["action"],
                    r["project"] or "",
                    r["branch"] or "",
                    f"#{r['pr_number']}" if r["pr_number"] else "",
                    (r["session_id"] or "")[:8],
                )
            console.print(table)
            return

        if stats_only:
            s = store.get_git_overlay_stats()
            console.print(f"[bold]Sessions with context:[/] {s['sessions_with_context']}")
            console.print(f"[bold]File interactions:[/] {s['file_interactions']}")
            console.print(f"[bold]Unique files tracked:[/] {s['unique_files']}")
            return

        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        from ..pipeline.git_overlay import run_git_overlay as _run

        console.print("[bold]Running git overlay...[/]")
        result = _run(
            vector_store=store,
            project=project,
            force=force,
            max_sessions=max_sessions,
        )
        console.print(f"[green]Done![/] Sessions: {result['sessions_processed']}, "
                      f"File interactions: {result['file_interactions_added']}")
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("group-operations")
def group_operations(
    project: str = typer.Option(
        None, "--project", "-p",
        help="Only process specific project name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-process sessions with existing operations"
    ),
    max_sessions: int = typer.Option(
        0, "--max", "-m",
        help="Max sessions to process (0=all)"
    ),
    stats_only: bool = typer.Option(
        False, "--stats", help="Show stats and exit"
    ),
    session: str = typer.Option(
        None, "--session", "-s",
        help="Show operations for a specific session"
    ),
) -> None:
    """Group chunks into logical operations (Phase 8a)."""
    from ..vector_store import VectorStore

    db_path = (
        Path.home() / ".local" / "share"
        / "zikaron" / "zikaron.db"
    )
    store = VectorStore(db_path)

    try:
        if session:
            ops = store.get_session_operations(session)
            if not ops:
                console.print(
                    f"[dim]No operations for session"
                    f" '{session[:8]}...'[/]"
                )
                return

            table = Table(
                title=f"Operations: {session[:8]}..."
            )
            table.add_column("Type", style="cyan")
            table.add_column("Summary", style="white")
            table.add_column("Steps", style="yellow")
            table.add_column("Outcome", style="green")
            table.add_column("Started", style="dim")

            for op in ops:
                outcome_style = {
                    "success": "[green]success[/]",
                    "failure": "[red]failure[/]",
                }.get(
                    op["outcome"],
                    f"[dim]{op['outcome']}[/]",
                )
                ts = (op["started_at"] or "")[:19]
                table.add_row(
                    op["operation_type"],
                    op["summary"],
                    str(op["step_count"]),
                    outcome_style,
                    ts,
                )
            console.print(table)
            return

        if stats_only:
            s = store.get_operations_stats()
            console.print(
                "[bold]Total operations:[/]"
                f" {s['total_operations']}"
            )
            console.print(
                "[bold]Sessions with operations:[/]"
                f" {s['sessions_with_operations']}"
            )
            if s["by_type"]:
                table = Table(title="By Type")
                table.add_column("Type", style="cyan")
                table.add_column("Count", style="yellow")
                for t, c in s["by_type"].items():
                    table.add_row(t, str(c))
                console.print(table)
            return

        import logging
        logging.basicConfig(
            level=logging.INFO, format="%(message)s"
        )

        from ..pipeline.operation_grouping import (
            run_operation_grouping as _run,
        )

        console.print(
            "[bold]Grouping operations...[/]"
        )
        result = _run(
            vector_store=store,
            project=project,
            force=force,
            max_sessions=max_sessions,
        )
        console.print(
            f"[green]Done![/] Sessions:"
            f" {result['sessions_processed']},"
            f" Operations: {result['operations_added']}"
        )
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("topic-chains")
def topic_chains(
    project: str = typer.Option(
        None, "--project", "-p",
        help="Filter by project name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Clear and rebuild chains"
    ),
    stats_only: bool = typer.Option(
        False, "--stats", help="Show stats and exit"
    ),
    file_query: str = typer.Option(
        None, "--file",
        help="Show chains for a specific file"
    ),
    regression: str = typer.Option(
        None, "--regression",
        help="Show regression analysis for a file"
    ),
) -> None:
    """Build topic chains + regression detection (Phase 8d)."""
    from ..vector_store import VectorStore

    db_path = (
        Path.home() / ".local" / "share"
        / "zikaron" / "zikaron.db"
    )
    store = VectorStore(db_path)

    try:
        if regression:
            result = store.get_file_regression(
                regression, project=project
            )
            if not result["timeline"]:
                console.print(
                    f"[dim]No interactions for"
                    f" '{regression}'[/]"
                )
                return

            console.print(
                f"[bold]Regression: {regression}[/]"
            )
            console.print(
                f"Timeline entries:"
                f" {len(result['timeline'])}"
            )

            if result["last_success"]:
                ls = result["last_success"]
                console.print(
                    f"[green]Last success:[/]"
                    f" {ls['timestamp']}"
                    f" (session {ls['session_id'][:8]},"
                    f" branch {ls.get('branch', '?')})"
                )
            else:
                console.print(
                    "[yellow]No successful operations"
                    " found[/]"
                )

            if result["changes_after"]:
                table = Table(
                    title="Changes After Last Success"
                )
                table.add_column(
                    "Timestamp", style="cyan"
                )
                table.add_column(
                    "Action", style="yellow"
                )
                table.add_column(
                    "Branch", style="magenta"
                )
                table.add_column(
                    "Session", style="dim"
                )
                for c in result["changes_after"][:20]:
                    ts = (c["timestamp"] or "")[:19]
                    table.add_row(
                        ts,
                        c["action"] or "?",
                        c.get("branch") or "",
                        (c["session_id"] or "")[:8],
                    )
                console.print(table)
            return

        if file_query:
            chains = store.get_file_chains(file_query)
            if not chains:
                console.print(
                    f"[dim]No chains for"
                    f" '{file_query}'[/]"
                )
                return

            table = Table(
                title=f"Topic Chains: {file_query}"
            )
            table.add_column(
                "Session A", style="cyan"
            )
            table.add_column(
                "Session B", style="green"
            )
            table.add_column(
                "Delta (hrs)", style="yellow"
            )
            table.add_column(
                "Shared", style="magenta"
            )
            table.add_column(
                "Branch A", style="dim"
            )
            table.add_column(
                "Branch B", style="dim"
            )
            for c in chains:
                delta = (
                    f"{c['time_delta_hours']:.1f}"
                    if c["time_delta_hours"] is not None
                    else "?"
                )
                table.add_row(
                    (c["session_a"] or "")[:8],
                    (c["session_b"] or "")[:8],
                    delta,
                    str(c["shared_actions"]),
                    c.get("branch_a") or "",
                    c.get("branch_b") or "",
                )
            console.print(table)
            return

        if stats_only:
            s = store.get_topic_chain_stats()
            console.print(
                f"[bold]Total chains:[/]"
                f" {s['total_chains']}"
            )
            console.print(
                f"[bold]Unique files:[/]"
                f" {s['unique_files']}"
            )
            return

        import logging
        logging.basicConfig(
            level=logging.INFO, format="%(message)s"
        )

        from ..pipeline.temporal_chains import (
            run_temporal_chains as _run,
        )

        console.print(
            "[bold]Building topic chains...[/]"
        )
        result = _run(
            vector_store=store,
            project=project,
            force=force,
        )
        console.print(
            f"[green]Done![/] Files:"
            f" {result['files_analyzed']},"
            f" Chains: {result['chains_created']}"
        )
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("plan-linking")
def plan_linking(
    project: str = typer.Option(
        None, "--project", "-p",
        help="Filter by project name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Clear and rebuild plan links"
    ),
    stats_only: bool = typer.Option(
        False, "--stats", help="Show stats and exit"
    ),
    plan_query: str = typer.Option(
        None, "--plan",
        help="Show sessions for a specific plan"
    ),
    session_query: str = typer.Option(
        None, "--session",
        help="Show plan info for a session ID"
    ),
    repo_root: str = typer.Option(
        None, "--repo",
        help="Repo root path (auto-detected)"
    ),
) -> None:
    """Link sessions to active plans (Phase 8c)."""
    from ..vector_store import VectorStore

    db_path = (
        Path.home() / ".local" / "share"
        / "zikaron" / "zikaron.db"
    )
    store = VectorStore(db_path)

    try:
        if session_query:
            ctx = store.get_session_context(session_query)
            if not ctx:
                # Try prefix match
                cursor = store.conn.cursor()
                rows = list(cursor.execute(
                    "SELECT session_id FROM session_context"
                    " WHERE session_id LIKE ?",
                    (f"{session_query}%",),
                ))
                if len(rows) == 1:
                    ctx = store.get_session_context(rows[0][0])
                elif len(rows) > 1:
                    console.print(
                        f"[yellow]Multiple sessions"
                        f" match '{session_query}':[/]"
                    )
                    for r in rows[:10]:
                        console.print(f"  {r[0]}")
                    return
            if not ctx:
                console.print(
                    f"[dim]No context for session"
                    f" '{session_query[:8]}'[/]"
                )
                return
            console.print(
                f"[bold]Session:[/] {ctx['session_id'][:8]}"
            )
            console.print(
                f"  Branch: {ctx.get('branch') or '?'}"
            )
            console.print(
                f"  PR: #{ctx.get('pr_number') or '?'}"
            )
            console.print(
                f"  Plan: {ctx.get('plan_name') or '(none)'}"
            )
            console.print(
                f"  Phase: {ctx.get('plan_phase') or '(none)'}"
            )
            console.print(
                f"  Story: {ctx.get('story_id') or '(none)'}"
            )
            return

        if plan_query:
            sessions = store.get_sessions_by_plan(
                plan_name=plan_query, project=project
            )
            if not sessions:
                console.print(
                    f"[dim]No sessions for plan"
                    f" '{plan_query}'[/]"
                )
                return

            table = Table(
                title=f"Sessions: {plan_query}"
            )
            table.add_column(
                "Session", style="cyan"
            )
            table.add_column(
                "Branch", style="green"
            )
            table.add_column(
                "PR", style="yellow"
            )
            table.add_column(
                "Phase", style="magenta"
            )
            table.add_column(
                "Started", style="dim"
            )
            for s in sessions:
                table.add_row(
                    (s["session_id"] or "")[:8],
                    s.get("branch") or "",
                    f"#{s['pr_number']}"
                    if s.get("pr_number") else "",
                    s.get("plan_phase") or "",
                    (s.get("started_at") or "")[:19],
                )
            console.print(table)
            return

        if stats_only:
            s = store.get_plan_linking_stats()
            console.print(
                f"[bold]Total sessions:[/]"
                f" {s['total_sessions']}"
            )
            console.print(
                f"[bold]Linked:[/]"
                f" {s['linked_sessions']}"
            )
            console.print(
                f"[bold]Unlinked:[/]"
                f" {s['unlinked_sessions']}"
            )
            if s["plans"]:
                table = Table(title="Plans")
                table.add_column(
                    "Plan", style="cyan"
                )
                table.add_column(
                    "Sessions", style="green"
                )
                for name, count in s["plans"].items():
                    table.add_row(name, str(count))
                console.print(table)
            return

        import logging
        logging.basicConfig(
            level=logging.INFO, format="%(message)s"
        )

        from ..pipeline.plan_linking import (
            run_plan_linking as _run,
        )

        console.print(
            "[bold]Linking sessions to plans...[/]"
        )
        result = _run(
            vector_store=store,
            repo_root=repo_root,
            project=project,
            force=force,
        )
        console.print(
            f"[green]Done![/] Checked:"
            f" {result['sessions_checked']},"
            f" Linked: {result['sessions_linked']}"
        )
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("export-obsidian")
def export_obsidian(
    vault_path: str = typer.Option(
        None, "--vault", "-v",
        help="Obsidian vault path"
        " (default: ~/.golems-brain/Zikaron)"
    ),
    project: str = typer.Option(
        None, "--project", "-p",
        help="Filter by project name"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing notes"
    ),
) -> None:
    """Export Zikaron data to Obsidian vault (Phase 9)."""
    from ..vector_store import VectorStore

    db_path = (
        Path.home() / ".local" / "share"
        / "zikaron" / "zikaron.db"
    )
    store = VectorStore(db_path)

    try:
        import logging
        logging.basicConfig(
            level=logging.INFO, format="%(message)s"
        )

        from ..pipeline.obsidian_export import (
            export_obsidian as _export,
        )

        console.print(
            "[bold]Exporting to Obsidian vault...[/]"
        )
        result = _export(
            vector_store=store,
            vault_path=vault_path,
            project=project,
            force=force,
        )
        console.print(
            f"[green]Done![/]"
            f" Sessions: {result['sessions']},"
            f" Files: {result['files']},"
            f" Plans: {result['plans']},"
            f" Dashboards: {result['dashboards']}"
        )
        vault = vault_path or str(
            Path.home() / ".golems-brain" / "Zikaron"
        )
        console.print(
            f"[dim]Vault: {vault}[/]"
        )
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)
    finally:
        store.close()


@app.command("index-fast", hidden=True)
def index_fast(
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
    """Index using new fast sqlite-vec backend."""
    try:
        from ..pipeline.extract import parse_jsonl
        from ..pipeline.classify import classify_content
        from ..pipeline.chunk import chunk_content
        from ..index_new import index_chunks_to_sqlite
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
        import time
        
        if not source.exists():
            rprint(f"[bold red]Error:[/] Source directory not found: {source}")
            raise typer.Exit(1)

        # Find JSONL files
        if project:
            project_dir = source / project
            if not project_dir.exists():
                rprint(f"[bold red]Error:[/] Project directory not found: {project_dir}")
                raise typer.Exit(1)
            jsonl_files = list(project_dir.glob("*.jsonl"))
            if not jsonl_files:
                rprint(f"[bold red]Error:[/] No JSONL files found in project: {project_dir}")
                raise typer.Exit(1)
        else:
            jsonl_files = list(source.rglob("*.jsonl"))
            if not jsonl_files:
                rprint(f"[bold red]Error:[/] No JSONL files found in: {source}")
                raise typer.Exit(1)

        rprint(f"[bold blue]זיכרון[/] - Fast Indexing: [bold]{len(jsonl_files)}[/] files")

        total_chunks = 0
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing files...", total=len(jsonl_files))

            for i, jsonl_file in enumerate(jsonl_files):
                proj_name = jsonl_file.parent.name if jsonl_file.parent != source else None

                # Parse, classify, and chunk each entry
                all_chunks = []
                for entry in parse_jsonl(jsonl_file):
                    classified = classify_content(entry)
                    if classified is not None:  # Skip noise entries
                        chunks = chunk_content(classified)
                        all_chunks.extend(chunks)

                if all_chunks:
                    # Index with progress callback
                    def progress_callback(embedded_count, total_embed):
                        pass  # Could update sub-progress here

                    indexed = index_chunks_to_sqlite(
                        all_chunks,
                        source_file=str(jsonl_file),
                        project=proj_name,
                        on_progress=progress_callback
                    )
                    total_chunks += indexed

                # Update progress
                elapsed = time.time() - start_time
                files_done = i + 1
                rate_per_min = (files_done / elapsed) * 60 if elapsed > 0 else 0
                chunks_per_min = (total_chunks / elapsed) * 60 if elapsed > 0 else 0

                progress.update(
                    task,
                    completed=files_done,
                    description=f"[cyan]{jsonl_file.name[:30]}[/] • {total_chunks:,} chunks • {rate_per_min:.1f} f/m • {chunks_per_min:.0f} c/m"
                )

        elapsed_total = time.time() - start_time
        rprint(f"\n[bold green]✓[/] Indexed [bold]{total_chunks:,}[/] chunks from [bold]{len(jsonl_files):,}[/] files in [bold]{elapsed_total/60:.1f}[/] minutes")

        # Sync enrichment stats to Supabase (best-effort)
        try:
            from ..pipeline.enrichment import _sync_stats_to_supabase, DEFAULT_DB_PATH
            from ..vector_store import VectorStore
            store = VectorStore(DEFAULT_DB_PATH)
            _sync_stats_to_supabase(store)
            store.close()
            rprint("[dim]Synced enrichment stats to Supabase[/]")
        except Exception:
            pass

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        raise typer.Exit(1)


@app.command("analyze-semantic")
def analyze_semantic(
    whatsapp_limit: int = typer.Option(
        5000, "--whatsapp-limit", "-w",
        help="Number of WhatsApp messages to analyze"
    ),
    claude_export: Path = typer.Option(
        None, "--claude-export", "-c",
        help="Path to Claude chat export JSON (optional)"
    ),
    output_dir: Path = typer.Option(
        Path.home() / ".golems-zikaron/style",
        "--output", "-o",
        help="Output directory for semantic style rules"
    ),
    min_cluster: int = typer.Option(
        10, "--min-cluster", "-m",
        help="Minimum messages per topic cluster"
    ),
) -> None:
    """Analyze communication style by topic using semantic clustering.

    Clusters your messages by topic (technical, casual, professional, etc.)
    and analyzes writing style patterns within each context. Generates
    context-aware rules for better cover letters and outreach.

    Requires: sentence-transformers, scikit-learn
    """
    try:
        from ..pipeline.extract_whatsapp import extract_whatsapp_messages
        from ..pipeline.analyze_communication import CommunicationAnalyzer

        rprint("[bold blue]זיכרון[/] - Semantic Style Analysis\n")

        analyzer = CommunicationAnalyzer()

        # Extract WhatsApp messages
        with console.status("[bold green]Extracting WhatsApp messages..."):
            try:
                messages = list(extract_whatsapp_messages(
                    limit=whatsapp_limit,
                    only_from_me=True,  # Only user's messages
                    exclude_groups=True
                ))
                analyzer.add_whatsapp_messages(messages)
                rprint(f"[green]✓[/] Extracted {len(messages)} WhatsApp messages")
            except FileNotFoundError as e:
                rprint("[yellow]Warning:[/] WhatsApp database not found")
                rprint(f"[dim]  {e}[/]")
            except Exception as e:
                rprint(f"[yellow]Warning:[/] Failed to extract WhatsApp: {e}")

        # Extract Claude chats if provided
        if claude_export and claude_export.exists():
            with console.status("[bold green]Extracting Claude conversations..."):
                try:
                    import json
                    with open(claude_export) as f:
                        data = json.load(f)
                    conversations = data if isinstance(data, list) else data.get("conversations", [])
                    analyzer.add_claude_conversations(conversations)
                    rprint(f"[green]✓[/] Extracted {len(conversations)} Claude conversations")
                except Exception as e:
                    rprint(f"[yellow]Warning:[/] Failed to extract Claude chats: {e}")
        else:
            rprint("[dim]No Claude export provided (use --claude-export)[/]")

        # Check if we have enough data
        if len(analyzer.user_messages) < min_cluster * 2:
            rprint(f"[bold red]Error:[/] Need at least {min_cluster * 2} messages, have {len(analyzer.user_messages)}")
            raise typer.Exit(1)

        rprint(f"\n[bold]Analyzing {len(analyzer.user_messages)} messages...[/]\n")

        # Run semantic analysis
        rules_path = analyzer.generate_semantic_rules(
            output_dir=output_dir,
            min_cluster_size=min_cluster,
        )

        if rules_path is None:
            rprint("[bold red]Error:[/] Semantic analysis failed")
            rprint("[dim]Install dependencies: pip install sentence-transformers scikit-learn[/]")
            raise typer.Exit(1)

        rprint("\n[bold green]✓[/] Semantic style rules generated!")
        rprint(f"  Markdown: [cyan]{rules_path}[/]")
        rprint(f"  JSON: [cyan]semantic-style-data.json[/]")

        # Show preview
        rprint("\n[bold]Preview:[/]\n")
        preview = rules_path.read_text()
        for line in preview.split('\n')[:30]:
            rprint(f"[dim]{line}[/]")
        if len(preview.split('\n')) > 30:
            rprint(f"[dim]... ({len(preview.split(chr(10))) - 30} more lines)[/]")

    except typer.Exit:
        raise
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command("brain-export")
def brain_export(
    output: str = typer.Option(
        None, "--output", "-o",
        help="Output directory (default: ~/.golems-brain)"
    ),
    project: str = typer.Option(
        None, "--project", "-p",
        help="Only include sessions from this project"
    ),
) -> None:
    """Generate brain graph (graph.json) for 3D visualization.

    Aggregates sessions → computes similarity → runs Leiden communities → UMAP 3D layout.
    Requires: pip install -e '.[brain]' (igraph, leidenalg, umap-learn)
    """
    try:
        from ..pipeline.brain_graph import generate_brain_graph
        import logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")

        path = generate_brain_graph(output_dir=output, project=project)
        rprint(f"[bold green]Brain graph exported to {path}[/]")

        import json
        meta_path = path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            rprint(f"  Nodes: {meta['node_count']}, Edges: {meta['edge_count']}")
            rprint(f"  Communities: {meta['community_counts']}")
    except ImportError as e:
        rprint(f"[bold red]Missing dependency:[/] {e}")
        rprint("[dim]Install brain extras: pip install -e '.[brain]'[/]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[bold red]Error:[/] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
