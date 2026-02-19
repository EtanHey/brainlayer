"""Main dashboard application using Rich TUI."""

import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
from rich import box

from ..vector_store import VectorStore
from .search import HybridSearchEngine
from .views import HomeView, MemoryView

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"


class DashboardApp:
    """Interactive dashboard for brainlayer memory management."""

    def __init__(self):
        self.console = Console()
        self.current_view = "home"
        self.search_engine = HybridSearchEngine()
        self.vector_store = None
        self.stats = {}

    def setup_database(self):
        """Initialize database connection using sqlite-vec."""
        try:
            self.vector_store = VectorStore(DEFAULT_DB_PATH)
            self.stats = self.vector_store.get_stats()
        except Exception as e:
            self.console.print(f"[red]Database error: {e}[/]")
            self.stats = {"total_chunks": 0, "projects": [], "content_types": []}
    
    def create_header(self) -> Panel:
        """Create dashboard header."""
        title = Text("זיכרון Dashboard", style="bold blue")
        subtitle = Text(f"Memory: {self.stats.get('total_chunks', 0):,} chunks", style="dim")
        
        nav_items = []
        views = [("home", "Home"), ("memory", "Memory"), ("jobs", "Jobs"), ("agents", "Agents")]
        
        for view_key, view_name in views:
            style = "bold white on blue" if view_key == self.current_view else "dim"
            nav_items.append(Text(f" {view_name} ", style=style))
        
        nav = Text(" | ").join(nav_items)
        
        header_content = Align.center(
            Text.assemble(title, "\n", subtitle, "\n\n", nav)
        )
        
        return Panel(header_content, box=box.ROUNDED, style="blue")
    
    def create_footer(self) -> Panel:
        """Create dashboard footer with controls."""
        controls = [
            "[bold]h[/] Home",
            "[bold]m[/] Memory", 
            "[bold]j[/] Jobs",
            "[bold]g[/] Agents",
            "[bold]q[/] Quit"
        ]
        
        footer_text = " • ".join(controls)
        return Panel(
            Align.center(footer_text),
            box=box.ROUNDED,
            style="dim"
        )
    
    def run_home_view(self) -> Panel:
        """Render home view with statistics."""
        view = HomeView(self.stats)
        return view.render()
    
    def run_memory_view(self) -> Panel:
        """Render memory view with search interface."""
        view = MemoryView(self.search_engine, self.vector_store, self.stats)
        return view.render()
    
    def run_jobs_view(self) -> Panel:
        """Render jobs view (placeholder)."""
        content = Text("Jobs view - Coming in Phase 3", style="dim italic")
        return Panel(
            Align.center(content),
            title="Jobs",
            box=box.ROUNDED
        )
    
    def run_agents_view(self) -> Panel:
        """Render agents view (placeholder)."""
        content = Text("Agents view - Coming in Phase 3", style="dim italic")
        return Panel(
            Align.center(content),
            title="Agents",
            box=box.ROUNDED
        )
    
    def render_dashboard(self) -> Layout:
        """Render the complete dashboard layout."""
        layout = Layout()
        
        layout.split_column(
            Layout(self.create_header(), name="header", size=7),
            Layout(name="main"),
            Layout(self.create_footer(), name="footer", size=3)
        )
        
        # Render current view
        if self.current_view == "home":
            main_content = self.run_home_view()
        elif self.current_view == "memory":
            main_content = self.run_memory_view()
        elif self.current_view == "jobs":
            main_content = self.run_jobs_view()
        elif self.current_view == "agents":
            main_content = self.run_agents_view()
        else:
            main_content = self.run_home_view()
        
        layout["main"].update(main_content)
        return layout
    
    def handle_input(self, key: str) -> bool:
        """Handle keyboard input. Returns True to continue, False to quit."""
        if key.lower() == 'q':
            return False
        elif key.lower() == 'h':
            self.current_view = "home"
        elif key.lower() == 'm':
            self.current_view = "memory"
        elif key.lower() == 'j':
            self.current_view = "jobs"
        elif key.lower() == 'g':
            self.current_view = "agents"
        
        return True
    
    def run(self):
        """Run the interactive dashboard."""
        self.console.print("[bold blue]Starting זיכרון Dashboard...[/]")
        
        # Setup database
        with self.console.status("[bold green]Connecting to database..."):
            self.setup_database()
        
        # Simple non-interactive version for now
        # In a full implementation, this would use keyboard input handling
        try:
            while True:
                self.console.clear()
                layout = self.render_dashboard()
                self.console.print(layout)
                
                # Simple input handling
                user_input = input("\nPress key (h/m/j/g/q): ").strip().lower()
                
                if not self.handle_input(user_input):
                    break
                    
        except (KeyboardInterrupt, EOFError):
            pass
        
        self.console.print("\n[dim]Dashboard closed.[/]")
