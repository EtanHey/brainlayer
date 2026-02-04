"""Dashboard views for Home and Memory interfaces."""

from typing import Dict, Any, Optional, List
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.console import Group
from rich.align import Align
from rich import box

from .search import HybridSearchEngine


class HomeView:
    """Home dashboard view showing system statistics."""
    
    def __init__(self, stats: Dict[str, Any]):
        self.stats = stats
    
    def render(self) -> Panel:
        """Render the home view with statistics."""
        # Create statistics table
        stats_table = Table(show_header=False, box=box.SIMPLE)
        stats_table.add_column("Metric", style="bold")
        stats_table.add_column("Value", style="cyan")
        
        total_chunks = self.stats.get("total_chunks", 0)
        projects = self.stats.get("projects", [])
        content_types = self.stats.get("content_types", [])
        
        stats_table.add_row("Total Chunks", f"{total_chunks:,}")
        stats_table.add_row("Projects", str(len(projects)))
        stats_table.add_row("Content Types", str(len(content_types)))
        
        # Create projects list
        projects_text = Text()
        if projects:
            for i, project in enumerate(projects[:5]):  # Show top 5
                if i > 0:
                    projects_text.append(" • ")
                projects_text.append(project, style="green")
            if len(projects) > 5:
                projects_text.append(f" • +{len(projects) - 5} more", style="dim")
        else:
            projects_text.append("No projects indexed", style="dim")
        
        # Create content types list
        types_text = Text()
        if content_types:
            for i, content_type in enumerate(content_types):
                if i > 0:
                    types_text.append(" • ")
                types_text.append(content_type, style="yellow")
        else:
            types_text.append("No content types", style="dim")
        
        # Combine into columns
        left_panel = Panel(stats_table, title="Statistics", box=box.ROUNDED)
        
        right_content = Text.assemble(
            "Projects:\n", projects_text, "\n\n",
            "Content Types:\n", types_text
        )
        right_panel = Panel(right_content, title="Collections", box=box.ROUNDED)
        
        columns = Columns([left_panel, right_panel], equal=True)
        
        # Status message
        if total_chunks == 0:
            status_msg = Text("No data indexed. Run 'zikaron index' to get started.", style="yellow")
        else:
            status_msg = Text(f"Ready to search {total_chunks:,} chunks across {len(projects)} projects", style="green")
        
        status_panel = Panel(Align.center(status_msg), box=box.ROUNDED, style="dim")
        
        # Combine all elements using Group (Text.assemble only works with text)
        main_content = Group(columns, Text(""), status_panel)

        return Panel(
            main_content,
            title="Home",
            box=box.ROUNDED
        )


class MemoryView:
    """Memory view with search interface and filtering."""

    def __init__(self, search_engine: HybridSearchEngine, vector_store, stats: Dict[str, Any]):
        self.search_engine = search_engine
        self.vector_store = vector_store  # sqlite-vec VectorStore (or None)
        self.stats = stats
        self.current_query = ""
        self.current_filter = None
        self.search_results = []
    
    def render(self) -> Panel:
        """Render the memory view with search interface."""
        # Search interface
        search_panel = self._render_search_interface()
        
        # Filters
        filters_panel = self._render_filters()
        
        # Results
        results_panel = self._render_results()
        
        # Combine into layout using Group (Text.assemble only works with text)
        top_row = Columns([search_panel, filters_panel], equal=True)

        main_content = Group(top_row, Text(""), results_panel)

        return Panel(
            main_content,
            title="Memory Search",
            box=box.ROUNDED
        )
    
    def _render_search_interface(self) -> Panel:
        """Render search input interface."""
        content = Text.assemble(
            "Search Query:\n",
            Text("Enter search terms to find relevant chunks", style="dim"), "\n\n",
            "Search Type: ", Text("Hybrid (BM25 + Semantic)", style="green"), "\n",
            "Status: ", Text("Ready", style="cyan")
        )
        
        return Panel(content, title="Search", box=box.ROUNDED)
    
    def _render_filters(self) -> Panel:
        """Render collection filters."""
        projects = self.stats.get("projects", [])
        content_types = self.stats.get("content_types", [])
        
        content = Text("Available Filters:\n\n")
        
        # Projects filter
        content.append("Projects:\n", style="bold")
        if projects:
            for project in projects[:3]:  # Show top 3
                content.append(f"• {project}\n", style="green")
            if len(projects) > 3:
                content.append(f"• +{len(projects) - 3} more\n", style="dim")
        else:
            content.append("• No projects\n", style="dim")
        
        content.append("\n")
        
        # Content types filter
        content.append("Content Types:\n", style="bold")
        if content_types:
            for ctype in content_types:
                content.append(f"• {ctype}\n", style="yellow")
        else:
            content.append("• No types\n", style="dim")
        
        return Panel(content, title="Filters", box=box.ROUNDED)
    
    def _render_results(self) -> Panel:
        """Render search results."""
        if not self.search_results:
            content = Align.center(
                Text("No search performed yet.\nEnter a query to see results.", style="dim italic")
            )
        else:
            # Create results table
            results_table = Table(show_header=True, box=box.SIMPLE)
            results_table.add_column("Score", width=8)
            results_table.add_column("Project", width=15)
            results_table.add_column("Type", width=12)
            results_table.add_column("Content", min_width=40)
            
            for i, result in enumerate(self.search_results[:5]):  # Show top 5
                score = f"{result.get('score', 0):.3f}"
                project = result.get('project', 'unknown')[:14]
                content_type = result.get('content_type', 'unknown')[:11]
                content_preview = result.get('content', '')[:80] + "..." if len(result.get('content', '')) > 80 else result.get('content', '')
                
                results_table.add_row(score, project, content_type, content_preview)
            
            content = results_table
        
        return Panel(content, title="Results", box=box.ROUNDED)
    
    def search(self, query: str, project_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform search and update results."""
        if not query.strip():
            self.search_results = []
            return []

        try:
            results = self.search_engine.search(
                self.vector_store,
                query,
                n_results=10,
                project_filter=project_filter
            )
            
            # Convert to display format
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            self.search_results = []
            for doc, meta, distance in zip(documents, metadatas, distances):
                self.search_results.append({
                    "content": doc,
                    "project": meta.get("project", "unknown"),
                    "content_type": meta.get("content_type", "unknown"),
                    "score": 1.0 - distance if distance is not None else 1.0
                })
            
            return self.search_results
            
        except Exception as e:
            print(f"Search error: {e}")
            self.search_results = []
            return []
