"""Centralized artifact storage for BrainLayer.

All docs.local, plans, research, and logs live here — not in project repos.
Files persist across worktrees, branches, and repo deletions.
To make something public, copy from storage into the git repo.
"""

from pathlib import Path
from typing import Optional

DEFAULT_BASE_DIR = Path.home() / ".local" / "share" / "brainlayer" / "storage"


class BrainStorage:
    """Manages centralized file storage organized by project."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _safe_path(self, *parts: str) -> Path:
        """Resolve a path and verify it stays within base_dir."""
        resolved = (self.base_dir / Path(*parts)).resolve()
        base_resolved = self.base_dir.resolve()
        if not str(resolved).startswith(str(base_resolved) + "/") and resolved != base_resolved:
            raise ValueError(f"Path escapes storage root: {'/'.join(parts)}")
        return resolved

    def get_project_dir(self, project: str) -> Path:
        """Get or create a project's storage directory with standard subdirs."""
        project_dir = self._safe_path(project)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "docs.local").mkdir(exist_ok=True)
        (project_dir / "plans").mkdir(exist_ok=True)
        return project_dir

    def store(self, project: str, relative_path: str, content: str) -> Path:
        """Store a file in a project's storage. Creates subdirs as needed."""
        self.get_project_dir(project)
        full_path = self._safe_path(project, relative_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return full_path

    def read(self, project: str, relative_path: str) -> str:
        """Read a file from a project's storage."""
        full_path = self._safe_path(project, relative_path)
        return full_path.read_text(encoding="utf-8")

    def list_projects(self) -> list[str]:
        """List all projects that have storage directories."""
        return sorted(d.name for d in self.base_dir.iterdir() if d.is_dir() and not d.name.startswith("."))

    def list_files(self, project: str, subdir: str = "") -> list[Path]:
        """List all files in a project's storage (or a subdirectory)."""
        if subdir:
            project_dir = self._safe_path(project, subdir)
        else:
            project_dir = self._safe_path(project)
        if not project_dir.exists():
            return []
        return sorted(f for f in project_dir.rglob("*") if f.is_file())

    def exists(self, project: str, relative_path: str) -> bool:
        """Check if a file exists in storage."""
        return self._safe_path(project, relative_path).exists()
