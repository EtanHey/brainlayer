"""Git Overlay Pipeline — Cross-reference sessions with git history.

Phase 8b: For each indexed session, extract git metadata (branch, commits,
PR number, files changed) and build file-interaction timelines.

Usage:
    from brainlayer.pipeline.git_overlay import run_git_overlay
    run_git_overlay(vector_store, project="-Users-username-Projects-myapp")
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def slug_to_path(slug: str) -> Optional[Path]:
    """Convert a project slug to a filesystem path.

    Example: '-Users-username-Projects-myapp' -> '/Users/username/Projects/myapp'
    """
    if not slug:
        return None
    # Replace leading '-' with '/', then '-' with '/'
    path_str = "/" + slug.lstrip("-").replace("-", "/")
    path = Path(path_str)
    if path.exists() and path.is_dir():
        return path
    return None


def slug_to_display(slug: str) -> str:
    """Convert project slug to display name.

    Example: '-Users-username-Projects-myapp' -> 'Projects/myapp'
    """
    return re.sub(r"^-Users-[^-]+-", "", slug).replace("-", "/")


def _git_cmd(cmd: List[str], cwd: Path, timeout: int = 10) -> Optional[str]:
    """Run a git command, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git"] + cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def is_git_repo(path: Path) -> bool:
    """Check if path is inside a git repository."""
    return _git_cmd(["rev-parse", "--git-dir"], path) is not None


def get_session_timestamps(jsonl_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Extract first and last timestamps from a session JSONL file."""
    first_ts = None
    last_ts = None
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = obj.get("timestamp")
                    if ts:
                        if first_ts is None:
                            first_ts = ts
                        last_ts = ts
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        pass
    return first_ts, last_ts


def extract_file_actions(jsonl_path: Path, session_id: str) -> List[Dict[str, Any]]:
    """Extract file interaction records from a session JSONL.

    Looks for tool_use blocks in assistant messages with file paths.
    """
    interactions: List[Dict[str, Any]] = []
    tool_to_action = {
        "Read": "read",
        "Edit": "edit",
        "Write": "write",
        "Glob": "search",
        "Grep": "search",
    }

    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("type") != "assistant":
                        continue
                    msg = obj.get("message", {})
                    content = msg.get("content", [])
                    timestamp = obj.get("timestamp")
                    if not isinstance(content, list):
                        continue
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") != "tool_use":
                            continue
                        tool_name = block.get("name", "")
                        action = tool_to_action.get(tool_name)
                        if not action:
                            continue
                        inp = block.get("input", {})
                        file_path = inp.get("file_path") or inp.get("path")
                        if file_path:
                            interactions.append(
                                {
                                    "file_path": file_path,
                                    "timestamp": timestamp,
                                    "session_id": session_id,
                                    "action": action,
                                }
                            )
                except json.JSONDecodeError:
                    continue
    except (OSError, IOError):
        pass
    return interactions


def get_git_context(
    repo_path: Path,
    start_ts: str,
    end_ts: str,
) -> Dict[str, Any]:
    """Get git context for a time range in a repo.

    Returns branch, commits, PR number, and files changed.
    """
    context: Dict[str, Any] = {
        "branch": None,
        "pr_number": None,
        "commit_shas": [],
        "files_changed": [],
    }

    # Current branch (at time of overlay, not session time)
    branch = _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    context["branch"] = branch

    # Commits in the time range
    log_output = _git_cmd(
        ["log", f"--after={start_ts}", f"--before={end_ts}", "--format=%H %s", "--all"],
        repo_path,
    )
    if log_output:
        for line in log_output.strip().split("\n"):
            if line.strip():
                parts = line.split(" ", 1)
                context["commit_shas"].append(parts[0])

    # Files changed in those commits
    if context["commit_shas"]:
        first_sha = context["commit_shas"][-1]  # oldest
        last_sha = context["commit_shas"][0]  # newest
        diff_output = _git_cmd(
            ["diff", "--name-only", f"{first_sha}~1", last_sha],
            repo_path,
        )
        if diff_output:
            context["files_changed"] = [f for f in diff_output.strip().split("\n") if f.strip()]

    # Try to extract PR number from branch name
    if branch and branch not in ("master", "main", "HEAD"):
        # Try gh CLI for PR number
        try:
            result = subprocess.run(
                ["gh", "pr", "list", "--head", branch, "--json", "number", "--limit", "1"],
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                prs = json.loads(result.stdout)
                if prs:
                    context["pr_number"] = prs[0]["number"]
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

    return context


def run_git_overlay(
    vector_store: Any,
    project: Optional[str] = None,
    force: bool = False,
    max_sessions: int = 0,
) -> Dict[str, int]:
    """Run git overlay on indexed sessions.

    Args:
        vector_store: VectorStore instance
        project: Filter to specific project slug
        force: Re-process sessions that already have context
        max_sessions: Limit number of sessions (0 = all)

    Returns:
        Dict with counts: sessions_processed, file_interactions_added
    """
    projects_dir = Path.home() / ".claude" / "projects"
    if not projects_dir.exists():
        logger.warning("No projects directory found at %s", projects_dir)
        return {"sessions_processed": 0, "file_interactions_added": 0}

    stats = {"sessions_processed": 0, "file_interactions_added": 0}
    processed = 0

    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        if project and proj_dir.name != project:
            continue

        # Resolve project path
        repo_path = slug_to_path(proj_dir.name)
        display_name = slug_to_display(proj_dir.name)
        has_git = repo_path and is_git_repo(repo_path)

        for jsonl_file in sorted(proj_dir.glob("*.jsonl")):
            if max_sessions and processed >= max_sessions:
                break

            session_id = jsonl_file.stem

            # Skip if already processed (unless force)
            if not force:
                existing = vector_store.get_session_context(session_id)
                if existing:
                    continue

            # Get timestamps
            start_ts, end_ts = get_session_timestamps(jsonl_file)
            if not start_ts:
                continue

            # Extract file interactions from JSONL
            file_actions = extract_file_actions(jsonl_file, session_id)
            for fa in file_actions:
                fa["project"] = display_name

            # Get git context if repo exists
            git_ctx = {}
            if has_git and repo_path:
                git_ctx = get_git_context(repo_path, start_ts, end_ts)

            # Store session context
            vector_store.store_session_context(
                session_id=session_id,
                project=display_name,
                branch=git_ctx.get("branch"),
                pr_number=git_ctx.get("pr_number"),
                commit_shas=git_ctx.get("commit_shas", []),
                files_changed=git_ctx.get("files_changed", []),
                started_at=start_ts,
                ended_at=end_ts,
            )

            # Store file interactions
            count = vector_store.store_file_interactions(file_actions)
            stats["file_interactions_added"] += count
            stats["sessions_processed"] += 1
            processed += 1

            logger.info(
                "Session %s: %s files, %d interactions, branch=%s",
                session_id[:8],
                len(git_ctx.get("files_changed", [])),
                count,
                git_ctx.get("branch", "n/a"),
            )

        if max_sessions and processed >= max_sessions:
            break

    return stats
