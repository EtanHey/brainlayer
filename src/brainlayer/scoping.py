"""Project scoping — auto-detect project from CWD using scopes.yaml config.

Reads ~/.config/brainlayer/scopes.yaml to map directory prefixes to project names.
Falls back to parsing CWD if no config exists.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCOPES_PATH = Path.home() / ".config" / "brainlayer" / "scopes.yaml"

# Cache parsed config
_cached_scopes: Optional[dict] = None
_cached_mtime: float = 0.0
_VALID_CONSUMERS = frozenset({"orchestrator", "lead", "worker", "coach"})
_DEFAULT_CONSUMER = "worker"
_COACH_PROJECT = "personal"


def _load_scopes() -> dict:
    """Load scopes.yaml, returning parsed dict. Caches by mtime."""
    global _cached_scopes, _cached_mtime

    if not _SCOPES_PATH.exists():
        return {}

    mtime = _SCOPES_PATH.stat().st_mtime
    if _cached_scopes is not None and mtime == _cached_mtime:
        return _cached_scopes

    try:
        import yaml

        with open(_SCOPES_PATH) as f:
            data = yaml.safe_load(f) or {}
        _cached_scopes = data
        _cached_mtime = mtime
        return data
    except ImportError:
        # Fall back to simple line parsing if PyYAML not installed
        return _parse_scopes_simple(_SCOPES_PATH)
    except Exception as e:
        logger.debug("Failed to load scopes config from %s: %s", _SCOPES_PATH, e)
        return {}


def _parse_scopes_simple(path: Path) -> dict:
    """Minimal scopes.yaml parser without PyYAML dependency."""
    global _cached_scopes, _cached_mtime
    scopes: dict = {"scopes": {}, "default": "all"}

    try:
        lines = path.read_text().splitlines()
        in_scopes = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped == "scopes:":
                in_scopes = True
                continue
            if stripped.startswith("default:"):
                scopes["default"] = stripped.split(":", 1)[1].strip().strip("\"'")
                in_scopes = False
                continue
            if in_scopes and ":" in stripped:
                # Parse "  ~/Gits/golems: golems" or "  ~/Gits/golems: \"golems\""
                key, _, val = stripped.partition(":")
                key = key.strip().strip("\"'")
                val = val.strip().strip("\"'")
                # Expand ~ in the key
                key = str(Path(key).expanduser())
                scopes["scopes"][key] = val

        _cached_scopes = scopes
        _cached_mtime = path.stat().st_mtime
        return scopes
    except Exception:
        return {"scopes": {}, "default": "all"}


def resolve_project_scope() -> Optional[str]:
    """Resolve the current project from CWD using scopes.yaml.

    Returns:
        Project name if matched, None if default is "all" or no match.
    """
    config = _load_scopes()
    scope_map = config.get("scopes", {})
    default = config.get("default", "all")

    if not scope_map:
        # No config — try CWD heuristic
        return _cwd_heuristic()

    cwd = os.getcwd()

    # Check each scope prefix (longest match first)
    matches = []
    for prefix, project in scope_map.items():
        expanded = str(Path(prefix).expanduser())
        if cwd == expanded or cwd.startswith(expanded + os.sep):
            matches.append((len(expanded), project))

    if matches:
        # Return longest (most specific) match
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    return None if default == "all" else default


@dataclass(frozen=True)
class ConsumerScope:
    """Retrieval-time visibility policy for a BrainLayer consumer role."""

    role: str
    project_filter: Optional[str]
    project_filters: tuple[str, ...] = ()
    source_filter: Optional[str] = None
    include_checkpoints: bool = False
    allow_null_project: bool = False
    deny_all: bool = False

    @classmethod
    def for_worker(
        cls,
        project: Optional[str],
        *,
        source_filter: Optional[str] = None,
        include_checkpoints: bool = False,
    ) -> "ConsumerScope":
        return cls(
            role="worker",
            project_filter=project,
            project_filters=_worker_project_filters(project),
            source_filter=source_filter,
            include_checkpoints=include_checkpoints,
            allow_null_project=False,
            deny_all=project is None,
        )

    @classmethod
    def for_lead(
        cls,
        project: Optional[str],
        *,
        source_filter: Optional[str] = None,
        include_checkpoints: bool = False,
    ) -> "ConsumerScope":
        root_project = _main_project_for(project)
        return cls(
            role="lead",
            project_filter=root_project,
            project_filters=_lead_project_filters(root_project),
            source_filter=source_filter,
            include_checkpoints=include_checkpoints,
            allow_null_project=False,
            deny_all=root_project is None,
        )

    @classmethod
    def for_orchestrator(
        cls,
        *,
        source_filter: Optional[str] = None,
        include_checkpoints: bool = False,
    ) -> "ConsumerScope":
        return cls(
            role="orchestrator",
            project_filter=None,
            source_filter=source_filter,
            include_checkpoints=include_checkpoints,
            allow_null_project=True,
            deny_all=False,
        )

    @classmethod
    def for_coach(
        cls,
        *,
        source_filter: Optional[str] = None,
        include_checkpoints: bool = True,
    ) -> "ConsumerScope":
        return cls(
            role="coach",
            project_filter=_COACH_PROJECT,
            source_filter=source_filter,
            include_checkpoints=include_checkpoints,
            allow_null_project=True,
            deny_all=False,
        )

    def cache_key(self) -> tuple:
        return (
            self.role,
            self.project_filter,
            self.project_filters,
            self.source_filter,
            self.include_checkpoints,
            self.allow_null_project,
            self.deny_all,
        )


def resolve_consumer_role(consumer: Optional[str] = None) -> str:
    """Resolve consumer role from explicit value or BRAINLAYER_CONSUMER.

    Missing or invalid values fail closed to worker, the most restrictive role.
    """
    raw = consumer if consumer is not None else os.environ.get("BRAINLAYER_CONSUMER")
    role = (raw or _DEFAULT_CONSUMER).strip().casefold()
    if role not in _VALID_CONSUMERS:
        logger.warning("Invalid BRAINLAYER_CONSUMER=%r; defaulting to worker", raw)
        return _DEFAULT_CONSUMER
    return role


def resolve_consumer_scope(
    *,
    project: Optional[str] = None,
    consumer: Optional[str] = None,
    source_filter: Optional[str] = None,
    include_checkpoints: bool = False,
) -> ConsumerScope:
    """Resolve the retrieval-time policy for the current BrainLayer consumer."""
    role = resolve_consumer_role(consumer)
    if role == "orchestrator":
        return ConsumerScope.for_orchestrator(
            source_filter=source_filter,
            include_checkpoints=include_checkpoints,
        )
    if role == "lead":
        return ConsumerScope.for_lead(
            project,
            source_filter=source_filter,
            include_checkpoints=include_checkpoints,
        )
    if role == "coach":
        return ConsumerScope.for_coach(source_filter=source_filter, include_checkpoints=True)
    return ConsumerScope.for_worker(
        project,
        source_filter=source_filter,
        include_checkpoints=include_checkpoints,
    )


def _worktree_map(config: Optional[dict] = None) -> dict[str, str]:
    """Return configured worktree-project -> main-project mappings."""
    raw = (config or _load_scopes()).get("worktrees", {})
    if not isinstance(raw, dict):
        return {}
    result: dict[str, str] = {}
    for worktree_project, main_project in raw.items():
        if isinstance(worktree_project, str) and isinstance(main_project, str):
            worktree = worktree_project.strip()
            main = main_project.strip()
            if worktree and main:
                result[worktree] = main
    return result


def _main_project_for(project: Optional[str], config: Optional[dict] = None) -> Optional[str]:
    if project is None:
        return None
    return _worktree_map(config).get(project, project)


def _worktrees_for_main(main_project: Optional[str], config: Optional[dict] = None) -> tuple[str, ...]:
    if main_project is None:
        return ()
    mapping = _worktree_map(config)
    return tuple(sorted(worktree for worktree, main in mapping.items() if main == main_project))


def _parallel_projects_for_lead(main_project: Optional[str], config: Optional[dict] = None) -> tuple[str, ...]:
    if main_project is None:
        return ()
    cfg = config or _load_scopes()
    raw = cfg.get("lead_parallel_projects", cfg.get("parallel_repos", {}))
    if not isinstance(raw, dict):
        return ()
    configured = raw.get(main_project, ())
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, (list, tuple)):
        return ()
    projects: list[str] = []
    for project in configured:
        if isinstance(project, str) and project.strip():
            projects.append(_main_project_for(project.strip(), cfg) or project.strip())
    return tuple(projects)


def _dedupe_projects(projects: list[str | None]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for project in projects:
        if project is None or project in seen:
            continue
        seen.add(project)
        deduped.append(project)
    return tuple(deduped)


def _worker_project_filters(project: Optional[str]) -> tuple[str, ...]:
    if project is None:
        return ()
    main_project = _main_project_for(project)
    if main_project and main_project != project:
        return _dedupe_projects([project, main_project])
    return (project,)


def _lead_project_filters(main_project: Optional[str]) -> tuple[str, ...]:
    if main_project is None:
        return ()
    projects: list[str | None] = [main_project, *_worktrees_for_main(main_project)]
    for parallel_project in _parallel_projects_for_lead(main_project):
        projects.extend([parallel_project, *_worktrees_for_main(parallel_project)])
    return _dedupe_projects(projects)


def _cwd_heuristic() -> Optional[str]:
    """Fallback: extract project name from CWD if it looks like ~/Gits/<project>."""
    cwd = os.getcwd()
    home = str(Path.home())

    # Check ~/Gits/<project> pattern
    gits_dir = os.path.join(home, "Gits")
    if cwd.startswith(gits_dir + os.sep):
        relative = cwd[len(gits_dir) + 1 :]
        # First segment is the project
        project = relative.split(os.sep)[0]
        if project:
            return project

    return None
