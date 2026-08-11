"""Lightweight Spotlight exclusion layout preflight."""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from .paths import SPOTLIGHT_EXCLUSION_MARKER, get_canonical_db_path, resolve_db_path

_DATA_CHILDREN = (
    "backups",
    "chromadb",
    "chromadb.backup",
    "enrichment-scratch",
    "experiments",
    "jsonl-backups",
    "logs",
    "prompts",
    "style",
    "storage",
)
_RUNTIME_CHILDREN = ("logs", "quarantine", "queue")
_UNSUPPORTED_RUNTIME_PATH_OVERRIDES = (
    "BRAINLAYER_BACKUP_LOG_PATH",
    "BRAINLAYER_BACKUP_STAGING_DIR",
    "BRAINLAYER_DRAIN_HEALTH_PATH",
    "BRAINLAYER_DRAIN_LOG_PATH",
    "BRAINLAYER_ENRICH_COST_DIR",
    "BRAINLAYER_JSONL_BACKUP_LOG_PATH",
    "BRAINLAYER_JSONL_BACKUP_STAGING_DIR",
    "BRAINLAYER_JSONL_BACKUP_STATE_PATH",
    "BRAINLAYER_LOG_DIR",
    "BRAINLAYER_OFFSETS_PATH",
    "BRAINLAYER_P0_COUNTER_DIR",
    "BRAINLAYER_QUEUE_DIR",
    "BRAINLAYER_WATCHER_HEALTH_PATH",
    "BRAINLAYER_WATCHER_OFFSETS_PATH",
    "BRAINLAYER_WATCHER_QUARANTINE_DIR",
    "BRAINLAYER_WRITER_HEARTBEAT_DIR",
    "BRAINLAYER_WRITER_TELEMETRY_PATH",
)


def _data_root_from_db_path(db_path: Path) -> Path:
    expanded = db_path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError(f"BRAINLAYER_DB must not be a symbolic link: {expanded}")
    return expanded.parent


def _reject_runtime_path_overrides(env_file: Path) -> None:
    from .config import configured_brainlayer_env_value

    for name in _UNSUPPORTED_RUNTIME_PATH_OVERRIDES:
        if configured_brainlayer_env_value(name, env_file) not in (None, ""):
            raise RuntimeError(
                f"{name} is not supported by the Spotlight preflight; remove the override before installation"
            )


def _validate_override_data_root(
    active_root: Path,
    canonical_root: Path,
    *,
    home: Path,
    ismount: Callable[[Path], bool],
) -> None:
    raw_active = active_root.expanduser()
    if not raw_active.is_absolute():
        raise RuntimeError(f"BRAINLAYER_DB must be an absolute path; got: {raw_active}")
    for component in (*reversed(raw_active.parents), raw_active):
        if component.is_symlink():
            raise RuntimeError(f"BRAINLAYER_DB parent must not contain a symbolic link: {raw_active}")
    active = raw_active.resolve(strict=False)
    canonical = canonical_root.expanduser().resolve(strict=False)
    if active == canonical:
        return
    unsafe_location = active == home.resolve() or active == Path(active.anchor) or ismount(active)
    if unsafe_location or active in canonical.parents or "brainlayer" not in active.name.casefold():
        raise RuntimeError(f"BRAINLAYER_DB must be inside a dedicated BrainLayer directory; unsafe parent: {active}")


def _validate_disjoint_roots(roots: Tuple[Path, ...]) -> None:
    resolved = tuple(root.expanduser().resolve(strict=False) for root in roots)
    for index, root in enumerate(resolved):
        for other in resolved[:index]:
            if root == other or root in other.parents or other in root.parents:
                raise RuntimeError(f"Spotlight exclusion roots must not overlap: {other} and {root}")


def _ensure_spotlight_excluded_root(root: Path, children: Tuple[str, ...] = ()) -> Path:
    resolved = root.expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / SPOTLIGHT_EXCLUSION_MARKER
    if not marker.is_file() and next(resolved.iterdir(), None) is not None:
        raise RuntimeError(f"existing runtime tree {resolved} requires the Spotlight exclusion migration runbook")
    marker.touch(exist_ok=True)
    for child in children:
        (resolved / child).mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_spotlight_excluded_layout(
    *,
    data_dir: Optional[Path] = None,
    env_file: Optional[Path] = None,
    runtime_dir: Optional[Path] = None,
    launchd_log_dir: Optional[Path] = None,
    counter_dir: Optional[Path] = None,
    resolve_db_path_fn: Callable[[], Path] = resolve_db_path,
    get_canonical_db_path_fn: Callable[[], Path] = get_canonical_db_path,
    home_fn: Callable[[], Path] = Path.home,
    ismount_fn: Callable[[Path], bool] = os.path.ismount,
) -> Tuple[Path, ...]:
    """Create marker-backed roots for every high-churn BrainLayer runtime path."""
    home = home_fn()
    if env_file is not None:
        _reject_runtime_path_overrides(env_file)
    if data_dir is not None:
        data_roots = (data_dir,)
    else:
        canonical_data_root = get_canonical_db_path_fn().parent
        if env_file is not None:
            from .config import configured_brainlayer_env_value

            configured_db = configured_brainlayer_env_value("BRAINLAYER_DB", env_file)
            active_data_root = _data_root_from_db_path(Path(configured_db)) if configured_db else canonical_data_root
        else:
            active_data_root = _data_root_from_db_path(resolve_db_path_fn())
        _validate_override_data_root(active_data_root, canonical_data_root, home=home, ismount=ismount_fn)
        data_roots = tuple(
            dict.fromkeys(root.expanduser().resolve(strict=False) for root in (active_data_root, canonical_data_root))
        )
    requested_roots = (
        *data_roots,
        runtime_dir or home / ".brainlayer",
        launchd_log_dir or home / "Library" / "Logs" / "brainlayer",
        counter_dir or home / ".brainlayer-p0-counter",
    )
    _validate_disjoint_roots(requested_roots)
    resolved_roots = tuple(root.expanduser() for root in requested_roots)
    for root in resolved_roots:
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise RuntimeError(f"runtime root {root} must be a directory")
        marker = root / SPOTLIGHT_EXCLUSION_MARKER
        if root.is_dir() and not marker.is_file() and next(root.iterdir(), None) is not None:
            raise RuntimeError(f"existing runtime tree {root} requires the Spotlight exclusion migration runbook")

    data_root_count = len(data_roots)
    return (
        *(_ensure_spotlight_excluded_root(root, _DATA_CHILDREN) for root in resolved_roots[:data_root_count]),
        _ensure_spotlight_excluded_root(resolved_roots[data_root_count], _RUNTIME_CHILDREN),
        _ensure_spotlight_excluded_root(resolved_roots[data_root_count + 1]),
        _ensure_spotlight_excluded_root(resolved_roots[data_root_count + 2]),
    )
