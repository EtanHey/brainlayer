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


def _validate_override_data_root(
    active_root: Path,
    canonical_root: Path,
    *,
    home: Path,
    ismount: Callable[[Path], bool],
) -> None:
    active = active_root.expanduser().resolve(strict=False)
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
    if data_dir is not None:
        data_roots = (data_dir,)
    else:
        active_data_root = resolve_db_path_fn().parent
        canonical_data_root = get_canonical_db_path_fn().parent
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
