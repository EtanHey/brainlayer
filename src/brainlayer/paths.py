"""Centralized data paths for BrainLayer.

Resolution order:
  1. BRAINLAYER_DB env var (full path to .db file)
  2. ~/.local/share/brainlayer/brainlayer.db (canonical path)
"""

import os
from pathlib import Path

_CANONICAL_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"
SPOTLIGHT_EXCLUSION_MARKER = ".metadata_never_index"


def is_spotlight_excluded(path: Path) -> bool:
    """Return whether *path* is beneath a marker-backed Spotlight exclusion."""
    resolved = path.expanduser().resolve(strict=False)
    return any((directory / SPOTLIGHT_EXCLUSION_MARKER).is_file() for directory in (resolved, *resolved.parents))


def _guard_test_runtime_path(path: Path, *, source: str) -> Path:
    """Fail closed when a pytest unit test resolves a production runtime path."""
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        return path

    provenance = os.environ.get("BRAINLAYER_TEST_PATH_PROVENANCE")
    if provenance == "live":
        return path
    if provenance != "pytest":
        raise RuntimeError(f"{source} resolved during pytest without test path provenance")

    protected_home_value = os.environ.get("BRAINLAYER_TEST_PROTECTED_HOME")
    if not protected_home_value:
        raise RuntimeError(f"{source} resolved during pytest without a protected-home provenance guard")

    resolved = path.expanduser().resolve(strict=False)
    protected_home = Path(protected_home_value).expanduser().resolve(strict=False)
    protected_roots = (
        protected_home / ".brainlayer",
        protected_home / ".local" / "share" / "brainlayer",
    )
    if any(resolved == root or root in resolved.parents for root in protected_roots):
        raise RuntimeError(f"{source} resolved production BrainLayer path during pytest: {resolved}")
    return path


def resolve_db_path() -> Path:
    """Resolve the BrainLayer database path without creating its parent."""
    env = os.environ.get("BRAINLAYER_DB")
    if env:
        return _guard_test_runtime_path(Path(env), source="BRAINLAYER_DB")

    return get_canonical_db_path()


def get_canonical_db_path() -> Path:
    """Return the canonical database path without creating its parent."""
    return _guard_test_runtime_path(_CANONICAL_DB_PATH, source="canonical database path")


def get_db_path() -> Path:
    """Resolve the DB path, creating only the canonical path's parent."""
    db_path = resolve_db_path()
    if not os.environ.get("BRAINLAYER_DB"):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


# Convenience: pre-resolved default without import-time filesystem mutation.
DEFAULT_DB_PATH = resolve_db_path()
