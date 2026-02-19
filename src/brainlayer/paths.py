"""Centralized data paths for BrainLayer.

The database currently lives at ~/.local/share/zikaron/ (legacy naming from
before the BrainLayer extraction). An env var override is supported for custom
installations.

Resolution order:
  1. BRAINLAYER_DB env var (full path to .db file)
  2. ~/.local/share/zikaron/zikaron.db (current production location)
  3. ~/.local/share/brainlayer/brainlayer.db (future default)
"""

import os
from pathlib import Path

# Legacy path (where data actually lives today)
_LEGACY_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"

# New canonical path (for fresh installs)
_CANONICAL_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"


def get_db_path() -> Path:
    """Resolve the BrainLayer database path.

    Checks BRAINLAYER_DB env var first, then falls back to whichever
    known path exists (preferring the legacy zikaron path if both exist,
    since that's where the real data is).
    """
    env = os.environ.get("BRAINLAYER_DB")
    if env:
        return Path(env)

    if _LEGACY_DB_PATH.exists():
        return _LEGACY_DB_PATH

    # Ensure parent dir exists for fresh installs
    _CANONICAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _CANONICAL_DB_PATH


# Convenience: pre-resolved default for import
DEFAULT_DB_PATH = get_db_path()
