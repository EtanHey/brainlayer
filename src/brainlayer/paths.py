"""Centralized data paths for BrainLayer.

Resolution order:
  1. BRAINLAYER_DB env var (full path to .db file)
  2. ~/.local/share/brainlayer/brainlayer.db (canonical path)
"""

import os
from pathlib import Path

_CANONICAL_DB_PATH = Path.home() / ".local" / "share" / "brainlayer" / "brainlayer.db"


def get_db_path() -> Path:
    """Resolve the BrainLayer database path.

    Checks BRAINLAYER_DB env var first, then uses the canonical path.
    """
    env = os.environ.get("BRAINLAYER_DB")
    if env:
        return Path(env)

    _CANONICAL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return _CANONICAL_DB_PATH


# Convenience: pre-resolved default for import
DEFAULT_DB_PATH = get_db_path()
