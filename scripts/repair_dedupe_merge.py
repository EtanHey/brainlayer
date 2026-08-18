#!/usr/bin/env python3
"""Archive exact content duplicates onto a survivor (rehearsal/live-window)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
# Prepend unconditionally. A plain "if not in sys.path" guard is not enough: an
# editable-install .pth can already have src on the path BEHIND site-packages,
# so the guard skips the insert and a stale installed copy of brainlayer wins.
while str(SRC_DIR) in sys.path:
    sys.path.remove(str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR))

from brainlayer.dedupe_merge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
