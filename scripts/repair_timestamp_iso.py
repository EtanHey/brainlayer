#!/usr/bin/env python3
"""Normalize TEXT timestamp columns to ISO-8601 UTC (rehearsal/live-window)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.timestamp_iso import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
