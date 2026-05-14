#!/usr/bin/env python3
"""Launchd wrapper for BrainLayer's single-writer drain daemon."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.drain import drain_once, main, run_daemon  # noqa: E402,F401

if __name__ == "__main__":
    raise SystemExit(main())
