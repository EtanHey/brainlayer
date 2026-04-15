#!/usr/bin/env python3
"""Repo script wrapper for BrainLayer cloud_backfill package module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from brainlayer.cloud_backfill import *  # noqa: F401,F403
from brainlayer.cloud_backfill import main


if __name__ == "__main__":
    raise SystemExit(main())
