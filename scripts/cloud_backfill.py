#!/usr/bin/env python3
"""Repo script wrapper for BrainLayer cloud_backfill package module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import brainlayer.cloud_backfill as _cloud_backfill

main = _cloud_backfill.main


def __getattr__(name: str):
    return getattr(_cloud_backfill, name)


if __name__ == "__main__":
    raise SystemExit(main())
