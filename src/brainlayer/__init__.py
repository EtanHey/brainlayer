"""BrainLayer (זיכרון) - Local knowledge pipeline for Claude Code conversations."""

from __future__ import annotations

__version__ = "1.5.11"
# Git sha the distributed package was built from; stamped by the release build, None in a source tree.
# scripts/sprint_gate.py reads it in "keg" mode to prove the served package is the code under test.
# Keep the annotation lazy (future import above): launchd/install.sh imports this under /usr/bin/python3.
__build_sha__: str | None = None
