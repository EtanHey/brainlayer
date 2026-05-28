#!/usr/bin/env python3
"""Seed the default hand-tuned BrainLayer agent ranking profiles."""

from __future__ import annotations

import sys

import apsw

from brainlayer.agent_profiles import DEFAULT_AGENT_PROFILES
from brainlayer.paths import get_db_path
from brainlayer.vector_store import VectorStore, WriterInUseError


def main() -> None:
    try:
        store = VectorStore(get_db_path())
    except WriterInUseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Stop BrainLayer writer services before seeding agent profiles.", file=sys.stderr)
        raise SystemExit(1) from exc
    except apsw.BusyError as exc:
        print("ERROR: Database is locked. Retry after other BrainLayer processes finish.", file=sys.stderr)
        raise SystemExit(1) from exc
    try:
        for agent_id, profile in DEFAULT_AGENT_PROFILES.items():
            store.set_agent_profile(agent_id, profile, notes="seed_agent_profiles.py default")
            print(f"seeded {agent_id}")
        print("Note: other running BrainLayer processes may keep cached search results for up to 60 seconds.")
        print("Profile updates are last-writer-wins; coordinate concurrent operator updates manually.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
