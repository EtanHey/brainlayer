#!/usr/bin/env python3
"""Inventory and replay BrainLayer docs.local fallback files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainlayer.fallback_replay import is_pending_entry, load_scope_map, parse_fallback_file, replay_entry
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory/replay BrainLayer fallback markdown files.")
    parser.add_argument("--gits-root", type=Path, default=Path.home() / "Gits")
    parser.add_argument("--scopes", type=Path, default=Path.home() / ".config" / "brainlayer" / "scopes.yaml")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--apply", action="store_true", help="Write pending structured files into BrainLayer.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum structured pending files to replay.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    scope_map = load_scope_map(args.scopes)
    structured, legacy = inventory(args.gits_root, scope_map=scope_map)
    pending = [entry for entry in structured if is_pending_entry(entry)]
    result: dict[str, object] = {
        "structured_count": len(structured),
        "pending_count": len(pending),
        "legacy_count": len(legacy),
        "pending": [str(entry.path) for entry in pending],
        "legacy": [str(path) for path in legacy],
        "replayed": [],
    }

    if args.apply:
        if len(pending) > args.limit:
            result["error"] = f"pending_count {len(pending)} exceeds --limit {args.limit}"
            _emit(result, as_json=args.json)
            return 2
        store = VectorStore(args.db)
        try:
            replayed = [
                replay_entry(
                    entry,
                    store_func=lambda **kwargs: store_memory(store=store, embed_fn=None, **kwargs),
                    replayed_by="brainlayer-replay-fallbacks",
                )
                for entry in pending
            ]
        finally:
            store.close()
        result["replayed"] = [
            {"path": str(item.path), "chunk_id": item.chunk_id, "error": item.error}
            for item in replayed
        ]
        if any(item.error for item in replayed):
            result["error"] = "one or more fallback replays failed"
            _emit(result, as_json=args.json)
            return 1

    _emit(result, as_json=args.json)
    return 0


def inventory(gits_root: Path, *, scope_map: dict[str, str]):
    structured = []
    legacy = []
    if not gits_root.exists():
        return structured, legacy
    for repo in sorted(path for path in gits_root.iterdir() if path.is_dir()):
        for path in sorted((repo / "docs.local" / "decisions").glob("*.md")):
            try:
                entry = parse_fallback_file(path, scope_map=scope_map)
            except Exception:
                legacy.append(path)
                continue
            if entry.frontmatter.get("intended_brain_store"):
                structured.append(entry)
        fallback_dir = repo / "docs.local" / "brain-store-fallback"
        if fallback_dir.exists():
            legacy.extend(sorted(path for path in fallback_dir.rglob("*.md") if path.is_file()))
    return structured, legacy


def _emit(result: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"structured fallback files: {result['structured_count']}")
    print(f"pending structured files: {result['pending_count']}")
    print(f"legacy fallback files: {result['legacy_count']}")
    for path in result["pending"]:
        print(f"PENDING {path}")
    for item in result["replayed"]:
        print(f"REPLAYED {item['path']} -> {item.get('chunk_id') or item.get('error')}")


if __name__ == "__main__":
    raise SystemExit(main())
