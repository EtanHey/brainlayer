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

from brainlayer.fallback_replay import (
    ReplayResult,
    inventory_fallbacks,
    legacy_entry_from_path,
    load_scope_map,
    queue_entry,
    queue_legacy_entry,
    replay_entry,
)
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.queue_io import enqueue_store
from brainlayer.store import store_memory
from brainlayer.vector_store import VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory/replay BrainLayer fallback markdown files.")
    parser.add_argument("--gits-root", type=Path, default=Path.home() / "Gits")
    parser.add_argument("--scopes", type=Path, default=Path.home() / ".config" / "brainlayer" / "scopes.yaml")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help="Queue directory for queued replay; defaults to the live queue, or <db parent>/queue for non-default --db.",
    )
    parser.add_argument("--apply", action="store_true", help="Write pending structured files into BrainLayer.")
    parser.add_argument(
        "--queue",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--direct-db-write",
        action="store_true",
        help="With --apply, bypass the durable queue and write structured files directly to the DB.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="With --apply, also enqueue legacy docs.local/brain-store-fallback markdown files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum structured pending and legacy fallback files to replay.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    scope_map = load_scope_map(args.scopes)
    inventory = inventory_fallbacks(args.gits_root, scope_map=scope_map)
    pending = inventory.pending
    result: dict[str, object] = {
        "structured_count": len(inventory.structured),
        "pending_count": len(pending),
        "legacy_count": len(inventory.legacy),
        "pending": [str(entry.path) for entry in pending],
        "legacy": [str(path) for path in inventory.legacy],
        "replayed": [],
        "legacy_replayed": [],
    }

    if args.apply:
        if args.legacy and args.direct_db_write:
            result["error"] = "--legacy requires queued replay"
            _emit(result, as_json=args.json)
            return 2
        legacy_entries = []
        legacy_replayed = []
        if args.legacy:
            for path in inventory.legacy:
                try:
                    legacy_entries.append(legacy_entry_from_path(path, scope_map=scope_map))
                except Exception as exc:
                    legacy_replayed.append(
                        ReplayResult(
                            path=path,
                            attempted=True,
                            chunk_id=None,
                            error=f"legacy parse failed: {exc}",
                        )
                    )
        replay_count = len(pending) + (len(inventory.legacy) if args.legacy else 0)
        if replay_count > args.limit:
            result["error"] = f"replay_count {replay_count} exceeds --limit {args.limit}"
            _emit(result, as_json=args.json)
            return 2
        if not args.direct_db_write:
            queue_dir = _queue_dir_for_target_db(args.db, args.queue_dir)

            def enqueue_for_target(**kwargs):
                if queue_dir is not None:
                    kwargs["queue_dir"] = queue_dir
                return enqueue_store(**kwargs)

            replayed = [
                queue_entry(
                    entry,
                    enqueue_func=enqueue_for_target,
                    replayed_by="brainlayer-replay-fallbacks",
                )
                for entry in pending
            ]
            legacy_replayed.extend(
                queue_legacy_entry(
                    entry,
                    enqueue_func=enqueue_for_target,
                    replayed_by="brainlayer-replay-fallbacks",
                )
                for entry in legacy_entries
            )
        else:
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
            legacy_replayed = []
        result["replayed"] = [
            {"path": str(item.path), "chunk_id": item.chunk_id, "error": item.error} for item in replayed
        ]
        result["legacy_replayed"] = [
            {"path": str(item.path), "chunk_id": item.chunk_id, "error": item.error} for item in legacy_replayed
        ]
        if any(item.error for item in [*replayed, *legacy_replayed]):
            result["error"] = "one or more fallback replays failed"
            _emit(result, as_json=args.json)
            return 1

    _emit(result, as_json=args.json)
    return 0


def _queue_dir_for_target_db(db_path: Path, queue_dir: Path | None) -> Path | None:
    if queue_dir is not None:
        return queue_dir.expanduser()
    resolved_db = db_path.expanduser().resolve()
    if resolved_db == DEFAULT_DB_PATH.expanduser().resolve():
        return None
    return resolved_db.parent / "queue"


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
    for item in result.get("legacy_replayed", []):
        print(f"REPLAYED_LEGACY {item['path']} -> {item.get('chunk_id') or item.get('error')}")


if __name__ == "__main__":
    raise SystemExit(main())
