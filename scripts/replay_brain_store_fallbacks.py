#!/usr/bin/env python3
"""Inventory and replay BrainLayer docs.local fallback files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from brainlayer.fallback_replay import (
    OUTCOME_ERROR,
    ReplayResult,
    inventory_fallbacks,
    legacy_entry_from_path,
    load_scope_map,
    queue_entry,
    queue_legacy_entry,
    replay_entry,
)
from brainlayer.paths import DEFAULT_DB_PATH, get_canonical_db_path
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
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="Write the JSON receipt (per-file outcome + chunk id) to this path as well as stdout.",
    )
    args = parser.parse_args()

    # Before load_scope_map/inventory_fallbacks: a refusal must not walk the real tree or read real
    # scopes on its way to saying no.
    if _refuse_marker_hazard(args):
        return 2

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
            _emit(result, as_json=args.json, receipt_path=args.receipt)
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
                            outcome=OUTCOME_ERROR,
                        )
                    )
        replay_count = len(pending) + (len(legacy_entries) if args.legacy else 0)
        if replay_count > args.limit:
            result["error"] = f"replay_count {replay_count} exceeds --limit {args.limit}"
            _emit(result, as_json=args.json, receipt_path=args.receipt)
            return 2
        if not args.direct_db_write:
            replayed, extra_legacy = _replay_via_queue(args, pending, legacy_entries)
            legacy_replayed.extend(extra_legacy)
        else:
            replayed = _replay_direct_to_db(args, pending)
            legacy_replayed = []
        result["replayed"] = [_receipt_row(item) for item in replayed]
        result["legacy_replayed"] = [_receipt_row(item) for item in legacy_replayed]
        result["outcome_counts"] = _outcome_counts([*replayed, *legacy_replayed])
        if any(item.error for item in [*replayed, *legacy_replayed]):
            result["error"] = "one or more fallback replays failed"
            _emit(result, as_json=args.json, receipt_path=args.receipt)
            return 1

    _emit(result, as_json=args.json, receipt_path=args.receipt)
    return 0


def marker_target_hazard(db_path: Path, gits_root: Path) -> str | None:
    """Refuse to mark the REAL fallback files while writing to a DB that is not the canonical one.

    The marker write is not DB-scoped, and nothing in `fallback_replay.py` reads a DB path at all:
    a replay stores, then writes `chunk_id` into the file under `--gits-root`, whatever `--db`
    pointed at. So `--db <copy> --direct-db-write` with the real gits-root permanently stamps every
    pending file with an id that exists only in a throwaway DB, `is_pending_entry` then answers
    "not pending", and the memories are hidden with nothing left to say they were never stored.

    That is a trap laid across the ratified procedure itself (2026-08-02: live-check against a COPY
    before merging anything that touches stored data), so it fails CLOSED. A live-check copies the
    DB *and* the files; the production drain uses the canonical DB *and* the real files. Both are
    allowed. Only the mismatch that hides data is refused.
    """
    real_gits_root = (Path.home() / "Gits").expanduser()
    if not _names_real_gits_tree(gits_root.expanduser(), real_gits_root):
        return None
    canonical = canonical_db_target()
    if _same_file(db_path.expanduser(), canonical):
        return None
    return _marker_refusal(db_path, canonical, real_gits_root)


def _same_file(left: Path, right: Path) -> bool:
    """Do these two paths name the same file ON THIS FILESYSTEM?

    Inode, not string. macOS is case-insensitive by default and is the fleet's primary host, so
    `~/.local/share/brainlayer/brainlayer.db` and `.../Brainlayer.db` are the same file while their
    `resolve()` strings differ -- which made the allowlist refuse a valid production drain.

    `os.path.normcase` does NOT help here: it is a no-op on darwin. Only the inode knows.
    """
    try:
        return os.path.samefile(left, right)
    except OSError:
        # One of them does not exist, so they cannot be the same file. Compare resolved strings for
        # the case where both are missing and merely spelled differently.
        return os.path.realpath(left) == os.path.realpath(right)


def _names_real_gits_tree(target: Path, real_gits_root: Path) -> bool:
    """Is `target` the real ~/Gits, or inside it? Inode-wise, so casing cannot dodge the guard.

    `resolve()` + `is_relative_to` is a string comparison: on APFS `~/gits` is `samefile` True with
    `~/Gits` yet compares unequal, so `--gits-root ~/gits` (or `~/gits/<repo>`) walked straight past
    the guard and marked production fallbacks with throwaway ids. ubuntu CI is case-sensitive and
    would never have surfaced it.

    A subtree's leaf may not exist yet, so the nearest EXISTING ancestor is what gets compared -- a
    case alias shows up at whatever level is real.
    """
    try:
        if _same_file(target, real_gits_root):
            return True
        for ancestor in [target, *target.parents]:
            if ancestor.exists() and _same_file(ancestor, real_gits_root):
                return True
        return target.resolve().is_relative_to(real_gits_root.resolve())
    except OSError:
        # Cannot prove it is outside the real tree => cannot prove it is safe. Fail closed.
        return True


def _replay_via_queue(
    args: argparse.Namespace, pending: list, legacy_entries: list
) -> tuple[list[ReplayResult], list[ReplayResult]]:
    """The default path: enqueue, and let the drain stay the single writer."""
    queue_dir = _queue_dir_for_target_db(args.db, args.queue_dir)

    def enqueue_for_target(**kwargs):
        if queue_dir is not None:
            kwargs["queue_dir"] = queue_dir
        return enqueue_store(**kwargs)

    replayed = [
        queue_entry(entry, enqueue_func=enqueue_for_target, replayed_by="brainlayer-replay-fallbacks")
        for entry in pending
    ]
    legacy_replayed = [
        queue_legacy_entry(entry, enqueue_func=enqueue_for_target, replayed_by="brainlayer-replay-fallbacks")
        for entry in legacy_entries
    ]
    return replayed, legacy_replayed


def _replay_direct_to_db(args: argparse.Namespace, pending: list) -> list[ReplayResult]:
    """`--direct-db-write`: bypass the queue. Guarded by marker_target_hazard before we get here."""
    store = VectorStore(args.db)
    try:
        return [
            replay_entry(
                entry,
                store_func=lambda **kwargs: store_memory(store=store, embed_fn=None, **kwargs),
                replayed_by="brainlayer-replay-fallbacks",
            )
            for entry in pending
        ]
    finally:
        store.close()


def _refuse_marker_hazard(args: argparse.Namespace) -> bool:
    """Emit the refusal receipt and say whether main() must stop. Kept out of main() so the
    pre-existing `C901 main is too complex` metric does not get worse for adding a safety gate."""
    if not args.apply:
        return False
    marker_hazard = marker_target_hazard(args.db, args.gits_root)
    if marker_hazard is None:
        return False
    _emit(
        {
            "structured_count": None,
            "pending_count": None,
            "legacy_count": None,
            "pending": [],
            "legacy": [],
            "replayed": [],
            "legacy_replayed": [],
            "error": marker_hazard,
        },
        as_json=args.json,
        receipt_path=args.receipt,
    )
    return True


def canonical_db_target() -> Path:
    """The canonical DB on disk — never `BRAINLAYER_DB`.

    `paths.DEFAULT_DB_PATH` is `resolve_db_path()` evaluated at import, so it becomes whatever
    `BRAINLAYER_DB` says. Allowlisting that means `BRAINLAYER_DB=<copy>` re-opens this exact trap
    through the env instead of through `--db`. Measured: with `BRAINLAYER_DB=/tmp/copy.db`,
    `DEFAULT_DB_PATH` is `/tmp/copy.db` while `get_canonical_db_path()` stays the real path. The
    allowlist has to be the second one.
    """
    return get_canonical_db_path().expanduser()


def _marker_refusal(db_path: Path, canonical: Path, real_gits_root: Path) -> str:
    return (
        f"refusing to replay: --db `{db_path}` is not the canonical DB (`{canonical}`), but "
        f"--gits-root is the real tree `{real_gits_root}` (or a subtree of it), so this run would "
        "mark the real fallback files as replayed with chunk ids that exist only in that DB — "
        "hiding those memories with no trace. Copy the fallback files too and pass "
        "--gits-root <copy>, or use the canonical DB."
    )


def _receipt_row(item: ReplayResult) -> dict[str, object]:
    return {
        "path": str(item.path),
        "outcome": item.outcome,
        "chunk_id": item.chunk_id,
        "error": item.error,
    }


def _outcome_counts(items: list[ReplayResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item.outcome] = counts.get(item.outcome, 0) + 1
    return dict(sorted(counts.items()))


def _queue_dir_for_target_db(db_path: Path, queue_dir: Path | None) -> Path | None:
    if queue_dir is not None:
        return queue_dir.expanduser()
    resolved_db = db_path.expanduser().resolve()
    if resolved_db == DEFAULT_DB_PATH.expanduser().resolve():
        return None
    return resolved_db.parent / "queue"


def _emit(result: dict[str, object], *, as_json: bool, receipt_path: Path | None = None) -> None:
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if receipt_path is not None:
        target = receipt_path.expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    if as_json:
        print(rendered)
        return
    # Text mode used to drop `error` entirely: a refusal and a clean run printed the same three
    # count lines, and only the exit code told them apart. An unreadable refusal is a silent one.
    error = result.get("error")
    if error:
        print(f"ERROR: {error}")
    print(f"structured fallback files: {result['structured_count']}")
    print(f"pending structured files: {result['pending_count']}")
    print(f"legacy fallback files: {result['legacy_count']}")
    for path in result["pending"]:
        print(f"PENDING {path}")
    for item in result["replayed"]:
        print(f"REPLAYED {item['path']} -> {_render_outcome(item)}")
    for item in result.get("legacy_replayed", []):
        print(f"REPLAYED_LEGACY {item['path']} -> {_render_outcome(item)}")
    counts = result.get("outcome_counts")
    if counts:
        print("outcomes: " + ", ".join(f"{name}={count}" for name, count in counts.items()))


def _render_outcome(item: dict[str, object]) -> str:
    outcome = str(item.get("outcome") or "unknown").upper()
    detail = item.get("chunk_id") or item.get("error") or "-"
    return f"{outcome} {detail}"


if __name__ == "__main__":
    raise SystemExit(main())
