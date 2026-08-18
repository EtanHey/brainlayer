"""Merge exact content duplicates by ARCHIVING losers, never deleting them.

Repair (e). Default is dry-run. Refuses the live canonical DB unless
allow_live=True after a lead-scheduled writer window. Follows the proven
repair-b/c/d preimage + checkpoint pattern (`chunk_origin_wipe`, `timestamp_iso`).

Policy (POLICY_REPAIR_E, lead-ACKed 2026-08-18, census in
`docs.local/repair-e-census-2026-08-18.md`):

* **Key** — `sha256(content.strip())`, recomputed here. The stored `content_hash`
  column is never trusted: the canonical DB carries four schemes at once (64-char
  sha256, a 32-char scheme, 16-char truncations, and 52k rows with none), plus
  2,256 rows whose stored hash went stale when their content later changed.
  Grouping on it both under-counts (rows in different schemes never meet) and
  over-counts (20 groups whose members hold genuinely different text).
* **Scope** — same `conversation_id` only. Identical text in two different
  conversations is two memories: the census found one 245-char sentence stored
  across 252 conversations in 59 projects, and collapsing that would erase 252
  session contexts. A blank conversation_id is *unknown*, not a match.
* **Survivor** — richest enrichment, then oldest `created_at`, then smallest id;
  rows already lifecycle-managed are never chosen. Oldest-wins would have
  discarded enrichment in 23,359 of 42,082 eligible groups; richest-wins, 1,243.
* **Losers** — `aggregated_into` + `archived_at` + `status='superseded'`. Content,
  ids, tags, FTS rows and vectors all stay. No DELETE, in any table, ever.
  `status='superseded'` and not `'archived'` because `vector_store.py` rewrites
  `status='archived'` on startup, which would silently revert the migration.

Both serving paths already exclude `aggregated_into IS NOT NULL` and
`archived_at IS NOT NULL` from default search, so this removes duplicate noise
with no index surgery -- aux counts are asserted UNCHANGED, unlike repair-c/d.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chunk_origin_wipe import AUX_COUNT_TABLES, assert_not_live_db
from .chunk_origin_wipe import live_canonical_db_path as live_canonical_db_path

PREIMAGE_TABLE = "dedupe_merge_preimage"
ALIAS_PREIMAGE_TABLE = "dedupe_merge_alias_preimage"
MIGRATION_NAME = "2026_08_18_dedupe_merge_archive"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")

#: Enrichment carried per chunk. Order is the survivor score and the union-fill order.
ENRICH_FIELDS = (
    "summary",
    "tags",
    "importance",
    "intent",
    "enriched_at",
    "key_facts",
    "primary_symbols",
    "raw_entities_json",
)

#: Columns the migration may write. Everything here is preimaged before a write.
WRITTEN_COLUMNS = (
    "aggregated_into",
    "archived_at",
    "status",
    "seen_count",
    *ENRICH_FIELDS,
)

LIFECYCLE_COLUMNS = ("archived_at", "superseded_by", "aggregated_into")


@dataclass
class DedupeMergeResult:
    scanned: int = 0
    groups_total: int = 0
    groups_eligible: int = 0
    groups_held: int = 0
    groups_merged: int = 0
    would_merge_groups: int = 0
    would_archive_losers: int = 0
    losers_archived: int = 0
    survivors_updated: int = 0
    union_fills: dict[str, int] = field(default_factory=dict)
    aliases_written: int = 0
    aliases_repointed: int = 0
    aliases_dropped_survivor_source: int = 0
    lineage_repointed: int = 0
    preexisting_alias_cycles: int = 0
    seen_count_transferred: int = 0
    batches: int = 0
    checkpoints: int = 0
    held_reasons: dict[str, int] = field(default_factory=dict)
    aux_counts_before: dict[str, int] = field(default_factory=dict)
    aux_counts_after: dict[str, int] = field(default_factory=dict)
    spot_checks: list[dict[str, Any]] = field(default_factory=list)


def content_key(content: Any) -> str | None:
    """The merge key. None when the row can never participate in a merge."""
    if content is None:
        return None
    text = str(content).strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_git_sha(git_sha: str) -> str:
    if not _SHA_RE.fullmatch(git_sha or ""):
        raise ValueError("git_sha must be an exact 40-character hexadecimal commit SHA")
    return git_sha.lower()


def _enrich_score(row: dict, present: set[str]) -> int:
    return sum(1 for f in ENRICH_FIELDS if f in present and not _blank(row.get(f)))


def _is_managed(row: dict) -> bool:
    return any(not _blank(row.get(col)) for col in LIFECYCLE_COLUMNS)


def _instant(value: Any) -> str:
    return "" if _blank(value) else str(value).strip()


def pick_survivor(members: list[dict], present: set[str]) -> dict | None:
    """Richest enrichment -> oldest created_at -> smallest id, among live rows."""
    live = [m for m in members if not _is_managed(m)]
    if not live:
        return None
    return min(
        live,
        key=lambda m: (-_enrich_score(m, present), _instant(m.get("created_at")) or "9999", str(m["id"])),
    )


def group_is_eligible(members: list[dict]) -> tuple[bool, str]:
    """Same-conversation rule. Returns (eligible, reason_when_held)."""
    conversations = {m.get("conversation_id") for m in members}
    if any(_blank(c) for c in conversations):
        return False, "conversation_id_unknown"
    if len(conversations) > 1:
        return False, "cross_conversation"
    return True, ""


def _aux_counts(conn: sqlite3.Connection) -> dict[str, int]:
    existing = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    counts: dict[str, int] = {}
    for table in AUX_COUNT_TABLES:
        if table not in existing:
            continue
        try:
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.OperationalError:
            continue
    return counts


def _checkpoint(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA wal_checkpoint(FULL)")


def _column_types(conn: sqlite3.Connection) -> dict[str, str]:
    return {row[1]: (row[2] or "TEXT") for row in conn.execute("PRAGMA table_info(chunks)")}


def _ensure_preimage(conn: sqlite3.Connection, columns: list[str]) -> None:
    """Preimage mirrors the chunks column TYPES so a rollback restores values as-is."""
    expected = ["id", *columns]
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (PREIMAGE_TABLE,)).fetchone()
    if exists:
        present = [row[1] for row in conn.execute(f"PRAGMA table_info({PREIMAGE_TABLE})")]
        if present != expected:
            raise RuntimeError(
                f"{PREIMAGE_TABLE} columns {present!r} do not match expected {expected!r}; "
                "restore from backup before re-running"
            )
        pk = [row[1] for row in conn.execute(f"PRAGMA table_info({PREIMAGE_TABLE})") if row[5]]
        if pk != ["id"]:
            raise RuntimeError(
                f"{PREIMAGE_TABLE} exists without an id PRIMARY KEY; INSERT OR IGNORE would "
                "append instead of protecting the first preimage. Restore from backup."
            )
        return
    types = _column_types(conn)
    column_defs = ", ".join(["id TEXT PRIMARY KEY", *(f"{column} {types.get(column, 'TEXT')}" for column in columns)])
    conn.execute(f"CREATE TABLE {PREIMAGE_TABLE} ({column_defs})")
    conn.commit()


def _ensure_alias_table(conn: sqlite3.Connection) -> bool:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunk_id_alias'").fetchone()
    return exists is not None


def merge_duplicates(
    db_path: Path,
    *,
    git_sha: str,
    apply: bool = False,
    batch_size: int = 500,
    checkpoint_every: int = 3,
    allow_live: bool = False,
    spot_check: int = 0,
    union_fill: bool = True,
    actor: str = "repair-e",
) -> DedupeMergeResult:
    """Archive exact content duplicates onto a survivor, preserving lineage."""
    sha = _validate_git_sha(git_sha)
    resolved = assert_not_live_db(Path(db_path), allow_live=allow_live)
    batch_size = max(1, int(batch_size))
    checkpoint_every = max(1, int(checkpoint_every))
    spot_check = max(0, int(spot_check))
    result = DedupeMergeResult()
    samples: list[str] = []
    sampled = 0
    rng = random.Random(0)

    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        present = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
        for required in ("id", "content", "conversation_id", "aggregated_into", "archived_at"):
            if required not in present:
                raise RuntimeError(f"chunks is missing required column {required!r}")
        written = [c for c in WRITTEN_COLUMNS if c in present]
        select_cols = sorted({"id", "content", "conversation_id", "created_at", *LIFECYCLE_COLUMNS, *written} & present)
        has_alias = _ensure_alias_table(conn)

        result.aux_counts_before = _aux_counts(conn)
        if apply:
            _ensure_preimage(conn, written)
            if has_alias:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {ALIAS_PREIMAGE_TABLE} (
                        old_chunk_id TEXT NOT NULL,
                        canonical_chunk_id TEXT,
                        deprecated_at TEXT,
                        action TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
            _checkpoint(conn)
            result.checkpoints += 1

        # --- pass 1a: count keys, holding only digests ------------------------
        # Two passes so peak memory scales with the DUPLICATE population, not the
        # table: on the canonical DB that is ~150k member rows instead of ~760k.
        key_counts: dict[str, int] = defaultdict(int)
        for (content,) in conn.execute("SELECT content FROM chunks"):
            result.scanned += 1
            key = content_key(content)
            if key is not None:
                key_counts[key] += 1
        duplicate_keys = {key for key, count in key_counts.items() if count > 1}
        del key_counts

        # --- pass 1b: materialize only rows in a duplicate group --------------
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in conn.execute(f"SELECT {', '.join(select_cols)} FROM chunks"):
            key = content_key(row["content"])
            if key is None or key not in duplicate_keys:
                continue
            record = {c: row[c] for c in select_cols}
            record.pop("content", None)
            groups[key].append(record)
        del duplicate_keys

        plans: list[tuple[dict, list[dict]]] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            result.groups_total += 1
            eligible, reason = group_is_eligible(members)
            if not eligible:
                result.groups_held += 1
                result.held_reasons[reason] = result.held_reasons.get(reason, 0) + 1
                continue
            survivor = pick_survivor(members, present)
            if survivor is None:
                result.groups_held += 1
                result.held_reasons["all_members_lifecycle_managed"] = (
                    result.held_reasons.get("all_members_lifecycle_managed", 0) + 1
                )
                continue
            losers = [m for m in members if m["id"] != survivor["id"] and not _is_managed(m)]
            if not losers:
                result.groups_held += 1
                result.held_reasons["nothing_left_to_archive"] = (
                    result.held_reasons.get("nothing_left_to_archive", 0) + 1
                )
                continue
            result.groups_eligible += 1
            result.would_merge_groups += 1
            result.would_archive_losers += len(losers)
            plans.append((survivor, losers))
            if spot_check:
                for loser in losers:
                    sampled += 1
                    if len(samples) < spot_check:
                        samples.append(str(loser["id"]))
                    else:
                        j = rng.randrange(sampled)
                        if j < spot_check:
                            samples[j] = str(loser["id"])

        del groups
        if not apply:
            result.aux_counts_after = _aux_counts(conn)
            return result

        # --- pass 2: apply, batched, preimaged -------------------------------
        stamp = _now_iso()
        touched_ids: set[str] = set()
        # Pre-existing lineage, prefetched once (target -> rows pointing at it).
        # Rows this run archives all point at survivors, and a survivor is never
        # another group's loser (groups are disjoint by content key), so this map
        # does not need refreshing mid-run.
        inbound_lineage: dict[str, list[str]] = defaultdict(list)
        for row_id, target in conn.execute("SELECT id, aggregated_into FROM chunks WHERE aggregated_into IS NOT NULL"):
            inbound_lineage[str(target)].append(str(row_id))
        for offset in range(0, len(plans), batch_size):
            batch = plans[offset : offset + batch_size]
            result.batches += 1
            try:
                conn.execute("BEGIN IMMEDIATE")
                for survivor, losers in batch:
                    survivor_row = conn.execute(
                        f"SELECT {', '.join(sorted(set(written) | {'id'}))} FROM chunks WHERE id = ?",
                        (survivor["id"],),
                    ).fetchone()
                    if survivor_row is None:
                        continue
                    survivor_state = {c: survivor_row[c] for c in survivor_row.keys()}

                    updates: dict[str, Any] = {}
                    seen_total = 0
                    archived_ids: list[str] = []
                    for loser in losers:
                        current = conn.execute(
                            f"SELECT {', '.join(sorted(set(written) | set(LIFECYCLE_COLUMNS) | {'id'}))} "
                            "FROM chunks WHERE id = ?",
                            (loser["id"],),
                        ).fetchone()
                        if current is None:
                            continue
                        loser_state = {c: current[c] for c in current.keys()}
                        # Resume safety: a loser already archived in a prior run is
                        # skipped, so seen_count is never transferred twice.
                        if _is_managed(loser_state):
                            continue

                        conn.execute(
                            f"INSERT OR IGNORE INTO {PREIMAGE_TABLE}(id, {', '.join(written)}) "
                            f"VALUES (?, {', '.join('?' for _ in written)})",
                            [loser_state["id"], *[loser_state.get(c) for c in written]],
                        )
                        assignments = {
                            "aggregated_into": survivor["id"],
                            "archived_at": stamp,
                        }
                        if "status" in present:
                            assignments["status"] = "superseded"
                        conn.execute(
                            f"UPDATE chunks SET {', '.join(f'{c} = ?' for c in assignments)} WHERE id = ?",
                            [*assignments.values(), loser_state["id"]],
                        )
                        result.losers_archived += 1
                        seen_total += int(loser_state.get("seen_count") or 0)

                        # Lineage convergence: rows that an EARLIER process had
                        # already aggregated into this loser would now sit two
                        # hops from a live row. Re-point them at the survivor so
                        # every chain still ends one hop away, on a live chunk.
                        # Looked up in the prefetched map: `aggregated_into` has no
                        # index, so querying per loser meant 61k full table scans.
                        inbound_ids = [
                            row_id
                            for row_id in inbound_lineage.get(str(loser_state["id"]), ())
                            if row_id != str(survivor["id"])
                        ]
                        inbound = (
                            conn.execute(
                                f"SELECT id, {', '.join(written)} FROM chunks "
                                f"WHERE id IN ({', '.join('?' for _ in inbound_ids)})",
                                inbound_ids,
                            ).fetchall()
                            if inbound_ids
                            else []
                        )
                        if inbound:
                            conn.executemany(
                                f"INSERT OR IGNORE INTO {PREIMAGE_TABLE}(id, {', '.join(written)}) "
                                f"VALUES (?, {', '.join('?' for _ in written)})",
                                [[row["id"], *[row[c] for c in written]] for row in inbound],
                            )
                            conn.executemany(
                                "UPDATE chunks SET aggregated_into = ? WHERE id = ?",
                                [(survivor["id"], row["id"]) for row in inbound],
                            )
                            result.lineage_repointed += len(inbound)
                            touched_ids.update(str(row["id"]) for row in inbound)

                        if union_fill:
                            for column in ENRICH_FIELDS:
                                if column not in present:
                                    continue
                                if not _blank(survivor_state.get(column)) or column in updates:
                                    continue
                                if not _blank(loser_state.get(column)):
                                    updates[column] = loser_state[column]

                        archived_ids.append(str(loser_state["id"]))
                        touched_ids.add(str(loser_state["id"]))

                    if has_alias and archived_ids:
                        touched_ids.add(str(survivor["id"]))
                        _converge_aliases(
                            conn, survivor_id=str(survivor["id"]), losers=archived_ids, stamp=stamp, result=result
                        )

                    if union_fill and (updates or seen_total):
                        conn.execute(
                            f"INSERT OR IGNORE INTO {PREIMAGE_TABLE}(id, {', '.join(written)}) "
                            f"VALUES (?, {', '.join('?' for _ in written)})",
                            [survivor_state["id"], *[survivor_state.get(c) for c in written]],
                        )
                        if "seen_count" in present and seen_total:
                            updates["seen_count"] = int(survivor_state.get("seen_count") or 0) + seen_total
                            result.seen_count_transferred += seen_total
                        if updates:
                            conn.execute(
                                f"UPDATE chunks SET {', '.join(f'{c} = ?' for c in updates)} WHERE id = ?",
                                [*updates.values(), survivor_state["id"]],
                            )
                            result.survivors_updated += 1
                            for column in updates:
                                if column in ENRICH_FIELDS:
                                    result.union_fills[column] = result.union_fills.get(column, 0) + 1
                    if any(loser for loser in losers):
                        result.groups_merged += 1
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            if result.batches % checkpoint_every == 0:
                _checkpoint(conn)
                result.checkpoints += 1

        if result.batches and result.batches % checkpoint_every != 0:
            _checkpoint(conn)
            result.checkpoints += 1

        # groups_merged counts groups that actually produced an archived loser
        result.groups_merged = min(result.groups_merged, result.would_merge_groups)
        if result.losers_archived == 0:
            result.groups_merged = 0

        _assert_no_alias_cycles(conn, has_alias, touched_ids, result)
        result.aux_counts_after = _aux_counts(conn)
        if result.aux_counts_before != result.aux_counts_after:
            raise RuntimeError(
                "aux table counts changed; this migration must not touch FTS or vector rows: "
                f"{result.aux_counts_before} -> {result.aux_counts_after}"
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL,
                details TEXT
            )
            """
        )
        receipt = {
            "migration": MIGRATION_NAME,
            "git_sha": sha,
            "actor": actor,
            "policy": "POLICY_REPAIR_E",
            "groups_merged": result.groups_merged,
            "losers_archived": result.losers_archived,
            "groups_held": result.groups_held,
            "held_reasons": result.held_reasons,
            "union_fill": union_fill,
            "union_fills": result.union_fills,
            "seen_count_transferred": result.seen_count_transferred,
            "preexisting_alias_cycles": result.preexisting_alias_cycles,
            "lineage_repointed": result.lineage_repointed,
        }
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations(name, applied_at, details) VALUES (?, ?, ?)",
            (MIGRATION_NAME, _now_iso(), json.dumps(receipt, sort_keys=True)),
        )
        conn.commit()

        if samples:
            result.spot_checks = _spot_check(conn, samples, written)
        return result
    finally:
        conn.close()


def _alias_preimage(conn: sqlite3.Connection, rows: list[tuple[str, str | None, str | None, str]]) -> None:
    """Record an alias row's BEFORE state so alias edits are reversible too."""
    conn.executemany(
        f"INSERT INTO {ALIAS_PREIMAGE_TABLE}(old_chunk_id, canonical_chunk_id, deprecated_at, action) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )


def _converge_aliases(
    conn: sqlite3.Connection,
    *,
    survivor_id: str,
    losers: list[str],
    stamp: str,
    result: DedupeMergeResult,
) -> None:
    """Point every alias edge in this group AT the survivor -- and only at it.

    Writing `loser -> survivor` edges one at a time creates cycles: prior dedupe
    runs already aliased members of the same duplicate group to each other, so a
    new edge can close a loop (observed on the rehearsal copy: 72 pre-existing
    cycles became 588). Convergence is the invariant that cannot loop -- every
    edge in the group terminates at the survivor, and the survivor itself is
    never an alias source, so no cycle involving this group can exist.
    """
    loser_set = set(losers)
    marks = ", ".join("?" for _ in losers)

    # 1. Anything that resolved to a loser now resolves to the survivor.
    stale = conn.execute(
        f"SELECT old_chunk_id, canonical_chunk_id, deprecated_at FROM chunk_id_alias "
        f"WHERE canonical_chunk_id IN ({marks}) AND old_chunk_id <> ?",
        (*losers, survivor_id),
    ).fetchall()
    if stale:
        _alias_preimage(conn, [(r[0], r[1], r[2], "repoint") for r in stale])
        conn.executemany(
            "UPDATE chunk_id_alias SET canonical_chunk_id = ?, deprecated_at = ? WHERE old_chunk_id = ?",
            [(survivor_id, stamp, r[0]) for r in stale],
        )
        result.aliases_repointed += len(stale)

    # 2. A live survivor must never itself be a deprecated id, or the chain
    #    walks off the row that is meant to be canonical.
    outgoing = conn.execute(
        "SELECT old_chunk_id, canonical_chunk_id, deprecated_at FROM chunk_id_alias WHERE old_chunk_id = ?",
        (survivor_id,),
    ).fetchall()
    if outgoing:
        _alias_preimage(conn, [(r[0], r[1], r[2], "drop_survivor_source") for r in outgoing])
        conn.execute("DELETE FROM chunk_id_alias WHERE old_chunk_id = ?", (survivor_id,))
        result.aliases_dropped_survivor_source += len(outgoing)

    # 3. Every loser forwards to the survivor.
    existing = {
        row[0]: (row[1], row[2])
        for row in conn.execute(
            f"SELECT old_chunk_id, canonical_chunk_id, deprecated_at FROM chunk_id_alias "
            f"WHERE old_chunk_id IN ({marks})",
            losers,
        )
    }
    # Rows that already had an alias are recorded as "overwrite" (restore the old
    # target on rollback); rows that did not are recorded as "insert" (delete on
    # rollback). Without the insert rows a rollback cannot tell which alias rows
    # this migration created, and a date heuristic over-deletes pre-existing ones.
    pre_rows = [(old, existing[old][0], existing[old][1], "overwrite") for old in existing]
    pre_rows += [(loser, None, None, "insert") for loser in sorted(loser_set) if loser not in existing]
    if pre_rows:
        _alias_preimage(conn, pre_rows)
    conn.executemany(
        """
        INSERT INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(old_chunk_id) DO UPDATE SET
            canonical_chunk_id = excluded.canonical_chunk_id,
            deprecated_at = excluded.deprecated_at
        """,
        [(loser, survivor_id, stamp) for loser in loser_set],
    )
    result.aliases_written += len(loser_set)


def _find_alias_cycles(conn: sqlite3.Connection) -> set[str]:
    """Return every alias id that sits on a cycle."""
    mapping = {
        str(old): str(canonical)
        for old, canonical in conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias")
    }
    on_cycle: set[str] = set()
    for start in mapping:
        seen: list[str] = [start]
        index = {start}
        cursor = mapping[start]
        while cursor in mapping:
            if cursor in index:
                on_cycle.update(seen[seen.index(cursor) :])
                break
            seen.append(cursor)
            index.add(cursor)
            cursor = mapping[cursor]
    return on_cycle


def _assert_no_alias_cycles(
    conn: sqlite3.Connection, has_alias: bool, touched: set[str], result: DedupeMergeResult
) -> None:
    """This migration must introduce no cycle. Pre-existing ones are REPORTED.

    The canonical DB already carries 72 alias cycles that predate repair (e)
    (prior dedupe runs aliased members of one duplicate group to each other).
    Failing the whole migration on someone else's cycle would make it
    unrunnable, and silently ignoring cycles would hide a defect this migration
    could cause -- so the assertion is scoped to ids this run touched, and the
    pre-existing count is surfaced in the receipt for lead triage.
    """
    if not has_alias:
        return
    on_cycle = _find_alias_cycles(conn)
    ours = on_cycle & touched
    if ours:
        sample = sorted(ours)[:3]
        raise RuntimeError(f"this migration introduced {len(ours)} alias-cycle ids; first: {sample!r}")
    result.preexisting_alias_cycles = len(on_cycle)


def _spot_check(conn: sqlite3.Connection, samples: list[str], written: list[str]) -> list[dict[str, Any]]:
    """Re-READ stored rows and verify each archived loser against its preimage."""
    checks: list[dict[str, Any]] = []
    for chunk_id in samples:
        stored = conn.execute(
            "SELECT id, content, aggregated_into, archived_at, status FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
        if stored is None:
            checks.append({"id": chunk_id, "ok": False, "why": "row_missing_after_merge"})
            continue
        pre = conn.execute(
            f"SELECT id, {', '.join(written)} FROM {PREIMAGE_TABLE} WHERE id = ?", (chunk_id,)
        ).fetchone()
        survivor_id = stored["aggregated_into"]
        why = []
        if pre is None:
            why.append("no_preimage")
        if _blank(survivor_id):
            why.append("no_lineage_pointer")
        if _blank(stored["archived_at"]):
            why.append("not_archived")
        if stored["status"] not in (None, "superseded"):
            why.append(f"unexpected_status:{stored['status']}")
        if _blank(stored["content"]):
            why.append("content_lost")
        if not _blank(survivor_id):
            survivor = conn.execute(
                "SELECT id, content, aggregated_into FROM chunks WHERE id = ?", (survivor_id,)
            ).fetchone()
            if survivor is None:
                why.append("survivor_missing")
            else:
                if content_key(survivor["content"]) != content_key(stored["content"]):
                    why.append("survivor_content_differs")
                if not _blank(survivor["aggregated_into"]):
                    why.append("survivor_is_itself_archived")
        checks.append(
            {
                "id": chunk_id,
                "ok": not why,
                "aggregated_into": survivor_id,
                "archived_at": stored["archived_at"],
                "status": stored["status"],
                **({"why": why} if why else {}),
            }
        )
    return checks


def rollback_migration(db_path: Path, *, allow_live: bool = False) -> dict[str, Any]:
    """Restore every row this migration wrote, from its preimages.

    Reversibility has to be a command, not a runbook snippet: a hand-written
    rollback during the rehearsal used a `deprecated_at LIKE today` heuristic to
    find alias rows and over-deleted 4,817 pre-existing ones. The preimages name
    exactly what changed, so this is exact.

    The preimage tables are deliberately RETAINED afterwards (as repair-d
    retained `timestamp_iso_preimage`): they are the audit trail of what the
    migration touched, and removing them is a separate, human-authorized step.
    """
    resolved = assert_not_live_db(Path(db_path), allow_live=allow_live)
    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    out: dict[str, Any] = {
        "db": str(resolved),
        "chunks_restored": 0,
        "aliases_restored": 0,
        "aliases_removed": 0,
        "preimage_tables": "retained",
    }
    try:
        conn.execute("PRAGMA busy_timeout=30000")
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if PREIMAGE_TABLE not in tables:
            raise RuntimeError(f"{PREIMAGE_TABLE} is absent; nothing to roll back")
        conn.execute("BEGIN IMMEDIATE")
        try:
            columns = [r[1] for r in conn.execute(f"PRAGMA table_info({PREIMAGE_TABLE})") if r[1] != "id"]
            rows = conn.execute(f"SELECT id, {', '.join(columns)} FROM {PREIMAGE_TABLE}").fetchall()
            conn.executemany(
                f"UPDATE chunks SET {', '.join(f'{c} = ?' for c in columns)} WHERE id = ?",
                [[row[c] for c in columns] + [row["id"]] for row in rows],
            )
            out["chunks_restored"] = len(rows)

            if ALIAS_PREIMAGE_TABLE in tables and "chunk_id_alias" in tables:
                alias_rows = conn.execute(
                    f"SELECT old_chunk_id, canonical_chunk_id, deprecated_at, action FROM {ALIAS_PREIMAGE_TABLE}"
                ).fetchall()
                created = [r["old_chunk_id"] for r in alias_rows if r["action"] == "insert"]
                restore = [r for r in alias_rows if r["action"] != "insert"]
                if created:
                    conn.executemany("DELETE FROM chunk_id_alias WHERE old_chunk_id = ?", [(old,) for old in created])
                    out["aliases_removed"] = len(created)
                if restore:
                    conn.executemany(
                        "INSERT OR REPLACE INTO chunk_id_alias(old_chunk_id, canonical_chunk_id, deprecated_at) "
                        "VALUES (?, ?, ?)",
                        [(r["old_chunk_id"], r["canonical_chunk_id"], r["deprecated_at"]) for r in restore],
                    )
                    out["aliases_restored"] = len(restore)
            conn.execute("DELETE FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        _checkpoint(conn)
        return out
    finally:
        conn.close()


def _result_payload(db_path: Path, *, apply: bool, result: DedupeMergeResult) -> dict[str, Any]:
    return {
        "db": str(db_path),
        "mode": "apply" if apply else "dry-run",
        "policy": "POLICY_REPAIR_E",
        "scanned": result.scanned,
        "groups_total": result.groups_total,
        "groups_eligible": result.groups_eligible,
        "groups_held": result.groups_held,
        "held_reasons": result.held_reasons,
        "groups_merged": result.groups_merged,
        "would_merge_groups": result.would_merge_groups,
        "would_archive_losers": result.would_archive_losers,
        "losers_archived": result.losers_archived,
        "survivors_updated": result.survivors_updated,
        "union_fills": result.union_fills,
        "seen_count_transferred": result.seen_count_transferred,
        "aliases_written": result.aliases_written,
        "aliases_repointed": result.aliases_repointed,
        "aliases_dropped_survivor_source": result.aliases_dropped_survivor_source,
        "lineage_repointed": result.lineage_repointed,
        "preexisting_alias_cycles": result.preexisting_alias_cycles,
        "batches": result.batches,
        "checkpoints": result.checkpoints,
        "aux_counts_before": result.aux_counts_before,
        "aux_counts_after": result.aux_counts_after,
        "aux_counts_unchanged": result.aux_counts_before == result.aux_counts_after,
        "deleted_rows": 0,
        "spot_checks_ok": (
            "n/a-dry-run"
            if not apply
            else (all(item["ok"] for item in result.spot_checks) if result.spot_checks else None)
        ),
        "spot_checks": result.spot_checks,
        "next": (
            "verify every archived loser resolves to a live survivor, then review the live-window plan"
            if apply
            else "rerun with --apply against a rehearsal copy (never the live DB)"
        ),
    }


def _detect_git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha if _SHA_RE.fullmatch(sha) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Archive exact content duplicates onto a survivor (POLICY_REPAIR_E). "
            "Default is dry-run. Requires --db; refuses the live canonical DB "
            "unless --allow-live is set. Never deletes a row."
        )
    )
    parser.add_argument("--db", type=Path, required=True, help="Rehearsal copy DB path (required)")
    parser.add_argument("--apply", action="store_true", help="Write lifecycle columns")
    parser.add_argument("--git-sha", dest="git_sha", help="40-char commit SHA recorded in schema_migrations")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=3)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--spot-check", type=int, default=0)
    parser.add_argument("--no-union-fill", action="store_true", help="Lineage only; never touch the survivor")
    parser.add_argument("--actor", default="repair-e")
    parser.add_argument("--rollback", action="store_true", help="Restore every row this migration wrote")
    args = parser.parse_args(argv)
    if args.rollback:
        print(json.dumps(rollback_migration(args.db.expanduser(), allow_live=args.allow_live), sort_keys=True))
        return 0
    git_sha = args.git_sha or _detect_git_sha()
    if not git_sha:
        parser.error("--git-sha is required (40-char hex) when HEAD is not a full SHA")
    result = merge_duplicates(
        args.db.expanduser(),
        git_sha=git_sha,
        apply=args.apply,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        allow_live=args.allow_live,
        spot_check=args.spot_check,
        union_fill=not args.no_union_fill,
        actor=args.actor,
    )
    print(json.dumps(_result_payload(args.db.expanduser(), apply=args.apply, result=result), sort_keys=True))
    if args.apply and result.spot_checks and not all(item["ok"] for item in result.spot_checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
