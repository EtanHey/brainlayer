"""Repair (e): archive-not-delete merge of exact content duplicates.

The policy these tests pin (POLICY_REPAIR_E, lead-ACKed 2026-08-18):
  * key    = sha256(content.strip()), recomputed; the stored content_hash column
             is never trusted (four schemes live in the canonical DB).
  * scope  = same conversation only. Identical text in two different
             conversations is two memories, not one.
  * survivor = richest enrichment -> oldest created_at -> smallest id.
  * losers = aggregated_into + archived_at + status='superseded'. NEVER deleted.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from brainlayer.dedupe_merge import (
    MIGRATION_NAME,
    PREIMAGE_TABLE,
    content_key,
    merge_duplicates,
)

GIT_SHA = "0" * 40

CHUNK_COLUMNS = """
    id TEXT PRIMARY KEY,
    content TEXT,
    content_hash TEXT,
    source TEXT,
    project TEXT,
    conversation_id TEXT,
    source_file TEXT,
    content_type TEXT,
    char_count INTEGER,
    created_at TEXT,
    summary TEXT,
    tags TEXT,
    importance REAL,
    intent TEXT,
    enriched_at TEXT,
    key_facts TEXT,
    primary_symbols TEXT,
    raw_entities_json TEXT,
    seen_count INTEGER DEFAULT 1,
    archived_at TEXT,
    superseded_by TEXT,
    aggregated_into TEXT,
    status TEXT DEFAULT 'active'
"""


def _db(tmp_path: Path, rows: list[dict]) -> Path:
    db_path = tmp_path / "merge.db"
    conn = sqlite3.connect(db_path)
    conn.execute(f"CREATE TABLE chunks ({CHUNK_COLUMNS})")
    conn.execute(
        "CREATE TABLE chunk_id_alias (old_chunk_id TEXT PRIMARY KEY, "
        "canonical_chunk_id TEXT NOT NULL, deprecated_at TEXT NOT NULL)"
    )
    conn.execute("CREATE TABLE chunk_fts_rowids (chunk_id TEXT PRIMARY KEY, fts_rowid INTEGER)")
    for row in rows:
        cols = ", ".join(row)
        marks = ", ".join("?" for _ in row)
        conn.execute(f"INSERT INTO chunks ({cols}) VALUES ({marks})", list(row.values()))
        conn.execute(
            "INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid) VALUES (?, ?)",
            (row["id"], len(row["id"])),
        )
    conn.commit()
    conn.close()
    return db_path


def _rows(db_path: Path) -> dict[str, sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out = {r["id"]: r for r in conn.execute("SELECT * FROM chunks")}
    conn.close()
    return out


def _pair(**overrides) -> list[dict]:
    """Two rows, same conversation, same text, second better enriched."""
    base = [
        {
            "id": "old-one",
            "content": "the same memory text",
            "conversation_id": "conv-1",
            "source_file": "a.jsonl",
            "source": "claude_code",
            "created_at": "2026-01-01T00:00:00Z",
            "char_count": 20,
            "seen_count": 1,
        },
        {
            "id": "new-two",
            "content": "the same memory text",
            "conversation_id": "conv-1",
            "source_file": "a.jsonl",
            "source": "realtime_watcher",
            "created_at": "2026-01-02T00:00:00Z",
            "summary": "a summary",
            "tags": '["x"]',
            "importance": 8,
            "seen_count": 3,
        },
    ]
    base[1].update(overrides)
    return base


# --- key ---------------------------------------------------------------------


def test_content_key_is_sha256_of_stripped_content():
    assert content_key("  hello \n") == hashlib.sha256(b"hello").hexdigest()
    assert content_key("hello") == content_key("\nhello\n  ")


def test_content_key_rejects_blank_content():
    assert content_key("") is None
    assert content_key("   \n ") is None
    assert content_key(None) is None


def test_stored_content_hash_is_never_trusted(tmp_path):
    """Rows sharing a stored hash but differing in content must NOT merge.

    The canonical DB holds 20 such groups: full sha256 values that went stale
    when the content was later mutated.
    """
    rows = [
        {
            "id": "stale-a",
            "content": "text one",
            "content_hash": "deadbeef" * 8,
            "conversation_id": "c",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "id": "stale-b",
            "content": "text two -- genuinely different",
            "content_hash": "deadbeef" * 8,
            "conversation_id": "c",
            "created_at": "2026-01-02T00:00:00Z",
        },
    ]
    db_path = _db(tmp_path, rows)
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 0
    after = _rows(db_path)
    assert all(r["aggregated_into"] is None for r in after.values())


def test_rows_split_across_hash_schemes_still_merge(tmp_path):
    """A 16-char legacy hash and a 64-char hash on identical text must merge."""
    rows = _pair()
    rows[0]["content_hash"] = "abcdef0123456789"
    rows[1]["content_hash"] = hashlib.sha256(b"the same memory text").hexdigest()
    db_path = _db(tmp_path, rows)
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 1


# --- scope: conversation identity ---------------------------------------------


def test_cross_conversation_duplicates_are_held(tmp_path):
    """The 252-conversation case: identical text in different sessions stays put."""
    rows = _pair()
    rows[1]["conversation_id"] = "conv-2"
    db_path = _db(tmp_path, rows)
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 0
    assert result.groups_held >= 1
    after = _rows(db_path)
    assert after["old-one"]["aggregated_into"] is None
    assert after["new-two"]["aggregated_into"] is None


def test_blank_conversation_id_is_not_a_match(tmp_path):
    """Two NULL conversation ids are unknown, not equal."""
    rows = _pair()
    rows[0]["conversation_id"] = None
    rows[1]["conversation_id"] = None
    db_path = _db(tmp_path, rows)
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 0


# --- survivor selection --------------------------------------------------------


def test_survivor_is_richest_enrichment_not_oldest(tmp_path):
    db_path = _db(tmp_path, _pair())
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 1
    after = _rows(db_path)
    assert after["new-two"]["aggregated_into"] is None, "richest row must survive"
    assert after["old-one"]["aggregated_into"] == "new-two"


def test_oldest_breaks_an_enrichment_tie(tmp_path):
    rows = _pair(summary=None, tags=None, importance=None)
    db_path = _db(tmp_path, rows)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    after = _rows(db_path)
    assert after["old-one"]["aggregated_into"] is None, "tie -> oldest survives"
    assert after["new-two"]["aggregated_into"] == "old-one"


def test_already_managed_rows_are_not_chosen_as_survivor(tmp_path):
    """The richest row is archived already, so a live row must survive instead."""
    rows = _pair()
    rows[1]["archived_at"] = "2026-01-05T00:00:00Z"  # richest, but managed
    rows.append(
        {
            "id": "live-three",
            "content": "the same memory text",
            "conversation_id": "conv-1",
            "source_file": "a.jsonl",
            "created_at": "2026-01-03T00:00:00Z",
            "seen_count": 1,
        }
    )
    db_path = _db(tmp_path, rows)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    after = _rows(db_path)
    assert after["old-one"]["aggregated_into"] is None, "oldest live row survives"
    assert after["live-three"]["aggregated_into"] == "old-one"
    assert after["new-two"]["aggregated_into"] is None, "an already-managed row is left alone"
    assert after["new-two"]["archived_at"] == "2026-01-05T00:00:00Z", "its lifecycle stamp is untouched"


def test_group_with_every_member_already_managed_is_skipped(tmp_path):
    rows = _pair()
    rows[0]["archived_at"] = "2026-01-05T00:00:00Z"
    rows[1]["aggregated_into"] = "somewhere-else"
    db_path = _db(tmp_path, rows)
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 0
    after = _rows(db_path)
    assert after["new-two"]["aggregated_into"] == "somewhere-else", "existing lineage untouched"


# --- loser treatment: archive, never delete -------------------------------------


def test_losers_are_archived_with_lineage_and_never_deleted(tmp_path):
    db_path = _db(tmp_path, _pair())
    before = set(_rows(db_path))
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    after = _rows(db_path)
    assert set(after) == before, "no row may disappear"
    loser = after["old-one"]
    assert loser["aggregated_into"] == "new-two"
    assert loser["archived_at"] is not None
    assert loser["status"] == "superseded"
    assert loser["content"] == "the same memory text", "content is retained for walkback"


def test_loser_status_is_superseded_not_archived(tmp_path):
    """vector_store startup rewrites status='archived'; writing it would revert."""
    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert _rows(db_path)["old-one"]["status"] == "superseded"


def test_fts_and_side_table_rows_are_untouched(tmp_path):
    """Archiving hides losers from default search without index surgery."""
    db_path = _db(tmp_path, _pair())
    conn = sqlite3.connect(db_path)
    before = conn.execute("SELECT COUNT(*) FROM chunk_fts_rowids").fetchone()[0]
    conn.close()
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    conn = sqlite3.connect(db_path)
    after = conn.execute("SELECT COUNT(*) FROM chunk_fts_rowids").fetchone()[0]
    conn.close()
    assert after == before
    assert result.aux_counts_before == result.aux_counts_after


# --- union fill ------------------------------------------------------------------


def test_union_fill_takes_loser_enrichment_into_blank_survivor_fields(tmp_path):
    rows = _pair()
    rows[0]["intent"] = "decision"
    rows[0]["key_facts"] = '{"k": 1}'
    db_path = _db(tmp_path, rows)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    survivor = _rows(db_path)["new-two"]
    assert survivor["intent"] == "decision"
    assert survivor["key_facts"] == '{"k": 1}'
    assert survivor["summary"] == "a summary", "survivor's own values stay"


def test_union_fill_never_overwrites_a_non_blank_survivor_field(tmp_path):
    rows = _pair()
    rows[0]["summary"] = "the loser summary"
    db_path = _db(tmp_path, rows)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert _rows(db_path)["new-two"]["summary"] == "a summary"


def test_seen_count_totals_are_preserved_on_the_survivor(tmp_path):
    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert _rows(db_path)["new-two"]["seen_count"] == 4  # 3 + 1


def test_union_fill_can_be_disabled(tmp_path):
    rows = _pair()
    rows[0]["intent"] = "decision"
    db_path = _db(tmp_path, rows)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True, union_fill=False)
    survivor = _rows(db_path)["new-two"]
    assert survivor["intent"] is None
    assert survivor["seen_count"] == 3, "no survivor mutation at all"


# --- aliases ----------------------------------------------------------------------


def test_alias_row_forwards_loser_id_to_survivor(tmp_path):
    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    conn = sqlite3.connect(db_path)
    alias = dict(conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall())
    conn.close()
    assert alias == {"old-one": "new-two"}


def test_existing_alias_pointing_at_a_loser_is_repointed(tmp_path):
    db_path = _db(tmp_path, _pair())
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO chunk_id_alias VALUES ('ancient', 'old-one', '2026-01-01T00:00:00Z')")
    conn.commit()
    conn.close()
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    conn = sqlite3.connect(db_path)
    alias = dict(conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall())
    conn.close()
    assert alias["ancient"] == "new-two", "alias chains must resolve to the survivor"
    assert alias["old-one"] == "new-two"


def test_no_alias_cycle_is_created(tmp_path):
    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    conn = sqlite3.connect(db_path)
    pairs = conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall()
    conn.close()
    for old, canonical in pairs:
        assert old != canonical
        seen = {old}
        cursor = canonical
        mapping = dict(pairs)
        while cursor in mapping:
            assert cursor not in seen, "alias cycle"
            seen.add(cursor)
            cursor = mapping[cursor]


# --- dry run / preimage / receipt --------------------------------------------------


def test_dry_run_is_the_default_and_writes_nothing(tmp_path):
    db_path = _db(tmp_path, _pair())
    result = merge_duplicates(db_path, git_sha=GIT_SHA)
    assert result.would_merge_groups == 1
    assert result.groups_merged == 0
    after = _rows(db_path)
    assert after["old-one"]["aggregated_into"] is None
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert PREIMAGE_TABLE not in tables


def test_preimage_captures_every_column_written(tmp_path):
    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    pre = {r["id"]: r for r in conn.execute(f"SELECT * FROM {PREIMAGE_TABLE}")}
    conn.close()
    assert set(pre) == {"old-one", "new-two"}
    assert pre["old-one"]["aggregated_into"] is None, "preimage holds the BEFORE value"
    assert pre["old-one"]["status"] == "active"
    assert pre["new-two"]["seen_count"] == 3


def test_rollback_from_preimage_restores_original_rows(tmp_path):
    db_path = _db(tmp_path, _pair())
    before = {k: dict(v) for k, v in _rows(db_path).items()}
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({PREIMAGE_TABLE})") if r[1] != "id"]
    for row in conn.execute(f"SELECT * FROM {PREIMAGE_TABLE}").fetchall():
        conn.execute(
            f"UPDATE chunks SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
            [row[c] for c in cols] + [row["id"]],
        )
    conn.commit()
    conn.close()
    assert {k: dict(v) for k, v in _rows(db_path).items()} == before


def test_receipt_records_migration_and_git_sha(tmp_path):
    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT name, details FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,)).fetchone()
    conn.close()
    assert row is not None
    assert GIT_SHA in row[1]


def test_git_sha_must_be_a_full_commit_sha(tmp_path):
    db_path = _db(tmp_path, _pair())
    with pytest.raises(ValueError):
        merge_duplicates(db_path, git_sha="abc123", apply=True)


def test_resume_is_idempotent(tmp_path):
    db_path = _db(tmp_path, _pair())
    first = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    second = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert first.groups_merged == 1
    assert second.groups_merged == 0, "a second pass must find nothing left to do"
    assert _rows(db_path)["new-two"]["seen_count"] == 4, "seen_count must not double-count"


def test_spot_checks_reread_stored_rows(tmp_path):
    db_path = _db(tmp_path, _pair())
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True, spot_check=5)
    assert result.spot_checks
    assert all(check["ok"] for check in result.spot_checks)
    assert any(check["id"] == "old-one" for check in result.spot_checks)


# --- live guard --------------------------------------------------------------------


def test_refuses_the_live_canonical_db(tmp_path, monkeypatch):
    from brainlayer import chunk_origin_wipe

    live = tmp_path / "live" / "brainlayer.db"
    live.parent.mkdir(parents=True)
    _db(tmp_path, _pair()).replace(live)
    # assert_not_live_db lives in chunk_origin_wipe and reads ITS globals.
    monkeypatch.setattr(chunk_origin_wipe, "live_canonical_db_path", lambda: live)
    monkeypatch.setattr(chunk_origin_wipe, "account_home", lambda: live.parent.parent)
    with pytest.raises(RuntimeError):
        merge_duplicates(live, git_sha=GIT_SHA, apply=True)
    # and it proceeds once a lead-scheduled window sets allow_live
    merge_duplicates(live, git_sha=GIT_SHA, apply=True, allow_live=True)


# --- alias convergence: the cycles the rehearsal exposed ------------------------
# The first rehearsal run turned 72 pre-existing alias cycles into 588, because
# prior dedupe runs had already aliased members of one duplicate group to each
# other and a new `loser -> survivor` edge closed the loop. Every alias edge in a
# group must terminate at the survivor, and the survivor may not be an alias source.


def _today_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _alias_rows(db_path: Path) -> dict[str, str]:
    conn = sqlite3.connect(db_path)
    rows = dict(conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall())
    conn.close()
    return rows


def test_survivor_is_never_left_as_an_alias_source(tmp_path):
    """An alias survivor -> loser would send lookups to an archived row."""
    db_path = _db(tmp_path, _pair())
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO chunk_id_alias VALUES ('new-two', 'old-one', '2026-01-01T00:00:00Z')")
    conn.commit()
    conn.close()
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    alias = _alias_rows(db_path)
    assert "new-two" not in alias, "the live survivor must not be a deprecated id"
    assert alias["old-one"] == "new-two"
    assert result.aliases_dropped_survivor_source == 1


def test_pre_existing_intra_group_alias_chain_is_converged(tmp_path):
    """Prior runs chained members of the same group; all edges must land on the survivor."""
    rows = _pair()
    for index in range(3, 6):
        rows.append(
            {
                "id": f"member-{index}",
                "content": "the same memory text",
                "conversation_id": "conv-1",
                "source_file": "a.jsonl",
                "created_at": f"2026-01-0{index}T00:00:00Z",
                "seen_count": 1,
            }
        )
    db_path = _db(tmp_path, rows)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO chunk_id_alias VALUES (?, ?, '2026-01-01T00:00:00Z')",
        [("member-3", "member-4"), ("member-4", "member-5"), ("member-5", "old-one")],
    )
    conn.commit()
    conn.close()

    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    alias = _alias_rows(db_path)
    assert set(alias.values()) == {"new-two"}, f"all edges must converge: {alias}"
    for old, canonical in alias.items():
        assert canonical not in alias, "a converged graph has no second hop"


def test_alias_cycle_introduced_by_this_migration_fails_loudly(tmp_path, monkeypatch):
    from brainlayer import dedupe_merge

    db_path = _db(tmp_path, _pair())

    def _sabotage(conn, *, survivor_id, losers, stamp, result):
        conn.executemany(
            "INSERT OR REPLACE INTO chunk_id_alias VALUES (?, ?, ?)",
            [(losers[0], survivor_id, stamp), (survivor_id, losers[0], stamp)],
        )

    monkeypatch.setattr(dedupe_merge, "_converge_aliases", _sabotage)
    with pytest.raises(RuntimeError, match="alias-cycle"):
        merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)


def test_pre_existing_cycle_is_reported_not_fatal(tmp_path):
    """A cycle among ids this run never touches must not block the migration."""
    db_path = _db(tmp_path, _pair())
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO chunk_id_alias VALUES (?, ?, '2026-01-01T00:00:00Z')",
        [("ghost-a", "ghost-b"), ("ghost-b", "ghost-a")],
    )
    conn.commit()
    conn.close()
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert result.groups_merged == 1
    assert result.preexisting_alias_cycles == 2


def test_alias_edits_are_reversible_from_their_preimage(tmp_path):
    from brainlayer.dedupe_merge import ALIAS_PREIMAGE_TABLE

    db_path = _db(tmp_path, _pair())
    conn = sqlite3.connect(db_path)
    conn.execute("INSERT INTO chunk_id_alias VALUES ('new-two', 'elsewhere', '2026-01-01T00:00:00Z')")
    conn.commit()
    conn.close()
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)

    conn = sqlite3.connect(db_path)
    recorded = conn.execute(f"SELECT old_chunk_id, canonical_chunk_id, action FROM {ALIAS_PREIMAGE_TABLE}").fetchall()
    conn.close()
    assert ("new-two", "elsewhere", "drop_survivor_source") in recorded


def test_existing_lineage_pointing_at_a_loser_is_repointed_to_the_survivor(tmp_path):
    """A chain must always end one hop away, on a live row.

    Rehearsal found 5 rows that an earlier aggregation had pointed at a chunk
    this migration then archived, leaving them two hops from anything live.
    """
    rows = _pair()
    rows.append(
        {
            "id": "older-aggregated",
            "content": "a different memory entirely",
            "conversation_id": "conv-9",
            "created_at": "2025-12-01T00:00:00Z",
            "aggregated_into": "old-one",
            "archived_at": "2025-12-02T00:00:00Z",
            "status": "superseded",
        }
    )
    db_path = _db(tmp_path, rows)
    result = merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    after = _rows(db_path)
    assert after["old-one"]["aggregated_into"] == "new-two"
    assert after["older-aggregated"]["aggregated_into"] == "new-two", "must not point at an archived row"
    assert result.lineage_repointed == 1
    survivor = after[after["older-aggregated"]["aggregated_into"]]
    assert survivor["aggregated_into"] is None and survivor["archived_at"] is None


def test_no_lineage_pointer_ends_on_an_archived_row(tmp_path):
    rows = _pair()
    rows.append(
        {
            "id": "older-aggregated",
            "content": "another memory",
            "conversation_id": "conv-9",
            "created_at": "2025-12-01T00:00:00Z",
            "aggregated_into": "old-one",
            "archived_at": "2025-12-02T00:00:00Z",
        }
    )
    db_path = _db(tmp_path, rows)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    after = _rows(db_path)
    for row in after.values():
        target = row["aggregated_into"]
        if target:
            assert after[target]["archived_at"] is None, f"{row['id']} points at an archived row"
            assert after[target]["aggregated_into"] is None


# --- rollback as a command -------------------------------------------------------


def _snapshot(db_path: Path) -> tuple[dict, dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    chunks = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM chunks")}
    alias = dict(conn.execute("SELECT old_chunk_id, canonical_chunk_id FROM chunk_id_alias").fetchall())
    conn.close()
    return chunks, alias


def test_rollback_restores_chunks_and_aliases_exactly(tmp_path):
    from brainlayer.dedupe_merge import rollback_migration

    rows = _pair()
    rows.append(
        {
            "id": "older-aggregated",
            "content": "unrelated memory",
            "conversation_id": "conv-9",
            "created_at": "2025-12-01T00:00:00Z",
            "aggregated_into": "old-one",
            "archived_at": "2025-12-02T00:00:00Z",
        }
    )
    db_path = _db(tmp_path, rows)
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "INSERT INTO chunk_id_alias VALUES (?, ?, '2026-01-01T00:00:00Z')",
        [("ancient", "old-one"), ("new-two", "elsewhere"), ("untouched", "somewhere")],
    )
    conn.commit()
    conn.close()

    before = _snapshot(db_path)
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    assert _snapshot(db_path) != before, "the migration must have changed something"

    result = rollback_migration(db_path)
    assert result["chunks_restored"] > 0
    assert _snapshot(db_path) == before, "rollback must restore byte-for-byte"


def test_rollback_does_not_touch_unrelated_alias_rows(tmp_path):
    """The hand-written rollback deleted by date and lost 4,817 pre-existing rows."""
    from brainlayer.dedupe_merge import rollback_migration

    db_path = _db(tmp_path, _pair())
    conn = sqlite3.connect(db_path)
    # a pre-existing alias carrying the SAME day stamp the migration will write
    conn.execute("INSERT INTO chunk_id_alias VALUES ('bystander', 'somewhere-else', ?)", (_today_stamp(),))
    conn.commit()
    conn.close()

    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    rollback_migration(db_path)
    alias = _alias_rows(db_path)
    assert alias.get("bystander") == "somewhere-else", "a same-day bystander must survive rollback"
    assert "old-one" not in alias, "aliases this migration created must be gone"


def test_rollback_removes_the_receipt(tmp_path):
    from brainlayer.dedupe_merge import rollback_migration

    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    rollback_migration(db_path)
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT 1 FROM schema_migrations WHERE name = ?", (MIGRATION_NAME,)).fetchone()
    conn.close()
    assert row is None


def test_rollback_retains_the_preimage_audit_trail(tmp_path):
    from brainlayer.dedupe_merge import rollback_migration

    db_path = _db(tmp_path, _pair())
    merge_duplicates(db_path, git_sha=GIT_SHA, apply=True)
    result = rollback_migration(db_path)
    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert PREIMAGE_TABLE in tables, "the preimage is the audit trail; removal is a separate human step"
    assert result["preimage_tables"] == "retained"
