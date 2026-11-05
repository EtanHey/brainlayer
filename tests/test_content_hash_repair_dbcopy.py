"""Adversarial DB-copy proof for #623 (stacked on #1637): both merge writers and the repair.

Every fixture is a private synthetic copy built with the production ``VectorStore``
schema (FTS triggers, vec0 tables, the content_hash index). Nothing here opens,
copies or names a real database.

The invariant under attack: after a content-changing merge, and after the repair,
``content_hash == sha256(content.strip())``; every other column, FTS row and
vector row is left byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time

import apsw
import pytest
import sqlite_vec

from brainlayer import content_hash_repair
from brainlayer.content_hash_repair import repair_merged_content_hashes
from brainlayer.dedupe import (
    _MERGED_MARKER,
    _merged_originals,
    _whitespace_normalized,
    compute_dedupe_fields,
    merge_duplicate_chunk,
    merge_existing_chunk_content,
)
from brainlayer.vector_store import VectorStore

CREATED = "2026-10-22T01:00:00Z"


def _sha(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _insert(conn, chunk_id, content, content_hash, *, source_file="synthetic.jsonl", project="synthetic"):
    conn.execute(
        """INSERT INTO chunks(id, content, metadata, source_file, project, content_hash, created_at, seen_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
        (chunk_id, content, json.dumps({"origin": chunk_id}), source_file, project, content_hash, CREATED),
    )


def _merge(conn, kind: str, chunk_id: str, content: str, n: int) -> bool:
    incoming = {"content": content, "created_at": f"2026-10-22T02:{n % 60:02d}:00Z"}
    conn.execute("BEGIN IMMEDIATE")
    try:
        if kind == "duplicate":
            changed = merge_duplicate_chunk(
                conn,
                canonical_id=chunk_id,
                duplicate_id=f"{chunk_id}-dup-{n}",
                incoming=incoming,
                mechanism="simhash",
                hamming_distance_value=3,
                ensure_schema=False,
            )
        else:
            changed = merge_existing_chunk_content(conn, chunk_id=chunk_id, incoming=incoming, ensure_schema=False)
        conn.execute("COMMIT")
        return changed
    except BaseException:
        conn.execute("ROLLBACK")
        raise


# Inputs chosen to hit every branch of _merged_content: framing escapes, legacy
# bodies, whitespace-only/empty incoming, surrounding whitespace, re-sent originals.
TRICKY = [
    "  copper lantern note with leading and trailing whitespace \n",
    "record separator ␞ and escape ␛ inside an original",
    "escaped-looking ␛R and ␛E sequences",
    "",
    "   \n\t ",
    "שלום unicode original — em dash, ñ, 漢字",
    _MERGED_MARKER + " an original that merely starts with the marker",
    "1. numbered list\n\n2. that looks like a legacy heading",
]


@pytest.mark.parametrize("kind", ["duplicate", "content"])
@pytest.mark.parametrize("seed", range(6))
def test_merge_sequences_keep_hash_equal_to_current_trimmed_content(tmp_path, kind, seed):
    rng = random.Random(seed)
    store = VectorStore(tmp_path / "seq.db")
    conn = store.conn
    try:
        start = rng.choice(
            [
                "plain starting body for the ledger",
                _MERGED_MARKER + "\n\n1. legacy first original\n\n2. legacy second original",
            ]
        )
        _insert(conn, "c", start, _sha(start))
        sent = []
        for n in range(12):
            incoming = rng.choice(TRICKY + sent + [f"fresh original {seed}-{n} about the amber footnote"])
            before_content, before_hash = conn.execute(
                "SELECT content, content_hash FROM chunks WHERE id='c'"
            ).fetchone()
            changed = _merge(conn, kind, "c", incoming, n)
            content, stored, dedupe_hash, simhash = conn.execute(
                "SELECT content, content_hash, dedupe_hash, simhash FROM chunks WHERE id='c'"
            ).fetchone()
            assert changed == (content != before_content)
            if changed:
                assert stored == _sha(content), f"seed={seed} step={n}: stale hash after content-changing merge"
            else:
                assert (content, stored) == (before_content, before_hash), "no-op merge touched content/hash"
            fields = compute_dedupe_fields(content, CREATED)
            assert (dedupe_hash, simhash) == (fields.dedupe_hash, fields.simhash)
            if incoming.strip():
                sent.append(incoming)
                # Retained original text: every non-empty original is still recoverable.
                normalized = {_whitespace_normalized(o) for o in _merged_originals(content)}
                assert (
                    _whitespace_normalized(incoming) in normalized
                    or _whitespace_normalized(content) == (_whitespace_normalized(incoming))
                    or start.startswith(_MERGED_MARKER)
                )
    finally:
        store.close()


def _stale_merged(conn, chunk_id: str, n: int) -> None:
    """The exact state the pre-fix writer left: merged body, hash of the pre-merge body."""
    original = f"original {chunk_id} about the copper lantern, revision {n}"
    _insert(conn, chunk_id, original, _sha(original))
    merge_existing_chunk_content(
        conn, chunk_id=chunk_id, incoming={"content": original + " plus a later amber footnote"}, ensure_schema=False
    )
    conn.execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (_sha(original), chunk_id))


def _snapshot(db) -> dict:
    conn = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    conn.enable_load_extension(True)
    conn.load_extension(sqlite_vec.loadable_path())
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(chunks)")]
        others = [c for c in cols if c != "content_hash"]
        rows = conn.execute(f"SELECT {', '.join(others)} FROM chunks ORDER BY id").fetchall()
        return {
            "rows": rows,
            "hashes": dict(conn.execute("SELECT id, content_hash FROM chunks").fetchall()),
            "fts": conn.execute("SELECT chunk_id, content FROM chunks_fts ORDER BY chunk_id").fetchall(),
            "vectors": conn.execute("SELECT chunk_id, embedding FROM chunk_vectors ORDER BY chunk_id").fetchall(),
            "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
            "indexes": sorted(r[1] for r in conn.execute("PRAGMA index_list(chunks)")),
        }
    finally:
        conn.close()


@pytest.fixture
def big_copy(tmp_path):
    """~2.4k-row synthetic copy: stale/fresh/NULL/legacy merged rows, plain stale rows, vectors."""
    db = tmp_path / "copy.db"
    store = VectorStore(db)
    conn = store.conn
    stale = [f"stale-{i:04d}" for i in range(900)]
    conn.execute("BEGIN IMMEDIATE")
    for i, chunk_id in enumerate(stale):
        _stale_merged(conn, chunk_id, i)
    for i in range(300):
        _insert(conn, f"fresh-{i:04d}", f"plain fresh row {i}", None)
        merge_existing_chunk_content(conn, chunk_id=f"fresh-{i:04d}", incoming={"content": f"other {i}"})
    for i in range(100):
        body = f"{_MERGED_MARKER}\n\n␞ 1. null hash {i}\n\n␞ 2. second {i}"
        _insert(conn, f"mnull-{i:04d}", body, None)
    for i in range(800):  # Ordinary rows: legacy-scheme, edited-later (stale) and canonical hashes.
        content = f"ordinary row {i} with sixteen chars"
        stored = [content[:16].encode().hex()[:16], _sha(content + " before an edit"), _sha(content)][i % 3]
        _insert(conn, f"plain-{i:04d}", content, stored)
    for chunk_id in (*stale[:50], "plain-0001", "plain-0002"):
        store._upsert_chunk_vector(conn.cursor(), chunk_id, [0.001 * (len(chunk_id) % 7)] * 1024)
    conn.execute("COMMIT")
    store.close()
    return db, stale


def test_bounded_batched_repair_converges_and_touches_only_content_hash(big_copy):
    db, stale = big_copy
    before = _snapshot(db)
    dry = repair_merged_content_hashes(db, batch_size=37, max_rows=250)
    assert (dry["stale"], dry["repaired"], dry["more_remaining"]) == (250, 0, True)
    assert _snapshot(db) == before

    runs = []
    while True:
        result = repair_merged_content_hashes(db, apply=True, batch_size=37, max_rows=250)
        runs.append(result)
        if result["stale"] == 0:
            break
        assert len(runs) < 10, "repair did not converge"
    assert [r["repaired"] for r in runs] == [250, 250, 250, 150, 0]
    assert all(r["skipped_identity_changed"] == 0 for r in runs)

    after = _snapshot(db)
    assert after["rows"] == before["rows"], "a column other than content_hash changed"
    assert (after["fts"], after["vectors"], after["indexes"]) == (before["fts"], before["vectors"], before["indexes"])
    assert after["integrity"] == "ok"
    conn = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        for chunk_id, content, stored in conn.execute("SELECT id, content, content_hash FROM chunks"):
            if chunk_id.startswith(("stale-", "fresh-")):
                assert stored == _sha(content), chunk_id
            else:  # NULL merged and every ordinary row: exactly as before.
                assert stored == before["hashes"][chunk_id], chunk_id
    finally:
        conn.close()
    assert repair_merged_content_hashes(db)["stale"] == 0


def test_mid_batch_fault_rolls_back_whole_batch_and_restart_finishes(big_copy):
    db, stale = big_copy
    conn = apsw.Connection(str(db))
    conn.execute(  # Fault on the 3rd row of the 2nd batch (batch_size=40, rowid order).
        f"""CREATE TRIGGER inject_fault BEFORE UPDATE OF content_hash ON chunks WHEN NEW.id = '{stale[42]}'
            BEGIN SELECT RAISE(ABORT, 'injected mid-batch fault'); END"""
    )
    conn.close()
    with pytest.raises(apsw.ConstraintError, match="injected mid-batch fault"):
        repair_merged_content_hashes(db, apply=True, batch_size=40, max_rows=200)
    hashes = _snapshot(db)["hashes"]
    reader = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    content = dict(reader.execute("SELECT id, content FROM chunks").fetchall())
    reader.close()
    assert all(hashes[c] == _sha(content[c]) for c in stale[:40]), "committed batch 1 lost"
    assert all(hashes[c] != _sha(content[c]) for c in stale[40:]), "faulted batch partially committed"

    conn = apsw.Connection(str(db))
    conn.execute("DROP TRIGGER inject_fault")
    conn.close()
    result = repair_merged_content_hashes(db, apply=True, batch_size=40, max_rows=5000)
    assert (result["stale"], result["repaired"]) == (len(stale) - 40, len(stale) - 40)
    assert repair_merged_content_hashes(db, apply=True)["repaired"] == 0


def test_row_merged_by_fixed_writer_between_scan_and_write_is_skipped_and_already_correct(big_copy, monkeypatch):
    db, stale = big_copy
    real_find = content_hash_repair._find_candidates

    def scan_then_concurrent_merge(conn, **kwargs):
        found = real_find(conn, **kwargs)
        writer = VectorStore(db)  # A second writer lands after the scan, before BEGIN.
        try:
            assert _merge(writer.conn, "content", stale[0], "a concurrent revision of the copper lantern", 0)
        finally:
            writer.close()
        return found

    monkeypatch.setattr(content_hash_repair, "_find_candidates", scan_then_concurrent_merge)
    result = repair_merged_content_hashes(db, apply=True, batch_size=100, max_rows=5000)
    assert (result["repaired"], result["skipped_identity_changed"]) == (len(stale) - 1, 1)
    reader = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    content, stored = reader.execute("SELECT content, content_hash FROM chunks WHERE id=?", (stale[0],)).fetchone()
    reader.close()
    assert "concurrent revision" in content and stored == _sha(content)
    monkeypatch.setattr(content_hash_repair, "_find_candidates", real_find)
    assert repair_merged_content_hashes(db)["stale"] == 0


def test_apply_subprocess_waits_for_a_held_writer_lock_then_completes(big_copy):
    db, stale = big_copy
    holder = apsw.Connection(str(db))
    holder.execute("BEGIN IMMEDIATE")
    env = {**os.environ, "BRAINLAYER_DB": str(db.parent / "not-the-copy.db")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "brainlayer.content_hash_repair", "--db", str(db), "--apply", "--max-rows", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        time.sleep(1.5)
        assert proc.poll() is None, f"repair did not wait on the lock: {proc.communicate()}"
        dry = subprocess.run(  # A read-only dry-run is not blocked by the writer (WAL).
            [sys.executable, "-m", "brainlayer.content_hash_repair", "--db", str(db)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        assert dry.returncode == 0, dry.stderr
        assert json.loads(dry.stdout)["stale"] == len(stale)
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    out, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err
    assert json.loads(out)["repaired"] == len(stale)


def test_copy_named_by_brainlayer_db_or_hardlinked_to_live_is_refused(big_copy, monkeypatch, tmp_path):
    db, _ = big_copy
    before = _snapshot(db)
    monkeypatch.setenv("BRAINLAYER_DB", str(db))
    with pytest.raises(RuntimeError, match="refusing to write the live"):
        repair_merged_content_hashes(db, apply=True)
    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "elsewhere.db"))
    live = tmp_path / "pretend-live.db"
    os.link(db, live)
    monkeypatch.setattr("brainlayer.chunk_origin_wipe.live_canonical_db_path", lambda: live)
    with pytest.raises(RuntimeError, match="refusing to write the live"):
        repair_merged_content_hashes(db, apply=True)
    assert _snapshot(db) == before
