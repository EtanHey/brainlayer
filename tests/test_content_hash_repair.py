"""#623 bounded repair: only stale merged rows, dry-run first, never the live DB."""

from __future__ import annotations

import hashlib

import apsw
import pytest

from brainlayer import content_hash_repair
from brainlayer.content_hash_repair import repair_merged_content_hashes
from brainlayer.dedupe import merge_existing_chunk_content
from brainlayer.vector_store import VectorStore


def _sha(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


def _insert(conn, chunk_id: str, content: str, content_hash: str | None) -> None:
    conn.execute(
        """INSERT INTO chunks(id, content, metadata, source_file, content_hash, created_at, seen_count)
           VALUES (?, ?, '{}', 'test.jsonl', ?, '2026-10-22T01:00:00Z', 1)""",
        (chunk_id, content, content_hash),
    )


def _stale_merge(conn, chunk_id: str, original: str) -> str:
    """Leave the exact pre-fix state: merged body, hash of the pre-merge body."""
    _insert(conn, chunk_id, original, _sha(original))
    merge_existing_chunk_content(
        conn, chunk_id=chunk_id, incoming={"content": original + " plus a later revision"}, ensure_schema=False
    )
    conn.execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (_sha(original), chunk_id))
    return conn.execute("SELECT content FROM chunks WHERE id = ?", (chunk_id,)).fetchone()[0]


def _hashes(db) -> dict[str, str | None]:
    conn = apsw.Connection(str(db), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        return dict(conn.execute("SELECT id, content_hash FROM chunks ORDER BY id").fetchall())
    finally:
        conn.close()


@pytest.fixture
def copy_db(tmp_path):
    db = tmp_path / "copy.db"
    store = VectorStore(db)
    conn = store.conn
    merged = {
        cid: _stale_merge(conn, cid, f"Original note {cid} about the copper lantern") for cid in ("m1", "m2", "m3")
    }
    _insert(conn, "plain-legacy", "Ordinary row with a legacy hash scheme", "abcd1234abcd1234")
    _insert(conn, "merged-null", "[Merged duplicate originals]\n\n␞ 1. a\n\n␞ 2. b", None)
    fts_before = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    store.close()
    return db, merged, fts_before


def test_dry_run_reports_and_writes_nothing(copy_db):
    db, merged, _ = copy_db
    before = _hashes(db)
    result = repair_merged_content_hashes(db)
    assert result["mode"] == "dry-run"
    assert result["stale"] == 3 and result["repaired"] == 0
    assert sorted(result["sample_ids"]) == ["m1", "m2", "m3"]
    assert _hashes(db) == before


def test_apply_repairs_only_stale_merged_rows_and_is_idempotent(copy_db):
    db, merged, fts_before = copy_db
    result = repair_merged_content_hashes(db, apply=True, batch_size=2)
    assert (result["stale"], result["repaired"], result["batches"]) == (3, 3, 2)
    after = _hashes(db)
    for chunk_id, content in merged.items():
        assert after[chunk_id] == _sha(content)
    assert after["plain-legacy"] == "abcd1234abcd1234"
    assert after["merged-null"] is None

    again = repair_merged_content_hashes(db, apply=True)
    assert (again["stale"], again["repaired"]) == (0, 0)
    assert _hashes(db) == after

    store = VectorStore(db)
    try:
        assert store.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == fts_before
        from brainlayer.drain import _apply_enrichment

        _apply_enrichment(
            store.conn, {"chunk_id": "m1", "content_hash": _sha(merged["m1"]), "enrichment": {"summary": "fresh"}}
        )
        assert store.conn.execute("SELECT summary FROM chunks WHERE id = 'm1'").fetchone()[0] == "fresh"
    finally:
        store.close()


def test_apply_is_bounded_per_run(copy_db):
    db, _, _ = copy_db
    first = repair_merged_content_hashes(db, apply=True, max_rows=2)
    assert (first["repaired"], first["more_remaining"]) == (2, True)
    second = repair_merged_content_hashes(db, apply=True, max_rows=2)
    assert (second["repaired"], second["more_remaining"]) == (1, False)


def test_row_changed_after_scan_is_skipped_not_stamped(copy_db, monkeypatch):
    db, _, _ = copy_db
    real_find = content_hash_repair._find_candidates

    def find_then_concurrent_edit(conn, **kwargs):
        found = real_find(conn, **kwargs)
        other = apsw.Connection(str(db))
        other.execute("UPDATE chunks SET content = content || ' edited' WHERE id = 'm2'")
        other.close()
        return found

    monkeypatch.setattr(content_hash_repair, "_find_candidates", find_then_concurrent_edit)
    result = repair_merged_content_hashes(db, apply=True)
    assert (result["repaired"], result["skipped_identity_changed"]) == (2, 1)


def test_refuses_live_db_and_missing_db(copy_db, monkeypatch, tmp_path):
    db, _, _ = copy_db
    before = _hashes(db)
    monkeypatch.setattr("brainlayer.chunk_origin_wipe._live_db_candidates", lambda: [db])
    with pytest.raises(RuntimeError, match="refusing to write the live BrainLayer DB"):
        repair_merged_content_hashes(db, apply=True)
    assert _hashes(db) == before

    missing = tmp_path / "absent.db"
    with pytest.raises(FileNotFoundError):
        repair_merged_content_hashes(missing, apply=True)
    assert not missing.exists()


def test_cli_defaults_to_dry_run(copy_db, capsys):
    db, _, _ = copy_db
    before = _hashes(db)
    assert content_hash_repair.main(["--db", str(db)]) == 0
    assert '"mode": "dry-run"' in capsys.readouterr().out
    assert _hashes(db) == before
