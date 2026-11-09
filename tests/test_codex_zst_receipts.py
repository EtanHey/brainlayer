"""Recovery outcomes remain durable and cover every dedup attribution."""

import json
import sqlite3

import pytest

from brainlayer.drain import drain_once
from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from brainlayer.vector_store import VectorStore
from tests.test_codex_zst_alias import replay, selected
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_receipts_are_required_exclusive_and_complete(compress, tmp_path, monkeypatch):
    db, queue = tmp_path / "x.db", tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    content = rollout().splitlines(keepends=True)[0] + b"".join(
        rollout(f"Remember distinct receipt item {n} tomorrow.").splitlines(keepends=True)[1] for n in range(8)
    )
    manifest = selected(tmp_path, [compress("first.jsonl.zst", content)])
    with pytest.raises(ValueError, match="receipts-out is required"):
        list(ingest_codex_manifest(manifest, db))
    assert not list(queue.glob("*.jsonl"))
    replay(manifest, db)
    drain_once(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", pause_sentinel_path=tmp_path / "absent")
    manifest = selected(tmp_path, [compress("second.jsonl.zst", content)])
    ledger = tmp_path / "receipt.jsonl"
    row = list(ingest_codex_manifest(manifest, db, receipts_out=ledger))[0]
    saved = json.loads(ledger.read_text())
    assert len(saved["all_duplicate_matches"]) == row["duplicate_content"] == 8
    assert len(row["duplicate_matches"]) == 5
    assert all(
        m["candidate_chunk_id"] and m["chunk_id"] and m["source_file"] and m["mechanism"] == "exact_hash"
        for m in saved["all_duplicate_matches"]
    )
    assert len({m["candidate_chunk_id"] for m in saved["all_duplicate_matches"]}) == 8
    assert saved["live_chunk_ids_at_start"] == [] and saved["queued_chunk_ids"] == []
    before = ledger.read_bytes()
    with pytest.raises(FileExistsError):
        list(ingest_codex_manifest(manifest, db, receipts_out=ledger))
    assert ledger.read_bytes() == before


def test_archived_same_id_reactivates_and_live_count_excludes_lifecycle(compress, tmp_path, monkeypatch):
    db, queue = tmp_path / "x.db", tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    manifest = selected(tmp_path, [compress("same.jsonl.zst", rollout())])
    args = dict(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", pause_sentinel_path=tmp_path / "absent")
    replay(manifest, db)
    drain_once(**args)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE chunks SET archived_at='2026-11-08'")
    row = replay(manifest, db)[0]
    assert row["live_chunks_at_start"] == 0 and row["queued"] == 1
    drain_once(**args)
    assert replay(manifest, db)[0]["live_chunks_at_start"] == 1
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE chunks SET superseded_by='replacement'")
    row = replay(manifest, db)[0]
    assert row["live_chunks_at_start"] == 0 and row["queued"] == 0
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE chunks SET superseded_by=NULL, aggregated_into='summary'")
    row = replay(manifest, db)[0]
    assert row["live_chunks_at_start"] == 0 and row["queued"] == 0
    ledger = tmp_path / "final.jsonl"
    list(ingest_codex_manifest(manifest, db, receipts_out=ledger))
    assert json.loads(ledger.read_text())["live_chunk_ids_at_start"] == []


def test_archived_same_id_with_live_duplicate_converges_and_is_in_ledger(compress, tmp_path, monkeypatch):
    db, queue = tmp_path / "x.db", tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    args = dict(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", pause_sentinel_path=tmp_path / "absent")
    text = "The quarterly capacity review moved to the second Tuesday of each month"
    a = selected(tmp_path, [compress("a.jsonl.zst", rollout(text))])
    replay(a, db)
    drain_once(**args)
    with sqlite3.connect(db) as conn:
        own = conn.execute("SELECT id FROM chunks").fetchone()[0]
        conn.execute("UPDATE chunks SET archived_at='2026-11-08'")
    other = tmp_path / "b"
    other.mkdir()
    replay(selected(other, [compress("b/b.jsonl.zst", rollout(text))]), db)
    drain_once(**args)
    with sqlite3.connect(db) as conn:
        live, seen = conn.execute("SELECT id, seen_count FROM chunks WHERE archived_at IS NULL").fetchone()
    for n in range(3):
        ledger = tmp_path / f"recovered-{n}.jsonl"
        row = list(ingest_codex_manifest(a, db, receipts_out=ledger))[0]
        assert row["queued"] == 0 and row["duplicate_content"] == 1
        match = json.loads(ledger.read_text())["all_duplicate_matches"][0]
        assert match["candidate_chunk_id"] == own and match["chunk_id"] == live
        drain_once(**args)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT seen_count FROM chunks WHERE id=?", (live,)).fetchone() == (seen,)
