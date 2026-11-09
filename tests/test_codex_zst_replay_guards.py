"""Bounded retries must advance past content already committed under another source."""

import json
import sqlite3

import pytest

from brainlayer.drain import drain_once
from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from brainlayer.vector_store import VectorStore
from tests.test_codex_zst_manifest import manifest
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


@pytest.mark.parametrize("count", [1, 383])
def test_dedup_retries_complete_without_seen_bumps(compress, tmp_path, monkeypatch, count):
    queue, db = tmp_path / "queue", tmp_path / "fixture.db"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    drain = dict(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", pause_sentinel_path=tmp_path / "absent")
    seed_count = min(count, 30)
    records = [
        rollout(f"Remember distinct window item {i} for tomorrow.").splitlines(keepends=True)[1] for i in range(count)
    ]
    header = rollout().splitlines(keepends=True)[0]
    seed = manifest(tmp_path, [compress("existing.jsonl.zst", header + b"".join(records[:seed_count]))])
    for _ in range(2):
        list(ingest_codex_manifest(seed, db))
        drain_once(**drain)
    selected = manifest(tmp_path, [compress("replay.jsonl.zst", header + b"".join(records))])
    total = 0
    for _ in range(20):
        receipt = list(ingest_codex_manifest(selected, db))[0]
        total += receipt["queued"]
        drain_once(**drain)
        if receipt["class"] == "accepted":
            break
    assert receipt["class"] == "accepted", receipt
    assert total == count - seed_count
    with sqlite3.connect(db) as conn:
        before = conn.execute("SELECT id, seen_count FROM chunks ORDER BY id").fetchall()
    monkeypatch.setattr("brainlayer.codex_zst.queue_has_capacity", lambda: False)
    retry = list(ingest_codex_manifest(selected, db))[0]
    assert retry["queued"] == 0 and retry["duplicate_content"] == seed_count
    assert retry["live_chunks_at_start"] == count - seed_count and retry["filtered"] == 0
    assert len(retry["duplicate_matches"]) == min(seed_count, 5)
    assert retry["duplicate_matches"][0]["source_file"].endswith("existing.jsonl")
    if count == 1:
        assert retry["state"] == retry["reason"] == "indexed_elsewhere_dedup"
    assert drain_once(**drain) == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT id, seen_count FROM chunks ORDER BY id").fetchall() == before
        assert len(before) == count and all(row[1] == 1 for row in before)


def test_machine_filter_receipts_never_drop_invalid_rows(compress, tmp_path):
    selected = manifest(tmp_path, [compress("machine.jsonl.zst", rollout())])
    template = json.loads(selected.read_text())
    selected.write_text(
        "".join(json.dumps(dict(template, machine=m)) + "\n" for m in ["M4", "M1", "m4", None, [], {}])
        + json.dumps({k: v for k, v in template.items() if k != "machine"})
        + "\n"
    )
    rows = list(ingest_codex_manifest(selected, tmp_path / "x.db", dry_run=True, machine="M4"))
    assert len(rows) == 7
    assert rows[0]["class"] == "readable"
    assert (rows[1]["class"], rows[1]["state"]) == ("filtered", "skipped_other_machine")
    assert all((r["class"], r["reason"]) == ("error", "invalid_machine") for r in rows[2:])


def test_machine_selector_rejects_typo_before_rows(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="manifest-machine must be M1 or M4"):
        list(ingest_codex_manifest(tmp_path / "absent", tmp_path / "x.db", dry_run=True, machine="m4"))


def test_blank_manifest_lines_and_quarantined_tail(compress, tmp_path):
    data = rollout() + b'{"partial":'
    selected = manifest(tmp_path, [compress("tail.jsonl.zst", data)])
    selected.write_text("\n" + selected.read_text() + "\n")
    rows = list(ingest_codex_manifest(selected, tmp_path / "x.db", dry_run=True))
    assert len(rows) == 1
    assert rows[0]["class"] == "readable" and rows[0]["reason"] == "quarantined_tail"
    assert rows[0]["would_insert"] == 1 and rows[0]["decoded_bytes_total"] == len(data)
