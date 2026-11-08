import json
import uuid

from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from brainlayer.watcher import OffsetRegistry
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def manifest(tmp_path, files):
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            dict(
                physical_path=str(path),
                logical_path=str(path.with_suffix("")),
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                inode=stat.st_ino,
                machine="fixture",
            )
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_dry_run_writes_nothing_and_pins_stats(compress, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    path = compress("fork.jsonl.zst", rollout())
    selected = manifest(tmp_path, [path])
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    receipts = list(ingest_codex_manifest(selected, tmp_path / "db.sqlite", dry_run=True))
    assert receipts[0]["decodable"] is True
    assert receipts[0]["records"] == 2
    assert receipts[0]["would_insert"] == 1
    assert receipts[0]["queued"] == 0
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    path.write_bytes(path.read_bytes() + b"extra")
    receipt = list(ingest_codex_manifest(selected, tmp_path / "db.sqlite", dry_run=True))[0]
    assert receipt["skip_reason"] == "source-stats-mismatch"
    assert receipt["decoded_bytes"] == 0


def test_manifest_replays_offset_only_forks_without_collapsing_native_id(compress, tmp_path, monkeypatch):
    import sqlite3

    from brainlayer.drain import drain_once
    from brainlayer.vector_store import VectorStore

    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    prefix = rollout("Shared parent prefix that should remain searchable.")

    def tail(text):
        return rollout(text).splitlines()[1] + b"\n"

    a = compress("parent.jsonl.zst", prefix + tail("Unique parent tail that must remain searchable after a fork."))
    b = compress("fork.jsonl.zst", prefix + tail("Unique fork tail that must survive the parent's native id."))
    compress("unlisted.jsonl.zst", rollout("This file must never enter the queue from the manifest."))
    db_path = tmp_path / "db.sqlite"
    VectorStore(db_path).close()
    args = dict(
        db_path=db_path,
        queue_dir=tmp_path / "queue",
        log_path=tmp_path / "drain.log",
        pause_sentinel_path=tmp_path / "absent-sentinel",
    )
    first = manifest(tmp_path, [a])
    assert (
        list(ingest_codex_manifest(first, db_path, receipts_out=tmp_path / f"{uuid.uuid4().hex}.receipts.jsonl"))[0][
            "queued"
        ]
        == 2
    )
    assert drain_once(**args) == 2
    # The parent's native id already exists in durable chunks. The fork is a new FILE.
    registry = OffsetRegistry(tmp_path / "offsets.json")
    registry.set(str(b.with_suffix("")), 10, 123)
    registry.flush()
    second = manifest(tmp_path, [b])
    assert (
        list(ingest_codex_manifest(second, db_path, receipts_out=tmp_path / f"{uuid.uuid4().hex}.receipts.jsonl"))[0][
            "queued"
        ]
        == 1
    )
    events = [json.loads(p.read_text()) for p in (tmp_path / "queue").glob("*.jsonl")]
    assert {e["source_file"] for e in events} == {str(b.with_suffix(""))}
    assert drain_once(**args) == 1
    assert OffsetRegistry(tmp_path / "offsets.json").get(str(b.with_suffix("")))[0] == 10
    selected = manifest(tmp_path, [a, b])
    list(ingest_codex_manifest(selected, db_path, receipts_out=tmp_path / f"{uuid.uuid4().hex}.receipts.jsonl"))
    replay = [json.loads(p.read_text()) for p in (tmp_path / "queue").glob("*.jsonl")]
    assert replay == []
    assert drain_once(**args) == len(replay)
    with sqlite3.connect(db_path) as db:
        # The writer's existing exact-content dedup merges the common prefix only.
        assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM chunks WHERE content LIKE '%Unique fork tail%'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM chunks WHERE content LIKE '%Unique parent tail%'").fetchone()[0] == 1


def test_truncated_manifest_can_queue_a_prefix_but_never_confirms_completion(compress, tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    archive = compress("truncated.jsonl.zst", rollout())
    archive.write_bytes(archive.read_bytes()[:-4])
    selected = manifest(tmp_path, [archive])
    original = OffsetRegistry(tmp_path / "offsets.json")
    original.set(str(archive.with_suffix("")), 10, 123)
    original.flush()
    before = (tmp_path / "offsets.json").read_bytes()
    receipt = list(
        ingest_codex_manifest(
            selected, tmp_path / "db.sqlite", receipts_out=tmp_path / f"{uuid.uuid4().hex}.receipts.jsonl"
        )
    )[0]
    assert receipt["decodable"] is False and "decode failed" in receipt["skip_reason"]
    assert receipt["state"] in {"error", "partially_queued"}
    assert (tmp_path / "offsets.json").read_bytes() == before
