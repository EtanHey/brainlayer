"""Writer merge aliases make manifest replay idempotent after content is rewritten."""

import inspect
import json
import sqlite3

import pytest

from brainlayer.drain import drain_once
from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from brainlayer.vector_store import VectorStore
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress

A = (
    "Please remember the deployment checklist covers database migrations rollback plans monitoring dashboards "
    "alert routing ownership signoff staging verification canary traffic shifting and postmortem templates for "
    "every service team"
)


def replay(selected, db, **kwargs):
    # The same behavioral guard also runs against the pre-receipts-out baseline.
    if "receipts_out" in inspect.signature(ingest_codex_manifest).parameters:
        import uuid

        kwargs["receipts_out"] = db.parent / f"{uuid.uuid4().hex}.receipts.jsonl"
    return list(ingest_codex_manifest(selected, db, **kwargs))


def selected(tmp_path, paths):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                dict(
                    physical_path=str(p),
                    logical_path=str(p.with_suffix("")),
                    size=p.stat().st_size,
                    mtime_ns=p.stat().st_mtime_ns,
                    inode=p.stat().st_ino,
                )
            )
            + "\n"
            for p in paths
        )
    )
    return manifest


@pytest.mark.parametrize("kind", ["simhash", "normalized_equal"])
def test_merge_alias_resolves_next_pass(compress, tmp_path, monkeypatch, kind):
    db, queue = tmp_path / "x.db", tmp_path / "queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    monkeypatch.setenv("BRAINLAYER_DRAIN_EMBED", "0")
    VectorStore(db).close()
    args = dict(db_path=db, queue_dir=queue, log_path=tmp_path / "drain.log", pause_sentinel_path=tmp_path / "absent")
    if kind == "simhash":
        replay(selected(tmp_path, [compress("existing.jsonl.zst", rollout(A))]), db)
        drain_once(**args)
        manifest = selected(tmp_path, [compress("replay.jsonl.zst", rollout(A + " romeo"))])
        expected = 1
    else:
        one = compress("one.jsonl.zst", rollout("Remember the alpha rollout plan for staging today"))
        two = compress("two.jsonl.zst", rollout("remember THE alpha rollout plan, for staging today!!"))
        manifest, expected = selected(tmp_path, [one, two]), 2
    assert sum(r["queued"] for r in replay(manifest, db, max_files=2)) == expected
    drain_once(**args)
    with sqlite3.connect(db) as conn:
        before = conn.execute("SELECT id, seen_count FROM chunks ORDER BY id").fetchall()
        assert len(before) == 1
        assert conn.execute("SELECT COUNT(*) FROM chunk_id_alias").fetchone()[0] == 1
    for _ in range(2):
        receipts = replay(manifest, db, max_files=2)
        assert sum(r["queued"] for r in receipts) == 0
        assert sum(r["duplicate_content"] for r in receipts) >= 1
        drain_once(**args)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT id, seen_count FROM chunks ORDER BY id").fetchall() == before
        assert (
            conn.execute("SELECT COUNT(*) FROM chunks c JOIN chunk_id_alias a ON c.id=a.old_chunk_id").fetchone()[0]
            == 0
        )
