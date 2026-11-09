import json
import sqlite3
import uuid

import pytest

from brainlayer.ingest.codex_manifest import ingest_codex_manifest
from brainlayer.watcher_bridge import FlushWatermarks
from tests.test_codex_zst_manifest import manifest
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def test_bad_manifest_rows_do_not_abort_later_receipts(compress, tmp_path):
    selected = manifest(tmp_path, [compress("valid.jsonl.zst", rollout())])
    selected.write_text("not-json\n{}\n[]\n" + selected.read_text())
    rows = list(ingest_codex_manifest(selected, tmp_path / "x.db", dry_run=True))
    assert len(rows) == 4
    assert [r["class"] for r in rows] == ["error", "error", "error", "readable"]
    assert all(r["reason"] for r in rows)
    assert rows[-1]["decoded_bytes_total"] == len(rollout())


def test_unresolved_provenance_is_deferred(compress, tmp_path, monkeypatch):
    selected = manifest(tmp_path, [compress("pending.jsonl.zst", rollout())])
    monkeypatch.setattr(
        "brainlayer.ingest.codex_manifest.create_flush_callback",
        lambda *a, **kw: lambda entries: FlushWatermarks({}, deferred_entries=entries),
    )
    row = list(ingest_codex_manifest(selected, tmp_path / "x.db", dry_run=True))[0]
    assert row["class"] == row["state"] == "deferred"
    assert row["reason"] == "provenance-unresolved"


@pytest.mark.parametrize("data,expected", [(b"", "empty"), (b"{}\n", "filtered"), (rollout(), "readable")])
def test_success_classes_and_total_size(compress, tmp_path, data, expected):
    selected = manifest(tmp_path, [compress("class.jsonl.zst", data)])
    row = list(ingest_codex_manifest(selected, tmp_path / "x.db", dry_run=True))[0]
    assert row["class"] == expected and row["reason"]
    assert row["decoded_bytes_total"] == len(data)


def test_all_456_rows_receive_exactly_one_class(compress, tmp_path):
    archive = compress("first.jsonl.zst", rollout())
    selected = manifest(tmp_path, [archive])
    template = json.loads(selected.read_text())
    rows = [
        dict(template, physical_path=str(tmp_path / f"{i}.jsonl.zst"), logical_path=str(tmp_path / f"{i}.jsonl"))
        for i in range(455)
    ]
    selected.write_text(selected.read_text() + "".join(json.dumps(r) + "\n" for r in rows))
    receipts = list(ingest_codex_manifest(selected, tmp_path / "x.db", dry_run=True, max_files=1))
    assert len(receipts) == 456
    assert receipts[0]["class"] == "readable"
    assert all(r["class"] == "deferred" and r["reason"] == "admission-budget" for r in receipts[1:])


def test_real_file_reuses_one_readonly_connection(compress, tmp_path, monkeypatch):
    from brainlayer.vector_store import VectorStore

    path = compress(
        "two.jsonl.zst",
        rollout("First distinct prompt with enough content to store.")
        + rollout("Second distinct prompt with enough content to store.").splitlines(keepends=True)[1],
    )
    db = tmp_path / "x.db"
    VectorStore(db).close()
    selected = manifest(tmp_path, [path])
    original, opened = sqlite3.connect, []

    def connect(*a, **kw):
        opened.append(a[0])
        return original(*a, **kw)

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    row = list(ingest_codex_manifest(selected, db, receipts_out=tmp_path / f"{uuid.uuid4().hex}.receipts.jsonl"))[0]
    assert row["queued"] == 2
    assert len([p for p in opened if p.startswith(db.resolve().as_uri())]) == 1
    assert row["class"] == "accepted"
