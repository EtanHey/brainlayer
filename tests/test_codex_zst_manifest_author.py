"""Manifest recovery preserves the live Codex author attribution contract."""

import json

import pytest

from brainlayer.ingest.codex_manifest import _read_entries, ingest_codex_manifest
from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import create_flush_callback
from tests.test_codex_realtime_author_model import MODEL_A, MODEL_B, _assistant, _encode, _meta, _say, _turn, _user
from tests.test_codex_zst_manifest import manifest
from tests.test_codex_zstd_stream import compress as _compress

compress = _compress


@pytest.fixture
def data():
    return _encode(
        [
            _meta(),
            _turn(MODEL_A),
            _assistant(_say(0)),
            _turn(MODEL_B),
            _assistant(_say(1)),
            _user(_say(2)),
            _turn("<synthetic>"),
            _assistant(_say(3)),
            _turn(MODEL_A),
            _turn({"invalid": True}),
            _assistant(_say(4)),
            _meta("fixture-provider-2"),
            _turn(MODEL_B),
            _assistant(_say(5)),
        ]
    )


def metadata(rows):
    keys = ("sender", "model", "model_source", "model_provider")
    return {
        r["content"]: json.dumps({k: r["metadata"][k] for k in keys if k in r["metadata"]}, sort_keys=True)
        for r in rows
    }


def events(queue):
    return [json.loads(line) for p in queue.glob("*.jsonl") for line in p.read_text().splitlines()]


def candidates(path, tmp_path):
    batches = list(_read_entries(path, 1048576, dict(records=0, decoded_bytes=0)))
    preview = []
    callback = create_flush_callback(tmp_path / "unused.db", arbitrated=True, preview_chunks=preview)
    for batch in batches:
        callback(batch)
    return preview


def test_exact_models_and_providers(compress, tmp_path, data):
    rows = candidates(compress("rollout.jsonl.zst", data), tmp_path)
    by_content = {r["content"]: r["metadata"] for r in rows}
    assert len(by_content) == 6
    for n, model in [(0, MODEL_A), (1, MODEL_B), (5, MODEL_B)]:
        assert by_content[_say(n)]["model"] == model
        assert by_content[_say(n)]["model_source"] == "codex_turn_context"
    for n in (2, 3, 4):
        assert "model" not in by_content[_say(n)]
        assert "model_source" not in by_content[_say(n)]
    assert by_content[_say(0)]["model_provider"] == "fixture-provider"
    assert by_content[_say(2)]["model_provider"] == "fixture-provider"
    assert by_content[_say(5)]["model_provider"] == "fixture-provider-2"


def test_plain_archive_reader_metadata(compress, tmp_path, monkeypatch, data):
    plain = tmp_path / "plain.jsonl"
    plain.write_bytes(data)
    archive = compress("archive.jsonl.zst", data)
    # The manifest reader consumes decoded bytes; supply the plain copy directly.
    import brainlayer.ingest.codex_manifest as module

    decoder = module.ZstdDecodedFile
    monkeypatch.setattr(module, "ZstdDecodedFile", lambda p: p.open("rb") if p == plain else decoder(p))
    left = metadata(candidates(plain, tmp_path))
    right = metadata(candidates(archive, tmp_path))
    assert left == right
    assert json.loads(right[_say(1)])["model"] == MODEL_B


def test_manifest_queue_and_dry_run_match_live_watcher(compress, tmp_path, monkeypatch, data):
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "no-t3.db"))
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    root = tmp_path / "plain"
    root.mkdir()
    (root / "rollout.jsonl").write_bytes(data)
    queue = tmp_path / "live-queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("codex", root)],
        registry_path=tmp_path / "offsets.json",
        on_flush=create_flush_callback(tmp_path / "live.db", arbitrated=True),
        batch_size=1,
    )
    try:
        watcher.poll_once()
        watcher.indexer.flush()
    finally:
        watcher.stop()
    live = metadata(events(queue))
    archive = compress("archive/rollout.jsonl.zst", data)
    selected = manifest(tmp_path, [archive])
    queue = tmp_path / "manifest-queue"
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
    receipt = list(ingest_codex_manifest(selected, tmp_path / "manifest.db", receipts_out=tmp_path / "receipts.jsonl"))[
        0
    ]
    assert receipt["queued"] == 6
    assert metadata(events(queue)) == live
    import brainlayer.ingest.codex_manifest as module

    original = module.create_flush_callback
    previews = []

    def capture(*args, **kwargs):
        callback = original(*args, **kwargs)

        def plan(entries):
            result = callback(entries)
            previews.extend(kwargs["preview_chunks"])
            return result

        return plan

    monkeypatch.setattr(module, "create_flush_callback", capture)
    assert list(ingest_codex_manifest(selected, tmp_path / "manifest.db", dry_run=True))[0]["would_insert"] == 6
    assert metadata(previews) == live
