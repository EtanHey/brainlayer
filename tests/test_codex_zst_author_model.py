"""#1651 author context must survive bounded decoded-prefix replay, too."""

import json

import pytest

from brainlayer.codex_zst import ZstdTailer
from brainlayer.watcher import CodexTurnContext, JSONLWatcher, WatchRoot, fold_codex_turn_context
from brainlayer.watcher_bridge import FlushWatermarks, create_flush_callback
from tests.test_codex_realtime_author_model import MODEL_A, MODEL_B, _assistant, _encode, _meta, _say, _turn, _user
from tests.test_codex_zst_prefix import fingerprint
from tests.test_codex_zstd_stream import compress as _compress

compress = _compress


def test_prefix_replay_seeds_author_context_before_resumed_messages(tmp_path, compress):
    prefix = _encode([_meta(), _turn(MODEL_A), _assistant(_say(0)), _turn(MODEL_B)])
    data = prefix + _encode([_assistant(_say(1))])
    path = compress("rollout.jsonl.zst", data)
    tailer = ZstdTailer(str(path), len(prefix), resume_fingerprint=fingerprint(data, len(prefix)))
    try:
        for _ in range(len(data)):
            lines = tailer.read_new_lines(max_bytes=17)
            if lines:
                break
        assert len(lines) == 1
        assert tailer.codex_turn_context == CodexTurnContext(0, len(prefix), MODEL_B, "fixture-provider")
    finally:
        tailer.close()


@pytest.mark.parametrize("replacement", [MODEL_B, "<synthetic>", None, "malformed"])
def test_plain_and_archive_resume_attribute_like_reading_from_zero(tmp_path, monkeypatch, compress, replacement):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_BYTES_PER_CYCLE", "71")
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "no-t3.db"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "unused-queue"))
    prefix = _encode([_meta(), _turn(MODEL_A), _assistant(_say(0))])
    if replacement == "malformed":
        prefix += b'{"type":"turn_context","payload":{broken\n'
    else:
        prefix += _encode([_turn(replacement)])
    suffix = _encode([_assistant(_say(1)), _user("A resumed human prompt has a provider but no author model.")])
    data = prefix + suffix
    results = []
    for compressed in (False, True):
        root = tmp_path / ("archive" if compressed else "plain")
        root.mkdir()
        logical = root / "rollout.jsonl"
        physical = compress("archive/rollout.jsonl.zst", data) if compressed else logical
        if not compressed:
            physical.write_bytes(data)
        sink = []

        def accept(batch):
            sink.extend(batch)
            return FlushWatermarks({e["_source_file"]: e["_line_end_offset"] for e in batch}, inserted=len(batch))

        watcher = JSONLWatcher(
            watch_roots=[WatchRoot("codex", root)],
            registry_path=root / "offsets.json",
            on_flush=accept,
            batch_size=1,
            max_lines_per_file=1,
        )
        # A persisted, fingerprint-verified offset after the latest context record.
        watcher.registry.set(
            str(logical), len(prefix), physical.stat().st_ino, fingerprint=fingerprint(data, len(prefix))
        )
        watcher.registry.flush()
        # L2b deliberately holds cold archives until L2c adds persisted admission.
        # Construct the resumed production tailer explicitly at this slice boundary.
        watcher._discover_jsonl_files()
        watcher._ensure_tailer(str(logical))
        try:
            for _ in range(len(data)):
                watcher.poll_once()
                watcher.indexer.flush()
                if len(sink) == 2:
                    break
            assert len(sink) == 2, (
                compressed,
                watcher._discover_jsonl_files(),
                watcher._file_ingestion_failures,
                watcher.registry._data,
            )
        finally:
            watcher.stop()
        queue = root / "queue"
        monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
        create_flush_callback(root / "scratch.db", arbitrated=True)(sink)
        events = [json.loads(line) for p in queue.glob("*.jsonl") for line in p.read_text().splitlines()]
        results.append(
            {
                e["content"]: {k: e["metadata"].get(k) for k in ("model", "model_source", "model_provider")}
                for e in events
            }
        )
    # Fold the complete record stream from zero with main's production semantics.
    context = fold_codex_turn_context(CodexTurnContext(0, 0), _meta())
    context = fold_codex_turn_context(context, _turn(MODEL_A))
    context = fold_codex_turn_context(context, _turn(None if replacement == "malformed" else replacement))
    expected = {
        _say(1): {
            "model": context.model,
            "model_source": "codex_turn_context" if context.model else None,
            "model_provider": context.model_provider,
        },
        "A resumed human prompt has a provider but no author model.": {
            "model": None,
            "model_source": None,
            "model_provider": context.model_provider,
        },
    }
    assert results == [expected, expected]
