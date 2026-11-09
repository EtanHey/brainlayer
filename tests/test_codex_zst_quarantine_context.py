"""bl-zst-r11 reviewer probes (scratch, untracked; moved to the plan folder after the run)."""

import json

import pytest

from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import FlushWatermarks, create_flush_callback
from tests.test_codex_realtime_author_model import MODEL_A, _assistant, _encode, _meta, _say, _turn, _user
from tests.test_codex_zstd_stream import compress as _compress

compress = _compress


def _run(tmp_path, monkeypatch, compress, data, label, want, budget="4096"):
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_BYTES_PER_CYCLE", budget)
    monkeypatch.setenv("BRAINLAYER_T3_STATE_DB", str(tmp_path / "no-t3.db"))
    monkeypatch.setenv("BRAINLAYER_WATCHER_QUARANTINE_DIR", str(tmp_path / "quarantine"))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "unused-queue"))
    results = {}
    for compressed in (False, True):
        root = tmp_path / f"{label}-{'archive' if compressed else 'plain'}"
        root.mkdir()
        logical = root / "rollout.jsonl"
        if compressed:
            compress(f"{root.name}/rollout.jsonl.zst", data)
        else:
            logical.write_bytes(data)
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
        watcher._discover_jsonl_files()
        watcher._ensure_tailer(str(logical))
        try:
            for _ in range(4 * len(data) + 50):
                watcher.poll_once()
                watcher.indexer.flush()
                if len(sink) >= want:
                    break
        finally:
            watcher.stop()
        queue = root / "queue"
        monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(queue))
        create_flush_callback(root / "scratch.db", arbitrated=True)(sink)
        events = [json.loads(line) for p in queue.glob("*.jsonl") for line in p.read_text().splitlines()]
        results["zst" if compressed else "plain"] = {
            e["content"][:40]: (e["metadata"].get("model"), e["metadata"].get("model_provider")) for e in events
        }
    print(label, json.dumps(results, indent=1))
    return results


def test_quarantined_record_mid_archive_keeps_author_context(tmp_path, monkeypatch, compress):
    """A malformed (non-context) record is quarantined; later entries must keep model and provider."""
    data = (
        _encode([_meta(), _turn(MODEL_A), _assistant(_say(0))])
        + b'{"type":"response_item","payload":{broken\n'
        + _encode([_assistant(_say(1)), _user("A later human prompt keeps the session provider.")])
    )
    results = _run(tmp_path, monkeypatch, compress, data, "quarantine", want=3)
    assert results["plain"] == results["zst"]


def test_non_object_line_mid_archive_keeps_author_context(tmp_path, monkeypatch, compress):
    """A valid but non-object JSON line advances the offset with no annotated line."""
    data = (
        _encode([_meta(), _turn(MODEL_A), _assistant(_say(0))])
        + b"[]\n"
        + _encode([_assistant(_say(1)), _user("A later human prompt keeps the session provider.")])
    )
    results = _run(tmp_path, monkeypatch, compress, data, "nonobject", want=3)
    assert results["plain"] == results["zst"]


@pytest.mark.parametrize("budget", ["1", "13", "64"])
def test_latest_model_across_tiny_budgets_from_zero(tmp_path, monkeypatch, compress, budget):
    from tests.test_codex_realtime_author_model import MODEL_B, MODEL_C

    data = _encode([_meta(), _turn(MODEL_A), _assistant(_say(0)), _turn(MODEL_B), _turn(MODEL_C), _assistant(_say(1))])
    results = _run(tmp_path, monkeypatch, compress, data, f"latest-{budget}", want=2, budget=budget)
    assert results["plain"] == results["zst"]
    assert {v[0] for v in results["zst"].values()} == {MODEL_A, MODEL_C}
