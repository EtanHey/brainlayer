import json
import os

import pytest

from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import FlushWatermarks
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def watcher(tmp_path, monkeypatch, *, enabled=True, cap=2, budget=1048576, max_lines=100):
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded" if enabled else "off")
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_FILES_PER_CYCLE", str(cap))
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_BYTES_PER_CYCLE", str(budget))
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    monkeypatch.delenv("BRAINLAYER_INGEST_DENYLIST", raising=False)
    entries = []

    def accept(batch):
        entries.extend(batch)
        return FlushWatermarks({e["_source_file"]: e["_line_end_offset"] for e in batch}, inserted=len(batch))

    instance = JSONLWatcher(
        watch_roots=[WatchRoot("codex", tmp_path)],
        registry_path=tmp_path / "offsets.json",
        on_flush=accept,
        max_lines_per_file=max_lines,
        batch_size=1,
    )
    return instance, entries


def test_off_is_plain_only_with_unchanged_entries(compress, tmp_path, monkeypatch):
    plain = tmp_path / "plain.jsonl"
    records = [json.loads(line) for line in rollout().splitlines()]
    records[1]["timestamp"] = "2026-01-01T00:00:00Z"
    plain.write_bytes(b"".join((json.dumps(line) + "\n").encode() for line in records))
    monkeypatch.setattr("brainlayer.watcher.time.time", lambda: 123.0)
    compress("hidden.jsonl.zst", rollout())
    instance, entries = watcher(tmp_path, monkeypatch, enabled=False)
    assert instance._discover_jsonl_files() == [str(plain)]
    assert instance.poll_once() == 1
    assert entries[0] == {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": records[1]["payload"]["content"][0]["text"]}]},
        "timestamp": "2026-01-01T00:00:00Z",
        "_provider": "codex",
        "_source_file": str(plain),
        "_line_end_offset": len(plain.read_bytes()),
        "_ingest_observed_at": 123.0,
        "_source_inode": plain.stat().st_ino,
        "_source_generation": 0,
    }
    assert instance.registry.get(str(plain))[0] == len(plain.read_bytes())


def test_bounded_caps_continue_across_cycles_and_skip_old_checkpoints(compress, tmp_path, monkeypatch):
    paths = [compress(f"{i}.jsonl.zst", rollout(str(i) + " unique archived prompt " * 3)) for i in range(4)]
    instance, entries = watcher(tmp_path, monkeypatch, cap=1)
    held = str(paths[0].with_suffix(""))
    instance.registry.set(held, 40, 123)
    for _ in range(3):
        before = len(entries)
        instance.poll_once()
        assert len(entries) - before == 1
    assert len(entries) == 3
    assert held not in instance._tailers  # no decoder was ever opened for this path
    instance.stop()


def test_transition_resumes_decoded_offset_and_reverse_preserves_identity(compress, tmp_path, monkeypatch):
    plain = tmp_path / "rollout.jsonl"
    prefix = rollout("Original prompt that was already ingested before compression.")
    plain.write_bytes(prefix)
    instance, entries = watcher(tmp_path, monkeypatch)
    assert instance.poll_once() == 1
    late = json.loads(rollout().splitlines()[1])
    late["payload"]["content"][0]["text"] = "A late record added before Codex compressed the rollout."
    suffix = (json.dumps(late) + "\n").encode()
    archive = compress("rollout.jsonl.zst", prefix + suffix)
    plain.unlink()
    assert instance.poll_once() == 1
    assert len(entries) == 2
    assert {e["_source_file"] for e in entries} == {str(plain)}
    assert instance.registry.get(str(plain))[0] == len(prefix + suffix)
    assert instance.poll_once() == 0
    plain.write_bytes(prefix + suffix)
    archive.unlink()
    assert instance.poll_once() == 0
    instance.stop()


def test_decoded_budget_utf8_boundaries_and_no_cold_reread(compress, tmp_path, monkeypatch):
    data = rollout("A multiline\nmessage with 🌱 שלום and enough content to keep.")
    path = compress("utf8.jsonl.zst", data)
    os.utime(path, (1, 1))  # preserved old mtimes never exclude archives
    instance, entries = watcher(tmp_path, monkeypatch, budget=1)
    reads = 0
    for _ in range(len(data) + 2):
        instance.poll_once()
        tailer = instance._tailers[str(path.with_suffix(""))]
        assert tailer.bytes_read <= 1
        reads += tailer.bytes_read
    assert len(entries) == 1
    assert reads == len(data)
    assert "🌱 שלום" in entries[0]["message"]["content"][0]["text"]
    from brainlayer import codex_zst

    monkeypatch.setattr(codex_zst, "ZstdDecodedFile", lambda *_: pytest.fail("unchanged archive decoded twice"))
    instance.poll_once()
    instance.stop()
