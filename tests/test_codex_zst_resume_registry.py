"""Registry pruning and representation changes preserve confirmed decoded proofs."""

from brainlayer.watcher import JSONLWatcher, WatchRoot
from brainlayer.watcher_bridge import FlushWatermarks
from tests.test_codex_zstd_stream import compress as _compress
from tests.test_codex_zstd_stream import rollout

compress = _compress


def watcher(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "bounded")
    monkeypatch.setenv("BRAINLAYER_QUEUE_DIR", str(tmp_path / "queue"))
    entries = []

    def accept(batch):
        entries.extend(batch)
        return FlushWatermarks({e["_source_file"]: e["_line_end_offset"] for e in batch}, inserted=len(batch))

    w = JSONLWatcher(
        watch_roots=[WatchRoot("codex", tmp_path)],
        registry_path=tmp_path / "offsets.json",
        on_flush=accept,
        batch_size=1,
    )
    return w, entries


def test_prune_keeps_archive_beside_plain(compress, tmp_path, monkeypatch):
    physical = compress("old.jsonl.zst", rollout())
    logical = str(physical.with_suffix(""))
    (tmp_path / "live.jsonl").write_bytes(rollout())
    w, _ = watcher(tmp_path, monkeypatch)
    w.registry.set(logical, 40, 123)
    w.poll_once()
    assert w.registry.get(logical) == (40, 123)
    w.stop()


def test_transition_carries_prefix_without_legacy_downgrade(compress, tmp_path, monkeypatch):
    path = tmp_path / "rollout.jsonl"
    data = rollout()
    path.write_bytes(data)
    w, entries = watcher(tmp_path, monkeypatch)
    w.poll_once()
    proof = w.registry.fingerprint(str(path))
    assert proof is not None
    archive = compress(path.name + ".zst", data)
    path.unlink()
    w._discover_jsonl_files()
    w._ensure_tailer(str(path))
    assert w.registry.fingerprint(str(path)) == proof
    for _ in range(3):
        w.poll_once()
    archive_proof = w.registry.fingerprint(str(path))
    path.write_bytes(data)
    archive.unlink()
    w._discover_jsonl_files()
    w._ensure_tailer(str(path))
    assert w.registry.fingerprint(str(path)) == archive_proof
    w.poll_once()
    assert len(entries) == 1
    w.stop()


def test_plain_tailer_survives_discovery_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINLAYER_CODEX_ZST_ADMISSION", "off")
    root = tmp_path / "projects" / "-repo"
    root.mkdir(parents=True)
    path = root / "s.jsonl"
    path.write_text('{"type":"user","message":{"role":"user","content":"A long enough user prompt to keep."}}\n')
    entries = []

    def accept(batch):
        entries.extend(batch)
        return FlushWatermarks({str(path): batch[-1]["_line_end_offset"]}, inserted=len(batch))

    w = JSONLWatcher(
        watch_roots=[WatchRoot("claude", root.parent)],
        registry_path=tmp_path / "offsets.json",
        on_flush=accept,
        batch_size=1000,
    )
    w.poll_once()
    tailer = w._tailers[str(path)]
    hidden = path.with_suffix(".hidden")
    path.rename(hidden)
    w.poll_once()
    assert w._tailers.get(str(path)) is tailer
    hidden.rename(path)
    w.poll_once()
    w.indexer.flush()
    assert len(entries) == 1
    assert w._physical_files == {}
    w.stop()
