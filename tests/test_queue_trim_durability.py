import json
from pathlib import Path

import pytest

from brainlayer import drain, queue_io


@pytest.mark.parametrize("boundary", ["write", "file_sync", "replace", "publish_sync", "unlink", "retire_sync"])
def test_trim_failure_preserves_remainder(tmp_path, monkeypatch, boundary):
    source = tmp_path / "watcher.jsonl"
    event = {"kind": "enrichment_update", "chunk_id": "pending"}
    source.write_text(json.dumps(event) + "\n")
    calls = []
    real_write = queue_io._write_synced_queue_file
    real_sync = queue_io._sync_queue_fd
    real_replace = Path.replace
    real_unlink = Path.unlink

    def hit(name):
        calls.append(name)
        if boundary == name:
            raise OSError(name)

    def write(path, text):
        hit("write")
        real_write(path, text)

    def sync(fd):
        hit("file_sync")
        real_sync(fd)

    def replace(path, target):
        hit("replace")
        return real_replace(path, target)

    def directory(path):
        hit("retire_sync" if "unlink" in calls else "publish_sync")

    def unlink(path, *args, **kwargs):
        if path == source:
            hit("unlink")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(queue_io, "_write_synced_queue_file", write)
    monkeypatch.setattr(queue_io, "_sync_queue_fd", sync)
    monkeypatch.setattr(queue_io, "_sync_directory", directory)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match=boundary):
        drain._preserve_remaining_events(source, [event])
    assert any(json.loads(p.read_text()) == event for p in tmp_path.glob("*.jsonl"))
    if "unlink" in calls:
        assert calls.index("publish_sync") < calls.index("unlink")
