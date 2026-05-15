import json
from pathlib import Path

import pytest


def _write_event(path: Path, content: str) -> None:
    path.write_text(json.dumps({"kind": "store_memory", "content": content}) + "\n", encoding="utf-8")


def test_queue_merge_dry_run_lists_copies_without_writing(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    dest.mkdir()
    _write_event(source / "m1-1.jsonl", "from m1")

    result = merge_queue_dirs(source, dest, dry_run=True)

    assert result.copied == ["m1-1.jsonl"]
    assert result.skipped_exact == []
    assert not (dest / "m1-1.jsonl").exists()


def test_queue_merge_is_idempotent_and_skips_exact_content(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    dest.mkdir()
    _write_event(source / "m1-1.jsonl", "from m1")

    first = merge_queue_dirs(source, dest)
    second = merge_queue_dirs(source, dest)

    assert first.copied == ["m1-1.jsonl"]
    assert second.copied == []
    assert second.skipped_exact == ["m1-1.jsonl"]
    assert len(list(dest.glob("*.jsonl"))) == 1


def test_queue_merge_renames_filename_collision_without_overwriting(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    dest.mkdir()
    _write_event(source / "same.jsonl", "m1 content")
    _write_event(dest / "same.jsonl", "m4 content")

    result = merge_queue_dirs(source, dest)

    assert result.collisions == ["same.jsonl"]
    assert len(result.collision_renames) == 1
    assert len(result.copied) == 1
    merged_name = result.copied[0]
    assert result.collision_renames == [("same.jsonl", merged_name)]
    assert merged_name.startswith("same-merge-")
    assert merged_name.endswith(".jsonl")
    assert (dest / "same.jsonl").read_text(encoding="utf-8").endswith('"m4 content"}\n')
    assert (dest / merged_name).read_text(encoding="utf-8").endswith('"m1 content"}\n')


def test_queue_merge_ignores_non_jsonl_files(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    dest.mkdir()
    (source / "README.txt").write_text("not a queue event", encoding="utf-8")
    _write_event(source / "event.jsonl", "from m1")

    result = merge_queue_dirs(source, dest)

    assert result.copied == ["event.jsonl"]
    assert result.skipped_non_jsonl == ["README.txt"]
    assert not (dest / "README.txt").exists()


def test_queue_merge_raises_on_nonexistent_source(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    with pytest.raises(NotADirectoryError):
        merge_queue_dirs(tmp_path / "missing", tmp_path / "dest")


def test_queue_merge_raises_when_source_equals_dest(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()

    with pytest.raises(ValueError, match="must be different"):
        merge_queue_dirs(queue_dir, queue_dir)


def test_queue_merge_creates_dest_if_missing(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    _write_event(source / "m1-1.jsonl", "from m1")

    result = merge_queue_dirs(source, dest)

    assert result.copied == ["m1-1.jsonl"]
    assert (dest / "m1-1.jsonl").exists()


def test_queue_merge_deduplicates_same_content_across_filenames(tmp_path):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    dest.mkdir()
    _write_event(source / "a.jsonl", "same content")
    _write_event(source / "b.jsonl", "same content")

    result = merge_queue_dirs(source, dest)

    assert result.copied == ["a.jsonl"]
    assert result.skipped_exact == ["b.jsonl"]
    assert len(list(dest.glob("*.jsonl"))) == 1


def test_queue_merge_skips_destination_file_deleted_by_drain(tmp_path, monkeypatch):
    from brainlayer.queue_merge import merge_queue_dirs

    source = tmp_path / "airdrop-queue"
    dest = tmp_path / "live-queue"
    source.mkdir()
    dest.mkdir()
    _write_event(source / "m1-1.jsonl", "from m1")
    _write_event(dest / "vanished.jsonl", "drain deletes this during hash scan")
    original_read_bytes = Path.read_bytes

    def flaky_read_bytes(path: Path) -> bytes:
        if path.name == "vanished.jsonl":
            raise FileNotFoundError(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)

    result = merge_queue_dirs(source, dest)

    assert result.copied == ["m1-1.jsonl"]
