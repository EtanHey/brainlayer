import json
from pathlib import Path


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
    assert len(result.copied) == 1
    merged_name = result.copied[0]
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
