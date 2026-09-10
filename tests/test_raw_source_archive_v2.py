from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from brainlayer.raw_source_archive import (
    RawArchiveCorruptError,
    RawSourceArchive,
    RawSourceChangedError,
)


def archive(tmp_path: Path, *, chunk_bytes: int = 4, max_bytes: int = 4096) -> RawSourceArchive:
    return RawSourceArchive(tmp_path / "raw-v2", chunk_bytes=chunk_bytes, max_source_bytes=max_bytes)


def body(store: RawSourceArchive, receipt: dict) -> bytes:
    output = io.BytesIO()
    store.reconstruct(receipt, output)
    return output.getvalue()


def manifest(store: RawSourceArchive, receipt: dict) -> dict:
    return json.loads(Path(receipt["manifest_path"]).read_bytes())


def test_v2_append_manifests_store_only_delta_and_iterative_reader_reconstructs_history(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcd")
    store = archive(tmp_path)
    receipts = [store.capture(source, "synthetic")]
    for addition in (b"e", b"fg", b"hij"):
        with source.open("ab") as target:
            target.write(addition)
        receipts.append(store.capture(source, "synthetic"))

    root = manifest(store, receipts[0])
    assert root["format_version"] == 2
    assert root["parent_manifest_sha256"] is None
    assert root["generation_root_manifest_sha256"] is None
    assert len(root["generation_nonce"]) == 64
    assert sum(item["size"] for item in root["segments"]) == 4
    for previous, receipt, addition in zip(receipts[:-1], receipts[1:], (b"e", b"fg", b"hij"), strict=True):
        item = manifest(store, receipt)
        assert item["parent_manifest_sha256"] == previous["manifest_sha256"]
        assert item["generation_root_manifest_sha256"] == receipts[0]["manifest_sha256"]
        assert item["generation_nonce"] is None
        assert item["delta_start"] == previous["size"]
        assert sum(segment["size"] for segment in item["segments"]) == len(addition)
    assert [body(store, receipt) for receipt in receipts] == [b"abcd", b"abcde", b"abcdefg", b"abcdefghij"]
    assert store.is_prefix(receipts[0], receipts[-1]) is True


def test_unchanged_capture_reuses_version_but_returns_fresh_source_observation(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"same")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    manifest_count = len(list(store.manifests.iterdir()))
    head = next(store.sources.glob("*/HEAD.json"))
    head_count = len(list(head.parent.glob("HEAD.json")))
    changed_ns = source.stat().st_mtime_ns + 10_000_000
    os.utime(source, ns=(changed_ns, changed_ns))

    second = store.capture(source, "synthetic")

    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert second["sha256"] == first["sha256"]
    assert second["source_stat"] != first["source_stat"]
    assert second["source_stat"]["mtime_ns"] == source.stat().st_mtime_ns
    assert len(list(store.manifests.iterdir())) == manifest_count
    assert len(list(head.parent.glob("HEAD.json"))) == head_count == 1


def test_nonprefix_rewrite_starts_new_generation_and_differing_roots_are_conservative(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"prefix")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    source.write_bytes(b"prefix-and-more")
    second = store.capture(source, "synthetic")
    source.write_bytes(b"prefix")
    third = store.capture(source, "synthetic")
    with source.open("ab") as target:
        target.write(b"-and-more-again")
    fourth = store.capture(source, "synthetic")

    assert store.is_prefix(first, second) is True
    assert manifest(store, third)["parent_manifest_sha256"] is None
    assert third["manifest_sha256"] != first["manifest_sha256"]
    assert store.is_prefix(first, third) is False
    assert store.is_prefix(first, fourth) is False
    assert body(store, first) == body(store, third) == b"prefix"
    assert body(store, second) == b"prefix-and-more"
    assert body(store, fourth) == b"prefix-and-more-again"


@pytest.mark.parametrize("changed", [False, True])
@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_capture_refuses_unreadable_retained_prefix(tmp_path: Path, changed: bool, damage: str) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    first_segment = manifest(store, first)["segments"][0]
    retained = store.objects / first_segment["sha256"]
    if damage == "missing":
        retained.unlink()
    else:
        retained.chmod(0o600)
        retained.write_bytes(b"evil")
    if changed:
        with source.open("ab") as target:
            target.write(b"append")
    manifest_count = len(list(store.manifests.iterdir()))

    with pytest.raises(RawArchiveCorruptError, match="chunk"):
        store.capture(source, "synthetic")
    assert len(list(store.manifests.iterdir())) == manifest_count
    head = json.loads(next(store.sources.glob("*/HEAD.json")).read_bytes())
    assert head["manifest_sha256"] == first["manifest_sha256"]


def _legacy_receipt(store: RawSourceArchive, source: Path, provider: str, raw: bytes) -> dict:
    chunks = []
    for start in range(0, len(raw), store.chunk_bytes):
        data = raw[start : start + store.chunk_bytes]
        digest = hashlib.sha256(data).hexdigest()
        store._publish_chunk(digest, data)
        chunks.append({"sha256": digest, "size": len(data)})
    info = source.stat()
    source_stat = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": len(raw),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }
    document = {
        "format_version": 1,
        "provider": provider,
        "source_path": str(source.absolute()),
        "source_stat": source_stat,
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "chunks": chunks,
    }
    encoded = store._canonical_manifest(document)
    digest = hashlib.sha256(encoded).hexdigest()
    path = store.manifests / f"{digest}.json"
    store._publish_bytes(path, encoded, prefix=".legacy-")
    return {
        "manifest_sha256": digest,
        "manifest_path": str(path),
        "source_path": str(source.absolute()),
        "provider": provider,
        "sha256": document["sha256"],
        "size": len(raw),
        "source_stat": source_stat,
    }


def test_referenced_v1_receipts_remain_readable_and_cross_format_prefix_works(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"legacy")
    store = archive(tmp_path)
    legacy = _legacy_receipt(store, source, "synthetic", b"legacy")
    source.write_bytes(b"legacy-plus")
    legacy_later = _legacy_receipt(store, source, "synthetic", b"legacy-plus")
    current = store.capture(source, "synthetic")

    assert body(store, legacy) == b"legacy"
    assert body(store, legacy_later) == b"legacy-plus"
    assert body(store, current) == b"legacy-plus"
    assert store.is_prefix(legacy, legacy_later) is True
    assert store.is_prefix(legacy, current) is True


@pytest.mark.parametrize("claimed", [2, True, 2.0])
def test_legacy_receipt_cannot_spoof_v2_format(tmp_path: Path, claimed: object) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"legacy")
    store = archive(tmp_path)
    receipt = _legacy_receipt(store, source, "synthetic", b"legacy")
    receipt["format_version"] = claimed

    with pytest.raises(RawArchiveCorruptError, match="legacy receipt"):
        store.open_snapshot(receipt)


@pytest.mark.parametrize("claimed", [True, 2.0, 1, None])
def test_v2_receipt_requires_exact_integer_format_version(tmp_path: Path, claimed: object) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"v2")
    store = archive(tmp_path)
    receipt = store.capture(source, "synthetic")
    receipt["format_version"] = claimed

    with pytest.raises(RawArchiveCorruptError, match="format_version"):
        store.open_snapshot(receipt)


def test_v2_receipt_requires_immutable_root_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"bytes")
    store = archive(tmp_path)
    receipt = store.capture(source, "synthetic")
    marker = next(store.sources.glob("*/roots/*.ref"))
    marker.unlink()

    with pytest.raises(RawArchiveCorruptError, match="root marker is missing"):
        store.open_snapshot(receipt)


def test_first_generation_root_marker_failure_returns_no_receipt_and_retry_keeps_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"first")
    store = archive(tmp_path)
    original = store._publish_root_marker

    def fail_marker(_state: Path, _root_sha: str) -> None:
        raise OSError("root marker crash")

    monkeypatch.setattr(store, "_publish_root_marker", fail_marker)
    with pytest.raises(OSError, match="root marker crash"):
        store.capture(source, "synthetic")
    assert len(list(store.manifests.iterdir())) == 1
    assert list(store.sources.glob("*/HEAD.json")) == []
    assert list(store.sources.glob("*/roots/*.ref")) == []

    monkeypatch.setattr(store, "_publish_root_marker", original)
    receipt = store.capture(source, "synthetic")
    assert body(store, receipt) == b"first"
    assert len(list(store.manifests.iterdir())) == 2


def test_head_replacement_failure_keeps_prior_receipt_and_retry_reuses_child_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"first")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    with source.open("ab") as target:
        target.write(b"-append")
    original = store._replace_head

    def fail_head(_state: Path, _receipt: dict) -> None:
        raise OSError("head replacement crash")

    monkeypatch.setattr(store, "_replace_head", fail_head)
    with pytest.raises(OSError, match="head replacement crash"):
        store.capture(source, "synthetic")
    assert body(store, first) == b"first"
    manifest_count = len(list(store.manifests.iterdir()))

    monkeypatch.setattr(store, "_replace_head", original)
    second = store.capture(source, "synthetic")
    assert body(store, second) == b"first-append"
    assert len(list(store.manifests.iterdir())) == manifest_count


def test_path_replacement_during_nonprefix_second_pass_refuses_new_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    source.write_bytes(b"ABCDEFGH")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    original = store._publish_chunk
    replaced = False

    def replace(sha256: str, data: bytes) -> None:
        nonlocal replaced
        original(sha256, data)
        if not replaced:
            replaced = True
            replacement.replace(source)

    monkeypatch.setattr(store, "_publish_chunk", replace)
    with pytest.raises(RawSourceChangedError):
        store.capture(source, "synthetic")
    assert body(store, first) == b"abcdefgh"
    head = json.loads(next(store.sources.glob("*/HEAD.json")).read_bytes())
    assert head["manifest_sha256"] == first["manifest_sha256"]


def test_first_append_and_unchanged_capture_write_costs_are_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"a" * 32)
    store = archive(tmp_path, chunk_bytes=64)
    writes: list[tuple[str, int]] = []
    original_publish = store._publish_bytes
    original_replace = store._replace_bytes

    def publish(path: Path, data: bytes, *, prefix: str) -> None:
        writes.append((prefix, len(data)))
        original_publish(path, data, prefix=prefix)

    def replace(path: Path, data: bytes, *, prefix: str) -> None:
        writes.append((prefix, len(data)))
        original_replace(path, data, prefix=prefix)

    monkeypatch.setattr(store, "_publish_bytes", publish)
    monkeypatch.setattr(store, "_replace_bytes", replace)
    first = store.capture(source, "synthetic")
    first_cost = list(writes)
    writes.clear()
    with source.open("ab") as target:
        target.write(b"b" * 7)
    second = store.capture(source, "synthetic")
    append_cost = list(writes)
    writes.clear()
    third = store.capture(source, "synthetic")

    assert (".chunk-", 32) in first_cost
    assert [size for prefix, size in append_cost if prefix == ".chunk-"] == [7]
    assert not any(prefix == ".root-" for prefix, _size in append_cost)
    assert writes == []
    assert third == second
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_source_lock_serializes_concurrent_unchanged_captures(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"one-version")
    store = archive(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda _index: store.capture(source, "synthetic"), range(16)))

    assert len({receipt["manifest_sha256"] for receipt in receipts}) == 1
    assert len(list(store.manifests.iterdir())) == 1
    assert len(list(store.sources.glob("*/roots/*.ref"))) == 1


def test_generated_storage_probe_retains_only_initial_body_plus_append_segments(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"a" * (1024 * 1024))
    store = RawSourceArchive(tmp_path / "raw-v2", max_source_bytes=2 * 1024 * 1024)
    receipts = [store.capture(source, "synthetic")]
    for index in range(32):
        addition = index.to_bytes(4, "big") * 256
        with source.open("ab") as target:
            target.write(addition)
        receipts.append(store.capture(source, "synthetic"))

    logical_objects = sum(path.stat().st_size for path in store.objects.iterdir())
    logical_manifests = sum(path.stat().st_size for path in store.manifests.iterdir())
    allocated_archive = sum(path.stat().st_blocks * 512 for path in store.root.rglob("*") if path.is_file())
    assert logical_objects == 1024 * 1024 + 32 * 1024
    assert logical_manifests < 33 * 1024
    assert body(store, receipts[0]) == b"a" * (1024 * 1024)
    assert body(store, receipts[-1]) == source.read_bytes()
    assert allocated_archive >= logical_objects + logical_manifests


def test_1024_append_history_walk_is_iterative_and_exposes_quadratic_cumulative_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"root")
    store = archive(tmp_path, chunk_bytes=4096, max_bytes=8192)
    reads = {"manifest": 0, "object": 0, "head": 0, "root": 0}
    original = store._read_regular_bounded

    def measured(path: Path, expected_max: int) -> bytes:
        if path.parent == store.manifests:
            reads["manifest"] += 1
        elif path.parent == store.objects:
            reads["object"] += 1
        elif path.name == "HEAD.json":
            reads["head"] += 1
        elif path.suffix == ".ref":
            reads["root"] += 1
        return original(path, expected_max)

    monkeypatch.setattr(store, "_read_regular_bounded", measured)
    receipt = store.capture(source, "synthetic")
    for index in range(1024):
        with source.open("ab") as target:
            target.write(index.to_bytes(4, "big"))
        receipt = store.capture(source, "synthetic")

    capture_reads = dict(reads)
    expected_source_bytes_read = sum(4 + 4 * index for index in range(1025))
    allocated_archive = sum(path.stat().st_blocks * 512 for path in store.root.rglob("*") if path.is_file())
    assert capture_reads == {"manifest": 527873, "object": 526850, "head": 1025, "root": 5122}
    assert expected_source_bytes_read == 2_103_300
    assert allocated_archive > receipt["size"]

    before_reconstruct = dict(reads)
    assert body(store, receipt) == source.read_bytes()
    assert {key: reads[key] - before_reconstruct[key] for key in reads} == {
        "manifest": 1025,
        "object": 1025,
        "head": 0,
        "root": 1,
    }
