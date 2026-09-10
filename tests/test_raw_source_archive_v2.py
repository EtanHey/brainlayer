from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from brainlayer.raw_source_archive import RawSourceArchive


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
    assert store.is_prefix(first, fourth) is False
    assert body(store, first) == body(store, third) == b"prefix"
    assert body(store, second) == b"prefix-and-more"
    assert body(store, fourth) == b"prefix-and-more-again"


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
    current = store.capture(source, "synthetic")

    assert body(store, legacy) == b"legacy"
    assert body(store, current) == b"legacy-plus"
    assert store.is_prefix(legacy, current) is True


def test_v2_receipt_requires_immutable_root_marker(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"bytes")
    store = archive(tmp_path)
    receipt = store.capture(source, "synthetic")
    marker = next(store.sources.glob("*/roots/*.ref"))
    marker.unlink()

    with pytest.raises(FileNotFoundError):
        store.open_snapshot(receipt)
