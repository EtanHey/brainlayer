from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from brainlayer.raw_source_archive import (
    RawArchiveCorruptError,
    RawSourceArchive,
    RawSourceChangedError,
    RawSourceTooLargeError,
)


def archive(tmp_path: Path, *, chunk_bytes: int = 4, max_bytes: int = 1024) -> RawSourceArchive:
    return RawSourceArchive(tmp_path / "raw", chunk_bytes=chunk_bytes, max_source_bytes=max_bytes)


def reconstruct(store: RawSourceArchive, receipt: dict) -> bytes:
    output = io.BytesIO()
    result = store.reconstruct(receipt, output)
    assert result == {"sha256": receipt["sha256"], "size": receipt["size"]}
    return output.getvalue()


def payload_bytes(store: RawSourceArchive) -> int:
    return sum(path.stat().st_size for path in store.objects.iterdir())


def test_capture_reconstructs_arbitrary_bytes_and_bounds_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    raw = b'{"ok":true}\n\xff\x00partial'
    source.write_bytes(raw)
    store = archive(tmp_path)

    receipt = store.capture(source, "synthetic")
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert reconstruct(store, receipt) == raw

    with store.open_snapshot(receipt) as snapshot:
        assert snapshot.read(3) == raw[:3]
        assert snapshot.seek(4) == 4
        assert snapshot.read(5) == raw[4:9]
        assert snapshot.seek(-2, 2) == len(raw) - 2
        assert snapshot.read(2) == raw[-2:]
        with pytest.raises(ValueError, match="outside"):
            snapshot.seek(len(raw) + 1)
        with pytest.raises(ValueError, match="between"):
            snapshot.read(-1)
        with pytest.raises(ValueError, match="between"):
            snapshot.read(snapshot.MAX_READ_BYTES + 1)


def test_append_reuses_prefix_chunks_and_versions_remain_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefghij")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    before_payload = payload_bytes(store)

    with source.open("ab") as handle:
        handle.write(b"klmnop")
    second = store.capture(source, "synthetic")
    added_payload = payload_bytes(store) - before_payload

    assert added_payload <= 6 + store.chunk_bytes
    assert reconstruct(store, first) == b"abcdefghij"
    assert reconstruct(store, second) == b"abcdefghijklmnop"
    assert first["manifest_sha256"] != second["manifest_sha256"]


def test_replay_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")
    object_count = len(list(store.objects.iterdir()))
    manifest_count = len(list(store.manifests.iterdir()))

    second = store.capture(source, "synthetic")

    assert second == first
    assert len(list(store.objects.iterdir())) == object_count
    assert len(list(store.manifests.iterdir())) == manifest_count


def test_truncation_and_replacement_preserve_prior_versions(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"first-version")
    store = archive(tmp_path)
    first = store.capture(source, "synthetic")

    source.write_bytes(b"short")
    second = store.capture(source, "synthetic")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    replacement.replace(source)
    third = store.capture(source, "synthetic")

    assert reconstruct(store, first) == b"first-version"
    assert reconstruct(store, second) == b"short"
    assert reconstruct(store, third) == b"replacement"


def test_source_mutation_during_capture_refuses_mixed_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    store = archive(tmp_path)
    original = store._publish_chunk
    changed = False

    def mutate(sha256: str, data: bytes) -> None:
        nonlocal changed
        original(sha256, data)
        if not changed:
            changed = True
            with source.open("ab") as handle:
                handle.write(b"x")

    monkeypatch.setattr(store, "_publish_chunk", mutate)
    with pytest.raises(RawSourceChangedError):
        store.capture(source, "synthetic")
    assert list(store.manifests.iterdir()) == []


def test_path_replacement_during_capture_refuses_mixed_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replacement")
    store = archive(tmp_path)
    original = store._publish_chunk
    changed = False

    def replace_path(sha256: str, data: bytes) -> None:
        nonlocal changed
        original(sha256, data)
        if not changed:
            changed = True
            replacement.replace(source)

    monkeypatch.setattr(store, "_publish_chunk", replace_path)
    with pytest.raises(RawSourceChangedError):
        store.capture(source, "synthetic")
    assert list(store.manifests.iterdir()) == []


def test_crash_after_chunks_before_manifest_is_replay_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    store = archive(tmp_path)
    original = store._publish_bytes
    failed = False

    def crash(path: Path, data: bytes, *, prefix: str) -> None:
        nonlocal failed
        if path.parent == store.manifests and not failed:
            failed = True
            raise OSError("synthetic crash")
        original(path, data, prefix=prefix)

    monkeypatch.setattr(store, "_publish_bytes", crash)
    with pytest.raises(OSError, match="synthetic crash"):
        store.capture(source, "synthetic")
    assert len(list(store.objects.iterdir())) == 2
    assert list(store.manifests.iterdir()) == []

    monkeypatch.setattr(store, "_publish_bytes", original)
    receipt = store.capture(source, "synthetic")
    assert reconstruct(store, receipt) == b"abcdefgh"
    assert len(list(store.objects.iterdir())) == 2


def test_corrupt_preexisting_chunk_refuses_capture(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcd")
    store = archive(tmp_path)
    digest = hashlib.sha256(b"abcd").hexdigest()
    (store.objects / digest).write_bytes(b"evil")

    with pytest.raises(RawArchiveCorruptError, match="chunk digest mismatch"):
        store.capture(source, "synthetic")


def test_corrupt_chunk_refuses_snapshot_read(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"abcdefgh")
    store = archive(tmp_path)
    receipt = store.capture(source, "synthetic")
    first_digest = hashlib.sha256(b"abcd").hexdigest()
    first_object = store.objects / first_digest
    first_object.chmod(0o600)
    first_object.write_bytes(b"evil")

    with store.open_snapshot(receipt) as snapshot:
        with pytest.raises(RawArchiveCorruptError, match="chunk"):
            snapshot.read(1)


def test_source_limit_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"12345")
    store = archive(tmp_path, max_bytes=4)

    with pytest.raises(RawSourceTooLargeError):
        store.capture(source, "synthetic")
    assert list(store.manifests.iterdir()) == []


def test_symlink_source_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"bytes")
    source = tmp_path / "source.jsonl"
    source.symlink_to(target)
    store = archive(tmp_path)

    with pytest.raises(OSError):
        store.capture(source, "synthetic")
