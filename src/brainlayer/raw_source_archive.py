"""Durable raw-source versions for projection from immutable bytes.

Mechanism only: callers explicitly construct and inject an archive.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, BinaryIO

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class RawArchiveError(Exception):
    pass


class RawSourceChangedError(RawArchiveError):
    pass


class RawSourceTooLargeError(RawArchiveError):
    pass


class RawArchiveCorruptError(RawArchiveError):
    pass


class RawSnapshot:
    """Seekable bounded reader over a verified ordered chunk manifest."""

    MAX_READ_BYTES = 1024 * 1024

    def __init__(self, archive: RawSourceArchive, manifest: dict[str, Any]) -> None:
        self._archive, self._chunks = archive, manifest["chunks"]
        self._ends: list[int] = []
        total = 0
        for chunk in self._chunks:
            total += chunk["size"]
            self._ends.append(total)
        self._length, self._position = manifest["size"], 0
        self._cached_index: int | None = None
        self._cached_data = b""
        self.closed = False

    def __enter__(self) -> RawSnapshot:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True
        self._cached_data = b""

    def tell(self) -> int:
        self._ensure_open()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._ensure_open()
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("snapshot offset must be an integer")
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError("invalid whence")
        if not 0 <= position <= self._length:
            raise ValueError("snapshot seek outside version bounds")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._ensure_open()
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("snapshot read size must be an integer")
        if size < 0 or size > self.MAX_READ_BYTES:
            raise ValueError(f"snapshot read size must be between 0 and {self.MAX_READ_BYTES}")
        if size == 0 or self._position == self._length:
            return b""
        end = self._length if size == -1 else min(self._length, self._position + size)
        output = bytearray()
        while self._position < end:
            index = bisect.bisect_right(self._ends, self._position)
            chunk_start = 0 if index == 0 else self._ends[index - 1]
            data = self._load_chunk(index)
            within = self._position - chunk_start
            take = min(len(data) - within, end - self._position)
            if take <= 0:
                raise RawArchiveCorruptError("chunk manifest cannot satisfy snapshot read")
            output.extend(data[within : within + take])
            self._position += take
        return bytes(output)

    def _load_chunk(self, index: int) -> bytes:
        if self._cached_index != index:
            chunk = self._chunks[index]
            self._cached_data = self._archive._read_verified_chunk(chunk["sha256"], chunk["size"])
            self._cached_index = index
        return self._cached_data

    def _ensure_open(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed raw snapshot")


class RawSourceArchive:
    """Append-only content-addressed chunks and source-version manifests."""

    FORMAT_VERSION = 1
    DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
    DEFAULT_MAX_SOURCE_BYTES = 8 * 1024 * 1024 * 1024

    def __init__(
        self,
        root: str | Path,
        *,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    ) -> None:
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        if isinstance(max_source_bytes, bool) or not isinstance(max_source_bytes, int) or max_source_bytes < 0:
            raise ValueError("max_source_bytes must be a nonnegative integer")
        self.root = Path(root).absolute()
        self.chunk_bytes = chunk_bytes
        self.max_source_bytes = max_source_bytes
        self.objects = self.root / "objects"
        self.manifests = self.root / "manifests"
        for directory in (self.root, self.objects, self.manifests):
            self._private_dir(directory)

    def capture(self, path: str | Path, provider: str) -> dict[str, Any]:
        """Capture one stable whole-file version and return its durable receipt."""
        if not isinstance(provider, str) or not provider or len(provider) > 128:
            raise ValueError("provider must be a nonempty string of at most 128 characters")
        source_path = Path(path).absolute()
        chunks: list[dict[str, Any]] = []
        version_hash, size = hashlib.sha256(), 0
        with self._open_regular(source_path) as source:
            before = os.fstat(source.fileno())
            if before.st_size > self.max_source_bytes:
                raise RawSourceTooLargeError(f"source exceeds {self.max_source_bytes} bytes")
            while True:
                data = source.read(min(self.chunk_bytes, self.max_source_bytes - size + 1))
                if not data:
                    break
                size += len(data)
                if size > self.max_source_bytes:
                    raise RawSourceTooLargeError(f"source exceeds {self.max_source_bytes} bytes")
                version_hash.update(data)
                chunk_sha = hashlib.sha256(data).hexdigest()
                self._publish_chunk(chunk_sha, data)
                chunks.append({"sha256": chunk_sha, "size": len(data)})
            after = os.fstat(source.fileno())
            try:
                current = source_path.lstat()
            except OSError as error:
                raise RawSourceChangedError("source path disappeared during capture") from error
            signature = self._signature(before)
            if signature != self._signature(after) or signature != self._signature(current):
                raise RawSourceChangedError("source changed during capture")

        manifest = {
            "format_version": self.FORMAT_VERSION,
            "provider": provider,
            "source_path": str(source_path),
            "source_stat": {
                "device": before.st_dev,
                "inode": before.st_ino,
                "size": before.st_size,
                "mtime_ns": before.st_mtime_ns,
                "ctime_ns": before.st_ctime_ns,
            },
            "size": size,
            "sha256": version_hash.hexdigest(),
            "chunks": chunks,
        }
        raw = self._canonical_manifest(manifest)
        manifest_sha = hashlib.sha256(raw).hexdigest()
        manifest_path = self.manifests / f"{manifest_sha}.json"
        self._publish_bytes(manifest_path, raw, prefix=".manifest-")
        if self._load_manifest(manifest_sha) != manifest:
            raise RawArchiveCorruptError("published manifest differs from captured version")
        return {
            "manifest_sha256": manifest_sha,
            "manifest_path": str(manifest_path),
            "source_path": str(source_path),
            "provider": provider,
            "sha256": manifest["sha256"],
            "size": size,
            "source_stat": dict(manifest["source_stat"]),
        }

    def open_snapshot(self, receipt: dict[str, Any]) -> RawSnapshot:
        return RawSnapshot(self, self._manifest_for_receipt(receipt))

    def _manifest_for_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(receipt, dict):
            raise TypeError("receipt must be an object")
        manifest = self._load_manifest(receipt.get("manifest_sha256"))
        for key in ("source_path", "provider", "sha256", "size", "source_stat"):
            if receipt.get(key) != manifest.get(key):
                raise RawArchiveCorruptError(f"receipt {key} does not match manifest")
        return manifest

    def is_prefix(self, earlier: dict[str, Any], later: dict[str, Any]) -> bool:
        """Return whether the earlier verified version is an exact prefix of later."""
        old = self._manifest_for_receipt(earlier)
        new = self._manifest_for_receipt(later)
        if old["source_path"] != new["source_path"] or old["provider"] != new["provider"]:
            return False
        if old["size"] > new["size"]:
            return False
        if old["size"] == new["size"]:
            return old["sha256"] == new["sha256"]
        old_chunks = old["chunks"]
        new_chunks = new["chunks"]
        if len(new_chunks) < len(old_chunks):
            return False
        for index, old_chunk in enumerate(old_chunks):
            new_chunk = new_chunks[index]
            old_data = self._read_verified_chunk(old_chunk["sha256"], old_chunk["size"])
            new_data = self._read_verified_chunk(new_chunk["sha256"], new_chunk["size"])
            if not new_data.startswith(old_data):
                return False
        return True

    def reconstruct(self, receipt: dict[str, Any], target: BinaryIO) -> dict[str, Any]:
        digest, size = hashlib.sha256(), 0
        with self.open_snapshot(receipt) as source:
            while block := source.read(min(self.chunk_bytes, RawSnapshot.MAX_READ_BYTES)):
                target.write(block)
                digest.update(block)
                size += len(block)
        if size != receipt["size"] or digest.hexdigest() != receipt["sha256"]:
            raise RawArchiveCorruptError("reconstructed version digest mismatch")
        return {"sha256": digest.hexdigest(), "size": size}

    def _load_manifest(self, manifest_sha: Any) -> dict[str, Any]:
        self._require_sha(manifest_sha)
        raw = self._read_regular_bounded(self.manifests / f"{manifest_sha}.json", 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != manifest_sha:
            raise RawArchiveCorruptError("manifest digest mismatch")
        try:
            manifest = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise RawArchiveCorruptError("invalid manifest JSON") from error
        self._validate_manifest(manifest)
        if self._canonical_manifest(manifest) != raw:
            raise RawArchiveCorruptError("manifest is not canonical")
        return manifest

    def _validate_manifest(self, manifest: Any) -> None:
        keys = {"format_version", "provider", "source_path", "source_stat", "size", "sha256", "chunks"}
        if not isinstance(manifest, dict) or set(manifest) != keys:
            raise RawArchiveCorruptError("invalid manifest shape")
        if manifest["format_version"] != self.FORMAT_VERSION:
            raise RawArchiveCorruptError("unsupported manifest version")
        if not isinstance(manifest["provider"], str) or not manifest["provider"]:
            raise RawArchiveCorruptError("invalid manifest provider")
        if not isinstance(manifest["source_path"], str) or not manifest["source_path"]:
            raise RawArchiveCorruptError("invalid manifest source path")
        if type(manifest["size"]) is not int or not 0 <= manifest["size"] <= self.max_source_bytes:
            raise RawArchiveCorruptError("invalid manifest size")
        self._require_sha(manifest["sha256"], error=RawArchiveCorruptError)
        source_stat = manifest["source_stat"]
        stat_keys = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
        if not isinstance(source_stat, dict) or set(source_stat) != stat_keys:
            raise RawArchiveCorruptError("invalid manifest source stat")
        if any(type(source_stat[key]) is not int or source_stat[key] < 0 for key in stat_keys):
            raise RawArchiveCorruptError("invalid manifest source stat value")
        if source_stat["size"] != manifest["size"]:
            raise RawArchiveCorruptError("source stat size differs from manifest")
        chunks = manifest["chunks"]
        if not isinstance(chunks, list):
            raise RawArchiveCorruptError("invalid manifest chunks")
        total = 0
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or set(chunk) != {"sha256", "size"}:
                raise RawArchiveCorruptError("invalid chunk entry")
            self._require_sha(chunk["sha256"], error=RawArchiveCorruptError)
            if type(chunk["size"]) is not int or not 0 < chunk["size"] <= self.chunk_bytes:
                raise RawArchiveCorruptError("invalid chunk size")
            if index + 1 < len(chunks) and chunk["size"] != self.chunk_bytes:
                raise RawArchiveCorruptError("non-final chunk is not full sized")
            total += chunk["size"]
        if total != manifest["size"] or (not chunks and manifest["size"] != 0):
            raise RawArchiveCorruptError("chunk lengths do not match manifest")

    def _publish_chunk(self, sha256: str, data: bytes) -> None:
        self._require_sha(sha256)
        self._publish_bytes(self.objects / sha256, data, prefix=".chunk-")
        self._read_verified_chunk(sha256, len(data))

    def _read_verified_chunk(self, sha256: str, expected_size: int) -> bytes:
        self._require_sha(sha256)
        raw = self._read_regular_bounded(self.objects / sha256, expected_size)
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != sha256:
            raise RawArchiveCorruptError(f"chunk digest mismatch: {sha256}")
        return raw

    def _publish_bytes(self, path: Path, data: bytes, *, prefix: str) -> None:
        fd, name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fchmod(target.fileno(), 0o400)
                os.fsync(target.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            self._sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
            self._sync_dir(temporary.parent)

    @staticmethod
    def _canonical_manifest(manifest: dict[str, Any]) -> bytes:
        return (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("ascii")

    @staticmethod
    def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    @staticmethod
    def _open_regular(path: Path) -> BinaryIO:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RawArchiveError(f"not a regular file: {path}")
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise

    def _read_regular_bounded(self, path: Path, expected_max: int) -> bytes:
        with self._open_regular(path) as source:
            data = source.read(expected_max + 1)
            if len(data) > expected_max or source.read(1):
                raise RawArchiveCorruptError(f"archive object exceeds bound: {path.name}")
            return data

    @staticmethod
    def _require_sha(value: Any, *, error: type[Exception] = ValueError) -> None:
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise error("expected lowercase SHA-256 digest")

    def _private_dir(self, path: Path) -> None:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                raise RawArchiveError(f"archive directory is not owned: {path}")
        os.chmod(path, 0o700)
        self._sync_dir(path)
        self._sync_dir(path.parent)

    @staticmethod
    def _sync_dir(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
