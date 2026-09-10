"""Durable raw-source versions for projection from immutable bytes.

Mechanism only: callers explicitly construct and inject an archive.
"""

from __future__ import annotations

import bisect
import contextlib
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Iterator

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
RAW_ARCHIVE_PATH_ENV = "BRAINLAYER_RAW_ARCHIVE_PATH"
RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV = "BRAINLAYER_RAW_ARCHIVE_MAX_SOURCE_BYTES"
RAW_ARCHIVE_DIRECTORY_NAME = "raw-source-archive-v1"


class RawArchiveError(Exception):
    pass


class RawSourceChangedError(RawArchiveError):
    pass


class RawSourceTooLargeError(RawArchiveError):
    pass


class RawArchiveCorruptError(RawArchiveError):
    pass


class RawSnapshot:
    """Seekable bounded reader over verified ordered archive objects."""

    MAX_READ_BYTES = 1024 * 1024

    def __init__(self, archive: RawSourceArchive, manifest: dict[str, Any]) -> None:
        self._archive = archive
        self._chunks = archive._snapshot_chunks(manifest)
        self._ends: list[int] = []
        total = 0
        for chunk in self._chunks:
            total += chunk["size"]
            self._ends.append(total)
        self._length, self._position = manifest["size"], 0
        if total != self._length:
            raise RawArchiveCorruptError("archive objects do not cover snapshot length")
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
            start = 0 if index == 0 else self._ends[index - 1]
            data = self._load_chunk(index)
            within = self._position - start
            take = min(len(data) - within, end - self._position)
            if take <= 0:
                raise RawArchiveCorruptError("archive objects cannot satisfy snapshot read")
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
    """Append-only source versions using delta manifests and content-addressed objects."""

    FORMAT_VERSION = 2
    LEGACY_FORMAT_VERSION = 1
    DEFAULT_CHUNK_BYTES = 4 * 1024 * 1024
    DEFAULT_MAX_SOURCE_BYTES = 8 * 1024 * 1024 * 1024
    MAX_CHAIN_MANIFESTS = 1_000_000
    MAX_MANIFEST_BYTES = 1024 * 1024

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
        self.sources = self.root / "sources-v2"
        for directory in (self.root, self.objects, self.manifests, self.sources):
            self._private_dir(directory)

    def capture(self, path: str | Path, provider: str) -> dict[str, Any]:
        """Capture one stable whole-file version and return its durable receipt."""
        if not isinstance(provider, str) or not provider or len(provider) > 128:
            raise ValueError("provider must be a nonempty string of at most 128 characters")
        source_path = Path(path).absolute()
        state = self._source_state(source_path, provider)
        with self._source_lock(state):
            return self._capture_locked(source_path, provider, state)

    def _capture_locked(self, source_path: Path, provider: str, state: Path) -> dict[str, Any]:
        previous = self._load_head(state)
        previous_manifest = self._manifest_for_receipt(previous) if previous is not None else None
        if previous is not None:
            self._verify_snapshot_digest(previous)
        version_hash, size = hashlib.sha256(), 0
        prefix = True
        prefix_hash = hashlib.sha256()
        prefix_decided = previous is None or previous["size"] == 0
        segments: list[dict[str, Any]] = []

        with self._open_regular(source_path) as source:
            before = os.fstat(source.fileno())
            if before.st_size > self.max_source_bytes:
                raise RawSourceTooLargeError(
                    f"source {source_path} size {before.st_size} exceeds cap {self.max_source_bytes} bytes"
                )
            while True:
                data = source.read(min(self.chunk_bytes, RawSnapshot.MAX_READ_BYTES, self.max_source_bytes - size + 1))
                if not data:
                    break
                block_start = size
                size += len(data)
                if size > self.max_source_bytes:
                    raise RawSourceTooLargeError(
                        f"source {source_path} observed size over cap {self.max_source_bytes} bytes during capture"
                    )
                version_hash.update(data)
                suffix = data
                if previous is not None and block_start < previous["size"]:
                    compare_size = min(len(data), previous["size"] - block_start)
                    prefix_hash.update(data[:compare_size])
                    suffix = data[compare_size:]
                    if size >= previous["size"]:
                        prefix = prefix_hash.hexdigest() == previous["sha256"]
                        prefix_decided = True
                if prefix and prefix_decided and suffix:
                    segments.extend(self._publish_segments(suffix))
            if previous is not None and size < previous["size"]:
                prefix = False
                segments.clear()

            after = os.fstat(source.fileno())
            try:
                current = source_path.lstat()
            except OSError as error:
                raise RawSourceChangedError("source path disappeared during capture") from error
            signature = self._signature(before)
            if signature != self._signature(after) or signature != self._signature(current) or size != before.st_size:
                raise RawSourceChangedError("source changed during capture")

            if not prefix:
                source.seek(0)
                segments = []
                remaining = size
                while remaining:
                    data = source.read(min(self.chunk_bytes, remaining))
                    if not data:
                        raise RawSourceChangedError("source became unreadable during capture")
                    segments.extend(self._publish_segments(data))
                    remaining -= len(data)
                try:
                    final_current = source_path.lstat()
                except OSError as error:
                    raise RawSourceChangedError("source path disappeared during capture") from error
                if signature != self._signature(os.fstat(source.fileno())) or signature != self._signature(
                    final_current
                ):
                    raise RawSourceChangedError("source changed during capture")

        digest = version_hash.hexdigest()
        source_stat = self._source_stat(before)
        if previous is not None and prefix and size == previous["size"] and digest == previous["sha256"]:
            receipt = dict(previous)
            receipt["source_stat"] = source_stat
            if receipt != previous:
                self._replace_head(state, receipt)
            return receipt

        parent_sha = previous["manifest_sha256"] if prefix and previous is not None else None
        root_sha = previous_manifest["generation_root_manifest_sha256"] if parent_sha else None
        if parent_sha and root_sha is None:
            root_sha = parent_sha
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "provider": provider,
            "source_path": str(source_path),
            "generation_root_manifest_sha256": root_sha,
            "parent_manifest_sha256": parent_sha,
            "generation_nonce": None if parent_sha else secrets.token_hex(32),
            "delta_start": previous["size"] if parent_sha and previous is not None else 0,
            "size": size,
            "sha256": digest,
            "segments": segments,
        }
        raw = self._canonical_manifest(manifest)
        manifest_sha = hashlib.sha256(raw).hexdigest()
        manifest_path = self.manifests / f"{manifest_sha}.json"
        self._publish_bytes(manifest_path, raw, prefix=".manifest-")
        if self._load_manifest(manifest_sha) != manifest:
            raise RawArchiveCorruptError("published manifest differs from captured version")
        generation_root = root_sha or manifest_sha
        self._publish_root_marker(state, generation_root)
        receipt = {
            "format_version": self.FORMAT_VERSION,
            "manifest_sha256": manifest_sha,
            "manifest_path": str(manifest_path),
            "source_path": str(source_path),
            "provider": provider,
            "sha256": digest,
            "size": size,
            "source_stat": source_stat,
        }
        self._replace_head(state, receipt)
        return receipt

    def open_snapshot(self, receipt: dict[str, Any]) -> RawSnapshot:
        return RawSnapshot(self, self._manifest_for_receipt(receipt))

    def _manifest_for_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(receipt, dict):
            raise TypeError("receipt must be an object")
        manifest_sha = receipt.get("manifest_sha256")
        manifest = self._load_manifest(manifest_sha)
        if receipt.get("manifest_path") != str(self.manifests / f"{manifest_sha}.json"):
            raise RawArchiveCorruptError("receipt manifest_path does not match archive")
        for key in ("source_path", "provider", "sha256", "size"):
            if receipt.get(key) != manifest.get(key):
                raise RawArchiveCorruptError(f"receipt {key} does not match manifest")
        if manifest["format_version"] == self.LEGACY_FORMAT_VERSION:
            if "format_version" in receipt:
                raise RawArchiveCorruptError("legacy receipt must not claim a format version")
            if receipt.get("source_stat") != manifest.get("source_stat"):
                raise RawArchiveCorruptError("receipt source_stat does not match manifest")
        else:
            if type(receipt.get("format_version")) is not int or receipt["format_version"] != self.FORMAT_VERSION:
                raise RawArchiveCorruptError("receipt format_version does not match manifest")
            self._validate_source_stat(receipt.get("source_stat"), manifest["size"])
            state = self._source_state_path(Path(manifest["source_path"]), manifest["provider"])
            self._verify_root_marker(
                state,
                self._generation_root(manifest, manifest_sha),
            )
        return manifest

    def is_prefix(self, earlier: dict[str, Any], later: dict[str, Any]) -> bool:
        """Return whether the earlier verified version is an exact prefix of later."""
        old = self._manifest_for_receipt(earlier)
        new = self._manifest_for_receipt(later)
        if old["source_path"] != new["source_path"] or old["provider"] != new["provider"]:
            return False
        if old["format_version"] == self.FORMAT_VERSION and new["format_version"] == self.FORMAT_VERSION:
            old_root = self._generation_root(old, earlier["manifest_sha256"])
            new_root = self._generation_root(new, later["manifest_sha256"])
            if old_root != new_root:
                return False
        if old["size"] > new["size"]:
            return False
        if old["size"] == new["size"]:
            return old["sha256"] == new["sha256"]
        if old["format_version"] == self.FORMAT_VERSION and new["format_version"] == self.FORMAT_VERSION:
            cursor = later["manifest_sha256"]
            steps = 0
            while cursor != earlier["manifest_sha256"]:
                item = self._load_manifest(cursor)
                if item["format_version"] != self.FORMAT_VERSION or steps >= self.MAX_CHAIN_MANIFESTS:
                    raise RawArchiveCorruptError("invalid or overlong v2 prefix chain")
                parent = item.get("parent_manifest_sha256")
                if parent is None or item["size"] <= old["size"]:
                    return False
                cursor = parent
                steps += 1
            self._verify_snapshot_digest(earlier)
            return True
        return self._stream_prefix(earlier, later)

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

    def _snapshot_chunks(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        if manifest["format_version"] == self.LEGACY_FORMAT_VERSION:
            return list(manifest["chunks"])
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        current = manifest
        while True:
            manifest_sha = hashlib.sha256(self._canonical_manifest(current)).hexdigest()
            if manifest_sha in seen or len(chain) >= self.MAX_CHAIN_MANIFESTS:
                raise RawArchiveCorruptError("manifest chain is cyclic or exceeds bound")
            seen.add(manifest_sha)
            chain.append(current)
            parent = current["parent_manifest_sha256"]
            if parent is None:
                root_sha = manifest_sha
                break
            current = self._load_manifest(parent)
            if current["format_version"] != self.FORMAT_VERSION:
                raise RawArchiveCorruptError("v2 manifest chain references another format")
        chain.reverse()
        expected_start = 0
        chunks: list[dict[str, Any]] = []
        for index, item in enumerate(chain):
            if item["provider"] != manifest["provider"] or item["source_path"] != manifest["source_path"]:
                raise RawArchiveCorruptError("manifest chain changes source identity")
            if item["delta_start"] != expected_start:
                raise RawArchiveCorruptError("manifest chain has a discontinuous delta")
            expected_root = None if index == 0 else root_sha
            if item["generation_root_manifest_sha256"] != expected_root:
                raise RawArchiveCorruptError("manifest chain has an invalid generation root")
            chunks.extend(item["segments"])
            expected_start = item["size"]
        if expected_start != manifest["size"]:
            raise RawArchiveCorruptError("manifest chain size mismatch")
        return chunks

    def _load_manifest(self, manifest_sha: Any) -> dict[str, Any]:
        self._require_sha(manifest_sha)
        raw = self._read_regular_bounded(self.manifests / f"{manifest_sha}.json", self.MAX_MANIFEST_BYTES)
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
        if not isinstance(manifest, dict):
            raise RawArchiveCorruptError("invalid manifest shape")
        if manifest.get("format_version") == self.LEGACY_FORMAT_VERSION:
            self._validate_v1_manifest(manifest)
            return
        keys = {
            "format_version",
            "provider",
            "source_path",
            "generation_root_manifest_sha256",
            "parent_manifest_sha256",
            "generation_nonce",
            "delta_start",
            "size",
            "sha256",
            "segments",
        }
        if set(manifest) != keys or manifest.get("format_version") != self.FORMAT_VERSION:
            raise RawArchiveCorruptError("invalid or unsupported manifest shape")
        self._validate_manifest_identity(manifest)
        for key in ("generation_root_manifest_sha256", "parent_manifest_sha256"):
            if manifest[key] is not None:
                self._require_sha(manifest[key], error=RawArchiveCorruptError)
        if (manifest["parent_manifest_sha256"] is None) != (manifest["generation_root_manifest_sha256"] is None):
            raise RawArchiveCorruptError("root and parent fields disagree")
        nonce = manifest["generation_nonce"]
        if manifest["parent_manifest_sha256"] is None:
            self._require_sha(nonce, error=RawArchiveCorruptError)
        elif nonce is not None:
            raise RawArchiveCorruptError("child manifest contains a generation nonce")
        if type(manifest["delta_start"]) is not int or not 0 <= manifest["delta_start"] <= manifest["size"]:
            raise RawArchiveCorruptError("invalid manifest delta start")
        segments = manifest["segments"]
        if not isinstance(segments, list):
            raise RawArchiveCorruptError("invalid manifest segments")
        total = 0
        for segment in segments:
            if not isinstance(segment, dict) or set(segment) != {"sha256", "size"}:
                raise RawArchiveCorruptError("invalid segment entry")
            self._require_sha(segment["sha256"], error=RawArchiveCorruptError)
            if type(segment["size"]) is not int or not 0 < segment["size"] <= self.chunk_bytes:
                raise RawArchiveCorruptError("invalid segment size")
            total += segment["size"]
        if manifest["delta_start"] + total != manifest["size"]:
            raise RawArchiveCorruptError("segment lengths do not match manifest delta")

    def _validate_v1_manifest(self, manifest: dict[str, Any]) -> None:
        keys = {"format_version", "provider", "source_path", "source_stat", "size", "sha256", "chunks"}
        if set(manifest) != keys:
            raise RawArchiveCorruptError("invalid legacy manifest shape")
        self._validate_manifest_identity(manifest)
        self._validate_source_stat(manifest["source_stat"], manifest["size"])
        chunks = manifest["chunks"]
        if not isinstance(chunks, list):
            raise RawArchiveCorruptError("invalid legacy manifest chunks")
        total = 0
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or set(chunk) != {"sha256", "size"}:
                raise RawArchiveCorruptError("invalid legacy chunk entry")
            self._require_sha(chunk["sha256"], error=RawArchiveCorruptError)
            if type(chunk["size"]) is not int or not 0 < chunk["size"] <= self.chunk_bytes:
                raise RawArchiveCorruptError("invalid legacy chunk size")
            if index + 1 < len(chunks) and chunk["size"] != self.chunk_bytes:
                raise RawArchiveCorruptError("non-final legacy chunk is not full sized")
            total += chunk["size"]
        if total != manifest["size"]:
            raise RawArchiveCorruptError("legacy chunk lengths do not match manifest")

    def _validate_manifest_identity(self, manifest: dict[str, Any]) -> None:
        if not isinstance(manifest["provider"], str) or not manifest["provider"]:
            raise RawArchiveCorruptError("invalid manifest provider")
        if not isinstance(manifest["source_path"], str) or not manifest["source_path"]:
            raise RawArchiveCorruptError("invalid manifest source path")
        if type(manifest["size"]) is not int or not 0 <= manifest["size"] <= self.max_source_bytes:
            raise RawArchiveCorruptError("invalid manifest size")
        self._require_sha(manifest["sha256"], error=RawArchiveCorruptError)

    def _publish_segments(self, data: bytes) -> list[dict[str, Any]]:
        segments = []
        for start in range(0, len(data), self.chunk_bytes):
            block = data[start : start + self.chunk_bytes]
            sha = hashlib.sha256(block).hexdigest()
            self._publish_chunk(sha, block)
            segments.append({"sha256": sha, "size": len(block)})
        return segments

    def _publish_chunk(self, sha256: str, data: bytes) -> None:
        self._require_sha(sha256)
        path = self.objects / sha256
        try:
            raw = self._read_regular_bounded(path, len(data))
        except FileNotFoundError:
            self._publish_bytes(path, data, prefix=".chunk-")
        else:
            if len(raw) != len(data) or hashlib.sha256(raw).hexdigest() != sha256:
                raise RawArchiveCorruptError(f"chunk digest mismatch: {sha256}")
            return
        self._read_verified_chunk(sha256, len(data))

    def _read_verified_chunk(self, sha256: str, expected_size: int) -> bytes:
        self._require_sha(sha256)
        try:
            raw = self._read_regular_bounded(self.objects / sha256, expected_size)
        except FileNotFoundError as error:
            raise RawArchiveCorruptError(f"chunk is missing: {sha256}") from error
        if len(raw) != expected_size or hashlib.sha256(raw).hexdigest() != sha256:
            raise RawArchiveCorruptError(f"chunk digest mismatch: {sha256}")
        return raw

    def _source_state(self, source_path: Path, provider: str) -> Path:
        state = self._source_state_path(source_path, provider)
        self._private_dir(state)
        self._private_dir(state / "roots")
        return state

    def _source_state_path(self, source_path: Path, provider: str) -> Path:
        identity = self._canonical_manifest({"provider": provider, "source_path": str(source_path)})
        return self.sources / hashlib.sha256(identity).hexdigest()

    def _load_head(self, state: Path) -> dict[str, Any] | None:
        path = state / "HEAD.json"
        try:
            raw = self._read_regular_bounded(path, self.MAX_MANIFEST_BYTES)
        except FileNotFoundError:
            if any((state / "roots").iterdir()):
                raise RawArchiveCorruptError("source generation exists without HEAD")
            return None
        try:
            receipt = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise RawArchiveCorruptError("invalid source HEAD") from error
        if self._canonical_manifest(receipt) != raw:
            raise RawArchiveCorruptError("source HEAD is not canonical")
        manifest = self._manifest_for_receipt(receipt)
        root = self._generation_root(manifest, receipt["manifest_sha256"])
        self._verify_root_marker(state, root)
        return receipt

    def _replace_head(self, state: Path, receipt: dict[str, Any]) -> None:
        self._replace_bytes(state / "HEAD.json", self._canonical_manifest(receipt), prefix=".head-")

    def _publish_root_marker(self, state: Path, root_sha: str) -> None:
        marker = state / "roots" / f"{root_sha}.ref"
        expected = (root_sha + "\n").encode("ascii")
        try:
            current = self._read_regular_bounded(marker, len(expected))
            if current != expected:
                raise RawArchiveCorruptError("generation root marker mismatch")
            return
        except FileNotFoundError:
            pass
        self._publish_bytes(marker, expected, prefix=".root-")
        self._verify_root_marker(state, root_sha)

    def _verify_root_marker(self, state: Path, root_sha: str) -> None:
        expected = (root_sha + "\n").encode("ascii")
        try:
            current = self._read_regular_bounded(state / "roots" / f"{root_sha}.ref", len(expected))
        except FileNotFoundError as error:
            raise RawArchiveCorruptError("generation root marker is missing") from error
        if current != expected:
            raise RawArchiveCorruptError("generation root marker mismatch")

    @contextlib.contextmanager
    def _source_lock(self, state: Path) -> Iterator[None]:
        fd = os.open(
            state / "LOCK",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                raise RawArchiveError("source archive lock is not an owned regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _generation_root(manifest: dict[str, Any], manifest_sha: str) -> str:
        return manifest.get("generation_root_manifest_sha256") or manifest_sha

    def _verify_snapshot_digest(self, receipt: dict[str, Any]) -> None:
        digest, size = hashlib.sha256(), 0
        with self.open_snapshot(receipt) as snapshot:
            while data := snapshot.read(min(self.chunk_bytes, RawSnapshot.MAX_READ_BYTES)):
                digest.update(data)
                size += len(data)
        if size != receipt["size"] or digest.hexdigest() != receipt["sha256"]:
            raise RawArchiveCorruptError("snapshot digest mismatch")

    def _stream_prefix(self, earlier: dict[str, Any], later: dict[str, Any]) -> bool:
        with self.open_snapshot(earlier) as old, self.open_snapshot(later) as new:
            remaining = earlier["size"]
            while remaining:
                amount = min(remaining, RawSnapshot.MAX_READ_BYTES, self.chunk_bytes)
                if old.read(amount) != new.read(amount):
                    return False
                remaining -= amount
        return True

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

    def _replace_bytes(self, path: Path, data: bytes, *, prefix: str) -> None:
        fd, name = tempfile.mkstemp(prefix=prefix, dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(data)
                target.flush()
                os.fchmod(target.fileno(), 0o400)
                os.fsync(target.fileno())
            os.replace(temporary, path)
            self._sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _canonical_manifest(manifest: dict[str, Any]) -> bytes:
        return (
            json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("ascii")

    @staticmethod
    def _signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    @staticmethod
    def _source_stat(info: os.stat_result) -> dict[str, int]:
        return {
            "device": info.st_dev,
            "inode": info.st_ino,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }

    @staticmethod
    def _validate_source_stat(value: Any, size: int) -> None:
        keys = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
        if not isinstance(value, dict) or set(value) != keys:
            raise RawArchiveCorruptError("invalid receipt source stat")
        if any(type(value[key]) is not int or value[key] < 0 for key in keys) or value["size"] != size:
            raise RawArchiveCorruptError("invalid receipt source stat value")

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


def raw_archive_path(db_path: str | Path, environ: dict[str, str] | None = None) -> Path:
    """Resolve the persistent archive root without creating it."""
    env = os.environ if environ is None else environ
    configured = env.get(RAW_ARCHIVE_PATH_ENV)
    if configured is None:
        return Path(db_path).expanduser().absolute().parent / RAW_ARCHIVE_DIRECTORY_NAME
    if not configured.strip():
        raise ValueError(f"{RAW_ARCHIVE_PATH_ENV} must not be empty")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{RAW_ARCHIVE_PATH_ENV} must be an absolute path")
    return path


def raw_archive_max_source_bytes(environ: dict[str, str] | None = None) -> int:
    """Resolve the strict per-source ceiling; malformed overrides never fall back."""
    env = os.environ if environ is None else environ
    raw = env.get(RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV, str(RawSourceArchive.DEFAULT_MAX_SOURCE_BYTES))
    if not raw.isdecimal() or (value := int(raw)) <= 0:
        raise ValueError(f"{RAW_ARCHIVE_MAX_SOURCE_BYTES_ENV} must be a positive decimal integer")
    return value


def create_raw_source_archive(db_path: str | Path, environ: dict[str, str] | None = None) -> RawSourceArchive:
    """Construct the required archive for one resolved BrainLayer database."""
    root = raw_archive_path(db_path, environ)
    max_source_bytes = raw_archive_max_source_bytes(environ)
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return RawSourceArchive(root, max_source_bytes=max_source_bytes)
