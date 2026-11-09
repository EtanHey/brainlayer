"""Bounded zstd CLI decoding; source archives are never expanded onto disk."""

import os
import shutil
import subprocess
from pathlib import Path

ZSTD_CANDIDATE_PATHS = ("/opt/homebrew/bin/zstd", "/usr/local/bin/zstd")


def resolve_zstd(override_env: str = "BRAINLAYER_BACKUP_ZSTD_PATH") -> str | None:
    override = os.environ.get(override_env)
    if override is not None:
        candidate = Path(override)
        if not candidate.is_absolute() or not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ValueError(f"{override_env} must name an absolute executable zstd: {override!r}")
        return str(candidate)
    for candidate in ZSTD_CANDIDATE_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("zstd")


def logical_jsonl_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix("") if path.name.endswith(".jsonl.zst") else path


class ZstdDecodedFile:
    """A forward-only decoded-byte stream, validating the decoder at EOF.

    Reads always have an explicit bound. stderr is discarded rather than accumulated:
    failures carry the path and exit status, and cannot deadlock on a full stderr pipe.
    Early close terminates and reaps our own child, including on parser exceptions.
    """

    def __init__(self, path: Path, max_record_bytes: int = 128 * 1024 * 1024):
        binary = resolve_zstd(
            "BRAINLAYER_INGEST_ZSTD_PATH"
            if "BRAINLAYER_INGEST_ZSTD_PATH" in os.environ
            else "BRAINLAYER_BACKUP_ZSTD_PATH"
        )
        if binary is None:
            raise OSError("zstd binary not found (Homebrew prefixes and PATH)")
        self.path = path
        self.max_record_bytes = max_record_bytes
        self.process = subprocess.Popen(
            [binary, "-d", "-q", "-c", "--", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        self.position = 0
        self.eof = False

    def read(self, size: int) -> bytes:
        if size <= 0:
            raise ValueError("decoded reads require a positive size")
        data = self.process.stdout.read(size)
        self.position += len(data)
        if not data:
            status = self.process.wait()
            if status:
                raise OSError(f"zstd decode failed for {self.path} (exit {status}); corrupt or truncated archive")
            self.eof = True
        return data

    def __iter__(self):
        pending = bytearray()
        while data := self.read(64 * 1024):
            pending.extend(data)
            start = 0
            while (end := pending.find(b"\n", start)) >= 0:
                if end - start > self.max_record_bytes:
                    raise OSError("decoded JSONL record exceeds the memory limit")
                yield bytes(pending[start : end + 1])
                start = end + 1
            del pending[:start]
            if len(pending) > self.max_record_bytes:
                raise OSError("decoded JSONL record exceeds the memory limit")
        if pending:
            yield bytes(pending)

    def close(self):
        self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
