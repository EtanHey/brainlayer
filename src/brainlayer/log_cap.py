"""Bound the launchd logs owned by BrainLayer jobs, including open append logs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import plistlib
import stat
import sys
from pathlib import Path
from xml.parsers.expat import ExpatError

DEFAULT_MAX_BYTES = 8 * 1024 * 1024
DEFAULT_KEEP_BYTES = 2 * 1024 * 1024


def _job_log_paths(agents_dir: Path) -> tuple[set[Path], list[str]]:
    paths: set[Path] = set()
    errors: list[str] = []
    for pattern in ("com.brainlayer.*.plist", "com.etanhey.brainlayer-*.plist"):
        for agent in agents_dir.glob(pattern):
            try:
                with agent.open("rb") as stream:
                    plist = plistlib.load(stream)
                if not isinstance(plist, dict):
                    raise ValueError("plist root is not a dictionary")
            except (OSError, ValueError, ExpatError) as exc:
                errors.append(f"{agent.stem}: {type(exc).__name__}: {exc}")
                continue
            if plist.get("Label") != agent.stem:
                continue
            for key in ("StandardOutPath", "StandardErrorPath"):
                value = plist.get(key)
                if isinstance(value, str):
                    path = Path(value).expanduser()
                    if path.is_absolute():
                        paths.add(path)
    return paths, errors


def _cap_file(path: Path, *, max_bytes: int, keep_bytes: int, line_safe: bool = False) -> bool:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"refusing symlink log path: {path}")
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_uid != os.getuid() or file_stat.st_nlink != 1:
        raise ValueError(f"refusing non-regular, non-owned, or linked log: {path}")
    if file_stat.st_size <= max_bytes:
        return False

    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        if line_safe:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(f"busy backup log; retry cap after append: {path}") from exc
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_uid != os.getuid() or opened_stat.st_nlink != 1:
            raise ValueError(f"log changed while opening: {path}")
        if opened_stat.st_size <= max_bytes:
            return False
        if line_safe and os.pread(fd, 1, opened_stat.st_size - 1) != b"\n":
            raise ValueError(f"incomplete backup record; refusing log cap: {path}")
        tail = os.pread(fd, keep_bytes, opened_stat.st_size - keep_bytes)
        if line_safe and os.pread(fd, 1, opened_stat.st_size - keep_bytes - 1) != b"\n":
            newline = tail.find(b"\n")
            tail = tail[newline + 1 :] if newline >= 0 else b""
        if line_safe and not tail:
            candidate = os.pread(fd, max_bytes, opened_stat.st_size - max_bytes)
            boundary = candidate.rfind(b"\n", 0, len(candidate) - 1)
            if candidate.endswith(b"\n") and boundary >= 0:
                tail = candidate[boundary + 1 :]
            elif candidate.endswith(b"\n") and os.pread(fd, 1, opened_stat.st_size - max_bytes - 1) == b"\n":
                tail = candidate
            else:
                raise ValueError(f"cap would retain no complete backup record: {path}")
        if line_safe:
            # Daily receipts carry provenance; JSONL archive receipts do not.
            # Preserve the latest real receipt even after diagnostic/test records.
            latest_real = None
            snapshots = {}
            with os.fdopen(os.dup(fd), "rb") as records:
                records.seek(0)
                while True:
                    offset = records.tell()
                    line = records.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(record, dict) and isinstance(record.get("snapshot"), str):
                        snapshots[record["snapshot"]] = (offset, record)
                    if isinstance(record, dict) and record.get("backup_log_provenance") == "real":
                        latest_real = offset
            for offset, record in snapshots.values():
                if record.get("local_verified") is True and not (record.get("uploaded") and record.get("verified")):
                    if opened_stat.st_size - offset > max_bytes:
                        raise ValueError(f"cap would erase pending local snapshot receipt: {path}")
                    if offset < opened_stat.st_size - len(tail):
                        tail = os.pread(fd, opened_stat.st_size - offset, offset)
            if latest_real is not None and latest_real < opened_stat.st_size - len(tail):
                if opened_stat.st_size - latest_real > max_bytes:
                    raise ValueError(f"cap would erase last real-provenance receipt: {path}")
                tail = os.pread(fd, opened_stat.st_size - latest_real, latest_real)
        os.ftruncate(fd, 0)  # Keep the inode held by launchd's O_APPEND descriptors.
        while tail:
            written = os.write(fd, tail)  # Append so a concurrent writer is never overwritten.
            if written <= 0:
                raise OSError(f"could not restore log tail: {path}")
            tail = tail[written:]
        os.fsync(fd)
        return True
    finally:
        os.close(fd)


def cap_job_logs(
    agents_dir: Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    keep_bytes: int = DEFAULT_KEEP_BYTES,
    manifest_path: Path | None = None,
    calendar_logs: Path | None = None,
    backup_logs: tuple[Path, ...] = (),
) -> list[Path]:
    """Trim oversized installed-job logs in place and return paths changed."""
    if not 0 < keep_bytes < max_bytes:
        raise ValueError("keep_bytes must be positive and less than max_bytes")
    if not agents_dir.is_dir():
        raise FileNotFoundError(f"LaunchAgents directory missing: {agents_dir}")
    paths, errors = _job_log_paths(agents_dir)
    if manifest_path is not None:
        from .paths import supervisor_socket_path
        from .supervisor_jobs import load_components

        for child in load_components(manifest_path, supervisor_socket_path()).values():
            paths.update(p for p in (child.stdout, child.stderr) if p is not None)
    if calendar_logs is not None:
        from .maintenance_calendar import NAMES

        paths.update(calendar_logs / f"{name}.{stream}.log" for name in NAMES for stream in ("out", "err"))
    paths.update(backup_logs)
    if not paths:
        errors.insert(0, f"no BrainLayer job log paths found in {agents_dir}")
    trimmed: list[Path] = []
    for path in sorted(paths):
        if _cap_file(path, max_bytes=max_bytes, keep_bytes=keep_bytes, line_safe=path in backup_logs):
            trimmed.append(path)
    if errors:
        raise RuntimeError("\n".join(errors))
    return trimmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents-dir", type=Path, default=Path.home() / "Library/LaunchAgents")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--keep-bytes", type=int, default=DEFAULT_KEEP_BYTES)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--calendar-logs", type=Path)
    parser.add_argument("--backup-log", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        cap_job_logs(
            args.agents_dir,
            max_bytes=args.max_bytes,
            keep_bytes=args.keep_bytes,
            manifest_path=args.manifest,
            calendar_logs=args.calendar_logs,
            backup_logs=tuple(args.backup_log),
        )
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
