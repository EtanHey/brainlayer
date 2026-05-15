"""Merge append-only BrainLayer queue directories safely."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class QueueMergeResult:
    copied: list[str] = field(default_factory=list)
    skipped_exact: list[str] = field(default_factory=list)
    skipped_non_jsonl: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)

    @property
    def total_actions(self) -> int:
        return len(self.copied) + len(self.skipped_exact) + len(self.skipped_non_jsonl)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_existing_hashes(queue_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not queue_dir.exists():
        return hashes
    for path in sorted(queue_dir.glob("*.jsonl")):
        if path.is_file():
            hashes[_sha256(path.read_bytes())] = path.name
    return hashes


def _collision_target(dest_dir: Path, source_name: str, content_hash: str, content: bytes) -> Path:
    source_path = Path(source_name)
    stem = source_path.stem
    suffix = source_path.suffix
    target = dest_dir / f"{stem}-merge-{content_hash[:12]}{suffix}"
    if not target.exists() or target.read_bytes() == content:
        return target
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}-merge-{content_hash[:12]}-{counter}{suffix}"
        if not candidate.exists() or candidate.read_bytes() == content:
            return candidate
        counter += 1


def _copy_atomic(source: Path, target: Path) -> None:
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


def merge_queue_dirs(source_dir: Path, dest_dir: Path, *, dry_run: bool = False) -> QueueMergeResult:
    """Union ``source_dir`` into ``dest_dir`` without deleting or overwriting queue files."""
    source_dir = source_dir.expanduser()
    dest_dir = dest_dir.expanduser()
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    if source_dir.resolve() == dest_dir.resolve():
        raise ValueError("source and destination queue directories must be different")

    result = QueueMergeResult()
    existing_hashes = _read_existing_hashes(dest_dir)

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(path for path in source_dir.iterdir() if path.is_file()):
        if source_path.suffix != ".jsonl":
            result.skipped_non_jsonl.append(source_path.name)
            continue

        content = source_path.read_bytes()
        content_hash = _sha256(content)
        if content_hash in existing_hashes:
            result.skipped_exact.append(source_path.name)
            continue

        target_path = dest_dir / source_path.name
        if target_path.exists():
            result.collisions.append(source_path.name)
            target_path = _collision_target(dest_dir, source_path.name, content_hash, content)

        result.copied.append(target_path.name)
        if not dry_run:
            _copy_atomic(source_path, target_path)
            existing_hashes[content_hash] = target_path.name

    return result
