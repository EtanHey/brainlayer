"""Replay docs.local BrainLayer fallback files with origin attribution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

_FALLBACK_LOCKS_LOCK = threading.Lock()
_FALLBACK_LOCKS: dict[Path, threading.Lock] = {}


@dataclass(frozen=True)
class FallbackEntry:
    path: Path
    frontmatter: dict[str, Any]
    body: str
    origin_repo_path: Path
    project: str


@dataclass(frozen=True)
class ReplayResult:
    path: Path
    attempted: bool
    chunk_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class FallbackInventory:
    structured: list[FallbackEntry]
    legacy: list[Path]

    @property
    def pending(self) -> list[FallbackEntry]:
        return [entry for entry in self.structured if is_pending_entry(entry)]

    def summary(self, *, sample_limit: int = 20) -> dict[str, Any]:
        pending = self.pending
        return {
            "green": len(pending) == 0 and len(self.legacy) == 0,
            "structured_count": len(self.structured),
            "pending_count": len(pending),
            "legacy_count": len(self.legacy),
            "pending_sample": [str(entry.path) for entry in pending[:sample_limit]],
            "legacy_sample": [str(path) for path in self.legacy[:sample_limit]],
        }


def load_scope_map(scopes_path: Path | None = None) -> dict[str, str]:
    path = scopes_path or Path.home() / ".config" / "brainlayer" / "scopes.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scopes = data.get("scopes") or {}
    return {str(Path(prefix).expanduser().resolve()): str(project) for prefix, project in scopes.items()}


def inventory_fallbacks(gits_root: Path, *, scope_map: dict[str, str]) -> FallbackInventory:
    structured: list[FallbackEntry] = []
    legacy: list[Path] = []
    if not gits_root.exists():
        return FallbackInventory(structured=structured, legacy=legacy)

    for repo in sorted(path for path in gits_root.iterdir() if path.is_dir()):
        for path in sorted((repo / "docs.local" / "decisions").glob("*.md")):
            try:
                entry = parse_fallback_file(path, scope_map=scope_map)
            except Exception:
                legacy.append(path)
                continue
            if entry.frontmatter.get("intended_brain_store"):
                structured.append(entry)

        fallback_dir = repo / "docs.local" / "brain-store-fallback"
        if fallback_dir.exists():
            for path in sorted(path for path in fallback_dir.rglob("*.md") if path.is_file()):
                try:
                    entry = legacy_entry_from_path(path, scope_map=scope_map)
                except Exception:
                    legacy.append(path)
                    continue
                if entry.frontmatter.get("intended_brain_store") and not is_pending_entry(entry):
                    continue
                legacy.append(path)

    return FallbackInventory(structured=structured, legacy=legacy)


def parse_fallback_file(path: Path, *, scope_map: dict[str, str] | None = None) -> FallbackEntry:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    origin_repo_path = _git_root(path)
    project = _resolve_project(
        path, frontmatter=frontmatter, origin_repo_path=origin_repo_path, scope_map=scope_map or {}
    )
    return FallbackEntry(
        path=path,
        frontmatter=frontmatter,
        body=body,
        origin_repo_path=origin_repo_path,
        project=project,
    )


def is_pending_entry(entry: FallbackEntry) -> bool:
    if not entry.frontmatter.get("intended_brain_store"):
        return False
    chunk_id = str(entry.frontmatter.get("chunk_id") or "").strip()
    if not chunk_id:
        return True
    replayed_body_sha256 = str(entry.frontmatter.get("replayed_body_sha256") or "").strip()
    if replayed_body_sha256:
        return replayed_body_sha256 != _body_sha256(entry.body)
    if chunk_id.startswith("fallback-") and chunk_id != _fallback_chunk_id(entry):
        return True
    return False


def replay_entry(
    entry: FallbackEntry,
    *,
    store_func: Callable[..., Any],
    replayed_by: str,
) -> ReplayResult:
    try:
        store_kwargs = {
            "content": entry.body,
            "memory_type": str(entry.frontmatter.get("memory_type") or "note"),
            "project": entry.project,
            "tags": _normalize_tags(entry.frontmatter.get("tags")),
            "importance": entry.frontmatter.get("importance"),
            "created_at": str(entry.frontmatter.get("timestamp") or ""),
            "fallback_source_path": str(entry.path),
            "origin_repo_path": str(entry.origin_repo_path),
            "replayed_by": replayed_by,
        }
        if entry.frontmatter.get("chunk_origin"):
            store_kwargs["chunk_origin"] = entry.frontmatter["chunk_origin"]
        result = store_func(**store_kwargs)
    except Exception as exc:
        try:
            _write_replay_attempt(entry, chunk_id=entry.frontmatter.get("chunk_id") or None)
        except Exception as write_exc:
            return ReplayResult(
                path=entry.path,
                attempted=True,
                chunk_id=None,
                error=f"store failed: {exc}; write_replay_attempt failed: {write_exc}",
            )
        return ReplayResult(path=entry.path, attempted=True, chunk_id=None, error=str(exc))

    chunk_id = _extract_chunk_id(result)
    if not chunk_id:
        error = "store result did not include a chunk_id"
        try:
            _write_replay_attempt(entry, chunk_id=None)
        except Exception as exc:
            error = f"{error}; write_replay_attempt failed: {exc}"
        return ReplayResult(path=entry.path, attempted=True, chunk_id=None, error=error)

    try:
        _write_replay_attempt(entry, chunk_id=chunk_id, trust_chunk_id=True)
    except Exception as exc:
        return ReplayResult(
            path=entry.path,
            attempted=True,
            chunk_id=chunk_id,
            error=f"stored chunk_id={chunk_id} but failed to update fallback frontmatter: {exc}",
        )

    return ReplayResult(path=entry.path, attempted=True, chunk_id=chunk_id)


def queue_entry(
    entry: FallbackEntry,
    *,
    enqueue_func: Callable[..., Any],
    replayed_by: str,
    source: str = "fallback-replay",
) -> ReplayResult:
    with _fallback_marker_file_lock(entry.path):
        latest = _latest_entry(entry)
        chunk_id = _fallback_chunk_id(latest)
        queued = str(latest.frontmatter.get("queued_chunk_id") or "").strip()
        queued_path = _queued_queue_path(latest)
        pending = is_pending_entry(latest)
        if queued == chunk_id and pending and (queued_path is None or queued_path.exists()):
            return ReplayResult(path=entry.path, attempted=True, chunk_id=chunk_id)

        stored = _stored_chunk_id(latest.frontmatter)
        if stored and not pending and _fallback_chunk_matches(latest, stored):
            return ReplayResult(path=entry.path, attempted=True, chunk_id=stored)
        if not pending:
            return ReplayResult(path=entry.path, attempted=False, chunk_id=None)

        try:
            queue_path = enqueue_func(
                content=latest.body,
                memory_type=str(latest.frontmatter.get("memory_type") or "note"),
                project=latest.project,
                tags=_normalize_tags(latest.frontmatter.get("tags")),
                importance=latest.frontmatter.get("importance"),
                created_at=_json_safe_scalar(latest.frontmatter.get("timestamp")) or None,
                source=source,
                chunk_id=chunk_id,
                fallback_source_path=str(latest.path),
                origin_repo_path=str(latest.origin_repo_path),
                replayed_by=replayed_by,
                chunk_origin=latest.frontmatter.get("chunk_origin"),
            )
        except Exception as exc:
            try:
                _write_replay_attempt_locked(latest, chunk_id=None)
            except Exception as write_exc:
                return ReplayResult(
                    path=entry.path,
                    attempted=True,
                    chunk_id=None,
                    error=f"queue failed: {exc}; write_replay_attempt failed: {write_exc}",
                )
            return ReplayResult(path=entry.path, attempted=True, chunk_id=None, error=str(exc))

        try:
            _write_queue_attempt_locked(latest, chunk_id=chunk_id, queue_path=queue_path)
        except Exception as exc:
            return ReplayResult(
                path=entry.path,
                attempted=True,
                chunk_id=chunk_id,
                error=f"queued chunk_id={chunk_id} but failed to update fallback frontmatter: {exc}",
            )

    return ReplayResult(path=entry.path, attempted=True, chunk_id=chunk_id)


def legacy_entry_from_path(path: Path, *, scope_map: dict[str, str] | None = None) -> FallbackEntry:
    entry = parse_fallback_file(path, scope_map=scope_map or {})
    frontmatter = dict(entry.frontmatter)
    frontmatter.setdefault("intended_brain_store", True)
    frontmatter["legacy_brain_store_fallback"] = True
    frontmatter.setdefault("importance", _infer_legacy_importance(entry.body))
    frontmatter.setdefault("tags", ["legacy-fallback", "brain-store-fallback", entry.project])
    frontmatter.setdefault("timestamp", _infer_legacy_timestamp(path))
    frontmatter.setdefault("reason", "legacy_brain_store_fallback")
    return FallbackEntry(
        path=entry.path,
        frontmatter=frontmatter,
        body=entry.body,
        origin_repo_path=entry.origin_repo_path,
        project=entry.project,
    )


def queue_legacy_entry(
    entry: FallbackEntry,
    *,
    enqueue_func: Callable[..., Any],
    replayed_by: str,
    source: str = "fallback-replay",
) -> ReplayResult:
    return queue_entry(entry, enqueue_func=enqueue_func, replayed_by=replayed_by, source=source)


def _fallback_chunk_id(entry: FallbackEntry) -> str:
    stable = json.dumps(
        {
            "path": os.path.relpath(entry.path.resolve(), entry.origin_repo_path.resolve()),
            "body": entry.body,
            "timestamp": _json_safe_scalar(entry.frontmatter.get("timestamp")),
            "project": entry.project,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "fallback-" + hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def _body_sha256(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def mark_fallback_stored(
    path: Path,
    *,
    chunk_id: str,
    project: str | None = None,
    origin_repo_path: Path | None = None,
) -> None:
    entry = legacy_entry_from_path(path) if _is_legacy_fallback_path(path) else parse_fallback_file(path)
    if project is not None or origin_repo_path is not None:
        entry = FallbackEntry(
            path=entry.path,
            frontmatter=entry.frontmatter,
            body=entry.body,
            origin_repo_path=origin_repo_path or entry.origin_repo_path,
            project=project or entry.project,
        )
    _write_replay_attempt(entry, chunk_id=chunk_id)


def _latest_entry(entry: FallbackEntry) -> FallbackEntry:
    try:
        latest = parse_fallback_file(entry.path)
    except Exception:
        return entry
    frontmatter = dict(latest.frontmatter)
    if _is_legacy_fallback_path(entry.path):
        for key, value in entry.frontmatter.items():
            frontmatter.setdefault(key, value)
    return FallbackEntry(
        path=latest.path,
        frontmatter=frontmatter,
        body=latest.body,
        origin_repo_path=entry.origin_repo_path,
        project=entry.project,
    )


def _stored_chunk_id(frontmatter: dict[str, Any]) -> str:
    return str(frontmatter.get("chunk_id") or "").strip()


def _fallback_chunk_matches(entry: FallbackEntry, chunk_id: Any) -> bool:
    text = str(chunk_id or "").strip()
    return not text.startswith("fallback-") or text == _fallback_chunk_id(entry)


def _is_legacy_fallback_path(path: Path) -> bool:
    return "brain-store-fallback" in path.parts


def _queued_queue_path(entry: FallbackEntry) -> Path | None:
    value = str(entry.frontmatter.get("queued_queue_path") or "").strip()
    if not value:
        return None
    return Path(value).expanduser()


def _resolved_queue_path(queue_path: Any) -> str | None:
    if not queue_path:
        return None
    return str(Path(queue_path).expanduser().resolve())


def _write_queue_attempt(entry: FallbackEntry, *, chunk_id: Any, queue_path: Any = None) -> None:
    with _fallback_marker_file_lock(entry.path):
        _write_queue_attempt_locked(entry, chunk_id=chunk_id, queue_path=queue_path)


def _write_queue_attempt_locked(entry: FallbackEntry, *, chunk_id: Any, queue_path: Any = None) -> None:
    latest = _latest_entry(entry)
    pending = is_pending_entry(latest)
    stored = _stored_chunk_id(latest.frontmatter)
    if stored and not pending and _fallback_chunk_matches(latest, stored):
        return
    if not pending:
        return
    if chunk_id and not _fallback_chunk_matches(latest, chunk_id):
        return
    updated = _frontmatter_with_resolved_project(latest)
    updated["retry_attempted"] = True
    updated["queued_chunk_id"] = chunk_id
    updated["queued_queue_path"] = _resolved_queue_path(queue_path)
    updated["chunk_id"] = None
    _write_frontmatter(latest.path, updated, latest.body)


def _write_replay_attempt(entry: FallbackEntry, *, chunk_id: Any, trust_chunk_id: bool = False) -> None:
    with _fallback_marker_file_lock(entry.path):
        _write_replay_attempt_locked(entry, chunk_id=chunk_id, trust_chunk_id=trust_chunk_id)


def _write_replay_attempt_locked(entry: FallbackEntry, *, chunk_id: Any, trust_chunk_id: bool = False) -> None:
    latest = _latest_entry(entry)
    pending = is_pending_entry(latest)
    stored = _stored_chunk_id(latest.frontmatter)
    if not chunk_id and not pending and stored and _fallback_chunk_matches(latest, stored):
        return
    if chunk_id and not pending:
        return
    if chunk_id and latest.body != entry.body:
        raise RuntimeError("fallback body changed before replay writeback")
    if chunk_id and not trust_chunk_id and not _fallback_chunk_matches(latest, chunk_id):
        raise RuntimeError("fallback chunk marker changed before replay writeback")
    if not pending:
        return
    updated = _frontmatter_with_resolved_project(latest)
    updated["retry_attempted"] = True
    updated["chunk_id"] = chunk_id or None
    if chunk_id:
        updated.pop("queued_chunk_id", None)
    if trust_chunk_id and chunk_id:
        updated["replayed_body_sha256"] = _body_sha256(latest.body)
    _write_frontmatter(latest.path, updated, latest.body)


@contextmanager
def _fallback_marker_file_lock(path: Path):
    lock_path = path.with_name(f".{path.name}.lock")
    resolved_lock_path = lock_path.resolve()
    with _FALLBACK_LOCKS_LOCK:
        thread_lock = _FALLBACK_LOCKS.setdefault(resolved_lock_path, threading.Lock())
    with thread_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a", encoding="utf-8") as lock_file:
            locked = False
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                locked = True
            except ImportError:
                pass
            except OSError:
                raise
            try:
                yield
            finally:
                if locked:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _frontmatter_with_resolved_project(entry: FallbackEntry) -> dict[str, Any]:
    updated = dict(entry.frontmatter)
    if entry.project and not updated.get("project"):
        updated["project"] = entry.project
    return updated


def _normalize_tags(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, list | tuple | set):
        return [_json_safe_scalar(value)]
    return [_json_safe_scalar(item) for item in value]


def _infer_legacy_importance(body: str) -> int:
    patterns = (
        r"\bimportance\s*[:=]\s*(10|[1-9])\b",
        r"\bimp\s*[:=]\s*(10|[1-9])\b",
        r"\[imp\s*:\s*(10|[1-9])\]",
        r"\(imp\s*(10|[1-9])\)",
    )
    for pattern in patterns:
        match = re.search(pattern, body, flags=re.I)
        if match:
            return int(match.group(1))
    return 7


def _infer_legacy_timestamp(path: Path) -> str:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", str(path))
    if match:
        return f"{match.group(1)}T00:00:00Z"
    return ""


def _json_safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    return str(value)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not re.match(r"\A---\r?\n", text):
        return {}, text
    match = re.match(r"\A---\r?\n(.*?)(?:\r?\n---(?:\r?\n|\Z))(.*)\Z", text, flags=re.S)
    if not match:
        return {}, text
    frontmatter_text = match.group(1)
    frontmatter = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(frontmatter, dict):
        frontmatter = {}
    raw_timestamp = _raw_frontmatter_scalar(frontmatter_text, "timestamp")
    if raw_timestamp:
        frontmatter["timestamp"] = raw_timestamp
    return frontmatter, match.group(2)


def _raw_frontmatter_scalar(frontmatter_text: str, key: str) -> str | None:
    prefix = f"{key}:"
    for line in frontmatter_text.splitlines():
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip().strip("\"'")
    return None


def _git_root(path: Path) -> Path:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            env=_git_env(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if output:
            return Path(output).resolve()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return _heuristic_repo_root(path)


def _git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def _heuristic_repo_root(path: Path) -> Path:
    current = path.resolve()
    docs_local_root: Path | None = None
    for parent in [current.parent, *current.parents]:
        if parent.name == "docs.local":
            docs_local_root = parent.parent
            continue
        if parent.name in {"decisions", "brain-store-fallback"}:
            continue
        if (parent / ".git").exists():
            return parent
    if docs_local_root is not None:
        return docs_local_root
    parts = current.parts
    if "Gits" in parts:
        index = parts.index("Gits")
        if len(parts) > index + 1:
            return Path(*parts[: index + 2])
    return current.parent


def _resolve_project(
    path: Path,
    *,
    frontmatter: dict[str, Any],
    origin_repo_path: Path,
    scope_map: dict[str, str],
) -> str:
    explicit = frontmatter.get("project") or frontmatter.get("scope")
    if explicit:
        return str(explicit)

    candidates = [str(origin_repo_path.resolve()), str(path.resolve())]
    matches: list[tuple[int, str]] = []
    for prefix, project in scope_map.items():
        expanded = str(Path(prefix).expanduser().resolve())
        for candidate in candidates:
            if candidate == expanded or candidate.startswith(expanded + os.sep):
                matches.append((len(expanded), project))
    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]
    return origin_repo_path.name


def _extract_chunk_id(result: Any) -> str | None:
    if isinstance(result, dict):
        value = result.get("id") or result.get("chunk_id")
        return str(value) if value else None
    value = getattr(result, "id", None) or getattr(result, "chunk_id", None)
    return str(value) if value else None


def _write_frontmatter(path: Path, frontmatter: dict[str, Any], body: str) -> None:
    rendered = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(f"---\n{rendered}\n---\n{body}", encoding="utf-8")
    tmp_path.replace(path)
