"""Replay docs.local BrainLayer fallback files with origin attribution."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml


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


def load_scope_map(scopes_path: Path | None = None) -> dict[str, str]:
    path = scopes_path or Path.home() / ".config" / "brainlayer" / "scopes.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scopes = data.get("scopes") or {}
    return {str(Path(prefix).expanduser().resolve()): str(project) for prefix, project in scopes.items()}


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
    return (
        bool(entry.frontmatter.get("intended_brain_store")) and not str(entry.frontmatter.get("chunk_id") or "").strip()
    )


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
        _write_replay_attempt(entry, chunk_id=chunk_id)
    except Exception as exc:
        return ReplayResult(
            path=entry.path,
            attempted=True,
            chunk_id=chunk_id,
            error=f"stored chunk_id={chunk_id} but failed to update fallback frontmatter: {exc}",
        )

    return ReplayResult(path=entry.path, attempted=True, chunk_id=chunk_id)


def _write_replay_attempt(entry: FallbackEntry, *, chunk_id: Any) -> None:
    updated = dict(entry.frontmatter)
    updated["retry_attempted"] = True
    updated["chunk_id"] = chunk_id or None
    _write_frontmatter(entry.path, updated, entry.body)


def _normalize_tags(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, list | tuple | set):
        return [value]
    return list(value)


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
