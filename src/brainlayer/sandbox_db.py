"""Sandboxed BrainLayer DB primitive for live eval isolation."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

from . import paths as brain_paths
from .isolation_proof import _embed
from .paths import _CANONICAL_DB_PATH
from .search_repo import clear_hybrid_search_cache
from .vector_store import VectorStore

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SEED_DIR = Path(__file__).resolve().parent / "data" / "sandbox_seeds"


class SandboxProdLeakError(RuntimeError):
    """Raised when a sandbox DB path points at the canonical production area."""


@dataclass(frozen=True)
class SandboxPaths:
    temp_dir: Path
    db_path: Path
    wal_path: Path
    shm_path: Path
    lease_path: Path


def _validate_name(value: str, *, label: str) -> str:
    if not value or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, dots, underscores, and dashes")
    return value


def sandbox_paths_for_token(token: str) -> SandboxPaths:
    """Return deterministic sandbox paths for a token."""
    safe_token = _validate_name(token, label="token")
    temp_dir = Path(tempfile.gettempdir()) / f"bl-sandbox-{safe_token}"
    db_path = temp_dir / "brainlayer.db"
    return SandboxPaths(
        temp_dir=temp_dir,
        db_path=db_path,
        wal_path=Path(f"{db_path}-wal"),
        shm_path=Path(f"{db_path}-shm"),
        lease_path=temp_dir / ".brainlayer-sandbox.lease",
    )


def _paths_for_db_path(db_path: Path) -> SandboxPaths:
    return SandboxPaths(
        temp_dir=db_path.parent,
        db_path=db_path,
        wal_path=Path(f"{db_path}-wal"),
        shm_path=Path(f"{db_path}-shm"),
        lease_path=db_path.parent / f".{db_path.name}.sandbox.lease",
    )


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _protected_db_residue_paths(db_path: Path) -> set[Path]:
    resolved_db = _resolved(db_path)
    return {
        resolved_db,
        _resolved(Path(f"{resolved_db}-wal")),
        _resolved(Path(f"{resolved_db}-shm")),
    }


def _assert_not_prod_path(
    db_path: Path,
    *,
    protected_db_path: Path | None = None,
    ignore_current_active_db: bool = False,
) -> None:
    resolved_db = _resolved(db_path)
    canonical_db = _resolved(_CANONICAL_DB_PATH)
    canonical_dir = _resolved(_CANONICAL_DB_PATH.parent)
    if resolved_db == canonical_db or resolved_db.is_relative_to(canonical_dir):
        raise SandboxProdLeakError(f"sandbox db path resolves inside production BrainLayer dir: {resolved_db}")
    active_db = _resolved(brain_paths.get_db_path())
    protected_paths = []
    if not ignore_current_active_db:
        protected_paths.append(active_db)
    if protected_db_path is not None:
        protected_paths.append(_resolved(protected_db_path))
    for protected_db in protected_paths:
        if protected_db is None:
            continue
        if resolved_db in _protected_db_residue_paths(protected_db):
            raise SandboxProdLeakError(f"sandbox db path resolves to active BrainLayer DB: {resolved_db}")


def _seed_path(seed: str) -> Path:
    safe_seed = _validate_name(seed, label="seed")
    path = _SEED_DIR / f"{safe_seed}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown sandbox seed set: {seed}")
    return path


def _load_seed(seed: str) -> list[dict[str, Any]]:
    payload = json.loads(_seed_path(seed).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"sandbox seed must be a list: {seed}")
    return [dict(item) for item in payload]


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lease_owner_pid(path: Path) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    try:
        return int(payload["pid"])
    except (KeyError, TypeError, ValueError):
        return None


def _seed_chunk_payload(*, seed: str, row: dict[str, Any]) -> dict[str, Any]:
    chunk_id = str(row["id"])
    content = str(row["content"])
    project = row.get("project")
    created_at = str(row["created_at"])
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {"sandbox_seed": seed},
        "source_file": f"sandbox-seed:{seed}",
        "project": project if project is None else str(project),
        "content_type": str(row.get("content_type", "assistant_text")),
        "char_count": len(content),
        "source": "sandbox_seed",
        "created_at": created_at,
        "tags": row.get("tags", []),
        "importance": row.get("importance"),
        "status": "active",
    }


def seed_sandbox_db(db_path: str | Path, seed: str) -> list[str]:
    """Create a fresh sandbox DB and load a deterministic named seed set."""
    path = Path(db_path)
    _assert_not_prod_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for residue in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            residue.unlink()
        except FileNotFoundError:
            pass

    rows = _load_seed(seed)
    store = VectorStore(path)
    try:
        chunks = [_seed_chunk_payload(seed=seed, row=row) for row in rows]
        embeddings = [_embed(chunk["content"]) for chunk in chunks]
        processed = store.upsert_chunks(chunks, embeddings)
        if processed != len(chunks):
            raise RuntimeError(f"seed {seed!r} inserted {processed}/{len(chunks)} chunks")
    finally:
        store.close()
    clear_hybrid_search_cache(path)
    return [str(chunk["id"]) for chunk in chunks]


class SandboxDB:
    """Context manager for a token-scoped throwaway BrainLayer database."""

    def __init__(self, *, seed: str, token: str, db_path: str | Path | None = None) -> None:
        self.seed = _validate_name(seed, label="seed")
        self.token = _validate_name(token, label="token")
        self._explicit_db_path = Path(db_path).expanduser() if db_path is not None else None
        self._owns_temp_dir = self._explicit_db_path is None
        self.paths = (
            sandbox_paths_for_token(self.token)
            if self._explicit_db_path is None
            else _paths_for_db_path(self._explicit_db_path)
        )
        self.db_path = self.paths.db_path
        self.env = {"BRAINLAYER_DB": str(self.db_path)}
        self.seeded_ids: list[str] = []
        self._protected_db_path = _resolved(brain_paths.get_db_path())
        self._patched_default_paths: list[tuple[object, str, Any]] = []
        self._old_env: str | None = None
        self._started = False
        self._lease_acquired = False

    def __enter__(self) -> SandboxDB:
        self.start()
        self._patch_default_db_paths()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _acquire_lease(self) -> None:
        try:
            fd = os.open(self.paths.lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError(f"sandbox token already exists and is already active: {self.token}") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as lease_file:
            lease_file.write(json.dumps({"pid": os.getpid(), "token": self.token}, sort_keys=True) + "\n")
        self._lease_acquired = True

    def _release_lease(self) -> None:
        try:
            self.paths.lease_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._lease_acquired = False

    def _patch_default_db_paths(self) -> None:
        self._patched_default_paths = []
        old_path = self._protected_db_path
        new_path = Path(self.db_path)
        for module in list(sys.modules.values()):
            if module is None or not getattr(module, "__name__", "").startswith("brainlayer"):
                continue
            if getattr(module, "DEFAULT_DB_PATH", None) != old_path:
                continue
            self._patched_default_paths.append((module, "DEFAULT_DB_PATH", old_path))
            setattr(module, "DEFAULT_DB_PATH", new_path)

    def _restore_default_db_paths(self) -> None:
        for module, attr, value in reversed(self._patched_default_paths):
            setattr(module, attr, value)
        self._patched_default_paths = []

    def _validate_owned_stop_lease(self) -> bool:
        if not self._owns_temp_dir:
            return self._started
        if self._started:
            return True
        owner_pid = _read_lease_owner_pid(self.paths.lease_path)
        if owner_pid is not None and owner_pid != os.getpid() and _pid_is_running(owner_pid):
            raise RuntimeError(f"sandbox token is active in another process: {self.token}")
        return True

    def start(self) -> SandboxDB:
        if self._started:
            raise RuntimeError("sandbox already started")
        if self._owns_temp_dir and self.paths.temp_dir.exists():
            if self.paths.lease_path.exists():
                raise RuntimeError(f"sandbox token already exists and is already active: {self.token}")
            if any(path.exists() for path in (self.paths.db_path, self.paths.wal_path, self.paths.shm_path)):
                raise RuntimeError(f"sandbox token already exists with live DB: {self.token}")
            if any(self.paths.temp_dir.iterdir()):
                raise RuntimeError(f"sandbox token already exists with residue: {self.token}")
        _assert_not_prod_path(self.db_path, protected_db_path=self._protected_db_path)
        if not self._owns_temp_dir and any(
            path.exists() for path in (self.paths.db_path, self.paths.wal_path, self.paths.shm_path)
        ):
            raise RuntimeError(f"sandbox db path already exists with live DB: {self.db_path}")
        self.paths.temp_dir.mkdir(parents=True, exist_ok=True)
        self._acquire_lease()
        try:
            self.seeded_ids = seed_sandbox_db(self.db_path, self.seed)
            self._old_env = os.environ.get("BRAINLAYER_DB")
            os.environ["BRAINLAYER_DB"] = str(self.db_path)
            self._started = True
        except Exception:
            self._release_lease()
            if self._owns_temp_dir:
                shutil.rmtree(self.paths.temp_dir, ignore_errors=True)
            raise
        return self

    def stop(self) -> None:
        lease_allows_token_stop = self._validate_owned_stop_lease()
        ignore_current_active_db = lease_allows_token_stop and _resolved(brain_paths.get_db_path()) == _resolved(
            self.db_path
        )
        _assert_not_prod_path(
            self.db_path,
            protected_db_path=None if ignore_current_active_db else self._protected_db_path,
            ignore_current_active_db=ignore_current_active_db,
        )
        clear_hybrid_search_cache(self.db_path)
        errors: list[str] = []
        for residue in (self.paths.db_path, self.paths.wal_path, self.paths.shm_path):
            try:
                residue.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"{residue}: {exc}")
        if not self._owns_temp_dir:
            try:
                self.paths.lease_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"{self.paths.lease_path}: {exc}")
        if self._owns_temp_dir:
            try:
                shutil.rmtree(self.paths.temp_dir)
            except FileNotFoundError:
                pass
            except OSError as exc:
                errors.append(f"{self.paths.temp_dir}: {exc}")

        residue_paths = [self.paths.db_path, self.paths.wal_path, self.paths.shm_path]
        if self._owns_temp_dir:
            residue_paths.append(self.paths.temp_dir)
        else:
            residue_paths.append(self.paths.lease_path)
        remaining = [str(path) for path in residue_paths if path.exists()]
        if remaining:
            errors.append(f"residue remains after sandbox teardown: {', '.join(remaining)}")

        if self._started:
            if self._old_env is None:
                os.environ.pop("BRAINLAYER_DB", None)
            else:
                os.environ["BRAINLAYER_DB"] = self._old_env
            self._started = False
            self._restore_default_db_paths()
        elif os.environ.get("BRAINLAYER_DB") == str(self.db_path):
            os.environ.pop("BRAINLAYER_DB", None)

        self._lease_acquired = False
        clear_hybrid_search_cache(self.db_path)
        if errors:
            raise RuntimeError("; ".join(errors))


def start_sandbox(*, seed: str, token: str) -> SandboxDB:
    return SandboxDB(seed=seed, token=token).start()


def stop_sandbox(*, token: str) -> None:
    sandbox = SandboxDB(seed="skill-eval-baseline", token=token)
    sandbox.stop()
