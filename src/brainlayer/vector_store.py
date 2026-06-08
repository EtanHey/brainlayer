"""SQLite-vec based vector store for fast search.

Thin facade: VectorStore inherits from focused mixin modules.
See search_repo.py, kg_repo.py, session_repo.py for the extracted methods.
"""

import atexit
import fcntl
import glob
import hashlib
import json
import os
import stat
import struct
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import apsw
import apsw.bestpractice
import sqlite_vec

from ._helpers import (  # noqa: I001
    _DEFAULT_MIN_CHARS as _DEFAULT_MIN_CHARS,
)
from ._helpers import (
    _SOURCE_MIN_CHARS as _SOURCE_MIN_CHARS,
)
from ._helpers import (
    _escape_fts5_query as _escape_fts5_query,
)
from ._helpers import (
    _safe_json_loads as _safe_json_loads,
)
from ._helpers import (
    serialize_f32 as serialize_f32,
)
from ._helpers import (
    source_aware_min_chars as source_aware_min_chars,
)
from .chunk_origin import (
    CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
    CHUNK_ORIGIN_UNKNOWN,
    detect_chunk_origin,
)
from .dedupe import (
    compute_dedupe_fields,
    ensure_dedupe_schema,
    find_duplicate,
    merge_duplicate_chunk,
    merge_existing_chunk_seen,
    resolve_chunk_id,
)
from .ingest_guard import reject_recursive_mcp_output
from .kg_repo import KGMixin
from .search_repo import SearchMixin
from .session_repo import SessionMixin

_DEFAULT_BUSY_TIMEOUT_MS = 30_000
_DEFAULT_READ_BUSY_TIMEOUT_MS = 5_000
_MAX_APSW_BUSY_TIMEOUT_MS = 2_147_483_647


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if value <= 0 or value > _MAX_APSW_BUSY_TIMEOUT_MS:
        return default
    return value


def _read_busy_timeout_ms() -> int:
    return _positive_int_env("BRAINLAYER_READ_BUSY_TIMEOUT_MS", _DEFAULT_READ_BUSY_TIMEOUT_MS)


def _set_busy_timeout_hook(conn: apsw.Connection) -> None:
    """Set busy_timeout on every new connection before any other hooks.

    APSW bestpractice hooks (connection_optimize) run PRAGMA optimize inside
    the Connection() constructor. Without busy_timeout set first, this PRAGMA
    fails with BusyError when other processes hold the DB lock.
    """
    timeout_ms = _read_busy_timeout_ms() if conn.readonly("main") else _DEFAULT_BUSY_TIMEOUT_MS
    conn.setbusytimeout(timeout_ms)


# Register busy_timeout hook BEFORE bestpractice hooks so it fires first.
# bestpractice.apply() adds hooks that run PRAGMA optimize inside Connection(),
# which needs busy_timeout active or it crashes under contention.
apsw.connection_hooks.insert(0, _set_busy_timeout_hook)
apsw.bestpractice.apply(apsw.bestpractice.recommended)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _read_mmap_bytes() -> int:
    return max(_int_env("BRAINLAYER_READ_MMAP_BYTES", 30_000_000_000), 0)


def _configure_reader_pragmas(conn: apsw.Connection) -> None:
    conn.setbusytimeout(_read_busy_timeout_ms())
    cursor = conn.cursor()
    cursor.execute("PRAGMA wal_autocheckpoint = 0")
    cursor.execute(f"PRAGMA mmap_size = {_read_mmap_bytes()}")
    cursor.execute(f"PRAGMA cache_size = {_read_cache_size_kb()}")
    cursor.execute("PRAGMA query_only = ON")


def _is_retryable_init_error(exc: BaseException) -> bool:
    if isinstance(exc, apsw.BusyError):
        return True
    if isinstance(exc, apsw.SchemaChangeError):
        return "vtable constructor failed" in str(exc).lower()
    return False


def _read_cache_size_kb() -> int:
    return -abs(_int_env("BRAINLAYER_READ_CACHE_KB", 64_000))


def _wal_autocheckpoint_pages() -> int:
    return max(_int_env("BRAINLAYER_WAL_AUTOCHECKPOINT", 10_000), 0)


def _wal_size_limit_bytes() -> int:
    """Cap the on-disk WAL file size (bytes). 0 disables (PRAGMA -1 = unlimited).

    Without this, journal_size_limit defaults to -1 and the WAL file is never
    truncated back after a checkpoint — it stays at its high-water mark. Under
    enrichment load with reader-pinned PASSIVE checkpoints the WAL grew to
    multi-GB (observed 2.4-3.9GB), which starved brain_store writes ("DB busy").
    256MB is generous headroom for the largest legitimate batch while bounding
    the file so checkpoints stay cheap and write windows reopen promptly.
    """
    return max(_int_env("BRAINLAYER_WAL_SIZE_LIMIT_BYTES", 256_000_000), 0)


class WriterInUseError(RuntimeError):
    """Raised when a live process already owns the writer pidfile."""


class VectorStore(SearchMixin, KGMixin, SessionMixin):
    """SQLite-vec based vector store.

    Core chunk CRUD and schema management live here.
    Search, KG, and session methods are inherited from mixin classes.
    """

    # Retry settings for DB init under contention (multiple MCP instances + enrichment)
    _INIT_MAX_RETRIES = max(_int_env("BRAINLAYER_INIT_MAX_RETRIES", 10), 1)
    _INIT_BASE_DELAY = 0.5  # seconds
    _INIT_MAX_DELAY = 30  # seconds
    _PIDFILE_REFS: dict[Path, int] = {}
    _PIDFILE_REF_PIDS: dict[Path, int] = {}
    _PIDFILE_REFS_LOCK = threading.Lock()
    _PIDFILE_ACQUIRE_LOCK = threading.Lock()
    _INIT_DB_LOCKS: dict[Path, threading.Lock] = {}
    _INIT_DB_LOCKS_LOCK = threading.Lock()

    def __init__(self, db_path: Path, readonly: bool = False):
        self.db_path = db_path
        self._writer_pidfile_acquired = False
        if not readonly:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fts5_health_cache: dict[str, Any] = {}
        self._retrieval_strengthening_pending: dict[str, dict[str, float]] = {}
        self._retrieval_strengthening_query_count = 0
        self._retrieval_strengthening_flush_threshold = 100
        self._retrieval_strengthening_lock = threading.Lock()
        self._checkpoint_count_cache: int | None = None
        self._checkpoint_count_cache_data_version: int | None = None
        self._audit_recursion_count_cache: int | None = None
        self._audit_recursion_count_cache_data_version: int | None = None
        self._readonly = readonly or (self.db_path.exists() and not os.access(self.db_path, os.W_OK))
        if self._readonly:
            self._init_readonly_db()
        else:
            self._acquire_writer_pidfile()
            try:
                with self._init_db_thread_lock():
                    self._init_db_with_retry()
            except Exception:
                self._release_writer_pidfile()
                raise

    def _init_db_thread_lock(self) -> threading.Lock:
        """Serialize same-process schema init for a DB path."""
        resolved_path = self.db_path.resolve()
        with self._INIT_DB_LOCKS_LOCK:
            lock = self._INIT_DB_LOCKS.get(resolved_path)
            if lock is None:
                lock = threading.Lock()
                self._INIT_DB_LOCKS[resolved_path] = lock
            return lock

    def _writer_pidfile_path(self) -> Path:
        pidfile_dir = Path(os.environ.get("BRAINLAYER_WRITER_PIDFILE_DIR", "/tmp")).expanduser()
        if not pidfile_dir.is_absolute():
            pidfile_dir = Path("/tmp") / pidfile_dir
        pidfile_dir = pidfile_dir.resolve()
        resolved_path = self.db_path.resolve()
        path_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
        return pidfile_dir / f"brainlayer-writer-{path_hash}-{resolved_path.name}.pid"

    def _acquire_writer_pidfile(self) -> None:
        pidfile = self._writer_pidfile_path()
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        with self._PIDFILE_ACQUIRE_LOCK:
            self._reclaim_stale_writer_pidfiles(pidfile)
            pid = os.getpid()

            with self._PIDFILE_REFS_LOCK:
                if self._PIDFILE_REFS.get(pidfile, 0) > 0:
                    ref_pid = self._PIDFILE_REF_PIDS.get(pidfile)
                    if ref_pid is not None and ref_pid != pid:
                        self._PIDFILE_REFS.pop(pidfile, None)
                        self._PIDFILE_REF_PIDS.pop(pidfile, None)
                    else:
                        owner_pid, owner_start_time = self._read_writer_pidfile_owner(pidfile)
                        if owner_pid != pid or not self._pidfile_owner_matches(owner_pid, owner_start_time):
                            raise WriterInUseError(
                                f"pidfile ref mismatch for {self.db_path}; refusing to clear active refs"
                            )
                        self._PIDFILE_REFS[pidfile] += 1
                        self._PIDFILE_REF_PIDS[pidfile] = pid
                        self._writer_pidfile_acquired = True
                        self._writer_pidfile_path_value = pidfile
                        atexit.register(self._release_writer_pidfile)
                        return

            max_create_attempts = 4
            for attempt in range(max_create_attempts):
                try:
                    fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX)
                        os.write(fd, self._writer_pidfile_payload(pid))
                    finally:
                        os.close(fd)
                    with self._PIDFILE_REFS_LOCK:
                        self._PIDFILE_REFS[pidfile] = self._PIDFILE_REFS.get(pidfile, 0) + 1
                        self._PIDFILE_REF_PIDS[pidfile] = pid
                    self._writer_pidfile_acquired = True
                    self._writer_pidfile_path_value = pidfile
                    atexit.register(self._release_writer_pidfile)
                    return
                except FileExistsError:
                    if self._handle_existing_writer_pidfile(pidfile, pid):
                        return
                    if attempt == max_create_attempts - 1:
                        raise
                    time.sleep(0.01 * (attempt + 1))

    def _reclaim_stale_writer_pidfiles(self, pidfile: Path) -> None:
        # Legacy pidfiles do not record the database path. Sweep sibling hashes
        # for the same DB basename, then use db_path metadata when it exists.
        pattern = f"brainlayer-writer-*-{glob.escape(self.db_path.resolve().name)}.pid"
        for candidate in pidfile.parent.glob(pattern):
            self._reclaim_stale_writer_pidfile(candidate)

    def _reclaim_stale_writer_pidfile(self, pidfile: Path) -> None:
        with self._PIDFILE_REFS_LOCK:
            if self._PIDFILE_REFS.get(pidfile, 0) > 0:
                return
        fd = self._open_writer_pidfile_readonly(pidfile)
        if fd is None:
            return
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            owner_pid, owner_start_time, owner_db_path = self._read_writer_pidfile_record_fd(fd)
            if owner_db_path is not None and not self._pidfile_db_path_matches(owner_db_path):
                return
            if owner_pid is not None and self._pidfile_owner_matches(owner_pid, owner_start_time):
                return
            try:
                path_stat = pidfile.stat()
            except FileNotFoundError:
                return
            if os.path.samestat(os.fstat(fd), path_stat):
                try:
                    pidfile.unlink()
                except FileNotFoundError:
                    pass
        finally:
            os.close(fd)

    def _handle_existing_writer_pidfile(self, pidfile: Path, pid: int) -> bool:
        fd = self._open_writer_pidfile_readonly(pidfile)
        if fd is None:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            other_pid, other_start_time = self._read_writer_pidfile_owner_fd(fd)
            if other_pid == pid and self._pidfile_owner_matches(other_pid, other_start_time):
                with self._PIDFILE_REFS_LOCK:
                    self._PIDFILE_REFS[pidfile] = self._PIDFILE_REFS.get(pidfile, 0) + 1
                    self._PIDFILE_REF_PIDS[pidfile] = pid
                self._writer_pidfile_acquired = True
                self._writer_pidfile_path_value = pidfile
                atexit.register(self._release_writer_pidfile)
                return True
            if other_pid is not None and self._pidfile_owner_matches(other_pid, other_start_time):
                raise WriterInUseError(f"another writer is using {self.db_path} (pid {other_pid})")
            try:
                path_stat = pidfile.stat()
            except FileNotFoundError:
                return False
            if os.path.samestat(os.fstat(fd), path_stat):
                try:
                    pidfile.unlink()
                except FileNotFoundError:
                    pass
            return False
        finally:
            os.close(fd)

    @staticmethod
    def _open_writer_pidfile_readonly(pidfile: Path) -> int | None:
        try:
            candidate_stat = pidfile.lstat()
        except OSError:
            return None
        if not stat.S_ISREG(candidate_stat.st_mode):
            return None

        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(pidfile, flags)
        except OSError:
            return None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                return None
        except OSError:
            os.close(fd)
            return None
        return fd

    @staticmethod
    def _read_writer_pidfile(pidfile: Path) -> int | None:
        return VectorStore._read_writer_pidfile_owner(pidfile)[0]

    @staticmethod
    def _read_writer_pidfile_owner(pidfile: Path) -> tuple[int | None, str | None]:
        try:
            pid, start_time, _db_path = VectorStore._parse_writer_pidfile_lines(
                pidfile.read_text(encoding="utf-8").splitlines()
            )
            return pid, start_time
        except (OSError, ValueError):
            return None, None

    @staticmethod
    def _read_writer_pidfile_fd(fd: int) -> int | None:
        return VectorStore._read_writer_pidfile_owner_fd(fd)[0]

    @staticmethod
    def _read_writer_pidfile_owner_fd(fd: int) -> tuple[int | None, str | None]:
        pid, start_time, _db_path = VectorStore._read_writer_pidfile_record_fd(fd)
        return pid, start_time

    @staticmethod
    def _read_writer_pidfile_record_fd(fd: int) -> tuple[int | None, str | None, str | None]:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            return VectorStore._parse_writer_pidfile_lines(os.read(fd, 512).decode("utf-8").splitlines())
        except (OSError, ValueError):
            return None, None, None

    @staticmethod
    def _parse_writer_pidfile_lines(lines: list[str]) -> tuple[int | None, str | None, str | None]:
        if not lines:
            return None, None, None
        pid = int(lines[0].strip())
        start_time = None
        db_path = None
        for line in lines[1:]:
            key, separator, value = line.strip().partition("=")
            if not separator:
                continue
            if key == "start_time":
                start_time = value.strip() or None
            elif key == "db_path":
                db_path = value.strip() or None
        return pid, start_time, db_path

    def _writer_pidfile_payload(self, pid: int) -> bytes:
        start_time = VectorStore._pid_start_time(pid)
        lines = [str(pid)]
        if start_time:
            lines.append(f"start_time={start_time}")
        lines.append(f"db_path={self.db_path.resolve()}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    def _pidfile_owner_matches(self, pid: int, start_time: str | None) -> bool:
        if not self._pid_is_alive(pid):
            return False
        if not start_time:
            return True
        current_start_time = self._pid_start_time(pid)
        return current_start_time is None or current_start_time == start_time

    def _pidfile_db_path_matches(self, db_path: str) -> bool:
        try:
            return Path(db_path).expanduser().resolve() == self.db_path.resolve()
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _pid_start_time(pid: int) -> str | None:
        try:
            stat_path = Path("/proc") / str(pid) / "stat"
            if stat_path.exists():
                fields = stat_path.read_text(encoding="utf-8").split()
                if len(fields) > 21:
                    return fields[21]
        except OSError:
            pass
        try:
            result = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(pid)],
                capture_output=True,
                check=False,
                text=True,
                timeout=1,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return " ".join(result.stdout.split()) or None

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _release_writer_pidfile(self) -> None:
        if not self._writer_pidfile_acquired:
            return

        pidfile = getattr(self, "_writer_pidfile_path_value", None)
        if pidfile is None:
            self._writer_pidfile_acquired = False
            return

        with self._PIDFILE_REFS_LOCK:
            refs = self._PIDFILE_REFS.get(pidfile, 0)
            if refs <= 1:
                if self._read_writer_pidfile(pidfile) == os.getpid():
                    try:
                        pidfile.unlink()
                    except FileNotFoundError:
                        pass
                self._PIDFILE_REFS.pop(pidfile, None)
                self._PIDFILE_REF_PIDS.pop(pidfile, None)
            else:
                self._PIDFILE_REFS[pidfile] = refs - 1

        self._writer_pidfile_acquired = False

    def _init_readonly_db(self) -> None:
        """Open an existing DB in readonly mode without running migrations."""
        self.conn = apsw.Connection(str(self.db_path), flags=apsw.SQLITE_OPEN_READONLY)
        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)
        _configure_reader_pragmas(self.conn)

        cursor = self.conn.cursor()
        existing_tables = {
            row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        }
        chunk_columns = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
        self._has_chunk_origin = "chunk_origin" in chunk_columns
        self._has_content_class = "content_class" in chunk_columns
        self._binary_index_available = "chunk_vectors_binary" in existing_tables
        self._trigram_fts_available = "chunks_fts_trigram" in existing_tables
        self._chunk_tags_available = "chunk_tags" in existing_tables
        self._local = threading.local()

    def _invalidate_checkpoint_count_cache(self) -> None:
        self._checkpoint_count_cache = None
        self._checkpoint_count_cache_data_version = None

    def _invalidate_audit_recursion_count_cache(self) -> None:
        self._audit_recursion_count_cache = None
        self._audit_recursion_count_cache_data_version = None

    def _invalidate_filtered_count_caches(self) -> None:
        self._invalidate_checkpoint_count_cache()
        self._invalidate_audit_recursion_count_cache()

    def _checkpoint_wal_full(self, cursor) -> None:
        try:
            cursor.execute("PRAGMA wal_checkpoint(FULL)")
        except apsw.Error:
            pass

    def _init_db_with_retry(self) -> None:
        """Initialize DB with retry on BusyError.

        Multiple BrainLayer processes (MCP instances, CLI, enrichment) may
        contend for write locks during DDL. Retry with exponential backoff
        instead of crashing on the first BusyError.
        """
        last_err = None
        start = time.monotonic()
        retry_budget = max(int(self._INIT_MAX_RETRIES), 1)
        for attempt in range(retry_budget):
            try:
                self._init_db()
                return
            except apsw.Error as e:
                if not _is_retryable_init_error(e):
                    raise
                last_err = e
                delay = min(self._INIT_BASE_DELAY * (2**attempt), self._INIT_MAX_DELAY)
                import sys

                elapsed = time.monotonic() - start
                print(
                    f"  DB init retryable SQLite error (attempt {attempt + 1}/{retry_budget}), "
                    f"elapsed {elapsed:.1f}s, retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise last_err  # type: ignore[misc]

    def _init_db(self) -> None:
        """Initialize database with vector extension."""
        self.conn = apsw.Connection(str(self.db_path))

        # Set busy timeout IMMEDIATELY via APSW native method — before any DDL.
        self.conn.setbusytimeout(10_000)  # 10 seconds

        self.conn.enableloadextension(True)
        self.conn.loadextension(sqlite_vec.loadable_path())
        self.conn.enableloadextension(False)

        cursor = self.conn.cursor()

        # WAL mode is persistent on the DB file — set it every time
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute(f"PRAGMA wal_autocheckpoint = {_wal_autocheckpoint_pages()}")
        # Bound the WAL file so it truncates back after each checkpoint instead of
        # staying at its high-water mark (default -1 = unlimited). See
        # _wal_size_limit_bytes() for the starvation root cause this prevents.
        _wal_limit = _wal_size_limit_bytes()
        cursor.execute(f"PRAGMA journal_size_limit = {_wal_limit if _wal_limit > 0 else -1}")

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                source_file TEXT NOT NULL,
                project TEXT,
                content_type TEXT,
                value_type TEXT,
                char_count INTEGER,
                source TEXT,
                sender TEXT,
                language TEXT,
                conversation_id TEXT,
                position INTEGER,
                context_summary TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                provenance_class TEXT,
                chunk_origin TEXT DEFAULT 'unknown',
                content_class TEXT DEFAULT 'knowledge'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL,
                details TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_profiles (
                agent_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at REAL NOT NULL,
                notes TEXT
            )
        """)
        atomic_brick_migration_applied = (
            cursor.execute("SELECT 1 FROM schema_migrations WHERE name = 'atomic_brick_chunks_v1'").fetchone()
            is not None
        )

        # Add columns if upgrading existing DB
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}
        atomic_brick_cols = {"brick_id", "source_uri", "status", "ingested_at", "topic_cluster"}
        needs_atomic_brick_backfill = (
            not atomic_brick_cols.issubset(existing_cols) or not atomic_brick_migration_applied
        )
        for col, typ in [
            ("source", "TEXT"),
            ("sender", "TEXT"),
            ("language", "TEXT"),
            ("conversation_id", "TEXT"),
            ("position", "INTEGER"),
            ("context_summary", "TEXT"),
            ("tags", "TEXT"),
            ("tag_confidence", "REAL"),
            ("summary", "TEXT"),
            ("summary_v2", "TEXT"),
            ("importance", "REAL"),
            ("intent", "TEXT"),
            ("enriched_at", "TEXT"),
            ("enrich_status", "TEXT"),
            ("enrichment_version", "TEXT DEFAULT '1.0'"),
            ("primary_symbols", "TEXT"),
            ("resolved_query", "TEXT"),
            ("key_facts", "TEXT"),
            ("resolved_queries", "TEXT"),
            ("raw_entities_json", "TEXT"),
            ("provenance_class", "TEXT"),
            ("epistemic_level", "TEXT"),
            ("version_scope", "TEXT"),
            ("debt_impact", "TEXT"),
            ("external_deps", "TEXT"),
            ("created_at", "TEXT"),
            ("sentiment_label", "TEXT"),
            ("sentiment_score", "REAL"),
            ("sentiment_signals", "TEXT"),
            ("half_life_days", "REAL DEFAULT 30.0"),
            ("last_retrieved", "REAL DEFAULT NULL"),
            ("retrieval_count", "INTEGER DEFAULT 0"),
            ("decay_score", "REAL DEFAULT 1.0"),
            ("pinned", "INTEGER DEFAULT 0"),
            ("archived", "INTEGER DEFAULT 0"),
            # Lifecycle columns (chunk lifecycle management)
            ("superseded_by", "TEXT"),
            ("aggregated_into", "TEXT"),
            ("archived_at", "TEXT"),
            ("chunk_origin", "TEXT DEFAULT 'unknown'"),
            ("content_class", "TEXT DEFAULT 'knowledge'"),
            ("seen_count", "INTEGER DEFAULT 1"),
            ("last_seen_at", "TEXT"),
            ("content_hash", "TEXT"),
            ("dedupe_hash", "TEXT"),
            ("simhash", "TEXT"),
            ("simhash_band_0", "TEXT"),
            ("simhash_band_1", "TEXT"),
            ("simhash_band_2", "TEXT"),
            ("simhash_band_3", "TEXT"),
            # Atomic-brick compatibility columns. Embeddings remain in sqlite-vec
            # virtual tables; these fields provide ledger/status metadata without
            # duplicating vector blobs inside chunks.
            ("brick_id", "TEXT"),
            ("source_uri", "TEXT"),
            ("status", "TEXT DEFAULT 'active'"),
            ("ingested_at", "INTEGER"),
            ("topic_cluster", "TEXT"),
        ]:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE chunks ADD COLUMN {col} {typ}")
        self._has_chunk_origin = True
        self._has_content_class = True
        ensure_dedupe_schema(self.conn)

        migration_name = "2026_05_16_fm6_chunk_origin"
        migration_done = cursor.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (migration_name,),
        ).fetchone()
        if migration_done is None:
            self._checkpoint_wal_full(cursor)
            cursor.execute(
                """
                UPDATE chunks
                SET chunk_origin = ?
                WHERE COALESCE(chunk_origin, ?) != ?
                  AND (
                    LOWER(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)))
                        LIKE '[precompact checkpoint]%'
                    OR LOWER(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)))
                        LIKE '# precompact checkpoint%'
                    OR LOWER(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)))
                        LIKE 'session-restore%'
                    OR LOWER(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)))
                        LIKE '# session-restore%'
                    OR LOWER(SUBSTR(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)), 1, 1024))
                        LIKE '%content="[precompact checkpoint]%'
                    OR LOWER(SUBSTR(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)), 1, 1024))
                        LIKE '%content=''[precompact checkpoint]%'
                    OR LOWER(SUBSTR(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)), 1, 1024))
                        LIKE '%"content": "[precompact checkpoint]%'
                    OR LOWER(SUBSTR(LTRIM(content, char(9) || char(10) || char(11) || char(12) || char(13) || char(32)), 1, 1024))
                        LIKE '%''content'': ''[precompact checkpoint]%'
                  )
                """,
                (
                    CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
                    CHUNK_ORIGIN_UNKNOWN,
                    CHUNK_ORIGIN_PRECOMPACT_CHECKPOINT,
                ),
            )
            backfilled = self.conn.changes()
            cursor.execute(
                "INSERT OR IGNORE INTO schema_migrations (name, applied_at, details) VALUES (?, ?, ?)",
                (
                    migration_name,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps({"backfilled_precompact_checkpoints": backfilled}),
                ),
            )
            self._checkpoint_wal_full(cursor)
            self._invalidate_filtered_count_caches()

        enrich_status_migration_name = "2026_05_26_enrich_status_v1"
        enrich_status_migration_done = cursor.execute(
            "SELECT 1 FROM schema_migrations WHERE name = ?",
            (enrich_status_migration_name,),
        ).fetchone()
        if enrich_status_migration_done is None:
            cursor.execute(
                """
                UPDATE chunks
                SET enriched_at = NULL
                WHERE enriched_at IS NOT NULL
                  AND TRIM(enriched_at) = ''
                """
            )
            cursor.execute(
                """
                UPDATE chunks
                SET enrich_status = 'success'
                WHERE enriched_at IS NOT NULL
                  AND TRIM(enriched_at) != ''
                  AND enriched_at NOT LIKE 'skipped:%'
                  AND enrich_status IS NULL
                """
            )
            success_backfilled = self.conn.changes()
            cursor.execute(
                """
                UPDATE chunks
                SET enrich_status = NULLIF(TRIM(SUBSTR(enriched_at, LENGTH('skipped:') + 1)), ''),
                    enriched_at = NULL
                WHERE enriched_at LIKE 'skipped:%'
                """
            )
            skipped_backfilled = self.conn.changes()
            cursor.execute(
                "INSERT OR IGNORE INTO schema_migrations (name, applied_at, details) VALUES (?, ?, ?)",
                (
                    enrich_status_migration_name,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(
                        {
                            "success_backfilled": success_backfilled,
                            "skipped_backfilled": skipped_backfilled,
                        }
                    ),
                ),
            )

        cursor.execute("""
            UPDATE chunks
            SET archived = 1
            WHERE value_type = 'ARCHIVED' AND COALESCE(archived, 0) = 0
        """)
        if needs_atomic_brick_backfill:
            self._backfill_atomic_brick_columns(cursor)
            cursor.execute("""
                INSERT OR REPLACE INTO schema_migrations(name, applied_at)
                VALUES ('atomic_brick_chunks_v1', datetime('now'))
            """)

        # Indexes for filtering
        for idx, col in [
            ("idx_chunks_source", "source"),
            ("idx_chunks_sender", "sender"),
            ("idx_chunks_conversation", "conversation_id"),
            ("idx_chunks_intent", "intent"),
            ("idx_chunks_importance", "importance"),
            ("idx_chunks_enriched", "enriched_at"),
            ("idx_chunks_enrich_status", "enrich_status"),
            ("idx_chunks_created", "created_at"),
            ("idx_chunks_sentiment", "sentiment_label"),
            ("idx_chunks_project", "project"),
            ("idx_chunks_content_type", "content_type"),
            ("idx_chunks_language", "language"),
            ("idx_chunks_chunk_origin", "chunk_origin"),
            ("idx_chunks_content_class", "content_class"),
            ("idx_chunks_source_uri", "source_uri"),
            ("idx_chunks_status", "status"),
            ("idx_chunks_ingested_at", "ingested_at"),
            ("idx_chunks_topic_cluster", "topic_cluster"),
        ]:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx} ON chunks({col})")
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_brick_id ON chunks(brick_id) WHERE brick_id IS NOT NULL"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_decay_score ON chunks(decay_score) WHERE archived = 0")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_last_retrieved ON chunks(last_retrieved) WHERE archived = 0"
        )

        # Vector table (1024 dims for bge-large-en-v1.5)
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)
        existing_tables = {
            row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        }
        self._binary_index_available = "chunk_vectors_binary" in existing_tables
        if not self._binary_index_available:
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors_binary USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding BIT[1024]
                    )
                """)
                self._binary_index_available = True
            except apsw.ReadOnlyError:
                self._binary_index_available = False

        # FTS5 full-text search — indexes content + enrichment metadata
        # for better keyword matches on summaries, tags, and resolved queries.
        _FTS5_COLUMNS = "content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id UNINDEXED"
        _TRIGRAM_TOKENIZER = "trigram"

        # Detect old single-column FTS5 schema and rebuild if needed.
        # FTS5 virtual tables can't be ALTERed — must drop and recreate.
        _needs_fts_rebuild = False
        try:
            fts_cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks_fts)")}
            if fts_cols and ("summary" not in fts_cols or "key_facts" not in fts_cols):
                _needs_fts_rebuild = True
        except Exception:
            pass  # Table doesn't exist yet, will be created below

        if _needs_fts_rebuild:
            cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
            cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
            cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
            cursor.execute("DROP TABLE IF EXISTS chunks_fts")

        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                {_FTS5_COLUMNS}
            )
        """)
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_trigram USING fts5(
                {_FTS5_COLUMNS},
                tokenize='{_TRIGRAM_TOKENIZER}'
            )
        """)
        self._trigram_fts_available = True
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_fts_rowids (
                chunk_id TEXT PRIMARY KEY,
                fts_rowid INTEGER,
                trigram_rowid INTEGER
            )
        """)
        cursor.execute("""
            INSERT OR IGNORE INTO chunk_fts_rowids(chunk_id, fts_rowid)
            SELECT chunk_id, rowid FROM chunks_fts WHERE chunk_id IS NOT NULL
        """)
        cursor.execute("""
            INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
            SELECT chunk_id, rowid FROM chunks_fts_trigram WHERE chunk_id IS NOT NULL
            ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid
        """)

        # FTS5 sync triggers — keep summary/tags/resolved_query in sync
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_insert")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (
                    new.content,
                    new.summary,
                    new.tags,
                    new.resolved_query,
                    new.key_facts,
                    new.resolved_queries,
                    new.id
                );
                INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
                VALUES (new.id, last_insert_rowid())
                ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_insert")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (
                    new.content,
                    new.summary,
                    new.tags,
                    new.resolved_query,
                    new.key_facts,
                    new.resolved_queries,
                    new.id
                );
                INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
                VALUES (new.id, last_insert_rowid())
                ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_delete")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts
                WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
                DELETE FROM chunks_fts_trigram
                WHERE rowid = (SELECT trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
                DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_delete")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts
                WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
                DELETE FROM chunks_fts_trigram
                WHERE rowid = (SELECT trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
                DELETE FROM chunk_fts_rowids WHERE chunk_id = old.id;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_update")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_update
            AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
                DELETE FROM chunks_fts
                WHERE rowid = (SELECT fts_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (
                    new.content,
                    new.summary,
                    new.tags,
                    new.resolved_query,
                    new.key_facts,
                    new.resolved_queries,
                    new.id
                );
                INSERT INTO chunk_fts_rowids(chunk_id, fts_rowid)
                VALUES (new.id, last_insert_rowid())
                ON CONFLICT(chunk_id) DO UPDATE SET fts_rowid = excluded.fts_rowid;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunks_fts_trigram_update")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunks_fts_trigram_update
            AFTER UPDATE OF content, summary, tags, resolved_query, key_facts, resolved_queries ON chunks BEGIN
                DELETE FROM chunks_fts_trigram
                WHERE rowid = (SELECT trigram_rowid FROM chunk_fts_rowids WHERE chunk_id = old.id);
                INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                VALUES (
                    new.content,
                    new.summary,
                    new.tags,
                    new.resolved_query,
                    new.key_facts,
                    new.resolved_queries,
                    new.id
                );
                INSERT INTO chunk_fts_rowids(chunk_id, trigram_rowid)
                VALUES (new.id, last_insert_rowid())
                ON CONFLICT(chunk_id) DO UPDATE SET trigram_rowid = excluded.trigram_rowid;
            END
        """)

        self._schema_user_version = cursor.execute("PRAGMA user_version").fetchone()[0]
        if os.environ.get("BRAINLAYER_REPAIR") == "1":
            self.repair_fts(rebuild_trigram=True)

        # ── Tag junction table (replaces json_each scanning) ──────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_tags (
                chunk_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (chunk_id, tag)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_tags_tag ON chunk_tags(tag)")

        # Sync triggers: keep chunk_tags in sync with chunks.tags JSON
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_insert")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_insert AFTER INSERT ON chunks
            WHEN new.tags IS NOT NULL AND json_valid(new.tags) = 1 BEGIN
                INSERT OR IGNORE INTO chunk_tags(chunk_id, tag)
                SELECT new.id, value FROM json_each(new.tags);
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_update")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_update
            AFTER UPDATE OF tags ON chunks
            WHEN new.tags IS NOT NULL AND json_valid(new.tags) = 1 BEGIN
                DELETE FROM chunk_tags WHERE chunk_id = new.id;
                INSERT OR IGNORE INTO chunk_tags(chunk_id, tag)
                SELECT new.id, value FROM json_each(new.tags);
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_update_clear")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_update_clear
            AFTER UPDATE OF tags ON chunks
            WHEN new.tags IS NULL OR json_valid(new.tags) = 0 BEGIN
                DELETE FROM chunk_tags WHERE chunk_id = new.id;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS chunk_tags_delete")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS chunk_tags_delete AFTER DELETE ON chunks BEGIN
                DELETE FROM chunk_tags WHERE chunk_id = old.id;
            END
        """)

        # Backfill chunk_tags from existing data (detects partial fills from crashes)
        tagged_chunks = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND json_valid(tags) = 1"
        ).fetchone()[0]
        backfilled_chunks = cursor.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunk_tags").fetchone()[0]
        if tagged_chunks > 0 and backfilled_chunks < tagged_chunks:
            cursor.execute("""
                INSERT OR IGNORE INTO chunk_tags(chunk_id, tag)
                SELECT c.id, j.value FROM chunks c, json_each(c.tags) j
                WHERE c.tags IS NOT NULL AND json_valid(c.tags) = 1
            """)
        self._chunk_tags_available = True

        # Session context table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_context (
                session_id TEXT PRIMARY KEY,
                project TEXT,
                branch TEXT,
                pr_number INTEGER,
                commit_shas TEXT,
                files_changed TEXT,
                started_at TEXT,
                ended_at TEXT,
                created_at TEXT
            )
        """)
        existing_sc_cols = {row[1] for row in cursor.execute("PRAGMA table_info(session_context)")}
        for col in ("plan_name", "plan_phase", "story_id"):
            if col not in existing_sc_cols:
                cursor.execute(f"ALTER TABLE session_context ADD COLUMN {col} TEXT")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                timestamp TEXT,
                session_id TEXT,
                action TEXT,
                chunk_id TEXT,
                project TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_interactions_path ON file_interactions(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_interactions_session ON file_interactions(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_context_project ON session_context(project)")

        # Operations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                operation_type TEXT,
                chunk_ids TEXT,
                summary TEXT,
                outcome TEXT,
                started_at TEXT,
                ended_at TEXT,
                step_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_session ON operations(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_operations_type ON operations(operation_type)")

        # Topic chains table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                session_a TEXT NOT NULL,
                session_b TEXT NOT NULL,
                shared_actions INTEGER DEFAULT 0,
                time_delta_hours REAL,
                project TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_chains_file ON topic_chains(file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_chains_session ON topic_chains(session_a)")

        # Session enrichment table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_enrichments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                file_path TEXT,
                enrichment_version TEXT NOT NULL DEFAULT '1.0',
                enrichment_model TEXT,
                enrichment_timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                session_start_time TEXT,
                session_end_time TEXT,
                duration_seconds INTEGER,
                message_count INTEGER NOT NULL DEFAULT 0,
                user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0,
                tool_call_count INTEGER NOT NULL DEFAULT 0,
                session_summary TEXT,
                primary_intent TEXT,
                outcome TEXT CHECK(outcome IN ('success','partial_success','failure','abandoned','ongoing')),
                complexity_score INTEGER CHECK(complexity_score BETWEEN 1 AND 10),
                session_quality_score INTEGER CHECK(session_quality_score BETWEEN 1 AND 10),
                decisions_made TEXT DEFAULT '[]',
                corrections TEXT DEFAULT '[]',
                learnings TEXT DEFAULT '[]',
                mistakes TEXT DEFAULT '[]',
                patterns TEXT DEFAULT '[]',
                topic_tags TEXT DEFAULT '[]',
                tool_usage_stats TEXT DEFAULT '[]',
                what_worked TEXT,
                what_failed TEXT,
                summary_embedding BLOB
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_enrichments_session ON session_enrichments(session_id)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_enrichments_project ON session_enrichments(primary_intent)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_enrichments_outcome ON session_enrichments(outcome)")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_enrichments_quality ON session_enrichments(session_quality_score)"
        )

        # Session enrichment FTS5
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS session_enrichments_fts USING fts5(
                session_summary, what_worked, what_failed, session_id UNINDEXED
            )
        """)

        # Phase commits table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS phase_commits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commit_hash TEXT NOT NULL,
                commit_message TEXT,
                phase_name TEXT,
                session_id TEXT,
                project TEXT,
                files_changed TEXT,
                confidence_score REAL,
                outcome TEXT,
                reversibility TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_commits_project ON phase_commits(project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_phase_commits_phase ON phase_commits(phase_name)")

        # Git learning tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS git_memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                repo TEXT NOT NULL,
                author TEXT,
                committed_at REAL NOT NULL,
                affected_files TEXT,
                strength REAL DEFAULT 1.0,
                half_life_days REAL DEFAULT 30.0,
                confidence REAL DEFAULT 0.7,
                retrieval_count INTEGER DEFAULT 0,
                invalidated_by TEXT,
                commit_message TEXT,
                tags TEXT,
                importance REAL DEFAULT 5.0,
                UNIQUE(repo, commit_hash)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_git_memories_repo ON git_memories(repo)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_git_memories_type ON git_memories(memory_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_git_memories_commit_time ON git_memories(committed_at)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migration_events (
                id TEXT PRIMARY KEY,
                from_pattern TEXT NOT NULL,
                to_pattern TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                repo TEXT NOT NULL,
                detected_at REAL NOT NULL,
                confidence REAL,
                memories_weakened INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_migration_events_repo ON migration_events(repo)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_migration_events_commit_hash ON migration_events(commit_hash)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS file_cochanges (
                file_a TEXT NOT NULL,
                file_b TEXT NOT NULL,
                repo TEXT NOT NULL,
                cochange_count INTEGER DEFAULT 1,
                last_cochange REAL,
                PRIMARY KEY (file_a, file_b, repo)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_cochanges_repo ON file_cochanges(repo)")

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS git_memories_fts USING fts5(
                content, commit_message, tags, git_memory_id UNINDEXED
            )
        """)
        cursor.execute("DROP TRIGGER IF EXISTS git_memories_fts_insert")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS git_memories_fts_insert AFTER INSERT ON git_memories BEGIN
                INSERT INTO git_memories_fts(content, commit_message, tags, git_memory_id)
                VALUES (new.content, new.commit_message, new.tags, new.id);
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS git_memories_fts_delete")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS git_memories_fts_delete AFTER DELETE ON git_memories BEGIN
                DELETE FROM git_memories_fts WHERE git_memory_id = old.id;
            END
        """)
        cursor.execute("DROP TRIGGER IF EXISTS git_memories_fts_update")
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS git_memories_fts_update
            AFTER UPDATE OF content, commit_message, tags ON git_memories BEGIN
                DELETE FROM git_memories_fts WHERE git_memory_id = old.id;
                INSERT INTO git_memories_fts(content, commit_message, tags, git_memory_id)
                VALUES (new.content, new.commit_message, new.tags, new.id);
            END
        """)

        # source_project_id column
        if "source_project_id" not in existing_cols:
            cursor.execute("ALTER TABLE chunks ADD COLUMN source_project_id TEXT")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source_project ON chunks(source_project_id)")

        # ── Chunk events audit table ──────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                action TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                by_whom TEXT,
                reason TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_events_chunk ON chunk_events(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_events_action ON chunk_events(action)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunk_events_timestamp ON chunk_events(timestamp)")

        # ── Health events audit table ─────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_events_type ON health_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_events_severity ON health_events(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_events_created ON health_events(created_at)")

        # ── Correction mining table ───────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS correction_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                entity_name TEXT,
                attribute TEXT,
                old_value TEXT,
                new_value TEXT,
                confidence REAL DEFAULT 0.0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_pairs_chunk ON correction_pairs(chunk_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_correction_pairs_pattern ON correction_pairs(pattern_type)")

        # ── Knowledge Graph tables ──────────────────────────────────────

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(entity_type, name)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_type ON kg_entities(entity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_name ON kg_entities(name)")

        kg_entity_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entities)")}
        if "user_verified" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN user_verified INTEGER DEFAULT 0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                UNIQUE(source_id, target_id, relation_type)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_source ON kg_relations(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_target ON kg_relations(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_type ON kg_relations(relation_type)")

        kg_rel_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_relations)")}
        if "user_verified" not in kg_rel_cols:
            cursor.execute("ALTER TABLE kg_relations ADD COLUMN user_verified INTEGER DEFAULT 0")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_chunks (
                entity_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                context TEXT,
                PRIMARY KEY (entity_id, chunk_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_ec_entity ON kg_entity_chunks(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_ec_chunk ON kg_entity_chunks(chunk_id)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_facts (
                entity_id TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT,
                status TEXT DEFAULT 'active',
                superseded_by TEXT,
                provenance_chunk_ids TEXT DEFAULT '[]',
                PRIMARY KEY (entity_id, fact_text)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_entity_facts_entity_status ON entity_facts(entity_id, status)")

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kg_vec_entities USING vec0(
                entity_id TEXT PRIMARY KEY,
                embedding FLOAT[1024]
            )
        """)

        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kg_entities_fts USING fts5(
                name, metadata, entity_id UNINDEXED
            )
        """)

        # KG FTS5 sync triggers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_insert AFTER INSERT ON kg_entities BEGIN
                INSERT INTO kg_entities_fts(name, metadata, entity_id)
                VALUES (new.name, new.metadata, new.id);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_delete AFTER DELETE ON kg_entities BEGIN
                DELETE FROM kg_entities_fts WHERE entity_id = old.id;
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS kg_entities_fts_update
            AFTER UPDATE OF name, metadata ON kg_entities BEGIN
                DELETE FROM kg_entities_fts WHERE entity_id = old.id;
                INSERT INTO kg_entities_fts(name, metadata, entity_id)
                VALUES (new.name, new.metadata, new.id);
            END
        """)

        # Entity aliases table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_entity_aliases (
                alias TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                alias_type TEXT DEFAULT 'name',
                created_at TEXT,
                PRIMARY KEY (alias, entity_id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_entity ON kg_entity_aliases(entity_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_lookup ON kg_entity_aliases(alias COLLATE NOCASE)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias_type ON kg_entity_aliases(alias_type)")

        # KG standard spec migrations
        for col, default in [
            ("canonical_name", "TEXT"),
            ("description", "TEXT"),
            ("confidence", "REAL DEFAULT 1.0"),
            ("importance", "REAL DEFAULT 0.5"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("expired_at", "TEXT"),
            ("group_id", "TEXT"),
        ]:
            if col not in kg_entity_cols:
                cursor.execute(f"ALTER TABLE kg_entities ADD COLUMN {col} {default}")

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_kg_entities_canonical ON kg_entities(canonical_name, entity_type)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_valid ON kg_entities(valid_from, valid_until)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_expired ON kg_entities(expired_at)")

        for col, default in [
            ("fact", "TEXT"),
            ("importance", "REAL DEFAULT 0.5"),
            ("valid_from", "TEXT"),
            ("valid_until", "TEXT"),
            ("expired_at", "TEXT"),
            ("source_chunk_id", "TEXT"),
        ]:
            if col not in kg_rel_cols:
                cursor.execute(f"ALTER TABLE kg_relations ADD COLUMN {col} {default}")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_validity ON kg_relations(valid_from, valid_until)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_relations_expired ON kg_relations(expired_at)")

        ec_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_chunks)")}
        if "mention_type" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN mention_type TEXT")

        # ── R49: Entity Contracts Schema ───────────────────────────────

        # R49: entity_contracts table — defines required/expected fields per entity type
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_contracts (
                entity_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                field_type TEXT NOT NULL DEFAULT 'text',
                requirement TEXT NOT NULL DEFAULT 'optional',
                description TEXT,
                PRIMARY KEY (entity_type, field_name)
            )
        """)

        # R49: entity_health table — completeness scores per entity
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_health (
                entity_name TEXT PRIMARY KEY,
                completeness_score REAL NOT NULL DEFAULT 0.0,
                health_level INTEGER NOT NULL DEFAULT 1,
                missing_required TEXT DEFAULT '[]',
                missing_expected TEXT DEFAULT '[]',
                chunk_count INTEGER NOT NULL DEFAULT 0,
                relationship_count INTEGER NOT NULL DEFAULT 0,
                last_scored_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            )
        """)

        # R49: entity_type_hierarchy — type taxonomy stored as data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_type_hierarchy (
                child_type TEXT PRIMARY KEY,
                parent_type TEXT,
                description TEXT
            )
        """)

        # R49: Seed type hierarchy with core types + subtypes
        _type_hierarchy_seed = [
            ("agent", "entity", "Autonomous AI agent or golem"),
            ("person", "entity", "Human individual"),
            ("tool", "entity", "Software tool or service"),
            ("project", "entity", "Software project or initiative"),
            ("concept", "entity", "Abstract concept, pattern, or domain idea"),
            ("topic", "concept", "Recurring subject or thematic area"),
            ("protocol", "topic", "Named workflow or protocol"),
            ("community", "entity", "Community, audience, or social group"),
            ("health_metric", "topic", "Health or wellness metric"),
            ("workflow", "concept", "Repeatable workflow or process"),
            ("device", "entity", "Hardware device or machine"),
            ("event", "entity", "Temporal event or occurrence"),
            ("organization", "entity", "Company or group"),
            ("source", "entity", "External content source: channel, podcast, blog, newsletter"),
            ("golem", "agent", "Specialized AI agent in the golems ecosystem"),
            ("platform", "tool", "Software platform or framework"),
            ("skill", "concept", "Reusable AI skill or capability"),
            ("decision", "concept", "Architectural or design decision"),
        ]
        for child, parent, desc in _type_hierarchy_seed:
            cursor.execute(
                "INSERT OR IGNORE INTO entity_type_hierarchy (child_type, parent_type, description) VALUES (?, ?, ?)",
                (child, parent, desc),
            )

        # R49: ALTER kg_entities — add entity_subtype, status
        if "entity_subtype" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN entity_subtype TEXT")
        if "status" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN status TEXT DEFAULT 'active'")
        if "parent_id" not in kg_entity_cols:
            cursor.execute("ALTER TABLE kg_entities ADD COLUMN parent_id TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_kg_entities_parent ON kg_entities(parent_id)")

        # R49: ALTER kg_entity_chunks — add relation_tier, weight
        if "relation_tier" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN relation_tier INTEGER DEFAULT 4")
        if "weight" not in ec_cols:
            cursor.execute("ALTER TABLE kg_entity_chunks ADD COLUMN weight REAL DEFAULT 0.25")

        # R49: Upgrade kg_entity_aliases — add valid_from, valid_to if missing
        alias_cols = {row[1] for row in cursor.execute("PRAGMA table_info(kg_entity_aliases)")}
        if "valid_to" not in alias_cols:
            cursor.execute("ALTER TABLE kg_entity_aliases ADD COLUMN valid_to TEXT")
        # valid_from may already exist from original schema (as created_at) — ensure both names exist
        if "valid_from" not in alias_cols:
            cursor.execute("ALTER TABLE kg_entity_aliases ADD COLUMN valid_from TEXT")

        # kg_current_facts view
        cursor.execute("DROP VIEW IF EXISTS kg_current_facts")
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS kg_current_facts AS
            SELECT * FROM kg_relations
            WHERE (valid_from IS NULL OR julianday(valid_from) <= julianday('now'))
              AND (valid_until IS NULL OR julianday(valid_until) >= julianday('now'))
              AND expired_at IS NULL
        """)

        # FTS5 backfill check — populate from chunks if FTS is empty (fresh rebuild or first run)
        fts_count = list(cursor.execute("SELECT COUNT(*) FROM chunks_fts"))[0][0]
        chunk_count = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        if chunk_count > 0 and fts_count == 0:
            cursor.execute("""
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
            """)

        # Thread-local storage for per-thread read connections.
        # APSW connections are NOT thread-safe — each thread needs its own.
        # This prevents "Connection is busy in another thread" when parallel
        # MCP tool calls (e.g., brain_search) hit the same VectorStore.
        self._local = threading.local()

    def _backfill_atomic_brick_columns(self, cursor) -> None:
        """Populate additive atomic-brick metadata from existing chunk fields."""
        cursor.execute("""
            UPDATE chunks
            SET brick_id = id
            WHERE brick_id IS NULL
        """)
        cursor.execute("""
            UPDATE chunks
            SET source_uri = source_file
            WHERE source_uri IS NULL AND source_file IS NOT NULL
        """)
        cursor.execute("""
            UPDATE chunks
            SET ingested_at = COALESCE(
                CAST(strftime('%s', created_at) AS INTEGER),
                CAST(strftime('%s', 'now') AS INTEGER)
            )
            WHERE ingested_at IS NULL
        """)
        cursor.execute("""
            UPDATE chunks
            SET status = CASE
                WHEN COALESCE(archived, 0) = 1
                  OR value_type = 'ARCHIVED'
                  OR archived_at IS NOT NULL
                    THEN 'archived'
                WHEN superseded_by IS NOT NULL
                  OR aggregated_into IS NOT NULL
                    THEN 'superseded'
                ELSE 'active'
            END
            WHERE status IS NULL
               OR status NOT IN ('active', 'superseded', 'archived')
               OR (
                    status = 'active'
                    AND (
                        COALESCE(archived, 0) = 1
                        OR value_type = 'ARCHIVED'
                        OR archived_at IS NOT NULL
                        OR superseded_by IS NOT NULL
                        OR aggregated_into IS NOT NULL
                    )
               )
        """)

    def repair_fts(self, *, rebuild_trigram: bool = True) -> dict[str, int]:
        """Run explicit FTS repair work outside normal startup."""
        for attempt in range(5):
            cursor = self.conn.cursor()
            repaired: dict[str, int] = {}
            transaction_started = False
            try:
                cursor.execute("PRAGMA wal_checkpoint(FULL)")
                cursor.execute("BEGIN IMMEDIATE")
                transaction_started = True
                if rebuild_trigram:
                    cursor.execute("DELETE FROM chunks_fts_trigram")
                    cursor.execute("""
                        INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                        SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
                    """)
                    repaired["chunks_fts_trigram"] = cursor.execute(
                        "SELECT COUNT(*) FROM chunks_fts_trigram"
                    ).fetchone()[0]
                cursor.execute("COMMIT")
                transaction_started = False
                cursor.execute("PRAGMA wal_checkpoint(FULL)")
                return repaired
            except apsw.BusyError:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                if attempt == 4:
                    raise
                time.sleep(0.2 * (2**attempt))
            except Exception:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                raise
        return {}

    def _get_read_conn(self) -> apsw.Connection:
        """Get or create a per-thread readonly connection."""
        conn = getattr(self._local, "read_conn", None)
        if conn is None:
            conn = apsw.Connection(str(self.db_path), flags=apsw.SQLITE_OPEN_READONLY)
            conn.enableloadextension(True)
            conn.loadextension(sqlite_vec.loadable_path())
            conn.enableloadextension(False)
            _configure_reader_pragmas(conn)
            self._local.read_conn = conn
        return conn

    def _read_cursor(self):
        """Return a cursor for read operations using a per-thread readonly connection."""
        return self._get_read_conn().cursor()

    def _log_health_event(self, event_type: str, severity: str, details: Dict[str, Any]) -> None:
        """Append a health event. Readonly stores skip writes silently."""
        if getattr(self, "_readonly", False):
            return
        try:
            self.conn.cursor().execute(
                "INSERT INTO health_events (event_type, severity, details) VALUES (?, ?, ?)",
                (event_type, severity, json.dumps(details, sort_keys=True)),
            )
        except (apsw.ReadOnlyError, apsw.BusyError):
            return

    def _get_fts5_counts(self) -> tuple[int, int]:
        """Read chunks and FTS counts using a single query on the readonly path."""
        row = (
            self._read_cursor()
            .execute("SELECT (SELECT COUNT(*) FROM chunks), (SELECT COUNT(*) FROM chunks_fts)")
            .fetchone()
        )
        return int(row[0]), int(row[1])

    @staticmethod
    def _build_fts5_health_result(chunk_count: int, fts_count: int, severity: str) -> Dict[str, Any]:
        """Shape a health payload from count data."""
        desync_pct = 0.0
        if chunk_count > 0:
            desync_pct = round(abs(chunk_count - fts_count) * 100.0 / chunk_count, 2)
        return {
            "synced": desync_pct <= 1.0,
            "chunk_count": chunk_count,
            "fts_count": fts_count,
            "desync_pct": desync_pct,
            "severity": severity,
        }

    def check_fts5_health(self, cache_ttl_seconds: int = 60) -> Dict[str, Any]:
        """Check FTS5 sync health with a short-lived cache for hot-path callers."""
        now = time.time()
        cache = self._fts5_health_cache
        if cache_ttl_seconds > 0 and cache.get("expires_at", 0) > now:
            return dict(cache["result"])

        chunk_count, fts_count = self._get_fts5_counts()
        desync_pct = 0.0 if chunk_count == 0 else abs(chunk_count - fts_count) * 100.0 / chunk_count

        if desync_pct > 20.0:
            self._log_health_event(
                "fts5_desync_critical",
                "emergency",
                {"chunk_count": chunk_count, "fts_count": fts_count, "desync_pct": round(desync_pct, 2)},
            )
            rebuild_result = self.rebuild_fts5()
            result = {
                "synced": rebuild_result["success"],
                "chunk_count": rebuild_result["chunk_count"],
                "fts_count": rebuild_result["fts_count"],
                "desync_pct": rebuild_result["desync_pct"],
                "severity": "emergency",
                "rebuild_triggered": True,
            }
        elif desync_pct > 5.0:
            result = self._build_fts5_health_result(chunk_count, fts_count, "critical")
            self._log_health_event("fts5_desync_critical", "critical", result)
        elif desync_pct > 1.0:
            result = self._build_fts5_health_result(chunk_count, fts_count, "warning")
            self._log_health_event("fts5_desync_warning", "warning", result)
        else:
            result = self._build_fts5_health_result(chunk_count, fts_count, "info")

        if cache_ttl_seconds > 0:
            self._fts5_health_cache = {"result": dict(result), "expires_at": now + cache_ttl_seconds}
        else:
            self._fts5_health_cache = {}
        return result

    def check_wal_health(self) -> Dict[str, Any]:
        """Check WAL size and passive checkpoint status."""
        wal_path = Path(f"{self.db_path}-wal")
        wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0
        wal_size_mb = wal_size_bytes / (1024 * 1024)

        checkpoint_row = self.conn.cursor().execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        checkpoint_status = {
            "busy": int(checkpoint_row[0]),
            "log_frames": int(checkpoint_row[1]),
            "checkpointed_frames": int(checkpoint_row[2]),
        }

        severity = "info"
        if wal_size_mb > 50:
            severity = "warning"
            self._log_health_event(
                "wal_bloat",
                "warning",
                {
                    "wal_path": str(wal_path),
                    "wal_size_bytes": wal_size_bytes,
                    "wal_size_mb": round(wal_size_mb, 2),
                    "checkpoint_status": checkpoint_status,
                },
            )

        return {
            "wal_path": str(wal_path),
            "wal_exists": wal_path.exists(),
            "wal_size_bytes": wal_size_bytes,
            "wal_size_mb": round(wal_size_mb, 2),
            "checkpoint_status": checkpoint_status,
            "severity": severity,
        }

    def deep_integrity_check(self) -> Dict[str, Any]:
        """Run an FTS integrity check plus a bounded spot-check of chunk IDs."""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")

        total_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        spot_check_count = min(100, total_chunks)
        sample_ids = [
            row[0]
            for row in cursor.execute(
                "SELECT id FROM chunks ORDER BY random() LIMIT ?",
                (spot_check_count,),
            )
        ]

        missing_chunk_ids: list[str] = []
        if sample_ids:
            placeholders = ", ".join("?" for _ in sample_ids)
            found_ids = {
                row[0]
                for row in cursor.execute(
                    f"SELECT chunk_id FROM chunks_fts WHERE chunk_id IN ({placeholders})",
                    sample_ids,
                )
            }
            missing_chunk_ids = sorted(set(sample_ids) - found_ids)

        ok = len(missing_chunk_ids) == 0
        details = {
            "spot_check_count": spot_check_count,
            "missing_chunk_ids": missing_chunk_ids,
        }
        if ok:
            self._log_health_event("fts5_integrity_ok", "info", details)
        else:
            self._log_health_event("integrity_fail", "critical", details)

        return {
            "ok": ok,
            "fts_integrity": "ok" if ok else "failed",
            "spot_check_count": spot_check_count,
            "missing_chunk_ids": missing_chunk_ids,
        }

    def rebuild_fts5(self) -> Dict[str, Any]:
        """Rebuild the FTS5 table and verify post-rebuild counts."""
        self._log_health_event("fts5_rebuild", "emergency", {"db_path": str(self.db_path)})
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        if getattr(self, "_trigram_fts_available", False):
            cursor.execute("INSERT INTO chunks_fts_trigram(chunks_fts_trigram) VALUES('rebuild')")
        chunk_count, fts_count = self._get_fts5_counts()
        if chunk_count != fts_count:
            cursor.execute("DELETE FROM chunks_fts")
            cursor.execute("""
                INSERT INTO chunks_fts(content, summary, tags, resolved_query, chunk_id)
                SELECT content, summary, tags, resolved_query, id FROM chunks
            """)
        if getattr(self, "_trigram_fts_available", False):
            trigram_count = cursor.execute("SELECT COUNT(*) FROM chunks_fts_trigram").fetchone()[0]
            if chunk_count != trigram_count:
                cursor.execute("DELETE FROM chunks_fts_trigram")
                cursor.execute("""
                    INSERT INTO chunks_fts_trigram(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
                    SELECT content, summary, tags, resolved_query, key_facts, resolved_queries, id FROM chunks
                """)
        try:
            cursor.execute("PRAGMA wal_checkpoint(PASSIVE)")
        except apsw.Error:
            pass

        self._fts5_health_cache = {}
        chunk_count, fts_count = self._get_fts5_counts()
        trigram_count = None
        fts_desync_pct = 0.0 if chunk_count == 0 else round(abs(chunk_count - fts_count) * 100.0 / chunk_count, 2)
        trigram_desync_pct = 0.0
        if getattr(self, "_trigram_fts_available", False):
            trigram_count = cursor.execute("SELECT COUNT(*) FROM chunks_fts_trigram").fetchone()[0]
            trigram_desync_pct = (
                0.0 if chunk_count == 0 else round(abs(chunk_count - trigram_count) * 100.0 / chunk_count, 2)
            )
        desync_pct = max(fts_desync_pct, trigram_desync_pct)
        return {
            "success": chunk_count == fts_count and (trigram_count is None or chunk_count == trigram_count),
            "chunk_count": chunk_count,
            "fts_count": fts_count,
            "trigram_count": trigram_count,
            "desync_pct": desync_pct,
        }

    def build_binary_index(self) -> int:
        """Backfill binary-quantized vectors from existing float vectors."""
        if not getattr(self, "_binary_index_available", False):
            raise RuntimeError("Binary vector index is unavailable on this database")
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO chunk_vectors_binary(chunk_id, embedding)
            SELECT chunk_id, vec_quantize_binary(embedding) FROM chunk_vectors
        """)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        return self.conn.changes()

    def _upsert_chunk_vector(self, cursor, chunk_id: str, embedding: List[float]) -> None:
        """Keep float and binary vector tables in sync for a chunk."""
        embedding_bytes = serialize_f32(embedding)
        cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        cursor.execute(
            """
            INSERT INTO chunk_vectors (chunk_id, embedding)
            VALUES (?, ?)
        """,
            (chunk_id, embedding_bytes),
        )
        if getattr(self, "_binary_index_available", False):
            cursor.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))
            cursor.execute(
                """
                INSERT INTO chunk_vectors_binary (chunk_id, embedding)
                VALUES (?, vec_quantize_binary(?))
            """,
                (chunk_id, embedding_bytes),
            )

    def _blend_chunk_vector(self, cursor, chunk_id: str, embedding: List[float]) -> List[float]:
        """Approximate a merged duplicate by preserving canonical and incoming vectors."""
        row = cursor.execute("SELECT embedding FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if not row or row[0] is None:
            return embedding
        existing_bytes = bytes(row[0])
        if len(existing_bytes) != len(embedding) * 4:
            return embedding
        existing = struct.unpack(f"{len(embedding)}f", existing_bytes)
        return [(float(left) + float(right)) / 2.0 for left, right in zip(existing, embedding)]

    def _chunk_vector_exists(self, cursor, chunk_id: str) -> bool:
        return bool(cursor.execute("SELECT 1 FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,)).fetchone())

    def _delete_chunk_vector(self, cursor, chunk_id: str) -> None:
        """Delete a chunk from both float and binary vector tables."""
        cursor.execute("DELETE FROM chunk_vectors WHERE chunk_id = ?", (chunk_id,))
        if getattr(self, "_binary_index_available", False):
            cursor.execute("DELETE FROM chunk_vectors_binary WHERE chunk_id = ?", (chunk_id,))

    # ── Chunk CRUD ──────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Upsert chunks with embeddings, returning the number of input chunks processed."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have same length")

        valid_pairs: list[tuple[Dict[str, Any], List[float]]] = []
        rejected_error: ValueError | None = None
        for chunk, embedding in zip(chunks, embeddings):
            try:
                reject_recursive_mcp_output(
                    chunk.get("content"),
                    chunk_id=chunk.get("id"),
                    source_file=chunk.get("source_file"),
                )
            except ValueError as exc:
                rejected_error = rejected_error or exc
                continue
            valid_pairs.append((chunk, embedding))
        if not valid_pairs and rejected_error is not None:
            raise rejected_error

        for attempt in range(5):
            cursor = self.conn.cursor()
            transaction_started = False
            try:
                cursor.execute("BEGIN IMMEDIATE")
                transaction_started = True
                for chunk, embedding in valid_pairs:
                    chunk_id = chunk["id"]
                    created_at = chunk.get("created_at") or datetime.now(timezone.utc).isoformat()
                    chunk = {**chunk, "created_at": created_at}
                    tags_value = chunk.get("tags")
                    tags_json = json.dumps(tags_value) if isinstance(tags_value, (list, dict)) else tags_value
                    duplicate, dedupe_fields = find_duplicate(
                        self.conn,
                        chunk_id=chunk_id,
                        content=chunk["content"],
                        created_at=created_at,
                        project=chunk.get("project"),
                        content_type=chunk.get("content_type"),
                    )
                    if duplicate is not None:
                        duplicate_row_exists = cursor.execute(
                            "SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)
                        ).fetchone()
                        content_changed = merge_duplicate_chunk(
                            self.conn,
                            canonical_id=duplicate.canonical_chunk_id,
                            duplicate_id=chunk_id,
                            incoming={
                                **chunk,
                                "tags": tags_json,
                                "created_at": created_at,
                                "last_seen_at": chunk.get("last_seen_at") or created_at,
                            },
                            mechanism=duplicate.mechanism,
                            hamming_distance_value=duplicate.hamming_distance,
                            archive_existing_duplicate=duplicate_row_exists is not None,
                        )
                        if content_changed:
                            merged_embedding = self._blend_chunk_vector(cursor, duplicate.canonical_chunk_id, embedding)
                            self._upsert_chunk_vector(cursor, duplicate.canonical_chunk_id, merged_embedding)
                        elif not self._chunk_vector_exists(cursor, duplicate.canonical_chunk_id):
                            self._upsert_chunk_vector(cursor, duplicate.canonical_chunk_id, embedding)
                        continue
                    if merge_existing_chunk_seen(
                        self.conn,
                        chunk_id=chunk_id,
                        incoming={
                            **chunk,
                            "tags": tags_json,
                            "created_at": created_at,
                            "last_seen_at": chunk.get("last_seen_at") or created_at,
                        },
                    ):
                        if not self._chunk_vector_exists(cursor, chunk_id):
                            self._upsert_chunk_vector(cursor, chunk_id, embedding)
                        continue

                    cursor.execute(
                        """
                        INSERT INTO chunks
                        (id, content, metadata, source_file, project,
                         content_type, value_type, char_count, source, created_at,
                         conversation_id, position, sender, chunk_origin, tags, importance,
                         half_life_days, seen_count, last_seen_at, dedupe_hash, simhash,
                         simhash_band_0, simhash_band_1, simhash_band_2, simhash_band_3,
                         brick_id, source_uri, status, ingested_at, topic_cluster)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            content = excluded.content,
                            metadata = excluded.metadata,
                            source_file = excluded.source_file,
                            project = excluded.project,
                            content_type = excluded.content_type,
                            value_type = excluded.value_type,
                            char_count = excluded.char_count,
                            source = excluded.source,
                            created_at = COALESCE(chunks.created_at, excluded.created_at),
                            conversation_id = COALESCE(excluded.conversation_id, chunks.conversation_id),
                            position = COALESCE(excluded.position, chunks.position),
                            sender = COALESCE(excluded.sender, chunks.sender),
                            tags = COALESCE(excluded.tags, chunks.tags),
                            importance = COALESCE(excluded.importance, chunks.importance),
                            half_life_days = MAX(COALESCE(chunks.half_life_days, 30.0), COALESCE(excluded.half_life_days, 30.0)),
                            seen_count = COALESCE(chunks.seen_count, 1),
                            last_seen_at = COALESCE(chunks.last_seen_at, excluded.last_seen_at),
                            dedupe_hash = excluded.dedupe_hash,
                            simhash = excluded.simhash,
                            simhash_band_0 = excluded.simhash_band_0,
                            simhash_band_1 = excluded.simhash_band_1,
                            simhash_band_2 = excluded.simhash_band_2,
                            simhash_band_3 = excluded.simhash_band_3,
                            brick_id = COALESCE(chunks.brick_id, excluded.brick_id),
                            source_uri = COALESCE(excluded.source_uri, chunks.source_uri),
                            status = CASE
                                WHEN excluded.status IS NOT NULL AND excluded.status != 'active' THEN excluded.status
                                WHEN chunks.status IS NULL THEN COALESCE(excluded.status, 'active')
                                ELSE chunks.status
                            END,
                            ingested_at = COALESCE(chunks.ingested_at, excluded.ingested_at),
                            topic_cluster = COALESCE(excluded.topic_cluster, chunks.topic_cluster),
                            chunk_origin = CASE
                                WHEN excluded.content != chunks.content
                                    THEN COALESCE(excluded.chunk_origin, 'unknown')
                                WHEN excluded.chunk_origin IS NOT NULL AND excluded.chunk_origin != 'unknown'
                                    THEN excluded.chunk_origin
                                WHEN chunks.chunk_origin IS NULL
                                    THEN COALESCE(excluded.chunk_origin, 'unknown')
                                ELSE chunks.chunk_origin
                            END
                    """,
                        (
                            chunk_id,
                            chunk["content"],
                            json.dumps(chunk["metadata"]),
                            chunk["source_file"],
                            chunk.get("project"),
                            chunk.get("content_type"),
                            chunk.get("value_type"),
                            chunk.get("char_count", 0),
                            chunk.get("source", "claude_code"),
                            chunk.get("created_at"),
                            chunk.get("conversation_id"),
                            chunk.get("position"),
                            chunk.get("sender"),
                            detect_chunk_origin(chunk.get("content"), chunk.get("chunk_origin")),
                            tags_json,
                            float(chunk["importance"]) if chunk.get("importance") is not None else None,
                            float(chunk["half_life_days"]) if chunk.get("half_life_days") is not None else None,
                            int(chunk.get("seen_count") or 1),
                            chunk.get("last_seen_at") or created_at,
                            dedupe_fields.dedupe_hash,
                            dedupe_fields.simhash,
                            dedupe_fields.bands[0],
                            dedupe_fields.bands[1],
                            dedupe_fields.bands[2],
                            dedupe_fields.bands[3],
                            chunk.get("brick_id", chunk_id),
                            chunk.get("source_uri") or chunk["source_file"],
                            chunk.get("status", "active"),
                            chunk.get("ingested_at") or int(time.time()),
                            chunk.get("topic_cluster"),
                        ),
                    )
                    self._upsert_chunk_vector(cursor, chunk_id, embedding)
                cursor.execute("COMMIT")
                transaction_started = False
                break
            except apsw.BusyError:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                if attempt == 4:
                    raise
                time.sleep(0.1 * (2**attempt))
            except Exception:
                if transaction_started:
                    cursor.execute("ROLLBACK")
                raise

        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        self._invalidate_filtered_count_caches()

        return len(valid_pairs)

    def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """Update fields on an existing chunk. Returns True if chunk was found."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
        if not rows:
            return False

        if content is not None:
            existing_source_file = cursor.execute("SELECT source_file FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            reject_recursive_mcp_output(
                content,
                chunk_id=chunk_id,
                source_file=existing_source_file[0] if existing_source_file else None,
            )
            created_at_row = cursor.execute("SELECT created_at FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            dedupe_fields = compute_dedupe_fields(content, created_at_row[0] if created_at_row else None)
            cursor.execute(
                """
                UPDATE chunks
                SET content = ?,
                    char_count = ?,
                    summary = ?,
                    chunk_origin = ?,
                    dedupe_hash = ?,
                    simhash = ?,
                    simhash_band_0 = ?,
                    simhash_band_1 = ?,
                    simhash_band_2 = ?,
                    simhash_band_3 = ?
                WHERE id = ?
                """,
                (
                    content,
                    len(content),
                    content[:200],
                    detect_chunk_origin(content),
                    dedupe_fields.dedupe_hash,
                    dedupe_fields.simhash,
                    dedupe_fields.bands[0],
                    dedupe_fields.bands[1],
                    dedupe_fields.bands[2],
                    dedupe_fields.bands[3],
                    chunk_id,
                ),
            )
        if tags is not None:
            cursor.execute(
                "UPDATE chunks SET tags = ? WHERE id = ?",
                (json.dumps(tags), chunk_id),
            )
        if importance is not None:
            cursor.execute(
                "UPDATE chunks SET importance = ? WHERE id = ?",
                (float(max(1, min(10, importance))), chunk_id),
            )
        if embedding is not None:
            self._upsert_chunk_vector(cursor, chunk_id, embedding)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        self._invalidate_filtered_count_caches()
        return True

    def archive_chunk(self, chunk_id: str) -> bool:
        """Soft-delete a chunk by setting value_type to ARCHIVED and archived_at."""
        cursor = self.conn.cursor()
        rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)))
        if not rows:
            return False
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "UPDATE chunks SET value_type = 'ARCHIVED', archived = 1, archived_at = ?, status = 'archived' WHERE id = ?",
            (now, chunk_id),
        )
        self._delete_chunk_vector(cursor, chunk_id)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        self._invalidate_filtered_count_caches()
        return True

    def supersede_chunk(self, old_chunk_id: str, new_chunk_id: str) -> bool:
        """Mark old chunk as superseded by new chunk. Removes old from vector index."""
        cursor = self.conn.cursor()
        old_rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (old_chunk_id,)))
        if not old_rows:
            return False
        new_rows = list(cursor.execute("SELECT id FROM chunks WHERE id = ?", (new_chunk_id,)))
        if not new_rows:
            return False
        cursor.execute(
            "UPDATE chunks SET superseded_by = ?, status = 'superseded' WHERE id = ?",
            (new_chunk_id, old_chunk_id),
        )
        self._delete_chunk_vector(cursor, old_chunk_id)
        from .search_repo import clear_hybrid_search_cache

        clear_hybrid_search_cache(getattr(self, "db_path", None))
        self._invalidate_filtered_count_caches()
        return True

    def get_chunk(self, chunk_id: str, *, include_archived: bool = False) -> Optional[Dict[str, Any]]:
        """Get a single active chunk by ID.

        Soft-deleted chunks are hidden by default: archived, superseded, aggregated,
        or non-active status rows return None unless include_archived=True.
        """
        read_conn = self._get_read_conn()
        cursor = read_conn.cursor()
        chunk_id = resolve_chunk_id(read_conn, chunk_id)
        cols = {row[1] for row in cursor.execute("PRAGMA table_info(chunks)")}

        def col_or_null(name: str, alias: str | None = None) -> str:
            output = alias or name
            if name in cols:
                return name
            return f"NULL AS {output}"

        archived_expr = "COALESCE(archived, 0) AS archived" if "archived" in cols else "0 AS archived"
        status_expr = "COALESCE(status, 'active') AS status" if "status" in cols else "'active' AS status"
        chunk_origin_expr = (
            "chunk_origin AS chunk_origin" if "chunk_origin" in cols else f"'{CHUNK_ORIGIN_UNKNOWN}' AS chunk_origin"
        )
        lifecycle_clauses = []
        if not include_archived:
            if "superseded_by" in cols:
                lifecycle_clauses.append("superseded_by IS NULL")
            if "aggregated_into" in cols:
                lifecycle_clauses.append("aggregated_into IS NULL")
            if "archived_at" in cols:
                lifecycle_clauses.append("archived_at IS NULL")
            if "archived" in cols:
                lifecycle_clauses.append("COALESCE(archived, 0) = 0")
            if "status" in cols:
                lifecycle_clauses.append("COALESCE(status, 'active') = 'active'")
        lifecycle_filter = "".join(f" AND {clause}" for clause in lifecycle_clauses)
        rows = list(
            cursor.execute(
                f"""SELECT id, content, metadata, source_file, project, content_type,
                      {col_or_null("value_type")}, {col_or_null("tags")},
                      {col_or_null("importance")}, {col_or_null("created_at")},
                      {col_or_null("summary")},
                      {col_or_null("superseded_by")}, {col_or_null("aggregated_into")},
                      {col_or_null("archived_at")}, {archived_expr}, {status_expr},
                      {col_or_null("brick_id")}, {col_or_null("source_uri")},
                      {col_or_null("ingested_at")}, {col_or_null("topic_cluster")},
                      {chunk_origin_expr}, {col_or_null("content_class")}
               FROM chunks WHERE id = ?
               {lifecycle_filter}
            """,
                (chunk_id,),
            )
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "id": r[0],
            "content": r[1],
            "metadata": r[2],
            "source_file": r[3],
            "project": r[4],
            "content_type": r[5],
            "value_type": r[6],
            "tags": r[7],
            "importance": r[8],
            "created_at": r[9],
            "summary": r[10],
            "superseded_by": r[11],
            "aggregated_into": r[12],
            "archived_at": r[13],
            "archived": r[14],
            "status": r[15],
            "brick_id": r[16],
            "source_uri": r[17],
            "ingested_at": r[18],
            "topic_cluster": r[19],
            "chunk_origin": r[20],
            "content_class": r[21],
        }

    def resolve_chunk_id(self, chunk_id: str) -> str:
        """Resolve a duplicate chunk alias within the 90-day grace period."""
        return resolve_chunk_id(self._get_read_conn(), chunk_id)

    # ── Chunk events audit ─────────────────────────────────────────────

    def record_event(
        self,
        chunk_id: str,
        action: str,
        by_whom: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> int:
        """Record an audit event for a chunk. Returns the event row ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO chunk_events (chunk_id, action, by_whom, reason) VALUES (?, ?, ?, ?)",
            (chunk_id, action, by_whom, reason),
        )
        return self.conn.last_insert_rowid()

    def get_chunk_events(
        self,
        chunk_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get audit events for a chunk, newest first."""
        cursor = self._read_cursor()
        rows = list(
            cursor.execute(
                "SELECT id, chunk_id, action, timestamp, by_whom, reason "
                "FROM chunk_events WHERE chunk_id = ? ORDER BY id DESC LIMIT ?",
                (chunk_id, limit),
            )
        )
        return [
            {
                "id": r[0],
                "chunk_id": r[1],
                "action": r[2],
                "timestamp": r[3],
                "by_whom": r[4],
                "reason": r[5],
            }
            for r in rows
        ]

    # ── Context manager ─────────────────────────────────────────────────

    def close(self) -> None:
        """Close database connections."""
        try:
            # Close thread-local read connection if it exists
            if hasattr(self, "_local"):
                read_conn = getattr(self._local, "read_conn", None)
                if read_conn is not None:
                    read_conn.close()
                    self._local.read_conn = None
            if hasattr(self, "conn"):
                self.conn.close()
        finally:
            self._release_writer_pidfile()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
