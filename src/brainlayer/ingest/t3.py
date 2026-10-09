"""Read-only ingestion for T3 Code's SQLite thread projection.

T3 is a live application database, not a JSONL transcript root.  This adapter
opens it with SQLite's read-only URI mode, validates the projection schema
before reading, and emits ordinary BrainLayer chunks with a distinct
``t3-thread`` provenance class.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..alarm import BrainLayerAlarm, raise_alarm

logger = logging.getLogger(__name__)

T3_SOURCE = "t3"
T3_PROVENANCE_CLASS = "t3-thread"
DEFAULT_T3_STATE_DB = Path("~/.t3/userdata/state.sqlite").expanduser()
DEFAULT_T3_HEALTH_PATH = Path("~/.local/share/brainlayer/t3-health.json").expanduser()

_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "projection_threads": frozenset({"thread_id", "project_id", "title", "created_at"}),
    "projection_thread_messages": frozenset({"message_id", "thread_id", "role", "text", "created_at"}),
    "projection_projects": frozenset({"project_id", "title"}),
    "provider_session_runtime": frozenset({"thread_id", "provider_name", "resume_cursor_json"}),
}

_V2_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "orchestration_v2_projection_threads": frozenset(
        {"thread_id", "project_id", "title", "created_at", "default_provider", "active_provider_thread_id"}
    ),
    "orchestration_v2_projection_messages": frozenset(
        {"message_id", "thread_id", "role", "created_at", "payload_json", "streaming"}
    ),
    "orchestration_v2_projection_provider_threads": frozenset({"provider_thread_id", "provider", "payload_json"}),
    "projection_projects": frozenset({"project_id", "title"}),
}


class T3PayloadError(RuntimeError):
    """A selected V2 projection contains an unreadable payload."""


def _v2_payload(raw: str, row_id: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise T3PayloadError(f"Invalid T3 V2 payload for {row_id}") from error
    if not isinstance(payload, dict):
        raise T3PayloadError(f"T3 V2 payload must be an object for {row_id}")
    return payload


@dataclass(frozen=True)
class T3Message:
    message_id: str
    thread_id: str
    role: str
    text: str
    created_at: str


@dataclass(frozen=True)
class T3Thread:
    thread_id: str
    project_id: str
    title: str
    created_at: str
    project_name: str | None = None
    messages: tuple[T3Message, ...] = ()
    provider_name: str | None = None
    provider_session_id: str | None = None
    mirrored: bool = False


@dataclass
class T3IngestionResult:
    threads_seen: int = 0
    threads_ingested: int = 0
    messages_seen: int = 0
    messages_ingested: int = 0
    chunks_planned: int = 0
    chunks_indexed: int = 0
    duplicates_accepted: int = 0
    messages_skipped: dict[str, int] = field(default_factory=dict)


class T3SchemaError(RuntimeError):
    """The live T3 projection no longer satisfies the reader contract."""

    def __init__(self, missing_tables: list[str], missing_columns: dict[str, list[str]]) -> None:
        self.missing_tables = missing_tables
        self.missing_columns = missing_columns
        super().__init__(
            "T3 schema drift: "
            + json.dumps(
                {"missing_tables": missing_tables, "missing_columns": missing_columns},
                sort_keys=True,
            )
        )


class T3Reader:
    """Read T3 projection rows without ever opening the source for writing."""

    def __init__(
        self,
        state_db_path: Path | str = DEFAULT_T3_STATE_DB,
        *,
        health_path: Path | str | None = DEFAULT_T3_HEALTH_PATH,
        projection_version: int = 1,
    ):
        if projection_version not in {1, 2}:
            raise ValueError("T3 projection_version must be 1 or 2")
        self.projection_version = projection_version
        self.state_db_path = Path(state_db_path).expanduser()
        self.health_path = Path(health_path).expanduser() if health_path is not None else None
        self._last_counts: dict[str, int] = {}
        self._failures: list[dict[str, Any]] = []

    def read_threads(self) -> list[T3Thread]:
        """Read a complete short snapshot of the T3 projections.

        Each query runs in SQLite autocommit mode and is fully materialized
        before the next query, so the adapter does not hold an application
        transaction open while indexing.
        """
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._validate_schema(conn, self.projection_version)
            threads = self._read_v2_snapshot(conn) if self.projection_version == 2 else self._read_snapshot(conn)
        except BrainLayerAlarm:
            raise
        except T3PayloadError as error:
            self._fail("t3_payload_invalid", "T3 V2 projection payload is unreadable", {}, error)
        except T3SchemaError as error:
            self._fail(
                "t3_schema_drift",
                "T3 schema no longer matches the ingestion reader contract",
                {
                    "missing_tables": error.missing_tables,
                    "missing_columns": error.missing_columns,
                },
                error,
            )
        except sqlite3.OperationalError as error:
            if "no such table" in str(error).lower() or "no such column" in str(error).lower():
                self._fail(
                    "t3_schema_drift",
                    "T3 schema changed while the reader was taking its snapshot",
                    {"sqlite_error": str(error)},
                    error,
                )
            self._fail(
                "t3_reader_failed",
                "T3 state database could not be read",
                {"state_db_path": str(self.state_db_path), "error_type": type(error).__name__},
                error,
            )
        except (OSError, sqlite3.Error) as error:
            self._fail(
                "t3_reader_failed",
                "T3 state database could not be read",
                {"state_db_path": str(self.state_db_path), "error_type": type(error).__name__},
                error,
            )
        else:
            self._last_counts = {
                "threads_seen": len(threads),
                "messages_seen": sum(len(thread.messages) for thread in threads),
                "mirrored_threads": sum(thread.mirrored for thread in threads),
            }
            self._write_health(alerting=False, alert_reasons=[])
            return threads
        finally:
            if conn is not None:
                conn.close()

        raise AssertionError("T3 reader failure path must raise")

    def _connect(self) -> sqlite3.Connection:
        if not self.state_db_path.is_file():
            raise FileNotFoundError(f"T3 state database not found: {self.state_db_path}")
        uri = f"file:{self.state_db_path}?mode=ro&immutable=0"
        conn = sqlite3.connect(uri, uri=True, timeout=1.0, isolation_level=None)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=1000")
        return conn

    @staticmethod
    def _validate_schema(conn: sqlite3.Connection, projection_version: int = 1) -> None:
        required_columns = _V2_REQUIRED_COLUMNS if projection_version == 2 else _REQUIRED_COLUMNS
        actual_tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        missing_tables = sorted(set(required_columns) - actual_tables)
        missing_columns: dict[str, list[str]] = {}
        for table, required in required_columns.items():
            if table not in actual_tables:
                continue
            actual_columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()}
            missing = sorted(required - actual_columns)
            if missing:
                missing_columns[table] = missing
        if missing_tables or missing_columns:
            raise T3SchemaError(missing_tables, missing_columns)

    @staticmethod
    def _read_snapshot(conn: sqlite3.Connection) -> list[T3Thread]:
        thread_rows = conn.execute(
            """
            SELECT t.thread_id, t.project_id, p.title, t.title, t.created_at
              FROM projection_threads AS t
              LEFT JOIN projection_projects AS p ON p.project_id = t.project_id
             ORDER BY t.created_at, t.thread_id
            """
        ).fetchall()
        message_rows = conn.execute(
            """
            SELECT message_id, thread_id, role, text, created_at
              FROM projection_thread_messages
             ORDER BY thread_id, created_at, message_id
            """
        ).fetchall()
        runtime_rows = conn.execute(
            """
            SELECT thread_id, provider_name, resume_cursor_json
              FROM provider_session_runtime
            """
        ).fetchall()

        messages_by_thread: dict[str, list[T3Message]] = {}
        for message_id, thread_id, role, text, created_at in message_rows:
            messages_by_thread.setdefault(thread_id, []).append(
                T3Message(
                    message_id=message_id,
                    thread_id=thread_id,
                    role=role,
                    text=text or "",
                    created_at=created_at,
                )
            )

        runtimes = {row[0]: row[1:] for row in runtime_rows}
        threads = []
        for thread_id, project_id, project_name, title, created_at in thread_rows:
            runtime = runtimes.get(thread_id)
            runtime_provider = runtime[0] if runtime else None
            runtime_session_id = _provider_session_id(runtime[1]) if runtime else None
            threads.append(
                T3Thread(
                    thread_id=thread_id,
                    project_id=project_id,
                    title=title,
                    created_at=created_at,
                    project_name=project_name,
                    messages=tuple(messages_by_thread.get(thread_id, ())),
                    provider_name=runtime_provider,
                    provider_session_id=runtime_session_id,
                    mirrored=runtime is not None,
                )
            )
        return threads

    @staticmethod
    def _read_v2_snapshot(conn: sqlite3.Connection) -> list[T3Thread]:
        """Read only the explicitly selected V2 projection; never fall back to legacy rows."""
        thread_rows = conn.execute(
            """
            SELECT t.thread_id, t.project_id, p.title, t.title, t.created_at,
                   t.default_provider, t.active_provider_thread_id
              FROM orchestration_v2_projection_threads AS t
              LEFT JOIN projection_projects AS p ON p.project_id = t.project_id
             ORDER BY t.created_at, t.thread_id
            """
        ).fetchall()
        message_rows = conn.execute(
            """
            SELECT message_id, thread_id, role, created_at, payload_json
              FROM orchestration_v2_projection_messages
             WHERE streaming = 0
             ORDER BY thread_id, created_at, message_id
            """
        ).fetchall()
        provider_rows = conn.execute(
            "SELECT provider_thread_id, provider, payload_json FROM orchestration_v2_projection_provider_threads"
        ).fetchall()
        providers = {}
        for provider_thread_id, provider, raw in provider_rows:
            payload = _v2_payload(raw, provider_thread_id)
            native_ref = payload.get("nativeThreadRef")
            if native_ref is not None and not isinstance(native_ref, dict):
                raise T3PayloadError(f"Invalid T3 V2 native thread reference for {provider_thread_id}")
            native_id = native_ref.get("nativeId") if native_ref is not None else None
            if native_id is not None and (
                not isinstance(native_id, str) or not native_id or native_id != native_id.strip()
            ):
                raise T3PayloadError(f"Invalid T3 V2 native thread ID for {provider_thread_id}")
            providers[provider_thread_id] = (provider, native_id)

        messages: dict[str, list[T3Message]] = {}
        for message_id, thread_id, role, created_at, raw in message_rows:
            payload = _v2_payload(raw, message_id)
            text = payload.get("text")
            if not isinstance(text, str):
                raise T3PayloadError(f"T3 V2 message text must be a string for {message_id}")
            messages.setdefault(thread_id, []).append(T3Message(message_id, thread_id, role, text, created_at))

        threads = []
        for thread_id, project_id, project_name, title, created_at, default_provider, active_provider in thread_rows:
            provider = providers.get(active_provider)
            threads.append(
                T3Thread(
                    thread_id=thread_id,
                    project_id=project_id,
                    title=title,
                    created_at=created_at,
                    project_name=project_name,
                    messages=tuple(messages.get(thread_id, ())),
                    provider_name=provider[0] if provider else default_provider,
                    provider_session_id=provider[1] if provider else None,
                    mirrored=provider is not None,
                )
            )
        return threads

    def _fail(self, code: str, message: str, context: dict[str, Any], error: BaseException) -> None:
        failure = {
            "code": code,
            "message": message,
            "error_type": type(error).__name__,
            "error": str(error),
            **context,
        }
        self._failures.append(failure)
        self._write_health(
            alerting=True, alert_reasons=["schema_drift" if code == "t3_schema_drift" else "reader_failure"]
        )
        try:
            raise_alarm(code, message, {"state_db_path": str(self.state_db_path), **context})
        except BrainLayerAlarm:
            raise
        raise error

    def _write_health(self, *, alerting: bool, alert_reasons: list[str]) -> None:
        if self.health_path is None:
            return
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "source": T3_SOURCE,
            "state_db_path": str(self.state_db_path),
            "projection_version": self.projection_version,
            "alerting": alerting,
            "alert_reasons": alert_reasons,
            "failures": list(self._failures),
            **self._last_counts,
        }
        try:
            self.health_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.health_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            temp_path.replace(self.health_path)
        except OSError:
            logger.exception("Failed to write T3 health snapshot to %s", self.health_path)


def _provider_session_id(resume_cursor_json: str | None) -> str | None:
    if not resume_cursor_json:
        return None
    try:
        value = json.loads(resume_cursor_json)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    provider_session_id = value.get("threadId")
    return str(provider_session_id) if provider_session_id else None


def _index_chunks(chunks, *, source_file: str, project: str | None, db_path: Path) -> int:
    from ..index_new import index_chunks_to_sqlite
    from ..system_prompt_guard import looks_like_system_prompt

    # Match the indexer's existing admission filter; an embedding batch can fail without raising.
    expected = sum(
        not chunk.metadata.get("is_system_prompt") and not looks_like_system_prompt(chunk.content) for chunk in chunks
    )
    indexed = index_chunks_to_sqlite(chunks, source_file=source_file, project=project, db_path=db_path)
    if indexed != expected:
        raise RuntimeError(
            f"T3 indexing incomplete: indexed {indexed} of {expected} eligible chunks; retain rows for replay"
        )
    return indexed


def _message_chunks(thread: T3Thread, message: T3Message) -> list:
    from ..pipeline.chunk import Chunk, chunk_content
    from ..pipeline.classify import ClassifiedContent, ContentType, ContentValue

    content_type = ContentType.USER_MESSAGE if message.role == "user" else ContentType.ASSISTANT_TEXT
    value = ContentValue.HIGH if message.role == "user" else ContentValue.MEDIUM
    base_metadata = {
        "chunk_id": f"t3:{thread.thread_id}:{message.message_id}:0",
        "conversation_id": thread.thread_id,
        "created_at": message.created_at,
        "allow_duplicate": True,
        "message_id": message.message_id,
        "project": thread.project_name,
        "provenance_class": T3_PROVENANCE_CLASS,
        "sender": message.role,
        "session_id": thread.thread_id,
        "source": T3_SOURCE,
        "source_uri": f"t3://thread/{thread.thread_id}/message/{message.message_id}",
        "t3_thread_id": thread.thread_id,
        "t3_title": thread.title,
        "t3_mirrored": thread.mirrored,
        "t3_provider_name": thread.provider_name,
        "t3_provider_session_id": thread.provider_session_id,
    }
    classified = ClassifiedContent(
        content=message.text,
        content_type=content_type,
        value=value,
        metadata=base_metadata,
    )
    chunks = chunk_content(classified)
    if not chunks:
        if not message.text.strip():
            return []
        chunks = [
            Chunk(
                content=message.text,
                content_type=content_type,
                value=value,
                metadata=base_metadata,
                char_count=len(message.text),
            )
        ]
    for index, chunk in enumerate(chunks):
        chunk.metadata = {
            **base_metadata,
            **chunk.metadata,
            "chunk_id": f"t3:{thread.thread_id}:{message.message_id}:{index}",
        }
    return chunks


def ingest_t3(
    state_db_path: Path | str = DEFAULT_T3_STATE_DB,
    *,
    db_path: Path | str | None = None,
    health_path: Path | str | None = DEFAULT_T3_HEALTH_PATH,
    dry_run: bool = False,
    projection_version: int = 1,
) -> T3IngestionResult:
    """Ingest all T3 threads, deliberately retaining mirrored content."""
    destination = Path(db_path).expanduser() if db_path is not None else None
    reader = T3Reader(state_db_path, health_path=health_path, projection_version=projection_version)
    threads = reader.read_threads()
    result = T3IngestionResult(
        threads_seen=len(threads),
        threads_ingested=len(threads),
        messages_seen=sum(len(thread.messages) for thread in threads),
        duplicates_accepted=sum(thread.mirrored for thread in threads),
    )
    chunks = []
    for thread in threads:
        for message in thread.messages:
            if message.role not in {"user", "assistant"}:
                result.messages_skipped["unsupported_role"] = result.messages_skipped.get("unsupported_role", 0) + 1
                continue
            message_chunks = _message_chunks(thread, message)
            if not message_chunks:
                result.messages_skipped["empty_text"] = result.messages_skipped.get("empty_text", 0) + 1
                continue
            result.messages_ingested += 1
            chunks.extend(message_chunks)
    result.chunks_planned = len(chunks)
    if chunks and not dry_run:
        if destination is None:
            from ..paths import DEFAULT_DB_PATH

            destination = DEFAULT_DB_PATH
        try:
            result.chunks_indexed = _index_chunks(
                chunks,
                source_file=str(Path(state_db_path).expanduser()),
                project=None,
                db_path=destination,
            )
        except BaseException as exc:
            # Preserve failure/cancellation semantics and any committed rows for stable-ID replay.
            # A successful projection read must not remain the final health result of a failed import.
            reader._last_counts["chunks_planned"] = result.chunks_planned
            reader._failures.append(
                {
                    "code": "t3_indexing_failure",
                    "message": "T3 chunk indexing did not complete",
                    "error_type": type(exc).__name__,
                }
            )
            reader._write_health(alerting=True, alert_reasons=["indexing_failure"])
            raise
    if reader.health_path is not None:
        reader._last_counts.update(
            {
                "threads_ingested": result.threads_ingested,
                "messages_ingested": result.messages_ingested,
                "chunks_planned": result.chunks_planned,
                "chunks_indexed": result.chunks_indexed,
                "duplicates_accepted": result.duplicates_accepted,
                "messages_skipped": sum(result.messages_skipped.values()),
            }
        )
        reader._write_health(alerting=False, alert_reasons=[])
    return result
