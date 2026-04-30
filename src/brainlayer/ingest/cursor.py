"""Cursor session ingestion adapter for BrainLayer."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

_MIN_USER_LEN = 15
_MIN_ASSISTANT_LEN = 30

_USER_QUERY_RE = re.compile(r"^\s*<user_query>\s*(.*?)\s*</user_query>\s*$", re.DOTALL)


def _extract_text(content) -> str:
    """Extract text content from Cursor message blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return ""


def _normalize_project_name(raw: str) -> str:
    """Convert encoded path-like project names to the repo leaf."""
    if not raw:
        return raw

    for marker in ("-Gits-", "-Desktop-", "-projects-", "-config-"):
        if marker in raw:
            return raw.split(marker)[-1]

    return raw


def _extract_project_from_path(file_path: Path) -> Optional[str]:
    parts = file_path.parts
    if "projects" in parts:
        idx = parts.index("projects") + 1
        if idx < len(parts):
            return _normalize_project_name(parts[idx])
    return None


def _file_mtime_iso(file_path: Path) -> str:
    return datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat()


def _clean_user_text(text: str) -> str:
    stripped = text.strip()
    match = _USER_QUERY_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def parse_cursor_session(file_path: Path) -> Iterator[dict]:
    """Parse a Cursor agent transcript JSONL into normalized entries."""
    session_id = file_path.stem
    project = _extract_project_from_path(file_path)
    fallback_ts = _file_mtime_iso(file_path)

    with open(file_path, "rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                line = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue

            role = line.get("role")
            message = line.get("message") or {}
            text = _extract_text(message.get("content")).strip()
            if not text:
                continue

            if role == "user":
                text = _clean_user_text(text)
                if len(text) < _MIN_USER_LEN:
                    continue
                yield {
                    "content": text,
                    "content_type": "user_message",
                    "session_id": session_id,
                    "timestamp": fallback_ts,
                    "project": project,
                    "source": "cursor",
                    "metadata": {
                        "session_id": session_id,
                        "sender": "user",
                        "source_file": str(file_path),
                    },
                }
                continue

            if role == "assistant":
                ctype = "ai_code" if "```" in text else "assistant_text"
                if ctype != "ai_code" and len(text) < _MIN_ASSISTANT_LEN:
                    continue
                yield {
                    "content": text,
                    "content_type": ctype,
                    "session_id": session_id,
                    "timestamp": fallback_ts,
                    "project": project,
                    "source": "cursor",
                    "metadata": {
                        "session_id": session_id,
                        "sender": "assistant",
                        "source_file": str(file_path),
                    },
                }


def ingest_cursor_session(
    file_path: Path,
    db_path: Optional[Path] = None,
    project_override: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Ingest a single Cursor session transcript."""
    from ..index_new import index_chunks_to_sqlite
    from ..pipeline.chunk import chunk_content
    from ..pipeline.classify import ClassifiedContent, ContentType, ContentValue

    type_map = {
        "user_message": (ContentType.USER_MESSAGE, ContentValue.HIGH),
        "assistant_text": (ContentType.ASSISTANT_TEXT, ContentValue.MEDIUM),
        "ai_code": (ContentType.AI_CODE, ContentValue.HIGH),
    }

    all_chunks: List = []
    project = project_override
    session_id: Optional[str] = None
    session_ts: Optional[str] = None

    for entry in parse_cursor_session(file_path):
        if project is None:
            project = entry.get("project")
        if session_id is None:
            session_id = entry.get("session_id")
        if session_ts is None:
            session_ts = entry.get("timestamp")

        content_type, value = type_map.get(entry["content_type"], (ContentType.ASSISTANT_TEXT, ContentValue.MEDIUM))
        classified = ClassifiedContent(
            content=entry["content"],
            content_type=content_type,
            value=value,
            metadata={**entry.get("metadata", {}), "source": "cursor"},
        )
        chunks = chunk_content(classified)
        all_chunks.extend(chunks)
        if verbose:
            print(f"  [{entry['content_type']}] {entry['content'][:80]!r}")

    if not all_chunks:
        logger.info("No indexable content in %s", file_path)
        return 0

    if dry_run:
        print(f"Dry run: {len(all_chunks)} chunks from {file_path.name} (not stored)")
        return len(all_chunks)

    if db_path is None:
        from ..paths import DEFAULT_DB_PATH

        db_path = DEFAULT_DB_PATH

    for chunk in all_chunks:
        chunk.metadata.setdefault("source", "cursor")

    indexed = index_chunks_to_sqlite(
        all_chunks,
        source_file=str(file_path),
        project=project,
        db_path=db_path,
        created_at=session_ts,
    )

    if session_id:
        try:
            from ..vector_store import VectorStore

            with VectorStore(db_path) as store:
                store.store_session_context(
                    session_id=session_id,
                    project=project or "cursor",
                    started_at=session_ts,
                    ended_at=None,
                )
        except Exception as exc:
            logger.debug("Could not store session context for %s: %s", session_id, exc)

    logger.info("Indexed %d chunks from %s (session %s, project %s)", indexed, file_path.name, session_id, project)
    return indexed


def ingest_cursor_dir(
    sessions_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    project_override: Optional[str] = None,
    since_days: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, int]:
    """Ingest Cursor agent transcript files under ~/.cursor/projects."""
    if sessions_dir is None:
        sessions_dir = Path.home() / ".cursor" / "projects"

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Cursor projects directory not found: {sessions_dir}")

    jsonl_files = sorted(sessions_dir.glob("**/agent-transcripts/**/*.jsonl"))
    if since_days is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - since_days * 86400
        jsonl_files = [f for f in jsonl_files if f.stat().st_mtime >= cutoff]

    if not jsonl_files:
        logger.info("No Cursor transcript files found in %s", sessions_dir)
        return 0, 0

    if not dry_run and db_path is None:
        from ..paths import DEFAULT_DB_PATH

        db_path = DEFAULT_DB_PATH

    already_indexed: set[str] = set()
    if not dry_run and db_path and db_path.exists():
        try:
            from ..vector_store import VectorStore

            with VectorStore(db_path) as store:
                cursor = store._read_cursor()
                rows = cursor.execute("SELECT DISTINCT source_file FROM chunks WHERE source = 'cursor'")
                already_indexed = {row[0] for row in rows}
        except Exception as exc:
            logger.debug("Could not check existing cursor chunks: %s", exc)

    files_processed = 0
    total_chunks = 0

    for file_path in jsonl_files:
        if str(file_path) in already_indexed:
            logger.debug("Skipping already-indexed %s", file_path.name)
            continue
        try:
            total_chunks += ingest_cursor_session(
                file_path,
                db_path=db_path,
                project_override=project_override,
                dry_run=dry_run,
                verbose=verbose,
            )
            files_processed += 1
        except Exception as exc:
            logger.warning("Failed to ingest %s: %s", file_path.name, exc)

    return files_processed, total_chunks
