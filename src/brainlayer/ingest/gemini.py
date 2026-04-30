"""Gemini CLI session ingestion adapter for BrainLayer."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

_MIN_USER_LEN = 15
_MIN_ASSISTANT_LEN = 30


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(part for part in parts if part)
    return ""


def _extract_project_from_path(file_path: Path) -> Optional[str]:
    parts = file_path.parts
    if "tmp" in parts:
        idx = parts.index("tmp") + 1
        if idx < len(parts):
            return parts[idx]
    return None


def _file_mtime_iso(file_path: Path) -> str:
    return datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat()


def parse_gemini_session(file_path: Path) -> Iterator[dict]:
    """Parse a Gemini session JSON into normalized entries."""
    with open(file_path) as fh:
        payload = json.load(fh)

    session_id = payload.get("sessionId") or file_path.stem
    project = _extract_project_from_path(file_path)
    session_start = payload.get("startTime") or _file_mtime_iso(file_path)

    for message in payload.get("messages") or []:
        message_type = message.get("type")
        if message_type not in {"user", "gemini"}:
            continue

        text = _extract_text(message.get("content")).strip()
        if not text:
            continue

        if message_type == "user":
            if len(text) < _MIN_USER_LEN:
                continue
            yield {
                "content": text,
                "content_type": "user_message",
                "session_id": session_id,
                "timestamp": message.get("timestamp") or session_start,
                "project": project,
                "source": "gemini",
                "metadata": {
                    "session_id": session_id,
                    "sender": "user",
                    "source_file": str(file_path),
                },
            }
            continue

        ctype = "ai_code" if "```" in text else "assistant_text"
        if ctype != "ai_code" and len(text) < _MIN_ASSISTANT_LEN:
            continue
        yield {
            "content": text,
            "content_type": ctype,
            "session_id": session_id,
            "timestamp": message.get("timestamp") or session_start,
            "project": project,
            "source": "gemini",
            "metadata": {
                "session_id": session_id,
                "sender": "assistant",
                "model": message.get("model"),
                "source_file": str(file_path),
            },
        }


def ingest_gemini_session(
    file_path: Path,
    db_path: Optional[Path] = None,
    project_override: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """Ingest a single Gemini session JSON file."""
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

    for entry in parse_gemini_session(file_path):
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
            metadata={**entry.get("metadata", {}), "source": "gemini"},
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
        chunk.metadata.setdefault("source", "gemini")

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
                    project=project or "gemini",
                    started_at=session_ts,
                    ended_at=None,
                )
        except Exception as exc:
            logger.debug("Could not store session context for %s: %s", session_id, exc)

    logger.info("Indexed %d chunks from %s (session %s, project %s)", indexed, file_path.name, session_id, project)
    return indexed


def ingest_gemini_dir(
    sessions_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
    project_override: Optional[str] = None,
    since_days: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[int, int]:
    """Ingest Gemini session files under ~/.gemini/tmp."""
    if sessions_dir is None:
        sessions_dir = Path.home() / ".gemini" / "tmp"

    if not sessions_dir.exists():
        raise FileNotFoundError(f"Gemini sessions directory not found: {sessions_dir}")

    session_files = sorted(sessions_dir.glob("**/chats/session-*.json"))
    if since_days is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - since_days * 86400
        session_files = [f for f in session_files if f.stat().st_mtime >= cutoff]

    if not session_files:
        logger.info("No Gemini session files found in %s", sessions_dir)
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
                rows = cursor.execute("SELECT DISTINCT source_file FROM chunks WHERE source = 'gemini'")
                already_indexed = {row[0] for row in rows}
        except Exception as exc:
            logger.debug("Could not check existing gemini chunks: %s", exc)

    files_processed = 0
    total_chunks = 0

    for file_path in session_files:
        if str(file_path) in already_indexed:
            logger.debug("Skipping already-indexed %s", file_path.name)
            continue
        try:
            total_chunks += ingest_gemini_session(
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
