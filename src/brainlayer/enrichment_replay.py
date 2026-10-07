"""Local replay of historical enrichment metadata; no model production."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

from .chunk_write import canonical_content_hash
from .provenance import derive_provenance_class
from .provenance_autosupersede import auto_supersede

logger = logging.getLogger(__name__)
_content_hash = canonical_content_hash
GEMINI_REALTIME_MODEL = os.environ.get("BRAINLAYER_GEMINI_REALTIME_MODEL", "gemini-2.5-flash-lite")


def _current_enrichment_backend() -> str:
    tier = _get_gemini_service_tier()
    return f"gemini-{tier}" if tier else "gemini"


def _current_auto_supersede_dry_run() -> bool | None:
    raw = str(os.environ.get("BRAINLAYER_AUTO_SUPERSEDE") or "").strip().lower()
    if raw in {"", "0", "false", "off", "no"}:
        return None
    return raw != "apply"


def _entity_name_from_payload(entity: Any) -> str:
    if isinstance(entity, dict):
        for key in ("name", "text", "entity", "label"):
            value = entity.get(key)
            if value:
                return str(value).strip()
        return ""
    return str(entity or "").strip()


def _maybe_auto_supersede_ingested_chunk(
    store,
    chunk: dict[str, Any],
    entities: list[Any],
    *,
    provenance_class: str,
) -> None:
    dry_run = _current_auto_supersede_dry_run()
    if dry_run is None:
        return

    for entity in entities:
        entity_name = _entity_name_from_payload(entity)
        if not entity_name:
            continue
        new_chunk = dict(chunk)
        new_chunk["entity"] = entity_name
        new_chunk["provenance_class"] = provenance_class
        try:
            report = auto_supersede(store, new_chunk, dry_run=dry_run, commit=False)
        except Exception:
            logger.exception("auto_supersede failed for chunk=%s entity=%s", chunk.get("id"), entity_name)
            continue
        mode = "dry_run" if dry_run else "apply"
        if (
            report.candidate_count
            or report.contradiction_count
            or report.would_supersede_count
            or report.pending_confirm_count
            or report.skipped_count
        ):
            logger.info(
                "auto_supersede %s entity=%s candidates=%s contradictions=%s would_supersede=%s superseded=%s "
                "pending_confirm=%s skipped=%s",
                mode,
                report.entity,
                report.candidate_count,
                report.contradiction_count,
                report.would_supersede_count,
                report.superseded_count,
                report.pending_confirm_count,
                report.skipped_reason or report.skipped_count,
            )


@contextmanager
def _savepoint(conn, name: str):
    cursor = conn.cursor()
    cursor.execute(f"SAVEPOINT {name}")
    try:
        yield
    except Exception:
        cursor.execute(f"ROLLBACK TO SAVEPOINT {name}")
        cursor.execute(f"RELEASE SAVEPOINT {name}")
        raise
    else:
        cursor.execute(f"RELEASE SAVEPOINT {name}")


def _get_gemini_service_tier() -> str:
    return os.environ.get("BRAINLAYER_GEMINI_SERVICE_TIER", "flex")


def _ensure_raw_entities_json_column(store) -> bool:
    """Ensure the raw_entities_json staging column exists on chunks."""
    try:
        store.conn.cursor().execute("SELECT raw_entities_json FROM chunks LIMIT 0")
        return True
    except Exception:
        try:
            store.conn.cursor().execute("ALTER TABLE chunks ADD COLUMN raw_entities_json TEXT")
            return True
        except Exception:
            return False


def _ensure_provenance_class_column(store) -> bool:
    """Ensure the provenance_class staging column exists on chunks."""
    try:
        store.conn.cursor().execute("SELECT provenance_class FROM chunks LIMIT 0")
        setattr(store, "_has_provenance_class", True)
        return True
    except Exception:
        try:
            store.conn.cursor().execute("ALTER TABLE chunks ADD COLUMN provenance_class TEXT")
            setattr(store, "_has_provenance_class", True)
            return True
        except Exception:
            return False


def _derive_chunk_provenance_class(chunk: dict[str, Any], content: str | None = None) -> str:
    source = str(chunk.get("source") or "").strip().lower()
    source_file = str(chunk.get("source_file") or "").strip().lower()
    if (source == "manual" and source_file in {"brainlayer-store", "brainbar-store", "brainlayer-queue"}) or (
        source == "mcp" and source_file == "brainlayer-queue"
    ):
        return "RAW-ETAN-DIRECT"
    text = (chunk.get("content") or "") if content is None else (content or "")
    return derive_provenance_class(
        content_type=chunk.get("content_type"),
        sender=chunk.get("sender"),
        text=text,
        prev_assistant_text=chunk.get("prev_assistant_text"),
    )
