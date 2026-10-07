"""Local replay of historical enrichment metadata; no model production."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any

from .chunk_write import canonical_content_hash
from .pipeline.cloud_scrub import scrub_llm_output
from .provenance import derive_provenance_class
from .provenance_autosupersede import auto_supersede
from .provenance_integration import enqueue_provenance_resolution_for_entities
from .writer_telemetry import start_writer_span

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


def _apply_enrichment_impl(
    store,
    chunk: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    chunk_origin: str | None = None,
    enrichment_model: str | None = None,
    enrichment_backend: str | None = None,
) -> None:
    should_promote_raw_entities = False
    enrichment = scrub_llm_output(enrichment)
    with _savepoint(store.conn, "enrichment_apply_provenance"):
        resolved_queries = enrichment.get("resolved_queries")
        legacy_resolved_query = enrichment.get("resolved_query")
        if not legacy_resolved_query and isinstance(resolved_queries, list) and resolved_queries:
            legacy_resolved_query = resolved_queries[0]
        model = str(enrichment_model or GEMINI_REALTIME_MODEL or "").strip()
        _ = chunk_origin

        update_kwargs = {
            "chunk_id": chunk["id"],
            "summary": enrichment.get("summary"),
            "tags": enrichment.get("tags"),
            "importance": enrichment.get("importance"),
            "intent": enrichment.get("intent"),
            "primary_symbols": enrichment.get("primary_symbols"),
            "resolved_query": legacy_resolved_query,
            "epistemic_level": enrichment.get("epistemic_level"),
            "version_scope": enrichment.get("version_scope"),
            "debt_impact": enrichment.get("debt_impact"),
            "external_deps": enrichment.get("external_deps"),
            "key_facts": enrichment.get("key_facts"),
            "resolved_queries": resolved_queries,
            "sentiment_label": enrichment.get("sentiment_label"),
            "sentiment_score": enrichment.get("sentiment_score"),
            "sentiment_signals": enrichment.get("sentiment_signals"),
            "enrichment_model": model or GEMINI_REALTIME_MODEL,
            "enrichment_backend": enrichment_backend or _current_enrichment_backend(),
        }
        enrichment_version = (enrichment.get("enrichment_metadata") or {}).get("prompt_version")
        if enrichment_version:
            update_kwargs["enrichment_version"] = enrichment_version
        store.update_enrichment(**update_kwargs)
        entities = enrichment.get("entities", [])
        # AIDEV-NOTE: raw entities persisted to chunks.raw_entities_json staging column;
        # R84b canonicalization pipeline will consume and populate kg_entities downstream.
        if _ensure_raw_entities_json_column(store):
            store.conn.cursor().execute(
                "UPDATE chunks SET raw_entities_json = ? WHERE id = ?",
                (json.dumps(entities), chunk["id"]),
            )
            try:
                from .vector_store import VectorStore

                should_promote_raw_entities = isinstance(store, VectorStore)
            except Exception:
                should_promote_raw_entities = False
        # Set content_hash after enrichment so dedup works next time
        content = chunk.get("content", "")
        if content:
            try:
                h = _content_hash(content)
                store.conn.cursor().execute("UPDATE chunks SET content_hash = ? WHERE id = ?", (h, chunk["id"]))
            except Exception:
                pass  # Non-critical — dedup still works on next index
        provenance_class = _derive_chunk_provenance_class(chunk, content)
        if _ensure_provenance_class_column(store):
            store.conn.cursor().execute(
                "UPDATE chunks SET provenance_class = ? WHERE id = ?",
                (provenance_class, chunk["id"]),
            )
        _maybe_auto_supersede_ingested_chunk(store, chunk, entities, provenance_class=provenance_class)
        enqueue_provenance_resolution_for_entities(store, entities, chunk_id=chunk["id"], commit=False)
    if should_promote_raw_entities:
        try:
            from .kg_promotion import promote_chunk_raw_entities

            promote_chunk_raw_entities(store, chunk["id"])
        except Exception:
            logger.debug("raw entity KG promotion skipped for %s", chunk["id"], exc_info=True)


def _apply_enrichment(
    store,
    chunk: dict[str, Any],
    enrichment: dict[str, Any],
    *,
    chunk_origin: str | None = None,
    enrichment_model: str | None = None,
    enrichment_backend: str | None = None,
) -> None:
    try:
        owns_transaction = bool(store.conn.getautocommit())
    except Exception:
        owns_transaction = True
    telemetry_span = start_writer_span(
        store.conn,
        db_path=getattr(store, "db_path", None),
        producer="enrichment",
        lane="enrichment",
        operation="apply",
        rows_planned=1,
        transaction_mode="savepoint",
    )
    try:
        _apply_enrichment_impl(
            store,
            chunk,
            enrichment,
            chunk_origin=chunk_origin,
            enrichment_model=enrichment_model,
            enrichment_backend=enrichment_backend,
        )
    except Exception as exc:
        telemetry_span.finish("rollback", error=f"{type(exc).__name__}: {exc}")
        raise
    telemetry_span.finish("commit" if owns_transaction else "completed")
