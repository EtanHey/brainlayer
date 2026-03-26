"""brain_enrich MCP handler — unified enrichment through a single tool."""

import asyncio
import json
import logging

from mcp.types import CallToolResult, TextContent

from ._shared import _error_result, _get_vector_store

logger = logging.getLogger(__name__)


async def _brain_enrich(
    mode: str = "realtime",
    limit: int = 25,
    since_hours: int = 24,
    backend: str = "mlx",
    parallel: int = 2,
    phase: str = "run",
    chunk_ids: list[str] | None = None,
    stats: bool = False,
) -> CallToolResult:
    """Handle brain_enrich tool call.

    Modes:
        realtime — Gemini 2.5 Flash-Lite, single-chunk, <600ms target
        batch   — Gemini Batch API, backlog processing
        local   — MLX/Ollama backend, offline/privacy mode
    """
    if mode not in ("realtime", "batch", "local"):
        return _error_result(f"Unknown mode: {mode}. Use realtime, batch, or local.")

    try:
        store = _get_vector_store()

        if stats:
            return await _enrich_stats(store)

        loop = asyncio.get_event_loop()

        if mode == "realtime":
            from ..enrichment_controller import enrich_realtime

            result = await loop.run_in_executor(
                None,
                lambda: enrich_realtime(
                    store=store,
                    limit=limit,
                    since_hours=since_hours,
                    chunk_ids=chunk_ids,
                ),
            )
        elif mode == "batch":
            from ..enrichment_controller import enrich_batch

            result = await loop.run_in_executor(
                None,
                lambda: enrich_batch(
                    store=store,
                    phase=phase,
                    limit=limit,
                ),
            )
        else:  # local
            from ..enrichment_controller import enrich_local

            result = await loop.run_in_executor(
                None,
                lambda: enrich_local(
                    store=store,
                    limit=limit,
                    parallel=parallel,
                    backend=backend,
                ),
            )

        output = {
            "mode": result.mode,
            "attempted": result.attempted,
            "enriched": result.enriched,
            "skipped": result.skipped,
            "failed": result.failed,
        }
        if result.errors:
            output["errors"] = result.errors[:10]  # Cap error list

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(output, indent=2))]
        )

    except Exception as e:
        logger.error("brain_enrich failed: %s", e)
        return _error_result(f"brain_enrich error: {e}")


async def _enrich_stats(store) -> CallToolResult:
    """Return enrichment progress statistics."""
    try:
        cursor = store._read_cursor()

        # Total chunks
        total = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Enriched
        enriched = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NOT NULL"
        ).fetchone()[0]

        # Unenriched (eligible — char_count >= 50)
        unenriched = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL AND char_count >= 50"
        ).fetchone()[0]

        # Skipped (too short)
        skipped = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL AND char_count < 50"
        ).fetchone()[0]

        # Recent enrichments (last 24h)
        recent = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at > datetime('now', '-24 hours')"
        ).fetchone()[0]

        result = {
            "total_chunks": total,
            "enriched": enriched,
            "unenriched_eligible": unenriched,
            "skipped_too_short": skipped,
            "enriched_pct": round(enriched / total * 100, 1) if total > 0 else 0,
            "enriched_last_24h": recent,
        }
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )
    except Exception as e:
        return _error_result(f"Stats query failed: {e}")
