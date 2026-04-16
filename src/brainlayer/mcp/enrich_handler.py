"""brain_enrich MCP handler — unified enrichment through a single tool."""

import asyncio
import logging
import os

from mcp.types import CallToolResult, TextContent

from ._format import format_digest_result
from ._shared import _error_result, _get_vector_store

DEFAULT_REALTIME_ENRICH_SINCE_HOURS = int(
    os.environ.get("BRAINLAYER_DEFAULT_ENRICH_SINCE_HOURS", "8760")
)

logger = logging.getLogger(__name__)


async def _brain_enrich(
    mode: str = "realtime",
    limit: int = 25,
    since_hours: int = DEFAULT_REALTIME_ENRICH_SINCE_HOURS,
    phase: str = "run",
    chunk_ids: list[str] | None = None,
    stats: bool = False,
) -> CallToolResult:
    """Handle brain_enrich tool call.

    Modes:
        realtime — Gemini 2.5 Flash-Lite, single-chunk, <600ms target
        batch   — Gemini Batch API, backlog processing
    """
    if mode not in ("realtime", "batch"):
        return _error_result(f"Unknown mode: {mode}. Use realtime or batch.")

    try:
        store = _get_vector_store()

        if stats:
            return await _enrich_stats(store)

        loop = asyncio.get_running_loop()

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

        output = {
            "mode": result.mode,
            "attempted": result.attempted,
            "enriched": result.enriched,
            "skipped": result.skipped,
            "failed": result.failed,
        }
        if result.errors:
            output["errors"] = result.errors[:10]  # Cap error list

        formatted = format_digest_result(output)
        return CallToolResult(content=[TextContent(type="text", text=formatted)])

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
        enriched = cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_at IS NOT NULL").fetchone()[0]

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
        pct = result["enriched_pct"]
        lines = [
            "\u250c\u2500 Enrichment Stats",
            f"\u2502 Total: {total:,}  Enriched: {enriched:,} ({pct}%)  Remaining: {unenriched:,}  Skipped: {skipped:,}",
            f"\u2502 Last 24h: {recent:,} enriched",
            "\u2514\u2500",
        ]
        return CallToolResult(content=[TextContent(type="text", text="\n".join(lines))])
    except Exception as e:
        return _error_result(f"Stats query failed: {e}")
