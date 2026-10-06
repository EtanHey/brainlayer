"""Retired enrichment handler compatibility and historical statistics."""

from mcp.types import CallToolResult, TextContent

from ._shared import _error_result


async def _brain_enrich(
    *_legacy_args: object,
    **_legacy_options: object,
) -> CallToolResult:
    """Accept legacy options solely to return a transport-free retirement error."""
    return _error_result("Enrichment has been retired. Local store, embeddings and search remain available.")


async def _enrich_stats(store) -> CallToolResult:
    """Return enrichment progress statistics."""
    try:
        cursor = store._read_cursor()

        # Total chunks
        total = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        # Enriched
        enriched = cursor.execute("SELECT COUNT(*) FROM chunks WHERE enrich_status = 'success'").fetchone()[0]

        # Unenriched (eligible — char_count >= 50)
        unenriched = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enriched_at IS NULL AND enrich_status IS NULL AND char_count >= 50"
        ).fetchone()[0]

        # Skipped terminal statuses
        skipped = cursor.execute(
            "SELECT COUNT(*) FROM chunks WHERE enrich_status IS NOT NULL AND enrich_status != 'success'"
        ).fetchone()[0]

        # Recent enrichments (last 24h)
        recent = cursor.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE datetime(enriched_at) > datetime('now', '-24 hours')
              AND enrich_status = 'success'
            """
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
