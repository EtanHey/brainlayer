"""Exercise actual installed retirement and saved-data entrypoints, without models."""

import asyncio
import json
from importlib import import_module


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def installed_surfaces(home):
    from typer.testing import CliRunner

    from brainlayer.chunk_write import insert_canonical_chunk
    from brainlayer.cli import app
    from brainlayer.drain import drain_once
    from brainlayer.enrichment_replay import _apply_enrichment
    from brainlayer.mcp.enrich_handler import _brain_enrich
    from brainlayer.vector_store import VectorStore
    from brainlayer.watcher_bridge import create_flush_callback

    # This module is installed by wheel force-include, not present under source src/.
    hotlane = import_module("brainlayer.launchd.hotlane_brainbar_daemon")
    surfaces = []
    for args in (["enrich"], ["enrich", "--mode", "batch", "--phase", "submit"], ["enrich", "--stats"]):
        result = CliRunner().invoke(app, args)
        require(result.exit_code == 1 and "retired" in result.output, "Installed CLI shim failed")
    surfaces.append("cli-retirement")
    for options in ({"mode": "realtime"}, {"mode": "batch", "phase": "submit"}, {"stats": True}):
        result = asyncio.run(_brain_enrich(**options))
        require(result.is_error and "retired" in str(result.content), "Library MCP shim failed")
    surfaces.append("library-mcp-retirement")
    path = home / "surfaces.db"
    store = VectorStore(path)
    try:
        insert_canonical_chunk(
            store.conn,
            {
                "id": "r11-hot",
                "content": "R11 local embedding and historical result fixture",
                "metadata": {},
                "source_file": "brainbar-store",
                "source": "mcp",
                "created_at": "2099-01-01",
            },
        )

        def retired_callback(*args, **kwargs):
            raise RuntimeError("Retired hotlane callback executed")

        cycle = hotlane.run_cycle(
            store=store,
            embed_fn=lambda text: [0.125] * 1024,
            recent_limit=5,
            backlog_batch=0,
            enrich_limit=25,
            enrich_since_hours=8760,
            enrich_fn=retired_callback,
        )
        require(
            cycle.embedded == 1 and cycle.enriched == cycle.enrich_attempted == 0,
            "Hotlane retired callback/vector contract failed",
        )
        require(
            store.conn.execute("SELECT id FROM chunk_vectors_rowids WHERE id='r11-hot'").fetchone(),
            "Hotlane vector missing",
        )
        surfaces.append("installed-hotlane-vector")
        chunk = store.get_chunk("r11-hot")
        _apply_enrichment(store, chunk, {"summary": "R11 historical replay", "tags": ["fixture"], "entities": []})
        require(store.get_chunk("r11-hot")["summary"] == "R11 historical replay", "Historical replay failed")
        surfaces.append("historical-replay")
    finally:
        store.close()
    queue = home / "surface-queue"
    queue.mkdir()
    event = queue / "001.jsonl"
    event.write_text(
        json.dumps(
            {
                "kind": "enrichment_update",
                "chunk_id": "r11-hot",
                "enrichment": {"summary": "R11 drained historical result"},
                "entities": [],
            }
        )
        + "\n"
    )
    count = drain_once(
        db_path=path,
        queue_dir=queue,
        log_path=home / "drain.log",
        pause_sentinel_path=home / "absent-pause",
        embed_fn=lambda text: [0.125] * 1024,
    )
    require(count == 1 and not event.exists(), "Actual saved-result drain failed")
    store = VectorStore(path)
    try:
        require(store.get_chunk("r11-hot")["summary"] == "R11 drained historical result", "Drained result not durable")
    finally:
        store.close()
    surfaces.append("saved-result-drain")
    # Arbitrated empty flush exercises the actual watcher callback without opening
    # another long-lived direct writer or manufacturing transcript data.
    flush = create_flush_callback(home / "watcher.db", arbitrated=True)
    watermarks = flush([])
    require(watermarks.inserted == 0, "Empty watcher callback failed")
    surfaces.append("watcher-flush")
    surfaces.append("drive-oauth-imports")
    return surfaces
