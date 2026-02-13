"""FastAPI daemon service for fast zikaron queries + dashboard API."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import apsw
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .vector_store import VectorStore
from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zikaron" / "zikaron.db"
SOCKET_PATH = Path("/tmp/zikaron.sock")
BRAIN_DIR = Path.home() / ".golems-brain"
API_COSTS_PATH = Path.home() / ".golems-zikaron" / "api_costs.jsonl"

# Global state
vector_store: Optional[VectorStore] = None
embedding_model = None
http_port: Optional[int] = None


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    n_results: int = 10
    project_filter: Optional[str] = None
    content_type_filter: Optional[str] = None
    source_filter: Optional[str] = None
    use_semantic: bool = True
    hybrid: bool = True


class SearchResponse(BaseModel):
    """Search response model."""
    ids: List[Optional[str]] = []
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[Optional[float]]
    total_time_ms: float


class StatsResponse(BaseModel):
    """Stats response model."""
    total_chunks: int
    projects: List[str]
    content_types: List[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan. Guards against double-init in dual server mode."""
    global vector_store, embedding_model

    # Guard: only initialize once (dual servers share the same app)
    if vector_store is not None:
        yield
        return

    # Startup
    logger.info("Starting zikaron daemon...")

    vector_store = VectorStore(DEFAULT_DB_PATH)
    logger.info(f"Loaded vector store: {vector_store.count()} chunks")

    embedding_model = get_embedding_model()
    logger.info(f"Loaded embedding model: {embedding_model.model_name}")

    try:
        embedding_model.embed_query("test query")
        logger.info("Model warmed up successfully")
    except Exception as e:
        logger.warning(f"Model warmup failed: {e}")

    yield

    # Shutdown (only close once)
    logger.info("Shutting down zikaron daemon...")
    if vector_store:
        vector_store.close()
        vector_store = None


app = FastAPI(
    title="Zikaron Daemon",
    description="Fast search daemon + dashboard API for zikaron knowledge base",
    version="0.2.0",
    lifespan=lifespan
)

# CORS — allow dashboard origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://etanheyman.com",
        "https://www.etanheyman.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Existing endpoints (search, stats, context)
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "chunks": vector_store.count() if vector_store else 0}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get collection statistics."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    stats = vector_store.get_stats()
    return StatsResponse(**stats)


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search the knowledge base."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    start_time = time.time()

    try:
        if request.hybrid and request.use_semantic:
            query_embedding = embedding_model.embed_query(request.query)
            results = vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=request.query,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter
            )
        elif request.use_semantic:
            query_embedding = embedding_model.embed_query(request.query)
            results = vector_store.search(
                query_embedding=query_embedding,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter
            )
        else:
            results = vector_store.search(
                query_text=request.query,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter
            )

        total_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            ids=results.get("ids", [[]])[0],
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
            total_time_ms=total_time_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/{chunk_id}")
async def get_context(chunk_id: str, before: int = 3, after: int = 3):
    """Get surrounding conversation context for a chunk."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    try:
        result = vector_store.get_context(chunk_id, before=before, after=after)
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Context lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Brain View endpoints
# ──────────────────────────────────────────────

@app.get("/brain/graph")
async def brain_graph():
    """Serve pre-generated brain graph.json."""
    graph_path = BRAIN_DIR / "graph.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="graph.json not found. Run: zikaron brain-export")
    return FileResponse(graph_path, media_type="application/json")


@app.get("/brain/metadata")
async def brain_metadata():
    """Stats about the brain graph (node count, last generated, etc)."""
    meta_path = BRAIN_DIR / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found. Run: zikaron brain-export")
    with open(meta_path) as f:
        return json.load(f)


@app.get("/brain/node/{node_id}")
async def brain_node_detail(node_id: str):
    """Detail for a specific brain graph node — sessions, files, operations."""
    graph_path = BRAIN_DIR / "graph.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="graph.json not found")

    with open(graph_path) as f:
        graph = json.load(f)

    for node in graph.get("nodes", []):
        if node.get("id") == node_id:
            return node

    raise HTTPException(status_code=404, detail=f"Node {node_id} not found")


# ──────────────────────────────────────────────
# Health / Service status endpoints
# ──────────────────────────────────────────────

@app.get("/health/services")
async def health_services():
    """Check status of ecosystem services: Ollama, Telegram bot, Railway, launchd."""

    def _check_service(cmd: List[str], timeout: int = 3) -> str:
        """Run a subprocess check (called via asyncio.to_thread to avoid blocking)."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return result.stdout.strip()
        except Exception:
            return ""

    # Run all checks concurrently via thread pool (non-blocking)
    ollama_fut = asyncio.to_thread(
        _check_service,
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:11434/api/tags"], 3
    )
    telegram_fut = asyncio.to_thread(
        _check_service,
        ["launchctl", "list", "com.golemszikaron.telegram"], 3
    )
    railway_fut = asyncio.to_thread(
        _check_service,
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
         "https://helpful-empathy-production-482d.up.railway.app/health"], 5
    )

    # Check all launchd services in one call (fast)
    launchd_fut = asyncio.to_thread(
        _check_service,
        ["bash", "-c", "launchctl list 2>/dev/null | grep -E 'golem|zikaron' || true"], 3
    )

    ollama_code, telegram_out, railway_code, launchd_out = await asyncio.gather(
        ollama_fut, telegram_fut, railway_fut, launchd_fut
    )

    # Parse launchd services
    launchd_services = {
        "nightshift": "com.golemszikaron.nightshift",
        "briefing": "com.golemszikaron.briefing",
        "healthcheck": "com.golemszikaron.healthcheck",
        "compactor": "com.golemszikaron.compactor",
        "bedtime_guardian": "com.golems.bedtime-guardian",
        "session_archiver": "com.golems.session-archiver",
        "auto_index": "com.golems.auto-index",
    }
    # Parse launchd list output — format: "PID\tExitStatus\tLabel"
    # PID is "-" for scheduled services not currently running (normal for cron-like jobs)
    launchd_statuses = {}
    for name, label in launchd_services.items():
        found = False
        for line in launchd_out.splitlines():
            if label in line:
                found = True
                parts = line.split("\t")
                pid = parts[0].strip() if parts else "-"
                exit_status = parts[1].strip() if len(parts) > 1 else "0"
                if pid != "-":
                    launchd_statuses[name] = {"status": "up"}
                elif exit_status == "0":
                    launchd_statuses[name] = {"status": "idle"}  # loaded, last run OK
                else:
                    launchd_statuses[name] = {"status": "error"}  # loaded, last run failed
                break
        if not found:
            launchd_statuses[name] = {"status": "not_loaded"}

    services = {
        "ollama": {"status": "up" if ollama_code == "200" else "down"},
        "telegram_bot": {"status": "up" if telegram_out and not telegram_out.startswith("-\t") else "down"},
        "railway": {"status": "up" if railway_code == "200" else "down"},
        "zikaron_daemon": {"status": "up", "chunks": vector_store.count() if vector_store else 0},
        **launchd_statuses,
    }

    return {"services": services}


# ──────────────────────────────────────────────
# Stats / Token usage endpoints
# ──────────────────────────────────────────────

@app.get("/stats/tokens")
async def stats_tokens(days: int = 7):
    """Token usage summary from Supabase llm_usage table."""
    from datetime import datetime, timedelta, timezone

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    entries = await asyncio.to_thread(
        _supabase_get, "llm_usage",
        f"select=model,source,input_tokens,output_tokens,cost_usd,tier,created_at&created_at=gte.{cutoff}&order=created_at.desc&limit=1000"
    )

    if not entries:
        return _stats_tokens_local(days)

    total_cost = sum(float(e.get("cost_usd", 0)) for e in entries)
    total_input = sum(e.get("input_tokens", 0) for e in entries)
    total_output = sum(e.get("output_tokens", 0) for e in entries)

    # Group by model
    by_model: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        model = e.get("model", "unknown")
        if model not in by_model:
            by_model[model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_model[model]["calls"] += 1
        by_model[model]["input_tokens"] += e.get("input_tokens", 0)
        by_model[model]["output_tokens"] += e.get("output_tokens", 0)
        by_model[model]["cost_usd"] += float(e.get("cost_usd", 0))

    # Group by day for charts
    by_day: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        day = e.get("created_at", "")[:10]  # YYYY-MM-DD
        if not day:
            continue
        if day not in by_day:
            by_day[day] = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        by_day[day]["calls"] += 1
        by_day[day]["input_tokens"] += e.get("input_tokens", 0)
        by_day[day]["output_tokens"] += e.get("output_tokens", 0)
        by_day[day]["cost_usd"] += float(e.get("cost_usd", 0))

    return {
        "days": days,
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "entry_count": len(entries),
        "by_model": {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in by_model.items()},
        "by_day": {k: {**v, "cost_usd": round(v["cost_usd"], 4)} for k, v in sorted(by_day.items())},
        "recent": entries[:20],
    }


def _stats_tokens_local(days: int) -> Dict[str, Any]:
    """Fallback: read from local api_costs.jsonl with date filtering."""
    if not API_COSTS_PATH.exists():
        return {"entries": [], "total_cost_usd": 0, "total_input_tokens": 0, "total_output_tokens": 0}

    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    entries: List[Dict[str, Any]] = []
    total_cost = 0.0
    total_input = 0
    total_output = 0

    with open(API_COSTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                # Filter by timestamp if present
                ts = entry.get("timestamp", "")
                if ts and ts < cutoff:
                    continue
                entries.append(entry)
                total_cost += entry.get("cost_usd", 0)
                total_input += entry.get("input_tokens", 0)
                total_output += entry.get("output_tokens", 0)
            except json.JSONDecodeError:
                continue

    return {
        "days": days,
        "entries": entries[-50:],
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "entry_count": len(entries),
    }


@app.get("/stats/enrichment")
async def stats_enrichment():
    """Zikaron enrichment progress — how many chunks have tags, summaries, importance."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    db_path = DEFAULT_DB_PATH
    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    cursor = conn.cursor()

    try:
        total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        has_tags = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND tags != ''"))[0][0]
        has_summary = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND summary != ''"))[0][0]
        has_importance = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE importance IS NOT NULL"))[0][0]
        has_intent = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE intent IS NOT NULL AND intent != ''"))[0][0]

        # Embeddings count
        try:
            has_embeddings = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors_rowids"))[0][0]
        except Exception:
            has_embeddings = 0

        # Projects breakdown
        projects = list(cursor.execute("""
            SELECT project, COUNT(*) as cnt
            FROM chunks
            WHERE project IS NOT NULL
            GROUP BY project
            ORDER BY cnt DESC
            LIMIT 20
        """))

        return {
            "total_chunks": total,
            "embeddings": {"count": has_embeddings, "pct": round(has_embeddings * 100 / total, 1) if total else 0},
            "tags": {"count": has_tags, "pct": round(has_tags * 100 / total, 1) if total else 0},
            "summaries": {"count": has_summary, "pct": round(has_summary * 100 / total, 1) if total else 0},
            "importance": {"count": has_importance, "pct": round(has_importance * 100 / total, 1) if total else 0},
            "intent": {"count": has_intent, "pct": round(has_intent * 100 / total, 1) if total else 0},
            "projects": [{"project": p, "chunks": c} for p, c in projects],
        }
    finally:
        conn.close()


# ──────────────────────────────────────────────
# Events + Service Runs
# ──────────────────────────────────────────────

_cached_ssl_ctx = None

def _supabase_ssl_ctx():
    """Get SSL context that works on macOS (uses certifi if available). Cached."""
    global _cached_ssl_ctx
    if _cached_ssl_ctx is not None:
        return _cached_ssl_ctx
    import ssl
    try:
        import certifi
        _cached_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        _cached_ssl_ctx = ssl.create_default_context()
    return _cached_ssl_ctx


def _supabase_get(path: str, params: str = "") -> list:
    """Fetch from Supabase REST API. Returns list of rows or empty on error."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        return []
    import urllib.request
    try:
        url = f"{supabase_url}/rest/v1/{path}{'?' + params if params else ''}"
        req = urllib.request.Request(url, headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        })
        with urllib.request.urlopen(req, timeout=5, context=_supabase_ssl_ctx()) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Supabase query failed ({path}): {e}")
        return []


@app.get("/events/recent")
async def events_recent(limit: int = 50):
    """Recent golem events from Supabase."""
    rows = await asyncio.to_thread(
        _supabase_get, "golem_events",
        f"select=actor,type,data,created_at&order=created_at.desc&limit={max(1, min(limit, 100))}"
    )
    return {"events": rows, "count": len(rows)}


@app.get("/stats/service-runs")
async def stats_service_runs(limit: int = 20):
    """Recent service runs from Supabase."""
    rows = await asyncio.to_thread(
        _supabase_get, "service_runs",
        f"select=service,started_at,ended_at,duration_ms,status,error&order=started_at.desc&limit={max(1, min(limit, 50))}"
    )
    return {"runs": rows, "count": len(rows)}


# ──────────────────────────────────────────────
# Server startup
# ──────────────────────────────────────────────

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """Main daemon entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Zikaron daemon")
    parser.add_argument("--http", type=int, default=None, help="Also serve on HTTP port (e.g. --http 8787)")
    args = parser.parse_args()

    global http_port
    http_port = args.http

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    setup_signal_handlers()

    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    if args.http:
        # Dual mode: unix socket + HTTP port
        asyncio.run(_run_dual(args.http))
    else:
        # Socket-only mode (backward compatible)
        config = uvicorn.Config(app, uds=str(SOCKET_PATH), log_level="info", access_log=False)
        server = uvicorn.Server(config)
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
        except Exception as e:
            logger.error(f"Daemon failed: {e}")
            sys.exit(1)


async def _run_dual(port: int):
    """Run both unix socket and HTTP servers concurrently."""
    socket_config = uvicorn.Config(app, uds=str(SOCKET_PATH), log_level="info", access_log=False)
    http_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info", access_log=False)

    socket_server = uvicorn.Server(socket_config)
    http_server = uvicorn.Server(http_config)

    logger.info(f"Starting dual mode: socket={SOCKET_PATH}, http=0.0.0.0:{port}")

    try:
        await asyncio.gather(
            socket_server.serve(),
            http_server.serve(),
        )
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")


if __name__ == "__main__":
    main()
