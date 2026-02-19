"""FastAPI daemon service for fast brainlayer queries + dashboard API."""

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
from typing import Any, Dict, List, Optional

import apsw
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .embeddings import get_embedding_model
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Default paths
from .paths import DEFAULT_DB_PATH

SOCKET_PATH = Path("/tmp/brainlayer.sock")
BRAIN_DIR = Path.home() / ".brainlayer-brain"
API_COSTS_PATH = Path.home() / ".local" / "share" / "brainlayer" / "api_costs.jsonl"

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
    logger.info("Starting brainlayer daemon...")

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
    logger.info("Shutting down brainlayer daemon...")
    if vector_store:
        vector_store.close()
        vector_store = None


app = FastAPI(
    title="BrainLayer Daemon",
    description="Fast search daemon + dashboard API for brainlayer knowledge base",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS — allow dashboard origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
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
                source_filter=request.source_filter,
            )
        elif request.use_semantic:
            query_embedding = embedding_model.embed_query(request.query)
            results = vector_store.search(
                query_embedding=query_embedding,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter,
            )
        else:
            results = vector_store.search(
                query_text=request.query,
                n_results=request.n_results,
                project_filter=request.project_filter,
                content_type_filter=request.content_type_filter,
                source_filter=request.source_filter,
            )

        total_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            ids=results.get("ids", [[]])[0],
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
            total_time_ms=total_time_ms,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")


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
        raise HTTPException(status_code=500, detail="Context lookup failed")


# ──────────────────────────────────────────────
# Brain View endpoints
# ──────────────────────────────────────────────


@app.get("/brain/graph")
async def brain_graph():
    """Serve pre-generated brain graph.json."""
    graph_path = BRAIN_DIR / "graph.json"
    if not graph_path.exists():
        raise HTTPException(status_code=404, detail="graph.json not found. Run: brainlayer brain-export")
    return FileResponse(graph_path, media_type="application/json")


@app.get("/brain/metadata")
async def brain_metadata():
    """Stats about the brain graph (node count, last generated, etc)."""
    meta_path = BRAIN_DIR / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found. Run: brainlayer brain-export")
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
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:11434/api/tags"],
        3,
    )
    mlx_fut = asyncio.to_thread(
        _check_service,
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8080/v1/models"],
        3,
    )
    telegram_fut = asyncio.to_thread(_check_service, ["launchctl", "list", "com.brainlayer.telegram"], 3)
    railway_fut = asyncio.to_thread(
        _check_service,
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8080/health"],
        5,
    )

    # Check all launchd services in one call (fast)
    launchd_fut = asyncio.to_thread(
        _check_service,
        ["bash", "-c", "launchctl list 2>/dev/null | grep -E 'brainlayer' || true"],
        3,
    )

    ollama_code, mlx_code, telegram_out, railway_code, launchd_out = await asyncio.gather(
        ollama_fut, mlx_fut, telegram_fut, railway_fut, launchd_fut
    )

    # Parse launchd services
    launchd_services = {
        "nightshift": "com.brainlayer.nightshift",
        "briefing": "com.brainlayer.briefing",
        "healthcheck": "com.brainlayer.healthcheck",
        "compactor": "com.brainlayer.compactor",
        "bedtime_guardian": "com.brainlayer.bedtime-guardian",
        "session_archiver": "com.brainlayer.session-archiver",
        "auto_index": "com.brainlayer.auto-index",
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
        "mlx": {"status": "up" if mlx_code == "200" else "down"},
        "telegram_bot": {"status": "up" if telegram_out and not telegram_out.startswith("-\t") else "down"},
        "railway": {"status": "up" if railway_code == "200" else "down"},
        "brainlayer_daemon": {
            "status": "up",
            "chunks": vector_store.count() if vector_store else 0,
        },
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
        _supabase_get,
        "llm_usage",
        f"select=model,source,input_tokens,output_tokens,cost_usd,tier,created_at&created_at=gte.{cutoff}&order=created_at.desc&limit=1000",
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
        return {
            "entries": [],
            "total_cost_usd": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

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
    """BrainLayer enrichment progress — how many chunks have tags, summaries, importance."""
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")

    db_path = DEFAULT_DB_PATH
    conn = apsw.Connection(str(db_path), flags=apsw.SQLITE_OPEN_READONLY)
    cursor = conn.cursor()

    try:
        total = list(cursor.execute("SELECT COUNT(*) FROM chunks"))[0][0]
        has_tags = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE tags IS NOT NULL AND tags != ''"))[0][0]
        has_summary = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE summary IS NOT NULL AND summary != ''"))[
            0
        ][0]
        has_importance = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE importance IS NOT NULL"))[0][0]
        has_intent = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE intent IS NOT NULL AND intent != ''"))[0][0]

        # Embeddings count
        try:
            has_embeddings = list(cursor.execute("SELECT COUNT(*) FROM chunk_vectors_rowids"))[0][0]
        except Exception:
            has_embeddings = 0

        # Projects breakdown
        projects = list(
            cursor.execute("""
            SELECT project, COUNT(*) as cnt
            FROM chunks
            WHERE project IS NOT NULL
            GROUP BY project
            ORDER BY cnt DESC
            LIMIT 20
        """)
        )

        return {
            "total_chunks": total,
            "embeddings": {
                "count": has_embeddings,
                "pct": round(has_embeddings * 100 / total, 1) if total else 0,
            },
            "tags": {"count": has_tags, "pct": round(has_tags * 100 / total, 1) if total else 0},
            "summaries": {
                "count": has_summary,
                "pct": round(has_summary * 100 / total, 1) if total else 0,
            },
            "importance": {
                "count": has_importance,
                "pct": round(has_importance * 100 / total, 1) if total else 0,
            },
            "intent": {
                "count": has_intent,
                "pct": round(has_intent * 100 / total, 1) if total else 0,
            },
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
        req = urllib.request.Request(
            url,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=5, context=_supabase_ssl_ctx()) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Supabase query failed ({path}): {e}")
        return []


@app.get("/events/recent")
async def events_recent(limit: int = 50):
    """Recent golem events from Supabase."""
    rows = await asyncio.to_thread(
        _supabase_get,
        "golem_events",
        f"select=actor,type,data,created_at&order=created_at.desc&limit={max(1, min(limit, 100))}",
    )
    return {"events": rows, "count": len(rows)}


@app.get("/stats/service-runs")
async def stats_service_runs(limit: int = 20):
    """Recent service runs from Supabase."""
    rows = await asyncio.to_thread(
        _supabase_get,
        "service_runs",
        f"select=service,started_at,ended_at,duration_ms,status,error&order=started_at.desc&limit={max(1, min(limit, 50))}",
    )
    return {"runs": rows, "count": len(rows)}


# ──────────────────────────────────────────────
# Content Pipeline
# ──────────────────────────────────────────────


@app.get("/content/pipeline-runs")
async def content_pipeline_runs(limit: int = 50):
    """Recent pipeline runs from Supabase."""
    rows = await asyncio.to_thread(
        _supabase_get,
        "pipeline_runs",
        f"select=id,pipeline_id,idea,idea_type,success,duration_ms,quality_score,user_feedback,output_format,error,created_at&order=created_at.desc&limit={max(1, min(limit, 100))}",
    )
    return {"runs": rows, "count": len(rows)}


@app.get("/content/pipeline-stats")
async def content_pipeline_stats():
    """Aggregate pipeline performance stats."""
    rows = await asyncio.to_thread(
        _supabase_get,
        "pipeline_runs",
        "select=pipeline_id,success,duration_ms,quality_score,idea_type&order=created_at.desc&limit=500",
    )
    # Aggregate by pipeline
    stats: dict[str, dict] = {}
    for r in rows:
        pid = r.get("pipeline_id", "unknown")
        if pid not in stats:
            stats[pid] = {
                "total": 0,
                "success": 0,
                "quality_sum": 0.0,
                "quality_count": 0,
                "duration_sum": 0,
                "idea_types": {},
            }
        s = stats[pid]
        s["total"] += 1
        if r.get("success"):
            s["success"] += 1
        qs = r.get("quality_score")
        if qs is not None:
            s["quality_sum"] += float(qs)
            s["quality_count"] += 1
        s["duration_sum"] += int(r.get("duration_ms") or 0)
        it = r.get("idea_type", "general")
        s["idea_types"][it] = s["idea_types"].get(it, 0) + 1

    result = []
    for pid, s in stats.items():
        top_types = sorted(s["idea_types"].items(), key=lambda x: -x[1])[:3]
        result.append(
            {
                "pipeline_id": pid,
                "total_runs": s["total"],
                "successful_runs": s["success"],
                "success_rate": round(s["success"] / s["total"], 2) if s["total"] > 0 else 0,
                "avg_quality": round(s["quality_sum"] / s["quality_count"], 2) if s["quality_count"] > 0 else None,
                "avg_duration_ms": round(s["duration_sum"] / s["total"]) if s["total"] > 0 else 0,
                "top_idea_types": [t[0] for t in top_types],
            }
        )
    return {"stats": result, "total_runs": len(rows)}


# ──────────────────────────────────────────────
# Backlog CRUD
# ──────────────────────────────────────────────


def _supabase_mutate(method: str, path: str, body: dict | None = None, params: str = "") -> dict | list | None:
    """POST/PATCH/DELETE to Supabase REST API. Returns parsed JSON or None."""
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if not supabase_url or not supabase_key:
        return None
    import urllib.request

    try:
        url = f"{supabase_url}/rest/v1/{path}{'?' + params if params else ''}"
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
        )
        with urllib.request.urlopen(req, timeout=5, context=_supabase_ssl_ctx()) as resp:
            return json.loads(resp.read())
    except Exception as e:
        logger.warning(f"Supabase {method} failed ({path}): {e}")
        return None


@app.get("/backlog/items")
async def backlog_list(project: str = "", status: str = ""):
    """List backlog items, optionally filtered by project and/or status."""
    from urllib.parse import quote

    params = "select=*&order=updated_at.desc&limit=200"
    if project:
        params += f"&project=eq.{quote(project, safe='')}"
    if status:
        params += f"&status=eq.{quote(status, safe='')}"
    rows = await asyncio.to_thread(_supabase_get, "backlog_items", params)
    return {"items": rows, "count": len(rows)}


@app.post("/backlog/items")
async def backlog_create(request: Request):
    """Create a new backlog item."""
    body = await request.json()
    # Validate required fields
    if not body.get("title"):
        return JSONResponse({"error": "title is required"}, status_code=400)
    item = {
        "title": body["title"],
        "project": body.get("project", ""),
        "description": body.get("description"),
        "status": body.get("status", "backlog"),
        "priority": body.get("priority", "medium"),
        "tags": body.get("tags", []),
        "created_by": body.get("created_by", "dashboard"),
    }
    result = await asyncio.to_thread(_supabase_mutate, "POST", "backlog_items", item)
    if result and len(result) > 0:
        return result[0]
    return JSONResponse({"error": "failed to create item"}, status_code=500)


@app.patch("/backlog/items/{item_id}")
async def backlog_update(item_id: str, request: Request):
    """Update a backlog item."""
    from urllib.parse import quote

    body = await request.json()
    # Only allow safe fields
    allowed = {"title", "description", "status", "priority", "tags", "project"}
    update = {k: v for k, v in body.items() if k in allowed}
    if not update:
        return JSONResponse({"error": "no valid fields to update"}, status_code=400)
    result = await asyncio.to_thread(
        _supabase_mutate, "PATCH", "backlog_items", update, f"id=eq.{quote(item_id, safe='')}"
    )
    if result and len(result) > 0:
        return result[0]
    return JSONResponse({"error": "item not found or update failed"}, status_code=404)


@app.delete("/backlog/items/{item_id}")
async def backlog_delete(item_id: str):
    """Delete a backlog item."""
    from urllib.parse import quote

    result = await asyncio.to_thread(
        _supabase_mutate, "DELETE", "backlog_items", None, f"id=eq.{quote(item_id, safe='')}"
    )
    if result is None:
        return JSONResponse({"error": "delete failed"}, status_code=500)
    return {"deleted": True, "count": len(result) if isinstance(result, list) else 0}


# ──────────────────────────────────────────────
# Dashboard Search & Session Detail
# ──────────────────────────────────────────────


@app.get("/dashboard/search")
async def dashboard_search(q: str = "", project: str = "", content_type: str = "", limit: int = 20):
    """Fast FTS5 text search across all chunks. Returns ranked results with snippets."""
    if not q.strip():
        return {"results": [], "query": q, "total": 0, "time_ms": 0}

    limit = max(1, min(limit, 100))
    start_time = time.time()

    def _run_search():
        import re

        conn = apsw.Connection(str(DEFAULT_DB_PATH), flags=apsw.SQLITE_OPEN_READONLY)
        cursor = conn.cursor()
        try:
            # Build FTS5 match expression: split words, join with AND
            # Strip FTS5 special chars except quotes
            words = re.findall(r"[a-zA-Z0-9_]+", q)
            if not words:
                return []
            match_expr = " AND ".join(f'"{w}"' for w in words)

            # Build WHERE clauses for filters
            where_parts = []
            params: list = []
            if project:
                where_parts.append("c.project LIKE ?")
                params.append(f"%{project}%")
            if content_type:
                where_parts.append("c.content_type = ?")
                params.append(content_type)
            where_clause = (" AND " + " AND ".join(where_parts)) if where_parts else ""

            sql = f"""
                SELECT c.id, c.content_type, c.project, c.conversation_id,
                       c.importance, c.tags, c.summary, c.intent,
                       snippet(chunks_fts, 0, '<mark>', '</mark>', '...', 40) as snippet,
                       fts.rank
                FROM chunks_fts fts
                JOIN chunks c ON c.id = fts.chunk_id
                WHERE chunks_fts MATCH ?{where_clause}
                ORDER BY fts.rank
                LIMIT ?
            """
            all_params = [match_expr] + params + [limit]
            rows = list(cursor.execute(sql, all_params))
            return rows
        except Exception as e:
            logger.warning(f"Dashboard search failed: {e}")
            return []
        finally:
            conn.close()

    rows = await asyncio.to_thread(_run_search)
    elapsed = (time.time() - start_time) * 1000

    def _sanitize_snippet(raw: str) -> str:
        """Escape HTML in FTS5 snippet except <mark> tags (defense-in-depth)."""
        import html

        escaped = html.escape(raw)
        return escaped.replace("&lt;mark&gt;", "<mark>").replace("&lt;/mark&gt;", "</mark>")

    results = []
    for row in rows:
        (
            chunk_id,
            content_type,
            proj,
            conv_id,
            importance,
            tags,
            summary,
            intent,
            snippet_text,
            rank,
        ) = row
        results.append(
            {
                "id": chunk_id,
                "content_type": content_type,
                "project": proj,
                "conversation_id": conv_id,
                "importance": importance,
                "tags": tags,
                "summary": summary,
                "intent": intent,
                "snippet": _sanitize_snippet(snippet_text) if snippet_text else "",
                "rank": rank,
            }
        )

    return {"results": results, "query": q, "total": len(results), "time_ms": round(elapsed, 1)}


@app.get("/session/{session_id:path}")
async def session_detail(session_id: str, page: int = 1, per_page: int = 50, content_type: str = ""):
    """Get session detail: chunks (paginated), files touched, metadata.

    Sessions are matched by conversation_id OR by chunk ID prefix (for newer chunks
    that don't have conversation_id set). The chunk ID format is '{jsonl_path}:{N}'.
    Optionally filter by content_type.
    """
    per_page = max(1, min(per_page, 200))
    offset = (max(1, page) - 1) * per_page

    def _get_session():
        conn = apsw.Connection(str(DEFAULT_DB_PATH), flags=apsw.SQLITE_OPEN_READONLY)
        cursor = conn.cursor()
        try:
            # Try conversation_id first, then fall back to ID prefix match
            total = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE conversation_id = ?", [session_id]))[0][0]

            if total == 0:
                # Fall back to ID prefix match (chunks where id starts with session_id:)
                prefix = session_id + ":"
                total = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE id LIKE ? || '%'", [prefix]))[0][0]
                if total == 0:
                    # Also try exact prefix without colon (e.g. the path itself)
                    total = list(cursor.execute("SELECT COUNT(*) FROM chunks WHERE id LIKE ? || '%'", [session_id]))[0][
                        0
                    ]
                    if total == 0:
                        return None
                    id_filter = ("id LIKE ? || '%'", [session_id])
                else:
                    id_filter = ("id LIKE ? || '%'", [prefix])
            else:
                id_filter = ("conversation_id = ?", [session_id])

            where, wparams = id_filter

            # Add type filter if specified
            type_where = ""
            type_params: list = []
            if content_type:
                type_where = " AND content_type = ?"
                type_params = [content_type]
                # Recalculate total with type filter
                total = list(
                    cursor.execute(
                        f"SELECT COUNT(*) FROM chunks WHERE {where}{type_where}",
                        wparams + type_params,
                    )
                )[0][0]

            # Paginated chunks
            chunks = list(
                cursor.execute(
                    f"""
                SELECT id, content_type, project, position, importance,
                       tags, summary, intent, content, source_file
                FROM chunks
                WHERE {where}{type_where}
                ORDER BY position ASC, rowid ASC
                LIMIT ? OFFSET ?
            """,
                    wparams + type_params + [per_page, offset],
                )
            )

            # Session context (if available)
            ctx = list(
                cursor.execute(
                    """
                SELECT session_id, project, branch, pr_number, commit_shas,
                       files_changed, started_at, ended_at, created_at,
                       plan_name, plan_phase, story_id
                FROM session_context WHERE session_id = ?
            """,
                    [session_id],
                )
            )

            # Unique files touched in this session
            files = list(
                cursor.execute(
                    f"""
                SELECT DISTINCT source_file
                FROM chunks
                WHERE {where} AND source_file IS NOT NULL AND source_file != ''
                ORDER BY source_file
            """,
                    wparams,
                )
            )

            # Content type distribution
            type_dist = list(
                cursor.execute(
                    f"""
                SELECT content_type, COUNT(*) as cnt
                FROM chunks
                WHERE {where}
                GROUP BY content_type
                ORDER BY cnt DESC
            """,
                    wparams,
                )
            )

            return {
                "total": total,
                "chunks": chunks,
                "context": ctx,
                "files": files,
                "type_distribution": type_dist,
            }
        finally:
            conn.close()

    data = await asyncio.to_thread(_get_session)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Format chunks
    formatted_chunks = []
    for c in data["chunks"]:
        cid, ctype, proj, pos, imp, tags, summary, intent, content, src = c
        formatted_chunks.append(
            {
                "id": cid,
                "content_type": ctype,
                "project": proj,
                "position": pos,
                "importance": imp,
                "tags": tags,
                "summary": summary,
                "intent": intent,
                "content": content[:2000] if content else None,  # Truncate large content
                "source_file": src,
            }
        )

    # Format session context
    ctx_data = None
    if data["context"]:
        row = data["context"][0]
        ctx_data = {
            "session_id": row[0],
            "project": row[1],
            "branch": row[2],
            "pr_number": row[3],
            "commit_shas": row[4],
            "files_changed": row[5],
            "started_at": row[6],
            "ended_at": row[7],
            "created_at": row[8],
            "plan_name": row[9] if len(row) > 9 else None,
            "plan_phase": row[10] if len(row) > 10 else None,
            "story_id": row[11] if len(row) > 11 else None,
        }

    return {
        "session_id": session_id,
        "total_chunks": data["total"],
        "page": page,
        "per_page": per_page,
        "chunks": formatted_chunks,
        "context": ctx_data,
        "files": [f[0] for f in data["files"]],
        "type_distribution": {t: c for t, c in data["type_distribution"]},
    }


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

    parser = argparse.ArgumentParser(description="BrainLayer daemon")
    parser.add_argument("--http", type=int, default=None, help="Also serve on HTTP port (e.g. --http 8787)")
    args = parser.parse_args()

    global http_port
    http_port = args.http

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
