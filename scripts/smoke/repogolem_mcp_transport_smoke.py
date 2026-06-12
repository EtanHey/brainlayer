#!/usr/bin/env python3
"""Representative repoGolem-context BrainLayer MCP transport smoke."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke BrainLayer MCP from representative repo contexts.")
    parser.add_argument(
        "--contexts",
        nargs="*",
        default=["brainlayer", "systems", "narrationlayer"],
        help="Repo names under ~/Gits to use as CWD contexts.",
    )
    parser.add_argument("--gits-root", type=Path, default=Path.home() / "Gits")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--live-store", action="store_true", help="Use normal store path instead of arbitrated queue.")
    args = parser.parse_args()

    results = []
    for name in args.contexts:
        cwd = REPO_ROOT if name == "brainlayer" else args.gits_root / name
        if not cwd.exists():
            results.append({"context": name, "cwd": str(cwd), "status": "SKIP", "reason": "cwd_missing"})
            continue
        result = await smoke_context(name, cwd=cwd, timeout=args.timeout, live_store=args.live_store)
        results.append(result)

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(item["status"] in {"PASS", "SKIP"} for item in results) else 1


async def smoke_context(name: str, *, cwd: Path, timeout: float, live_store: bool) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{SRC}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["BRAINLAYER_MCP_QUERY_TIMEOUT"] = str(max(1.0, timeout - 5.0))
    if not live_store:
        env["BRAINLAYER_ARBITRATED"] = "1"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from brainlayer.mcp import serve; serve()"],
        cwd=str(cwd),
        env=env,
    )
    marker = f"PHASE1-TRANSPORT-SMOKE-{name}-{int(time.time())}"
    try:
        async with asyncio.timeout(timeout):
            async with stdio_client(server) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_names = {tool.name for tool in tools.tools}
                    if "brain_search" not in tool_names or "brain_store" not in tool_names:
                        return {
                            "context": name,
                            "cwd": str(cwd),
                            "status": "FAIL",
                            "reason": "missing_tools",
                            "tools": sorted(tool_names),
                        }
                    search = await session.call_tool(
                        "brain_search",
                        {"query": "phase 1 transport smoke", "project": name, "num_results": 1},
                    )
                    _raise_if_tool_error(search, "brain_search")
                    store = await session.call_tool(
                        "brain_store",
                        {
                            "content": f"{marker}: representative MCP transport smoke from {cwd}",
                            "type": "note",
                            "project": name,
                            "tags": ["phase-1", "transport-smoke", name],
                            "importance": 3,
                        },
                    )
                    _raise_if_tool_error(store, "brain_store")
                    return {
                        "context": name,
                        "cwd": str(cwd),
                        "status": "PASS",
                        "store_mode": "live" if live_store else "arbitrated",
                        "marker": marker,
                        "search_text": _compact_tool_text(search),
                        "store_text": _compact_tool_text(store),
                    }
    except TimeoutError:
        return {"context": name, "cwd": str(cwd), "status": "FAIL", "reason": f"timeout_after_{timeout}s"}
    except Exception as exc:
        return {
            "context": name,
            "cwd": str(cwd),
            "status": "FAIL",
            "reason": str(exc) or exc.__class__.__name__,
            "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "traceback": "".join(traceback.format_exception(exc))[-4000:],
        }


def _raise_if_tool_error(result, tool_name: str) -> None:
    if getattr(result, "isError", False):
        raise RuntimeError(f"{tool_name} returned isError=true: {_compact_tool_text(result)}")
    text = _compact_tool_text(result).lower()
    if "transport closed" in text or "connection closed" in text:
        raise RuntimeError(f"{tool_name} transport closed: {_compact_tool_text(result)}")


def _compact_tool_text(result) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", "")
        if text:
            parts.append(text.replace("\n", " ")[:500])
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
