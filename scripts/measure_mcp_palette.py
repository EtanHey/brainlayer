#!/usr/bin/env python3
"""Measure a compact MCP tools payload in UTF-8 bytes and o200k tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import tiktoken


def extract_tools(payload: Any) -> list[dict[str, Any]]:
    """Extract a tools array from a raw list, result object, or JSON-RPC envelope."""
    tools: Any
    if isinstance(payload, list):
        tools = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tools"), list):
        tools = payload["tools"]
    elif (
        isinstance(payload, dict)
        and isinstance(payload.get("result"), dict)
        and isinstance(payload["result"].get("tools"), list)
    ):
        tools = payload["result"]["tools"]
    else:
        raise ValueError("Expected a tools array or a JSON-RPC result containing one")

    if not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("Every tools entry must be a JSON object")
    return tools


def canonical_tools_json(tools: list[dict[str, Any]]) -> str:
    return json.dumps(tools, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def measure_tools(tools: list[dict[str, Any]]) -> dict[str, int]:
    canonical = canonical_tools_json(tools)
    return {
        "tools": len(tools),
        "bytes": len(canonical.encode("utf-8")),
        "tokens": len(tiktoken.get_encoding("o200k_base").encode(canonical)),
    }


def _load_payload(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="-", help="JSON file path, or - for stdin")
    parser.add_argument("--max-bytes", type=int, help="Exit nonzero if the compact tools array exceeds this size")
    args = parser.parse_args(argv)

    try:
        measurement = measure_tools(extract_tools(_load_payload(args.path)))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(measurement, sort_keys=True))
    if args.max_bytes is not None and measurement["bytes"] > args.max_bytes:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
