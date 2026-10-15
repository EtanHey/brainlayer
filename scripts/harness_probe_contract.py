"""Offline contracts for the real-user-config harness probe; no model or DB access."""

import json
import os
import tomllib
from pathlib import Path

HARNESSES = ("claude", "codex-cli", "codex-app", "cursor", "gemini", "agy", "t3")
QUERY = "BrainLayer harness convergence probe"
# These surfaces need their own session-launch adapter; CLI success cannot stand in.
MANUAL = {
    "codex-app": "No app-thread adapter: launch a fresh Codex APP thread with the printed prompt; CLI is not app proof.",
    "t3": "No T3 thread adapter: use authenticated orchestration.dispatchCommand thread.turn.start on the running T3 server; CLI is not T3 proof.",
}


def command(harness, prompt):
    return {
        "claude": ["claude", "-p", "--verbose", "--output-format", "stream-json", prompt],
        "codex-cli": ["codex", "exec", "--json", prompt],
        "cursor": ["cursor-agent", "-p", "--output-format", "stream-json", prompt],
        "gemini": ["gemini", "-p", prompt, "--output-format", "stream-json"],
        "agy": ["agy", "-p", prompt, "--output-format", "stream-json"],
    }.get(harness)


def discover_configs(harness, home, cwd):
    """Inventory only; harness itself resolves precedence, trust and enabled state.

    Never copy arbitrary argv/env/URLs from a private config into the receipt.
    """
    codex_home = Path(os.environ.get("CODEX_HOME", home / ".codex"))
    paths = {
        "claude": [
            home / ".claude.json",
            home / ".claude/settings.json",
            home / ".claude/.mcp.json",
            cwd / ".mcp.json",
        ],
        "codex-cli": [codex_home / "config.toml", cwd / ".codex/config.toml"],
        "codex-app": [codex_home / "config.toml"],
        "cursor": [home / ".cursor/mcp.json", cwd / ".cursor/mcp.json"],
        "gemini": [home / ".gemini/settings.json", cwd / ".gemini/settings.json"],
        "agy": [home / ".gemini/config/mcp_config.json", home / ".gemini/antigravity-cli/settings.json"],
        "t3": [],
    }[harness]
    rows = []
    for path in paths:
        if not path.exists():
            continue
        try:
            data = tomllib.loads(path.read_text()) if path.suffix == ".toml" else json.loads(path.read_text())
            maps = [data.get("mcp_servers", data.get("mcpServers", {}))]
            maps.append(data.get("projects", {}).get(str(cwd), {}).get("mcpServers", {}))
            for servers in maps:
                for name, config in servers.items():
                    if "brain" in name.lower():
                        cmd = config.get("command", "")
                        rows.append(
                            {
                                "path": str(path),
                                "server": name,
                                "command": cmd
                                if isinstance(cmd, str) and " " not in cmd
                                else "[compound command redacted]",
                                "disabled": config.get("disabled", False) or config.get("enabled") is False,
                            }
                        )
        except (ValueError, OSError, AttributeError, TypeError) as exc:
            rows.append({"path": str(path), "error": type(exc).__name__})
    return rows


def parse_calls(output):
    """Only structured native tool records count. Never parse prose as evidence."""
    calls, pending = [], {}

    def add(name, args, result, ok=True):
        tool = str(name).split("__")[-1]
        if tool not in ("brain_search", "brain_store"):
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        encoded = json.dumps(result)
        failed = isinstance(result, dict) and (result.get("isError") or result.get("is_error") or result.get("error"))
        if tool == "brain_store":
            ok = ok and any(s in encoded for s in ("STORED", "DUPLICATE", "MERGED", "DEFERRED"))
        calls.append({"tool": tool, "args": args or {}, "ok": bool(ok and not failed)})

    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        step = event.get("step_update", {})
        if step.get("tool_name") == "call_mcp_tool" and step.get("state") in ("DONE", "ERROR"):
            info = step.get("tool_info", {})
            params = info.get("parameters", {})
            add(params.get("ToolName"), params.get("Arguments"), info.get("output"), step.get("state") == "DONE")
        item = event.get("item", {})
        if event.get("type") == "item.completed" and item.get("type") == "mcp_tool_call":
            add(item.get("tool"), item.get("arguments"), item.get("result"), item.get("status") == "completed")
        if event.get("type") == "tool_use":  # Gemini stream
            pending[event.get("tool_id")] = (event.get("tool_name"), event.get("parameters"))
        if event.get("type") == "tool_result" and event.get("tool_id") in pending:
            name, args = pending.pop(event["tool_id"])
            add(name, args, event.get("output"), event.get("status") == "success")
        if event.get("type") == "tool_call" and event.get("subtype") == "completed":
            call = event.get("tool_call", {}).get("mcpToolCall", {})
            args, result = call.get("args", {}), call.get("result", {})
            add(args.get("toolName"), args.get("args"), result.get("success"), "success" in result)
        message = event.get("message", {})
        for block in message.get("content", []) if isinstance(message, dict) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                pending[block.get("id")] = (block.get("name"), block.get("input"))
            if block.get("type") == "tool_result" and block.get("tool_use_id") in pending:
                name, args = pending.pop(block["tool_use_id"])
                add(name, args, block.get("content"), not block.get("is_error"))
    return calls


def verdict(legs):
    return (
        "PASS"
        if all(legs.get(k) is True for k in ("search", "store", "refound", "transport"))
        and (legs.get("ingest") is True or legs.get("ingest") == "NOT_APPLICABLE")
        else "FAIL"
    )


def exit_code(rows):
    return 0 if rows and all(row.get("status") == "PASS" for row in rows) else 1


def table(rows):
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")

    lines = [
        "| Harness | Result | Search | Store | Refound | Ingest | Transport | Seconds | Reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        legs = row.get("legs", {})
        values = [
            row.get("harness"),
            row.get("status"),
            *(legs.get(k, False) for k in ("search", "store", "refound", "ingest")),
            row.get("transport", "NOT_PROVEN"),
            row.get("seconds", 0),
            row.get("reason", ""),
        ]
        lines.append("| " + " | ".join(map(cell, values)) + " |")
    return "\n".join(lines)


def matching_hit(result, nonce, transcript=False, session_id=None):
    """Match content inside a result row, never the query heading or echoed args."""
    structured = result.get("structuredContent") or result.get("structured_content") or {}
    hits = structured.get("results", []) if isinstance(structured, dict) else []
    for hit in hits:
        source = str(hit.get("source_file", ""))
        if nonce in str(hit.get("content", "")) and (
            not transcript or source.endswith(".jsonl") and (not session_id or session_id in source)
        ):
            return True
    import re

    text = "\n".join(b.get("text", "") for b in result.get("content", []) if isinstance(b, dict))
    for block in re.split(r"(?m)^- ID: ", text)[1:]:
        # detail=full currently returns Source + Preview. Compact uses P:.
        source = re.search(r"(?m)^\s*Source:\s*(.+)$", block)
        body = re.search(r"(?ms)^\s*(?:Preview:|Content:|P:)\s*(.*)", block)
        if (
            body
            and nonce in body.group(1)
            and (
                not transcript
                or source
                and source.group(1).strip().endswith(".jsonl")
                and (not session_id or session_id in source.group(1))
            )
        ):
            return True
    return False


def session_id(output):
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            if event.get("event") == "init":
                return event.get("conversation_id")
            if event.get("type") == "thread.started":
                return event.get("thread_id")
            if event.get("type") in ("system", "init") and event.get("session_id"):
                return event["session_id"]
    return None
