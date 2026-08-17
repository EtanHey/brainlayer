"""Canonical BrainBar MCP socket wiring — one shape, shared by setup + doctor.

Target form (the only acceptable BrainLayer MCP transport):

    {"command": "socat", "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]}

Anything else that is a BrainLayer memory MCP entry (bridge, bun+bridge,
deleted brainlayer-mcp, python -m, …) must be migrated / flagged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SOCKET_MCP_SERVER: dict[str, Any] = {
    "command": "socat",
    "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
}

BRAINLAYER_SERVER_NAMES = frozenset({"brainlayer", "brainlayer-mcp", "brainbar", "brain-bar"})
_KNOWN_LEGACY_BASENAMES = frozenset({"brainlayer-mcp", "brainlayer-mcp-stdio-bridge"})


def mcp_command_basename(server: object) -> str | None:
    if not isinstance(server, dict):
        return None
    command = server.get("command")
    if isinstance(command, str) and command.strip():
        return Path(command).name
    if isinstance(command, dict):
        path = command.get("path")
        if isinstance(path, str) and path.strip():
            return Path(path).name
    return None


def is_canonical_socket_mcp_server(server: object) -> bool:
    """True only for socat STDIO UNIX-CONNECT:…/brainbar.sock (exact target shape)."""
    if not isinstance(server, dict):
        return False
    if mcp_command_basename(server) != "socat":
        return False
    args = server.get("args")
    if not isinstance(args, list):
        return False
    string_args = [arg for arg in args if isinstance(arg, str)]
    return "STDIO" in string_args and "UNIX-CONNECT:/tmp/brainbar.sock" in string_args


def is_brainlayer_mcp_entry(name: str, server: object) -> bool:
    """True when this MCP entry is BrainLayer memory wiring (by name or known transport)."""
    if name.lower() in BRAINLAYER_SERVER_NAMES:
        return True
    if not isinstance(server, dict):
        return False
    basename = mcp_command_basename(server)
    if basename in _KNOWN_LEGACY_BASENAMES:
        return True
    args = server.get("args")
    if isinstance(args, list):
        for arg in args:
            if not isinstance(arg, str):
                continue
            if Path(arg).name in _KNOWN_LEGACY_BASENAMES:
                return True
            if "brainlayer.mcp" in arg:
                return True
    if basename in {"python", "python3"} and isinstance(args, list):
        joined = " ".join(str(a) for a in args)
        if "brainlayer.mcp" in joined:
            return True
    if basename == "brainlayer" and isinstance(args, list) and any(str(a) == "serve" for a in args):
        return True
    return False


def needs_socket_migration(name: str, server: object) -> bool:
    """Shape-matcher: BrainLayer MCP entry that is not the canonical socket form."""
    return is_brainlayer_mcp_entry(name, server) and not is_canonical_socket_mcp_server(server)


def owned_mcp_config_paths(*, home: Path | None = None) -> tuple[Path, ...]:
    """Configs BrainLayer owns and may rewrite on setup/upgrade."""
    root = home if home is not None else Path.home()
    paths: list[Path] = [
        root / ".claude.json",
        root / ".cursor" / "mcp.json",
        root / ".gemini" / "settings.json",
        root / ".codex" / "config.toml",
    ]
    gits = root / "Gits"
    if gits.is_dir():
        for path in sorted(gits.rglob(".mcp.json")):
            if ".worktrees" in path.parts or ".git" in path.parts:
                continue
            paths.append(path)
    return tuple(paths)


def iter_json_mcp_servers(payload: object) -> list[tuple[str, object]]:
    if not isinstance(payload, dict):
        return []
    found: list[tuple[str, object]] = []
    servers = payload.get("mcpServers")
    if isinstance(servers, dict):
        found.extend((str(name), server) for name, server in servers.items())
    projects = payload.get("projects")
    if isinstance(projects, dict):
        for project_data in projects.values():
            if not isinstance(project_data, dict):
                continue
            nested = project_data.get("mcpServers")
            if isinstance(nested, dict):
                found.extend((str(name), server) for name, server in nested.items())
    return found


def iter_toml_mcp_servers(payload: object) -> list[tuple[str, object]]:
    """All mcp_servers.* entries (and mcpServers.* if present), not just .brainlayer."""
    if not isinstance(payload, dict):
        return []
    found: list[tuple[str, object]] = []
    for key in ("mcp_servers", "mcpServers"):
        servers = payload.get(key)
        if isinstance(servers, dict):
            found.extend((str(name), server) for name, server in servers.items())
    return found


def socket_server_preserving(server: object) -> dict[str, Any]:
    """Canonical socket transport, keeping unrelated keys (disabled/env/timeout)."""
    preserved: dict[str, Any] = {}
    if isinstance(server, dict):
        preserved = {k: v for k, v in server.items() if k not in {"command", "args", "path"}}
    return {**SOCKET_MCP_SERVER, **preserved}
