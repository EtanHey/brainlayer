"""BrainBar MCP config matching shared by setup and doctor.

The shipped reconnecting stdio bridge is the preferred agent transport.
Raw socat loses the agent's stdio transport when BrainBar restarts.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

SOCKET_MCP_SERVER: dict[str, Any] = {
    "command": "socat",
    "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
}

BRAINLAYER_SERVER_NAMES = frozenset({"brainlayer", "brainlayer-mcp", "brainbar", "brain-bar"})
_STDIO_BRIDGE_COMMAND = "brainlayer-mcp-stdio-bridge"
KEG_MCP_BRIDGE_PATHS = tuple(
    f"{prefix}/opt/brainlayer/bin/{_STDIO_BRIDGE_COMMAND}" for prefix in ("/opt/homebrew", "/usr/local")
)


def stable_mcp_bridge_command() -> str:
    """Keep the opt spelling across upgrades; never select a PATH shadow."""
    return next((p for p in KEG_MCP_BRIDGE_PATHS if Path(p).is_file()), KEG_MCP_BRIDGE_PATHS[0])


def bridge_path_risk(server: object) -> dict[str, str] | None:
    """Report unpinned, missing, or foreign bridge targets without reading config secrets."""
    if mcp_command_basename(server) != _STDIO_BRIDGE_COMMAND:
        return None
    command = server["command"]
    command = command.get("path", "") if isinstance(command, dict) else command
    selected = command if Path(command).is_absolute() else shutil.which(command)
    try:
        target = str(Path(selected).resolve()) if selected else "not found on PATH"
    except (OSError, RuntimeError):
        target = "unresolvable bridge target"
    reason = "bridge command is not a stable BrainLayer opt path"
    if command in KEG_MCP_BRIDGE_PATHS:
        keg_root = Path(command).parents[3] / "Cellar" / "brainlayer"
        if selected and Path(target).is_relative_to(keg_root) and Path(target).is_file() and os.access(target, os.X_OK):
            return None
        reason = "opt bridge does not resolve to an executable inside the BrainLayer keg"
    return {"configured_command": command, "resolved_target": target, "reason": reason}


def is_socat_brainbar_transport(server: object) -> bool:
    """Any socat argument naming BrainBar has the same restart lifetime risk."""
    if mcp_command_basename(server) != "socat":
        return False
    args = server.get("args")
    return isinstance(args, list) and any(isinstance(a, str) and "brainbar.sock" in a for a in args)


_SKIP_GITS_DIR_PARTS = frozenset({".worktrees", ".git", "docs.local", "node_modules", ".tmp"})
_HTTP_TRANSPORT_KEYS = frozenset({"type", "url", "headers"})


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
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return False
    return args == ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]


def is_canonical_stdio_bridge_mcp_server(server: object) -> bool:
    """True only for a stable opt-path bridge spelling, independent of installation state."""
    if not isinstance(server, dict):
        return False
    command = server.get("command")
    command = command.get("path") if isinstance(command, dict) else command
    return command in KEG_MCP_BRIDGE_PATHS


def is_acceptable_brainlayer_mcp_transport(server: object) -> bool:
    """Only the shipped bridge keeps an agent transport alive across restarts."""
    return is_canonical_stdio_bridge_mcp_server(server)


def is_brainlayer_mcp_entry(name: str, server: object) -> bool:
    """True only for MCP table entries BrainLayer owns by server name."""
    return name.lower() in BRAINLAYER_SERVER_NAMES


def needs_socket_migration(name: str, server: object) -> bool:
    """Owned BrainLayer entry that needs migration to the reconnecting bridge."""
    return is_brainlayer_mcp_entry(name, server) and not is_acceptable_brainlayer_mcp_transport(server)


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
        for repo_dir in sorted(gits.iterdir()):
            if not repo_dir.is_dir():
                continue
            if any(part in _SKIP_GITS_DIR_PARTS for part in repo_dir.parts):
                continue
            mcp_json = repo_dir / ".mcp.json"
            if mcp_json.is_file():
                paths.append(mcp_json)
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


def socket_server_preserving(server: object, *, bridge_command: str | None = None) -> dict[str, Any]:
    """Bridge transport, keeping unrelated keys (disabled/env/timeout)."""
    preserved: dict[str, Any] = {}
    if isinstance(server, dict):
        drop_keys = {"command", "args", "path", *_HTTP_TRANSPORT_KEYS}
        preserved = {k: v for k, v in server.items() if k not in drop_keys}
    command: Any = bridge_command or stable_mcp_bridge_command()
    if isinstance(server, dict) and isinstance(server.get("command"), dict):
        command = {**server["command"], "path": command, "args": []}
    return {"command": command, **preserved}


def _arg_public_shape(arg: object) -> str:
    if isinstance(arg, bool):
        return "bool"
    if isinstance(arg, int):
        return "int"
    if not isinstance(arg, str):
        return type(arg).__name__
    if arg.startswith("--") and "=" in arg:
        return arg.split("=", 1)[0] + "=…"
    if len(arg) > 40:
        return f"str(len={len(arg)})"
    return "str"


def mcp_server_public_summary(server: object) -> dict[str, Any]:
    """Safe MCP server shape for doctor JSON output (no env secrets)."""
    if not isinstance(server, dict):
        return {"shape": type(server).__name__}
    summary: dict[str, Any] = {}
    command = server.get("command")
    if command is not None:
        basename = mcp_command_basename(server)
        if basename:
            summary["command"] = basename
        else:
            summary["command"] = {"shape": type(command).__name__}
    args = server.get("args")
    if isinstance(args, list):
        summary["args_count"] = len(args)
        summary["args_shape"] = [_arg_public_shape(arg) for arg in args[:3]]
    transport_type = server.get("type")
    if isinstance(transport_type, str):
        summary["type"] = transport_type
    return summary
