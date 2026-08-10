"""Installable setup helpers for BrainLayer."""

from __future__ import annotations

import json
import os
import select
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

from .cli.wizard import DEFAULT_BRAINLAYER_CONFIG, write_gemini_env_file
from .paths import SPOTLIGHT_EXCLUSION_MARKER, get_db_path

DEFAULT_GOOGLE_API_KEY_OP_REF = "op://Private/Google AI/Gemini API key"
DEFAULT_MCP_PROTOCOL_VERSION = "2025-06-18"

_DATA_CHILDREN = (
    "backups",
    "chromadb",
    "chromadb.backup",
    "enrichment-scratch",
    "experiments",
    "jsonl-backups",
    "logs",
    "prompts",
    "style",
    "storage",
)
_RUNTIME_CHILDREN = ("logs", "quarantine", "queue")


def _ensure_spotlight_excluded_root(root: Path, children: tuple[str, ...] = ()) -> Path:
    resolved = root.expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    marker = resolved / SPOTLIGHT_EXCLUSION_MARKER
    if not marker.is_file() and next(resolved.iterdir(), None) is not None:
        raise RuntimeError(f"existing runtime tree {resolved} requires the Spotlight exclusion migration runbook")
    marker.touch(exist_ok=True)
    for child in children:
        (resolved / child).mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_spotlight_excluded_layout(
    *,
    data_dir: Path | None = None,
    runtime_dir: Path | None = None,
    launchd_log_dir: Path | None = None,
    counter_dir: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Create marker-backed roots for every high-churn BrainLayer runtime path."""
    requested_roots = (
        data_dir or get_db_path().parent,
        runtime_dir or Path.home() / ".brainlayer",
        launchd_log_dir or Path.home() / "Library" / "Logs" / "brainlayer",
        counter_dir or Path.home() / ".brainlayer-p0-counter",
    )
    resolved_roots = tuple(root.expanduser() for root in requested_roots)
    for root in resolved_roots:
        if (root.exists() or root.is_symlink()) and not root.is_dir():
            raise RuntimeError(f"runtime root {root} must be a directory")
        marker = root / SPOTLIGHT_EXCLUSION_MARKER
        if root.is_dir() and not marker.is_file() and next(root.iterdir(), None) is not None:
            raise RuntimeError(f"existing runtime tree {root} requires the Spotlight exclusion migration runbook")

    roots = (
        _ensure_spotlight_excluded_root(resolved_roots[0], _DATA_CHILDREN),
        _ensure_spotlight_excluded_root(resolved_roots[1], _RUNTIME_CHILDREN),
        _ensure_spotlight_excluded_root(resolved_roots[2]),
        _ensure_spotlight_excluded_root(resolved_roots[3]),
    )
    return roots


def get_default_env_file() -> Path:
    """Return the standard per-user BrainLayer env file."""
    return Path.home() / ".config" / "brainlayer" / "brainlayer.env"


def get_launchd_dir() -> Path:
    """Return the packaged launchd template directory, falling back to source checkout."""
    packaged = resources.files("brainlayer").joinpath("launchd")
    if packaged.is_dir():
        return Path(str(packaged))

    source = Path(__file__).resolve().parents[2] / "scripts" / "launchd"
    if source.is_dir():
        return source

    raise FileNotFoundError("BrainLayer launchd templates were not found")


def get_current_brainlayer_bin() -> str | None:
    """Return the current console-script path when setup was invoked by one."""
    argv0 = Path(sys.argv[0]).expanduser()
    if argv0.name == "brainlayer":
        candidate = argv0 if argv0.is_absolute() else Path.cwd() / argv0
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    found = shutil.which("brainlayer")
    if found:
        return found
    return None


def get_current_mcp_bridge_bin() -> str | None:
    """Return the bridge installed beside the active BrainLayer CLI, then fall back to PATH."""
    brainlayer_bin = get_current_brainlayer_bin()
    if brainlayer_bin:
        sibling = Path(brainlayer_bin).with_name("brainlayer-mcp-stdio-bridge")
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling)
    return shutil.which("brainlayer-mcp-stdio-bridge")


def get_default_mcp_config_paths() -> tuple[Path, ...]:
    home = Path.home()
    return (
        home / ".claude.json",
        home / ".cursor" / "mcp.json",
        home / ".gemini" / "settings.json",
    )


def _is_legacy_brainbar_socat(server: object) -> bool:
    if not isinstance(server, dict):
        return False
    command = server.get("command")
    args = server.get("args")
    if not isinstance(command, str) or Path(command).name != "socat" or not isinstance(args, list):
        return False
    string_args = [arg for arg in args if isinstance(arg, str)]
    return "STDIO" in string_args and "UNIX-CONNECT:/tmp/brainbar.sock" in string_args


def migrate_legacy_mcp_configs(
    config_paths: list[Path] | tuple[Path, ...] | None = None,
    *,
    bridge_command: str | None = None,
) -> list[Path]:
    """Replace known raw-socat BrainBar transports without touching unrelated MCP entries."""
    resolved_bridge = bridge_command
    changed: list[Path] = []
    paths = config_paths if config_paths is not None else get_default_mcp_config_paths()
    for config_path in paths:
        path = config_path.expanduser()
        if not path.is_file():
            continue
        target_path = path.resolve()
        try:
            payload = json.loads(target_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read MCP config {path}: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        servers = payload.get("mcpServers")
        if not isinstance(servers, dict):
            continue

        migrated = False
        for name, server in list(servers.items()):
            if not _is_legacy_brainbar_socat(server):
                continue
            if resolved_bridge is None:
                resolved_bridge = get_current_mcp_bridge_bin()
            if not resolved_bridge:
                raise FileNotFoundError("brainlayer-mcp-stdio-bridge was not found on PATH")
            migrated_server = dict(server)
            migrated_server["command"] = resolved_bridge
            migrated_server.pop("args", None)
            servers[name] = migrated_server
            migrated = True
        if not migrated:
            continue

        mode = target_path.stat().st_mode & 0o777
        temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.chmod(mode)
            os.replace(temporary, target_path)
        finally:
            temporary.unlink(missing_ok=True)
        changed.append(path)
    return changed


def verify_mcp_transport(*, bridge_command: str | None = None, timeout_seconds: float = 10.0) -> int:
    """Require initialize + tools/list + one successful tool call on one fresh bridge."""
    resolved_bridge = bridge_command or get_current_mcp_bridge_bin()
    if not resolved_bridge:
        raise FileNotFoundError("brainlayer-mcp-stdio-bridge was not found on PATH")
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": DEFAULT_MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "brainlayer-setup", "version": "1"},
        },
    }
    initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    tools_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    deadline = time.monotonic() + timeout_seconds

    def send(process: subprocess.Popen[bytes], payload: dict[str, object]) -> None:
        assert process.stdin is not None
        process.stdin.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        process.stdin.flush()

    def receive(process: subprocess.Popen[bytes], response_id: int) -> dict[str, object] | None:
        assert process.stdout is not None
        buffered = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                raise RuntimeError(f"MCP transport verification timed out after {timeout_seconds:g}s")
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                return None
            buffered.extend(chunk)
            while b"\n" in buffered:
                line, _, remainder = buffered.partition(b"\n")
                buffered = bytearray(remainder)
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("id") == response_id:
                    return payload

    try:
        process = subprocess.Popen(
            [resolved_bridge],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        raise RuntimeError(f"could not start MCP bridge: {exc}") from exc

    try:
        send(process, initialize_request)
        initialize = receive(process, 1)
        if not initialize or not isinstance(initialize.get("result"), dict):
            raise RuntimeError("MCP initialize response missing or invalid")
        send(process, initialized_notification)
        send(process, tools_request)
        tools_response = receive(process, 2)
        tools_result = tools_response.get("result") if tools_response else None
        tools = tools_result.get("tools") if isinstance(tools_result, dict) else None
        if not isinstance(tools, list) or not tools:
            raise RuntimeError("MCP tools/list response missing tools")
        tool_names = {
            tool.get("name") for tool in tools if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        if "brain_recall" in tool_names:
            smoke_tool = "brain_recall"
            smoke_arguments = {"mode": "stats"}
        elif "expand_palette" in tool_names:
            smoke_tool = "expand_palette"
            smoke_arguments: dict[str, object] = {}
        else:
            raise RuntimeError("MCP tools/list has no safe deployment smoke tool")
        send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": smoke_tool, "arguments": smoke_arguments},
            },
        )
        call_response = receive(process, 3)
        call_result = call_response.get("result") if call_response else None
        if not isinstance(call_result, dict) or call_result.get("isError") is True:
            raise RuntimeError(f"MCP deployment tool call failed: {smoke_tool}")
        return len(tools)
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def ensure_brainlayer_env(
    env_file: Path | None = None,
    *,
    google_api_key_op_ref: str | None = None,
    overwrite_google_key: bool = False,
) -> Path:
    """Create or update brainlayer.env with defaults and an op-backed Google key."""
    target = env_file or get_default_env_file()

    if google_api_key_op_ref:
        write_gemini_env_file(
            target,
            google_api_key=google_api_key_op_ref,
            secret_source="1password",
            overwrite=overwrite_google_key,
            enrichment_env=DEFAULT_BRAINLAYER_CONFIG,
        )
        return target

    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BrainLayer private config.",
        "# Add a 1Password-backed key before enabling cloud enrichment with:",
        f"# brainlayer setup --google-api-key-op-ref '{DEFAULT_GOOGLE_API_KEY_OP_REF}'",
        "",
    ]
    lines.extend(f"{key}={value}" for key, value in DEFAULT_BRAINLAYER_CONFIG.items())
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)
    return target


def install_launchd(
    target: str = "all",
    *,
    env_file: Path | None = None,
    launchd_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> None:
    """Run the packaged launchd installer, which owns bounded shutdown waits."""
    template_dir = launchd_dir or get_launchd_dir()
    install_script = template_dir / "install.sh"
    if not install_script.exists():
        raise FileNotFoundError(f"launchd installer not found: {install_script}")

    run_env = os.environ.copy()
    if extra_env:
        run_env.update(extra_env)
    run_env.setdefault("PYTHON_BIN", sys.executable)
    run_env.setdefault("BRAINLAYER_PYTHON", sys.executable)
    brainlayer_bin = get_current_brainlayer_bin()
    if brainlayer_bin is not None:
        run_env.setdefault("BRAINLAYER_BIN", brainlayer_bin)
    if env_file is not None:
        run_env["BRAINLAYER_ENV_FILE"] = str(env_file)

    try:
        subprocess.run([str(install_script), target], env=run_env, check=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        if timeout_seconds is None:
            raise TimeoutError("launchd installer timed out") from exc
        raise TimeoutError(f"launchd installer timed out after {timeout_seconds:g}s") from exc
