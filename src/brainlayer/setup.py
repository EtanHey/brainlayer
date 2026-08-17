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
from .mcp_socket_config import (
    iter_toml_mcp_servers,
    needs_socket_migration,
    owned_mcp_config_paths,
    socket_server_preserving,
)
from .paths import get_canonical_db_path, resolve_db_path
from .spotlight import ensure_spotlight_excluded_layout as _ensure_spotlight_excluded_layout

DEFAULT_GOOGLE_API_KEY_OP_REF = "op://Private/Google AI/Gemini API key"
DEFAULT_MCP_PROTOCOL_VERSION = "2025-06-18"


def ensure_spotlight_excluded_layout(
    *,
    data_dir: Path | None = None,
    env_file: Path | None = None,
    runtime_dir: Path | None = None,
    launchd_log_dir: Path | None = None,
    counter_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Create marker-backed roots for every high-churn BrainLayer runtime path."""
    return _ensure_spotlight_excluded_layout(
        data_dir=data_dir,
        env_file=env_file,
        runtime_dir=runtime_dir,
        launchd_log_dir=launchd_log_dir,
        counter_dir=counter_dir,
        resolve_db_path_fn=resolve_db_path,
        get_canonical_db_path_fn=get_canonical_db_path,
        home_fn=Path.home,
        ismount_fn=os.path.ismount,
    )


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
    """Configs BrainLayer owns and may rewrite on setup/upgrade."""
    return owned_mcp_config_paths()


def _backup_config_file(target_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d%H%M%S")
    backup = target_path.with_name(f"{target_path.name}.bak.{stamp}")
    backup.write_bytes(target_path.read_bytes())
    backup.chmod(target_path.stat().st_mode & 0o777)
    return backup


def _rewrite_json_mcp_servers(payload: dict) -> bool:
    changed = False
    servers = payload.get("mcpServers")
    if isinstance(servers, dict):
        for name, server in list(servers.items()):
            if needs_socket_migration(str(name), server):
                servers[name] = socket_server_preserving(server)
                changed = True
    projects = payload.get("projects")
    if isinstance(projects, dict):
        for project_data in projects.values():
            if not isinstance(project_data, dict):
                continue
            nested = project_data.get("mcpServers")
            if not isinstance(nested, dict):
                continue
            for name, server in list(nested.items()):
                if needs_socket_migration(str(name), server):
                    nested[name] = socket_server_preserving(server)
                    changed = True
    return changed


def _toml_servers_needing_migration(text: str) -> list[str]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return []
    try:
        payload = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    return [
        name
        for name, server in iter_toml_mcp_servers(payload)
        if needs_socket_migration(name, server)
    ]


def _rewrite_codex_mcp_servers_to_socket(text: str, names: list[str]) -> str:
    """Rewrite matching [mcp_servers.NAME] / [mcpServers.NAME] blocks to socket form."""
    import re

    result = text
    for name in names:
        for table in ("mcp_servers", "mcpServers"):
            pattern = re.compile(
                rf"(?ms)^(\[{re.escape(table)}\.{re.escape(name)}\]\s*\n)(.*?)(?=^\[|\Z)",
            )
            match = pattern.search(result)
            if not match:
                continue
            header, body = match.group(1), match.group(2)
            kept_lines = []
            for line in body.splitlines(keepends=True):
                stripped = line.lstrip()
                if stripped.startswith("command") or stripped.startswith("args"):
                    continue
                kept_lines.append(line)
            new_body = (
                'command = "socat"\n'
                'args = ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]\n' + "".join(kept_lines)
            )
            result = result[: match.start()] + header + new_body + result[match.end() :]
    return result


def migrate_legacy_mcp_configs(
    config_paths: list[Path] | tuple[Path, ...] | None = None,
    *,
    bridge_command: str | None = None,
) -> list[Path]:
    """Rewrite any non-socket BrainLayer MCP entry to the canonical socat form.

    Shape-matcher (not name-matcher): bun wrappers, stdio-bridge, deleted
    ``brainlayer-mcp``, and python entrypoints are all rewritten. Each rewritten
    file is backed up beside the original first. Unreadable/malformed files are
    skipped (doctor reports them); they do not abort the whole run.
    """
    del bridge_command  # socket form is the only rewrite target
    changed: list[Path] = []
    paths = config_paths if config_paths is not None else get_default_mcp_config_paths()
    for config_path in paths:
        path = config_path.expanduser()
        if not path.is_file():
            continue
        try:
            target_path = path.resolve()
            original_text = target_path.read_text(encoding="utf-8")
        except OSError:
            continue

        suffix = target_path.suffix.lower()
        if suffix == ".toml" or target_path.name == "config.toml":
            names = _toml_servers_needing_migration(original_text)
            if not names:
                continue
            new_text = _rewrite_codex_mcp_servers_to_socket(original_text, names)
            if new_text == original_text:
                continue
            try:
                _backup_config_file(target_path)
                mode = target_path.stat().st_mode & 0o777
                temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
                try:
                    temporary.write_text(new_text, encoding="utf-8")
                    temporary.chmod(mode)
                    os.replace(temporary, target_path)
                finally:
                    temporary.unlink(missing_ok=True)
            except OSError:
                continue
            changed.append(path)
            continue

        try:
            payload = json.loads(original_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if not _rewrite_json_mcp_servers(payload):
            continue

        try:
            _backup_config_file(target_path)
            mode = target_path.stat().st_mode & 0o777
            temporary = target_path.with_name(f".{target_path.name}.{os.getpid()}.tmp")
            try:
                temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                temporary.chmod(mode)
                os.replace(temporary, target_path)
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            continue
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
