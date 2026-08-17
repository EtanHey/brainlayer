"""Tests for setup-time MCP config migration."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

SOCKET_MCP = {
    "command": "socat",
    "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
}

MULTILINE_CODEX_TOML = (
    "[mcp_servers.brainlayer]\n"
    'command = "/Users/me/.bun/bin/bun"\n'
    "args = [\n"
    '  "/Users/me/.local/bin/brainlayer-mcp-stdio-bridge",\n'
    '  "extra",\n'
    "]\n"
    "\n"
    "[mcp_servers.other]\n"
    'command = "keep-me"\n'
    "command_timeout_sec = 30\n"
)


def test_setup_toml_migrator_preserves_multiline_args_and_unrelated_keys(tmp_path: Path):
    from brainlayer.setup import migrate_legacy_mcp_configs

    codex = tmp_path / "config.toml"
    codex.write_text(MULTILINE_CODEX_TOML, encoding="utf-8")
    report = migrate_legacy_mcp_configs([codex])
    assert report.changed == (codex,)
    payload = tomllib.loads(codex.read_text(encoding="utf-8"))
    assert payload["mcp_servers"]["brainlayer"]["command"] == "socat"
    assert payload["mcp_servers"]["brainlayer"]["args"] == SOCKET_MCP["args"]
    assert payload["mcp_servers"]["other"]["command"] == "keep-me"
    assert payload["mcp_servers"]["other"]["command_timeout_sec"] == 30


def test_setup_toml_migrator_aborts_on_invalid_rewrite(tmp_path: Path, monkeypatch):
    from brainlayer import setup as setup_module

    codex = tmp_path / "config.toml"
    codex.write_text(MULTILINE_CODEX_TOML, encoding="utf-8")
    original = codex.read_text(encoding="utf-8")

    def broken_rewrite(text: str, names: list[str]) -> str:
        return text.replace('command = "/Users/me/.bun/bin/bun"', 'command = "socat"\nargs = [')

    monkeypatch.setattr(setup_module, "_rewrite_codex_mcp_servers_to_socket", broken_rewrite)
    report = setup_module.migrate_legacy_mcp_configs([codex])
    assert report.changed == ()
    assert any(path == codex for path, _reason in report.skipped)
    assert codex.read_text(encoding="utf-8") == original


def test_setup_mcp_migration_reports_skipped_replace_failures(tmp_path: Path, monkeypatch):
    import brainlayer.setup as setup_helpers

    config_path = tmp_path / ".claude.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"brainlayer": {"command": "brainlayer-mcp"}}}) + "\n",
        encoding="utf-8",
    )

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise PermissionError("replace denied")

    monkeypatch.setattr(setup_helpers.os, "replace", fail_replace)
    report = setup_helpers.migrate_legacy_mcp_configs([config_path])
    assert report.changed == ()
    assert any(path == config_path for path, reason in report.skipped)
    assert "replace denied" in report.skipped[0][1]
