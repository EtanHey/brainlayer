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


def test_backup_config_files_with_same_basename_do_not_collide(tmp_path: Path, monkeypatch):
    from brainlayer import setup as setup_module

    home = tmp_path / "home"
    monkeypatch.setattr(
        setup_module,
        "_config_backup_dir",
        lambda: home / ".local" / "share" / "brainlayer" / "config-backups",
    )
    repo_a = tmp_path / "repoA" / ".mcp.json"
    repo_b = tmp_path / "repoB" / ".mcp.json"
    repo_a.parent.mkdir()
    repo_b.parent.mkdir()
    repo_a.write_text("from-a", encoding="utf-8")
    repo_b.write_text("from-b", encoding="utf-8")

    bak_a = setup_module._backup_config_file(repo_a)
    bak_b = setup_module._backup_config_file(repo_b)

    assert bak_a != bak_b
    assert bak_a.exists() and bak_b.exists()
    assert bak_a.read_text(encoding="utf-8") == "from-a"
    assert bak_b.read_text(encoding="utf-8") == "from-b"


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
