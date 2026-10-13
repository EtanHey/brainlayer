"""Tests for setup-time MCP config migration."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _scratch_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr("brainlayer.setup._require_mcp_bridge", lambda command: None)
    monkeypatch.setattr(
        "brainlayer.setup.get_current_mcp_bridge_bin",
        lambda: "/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge",
    )


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
    assert payload["mcp_servers"]["brainlayer"] == {
        "command": "/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"
    }
    assert payload["mcp_servers"]["other"]["command"] == "keep-me"
    assert payload["mcp_servers"]["other"]["command_timeout_sec"] == 30


def test_setup_toml_migrator_aborts_on_invalid_rewrite(tmp_path: Path, monkeypatch):
    from brainlayer import setup as setup_module

    codex = tmp_path / "config.toml"
    codex.write_text(MULTILINE_CODEX_TOML, encoding="utf-8")
    original = codex.read_text(encoding="utf-8")

    def broken_rewrite(text: str, names: list[str], bridge_command: str) -> str:
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


def test_migrate_socat_json_and_codex_to_bridge_with_backups(tmp_path: Path, monkeypatch):
    from brainlayer.setup import migrate_legacy_mcp_configs

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = tmp_path / "mcp.json"
    codex = tmp_path / "config.toml"
    server = {**SOCKET_MCP, "env": {"MCPLAYER_BRAINLAYER_FRONT": "/tmp/private-front.sock"}, "disabled": True}
    original = {
        "mcpServers": {"brainlayer": server, "other": SOCKET_MCP},
        "projects": {"repo": {"mcpServers": {"brainbar": server}}},
    }
    config.write_text(json.dumps(original))
    codex.write_text(
        '[mcp_servers.brainlayer]\ncommand="socat"\nargs=["STDIO","UNIX-CONNECT:/tmp/brainbar.sock"]\nstartup_timeout_sec=15\n[mcp_servers.brainlayer.env]\nMCPLAYER_BRAINLAYER_FRONT="/tmp/private-front.sock"\n[mcp_servers.other]\ncommand="keep-me"\n'
    )
    before = {p: p.read_bytes() for p in (config, codex)}
    report = migrate_legacy_mcp_configs(
        [config, codex], bridge_command="/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"
    )
    assert report.changed == (config, codex)
    assert report.skipped == ()
    migrated = json.loads(config.read_text())
    expected = {
        "command": "/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge",
        "env": server["env"],
        "disabled": True,
    }
    assert migrated["mcpServers"]["brainlayer"] == expected
    assert migrated["projects"]["repo"]["mcpServers"]["brainbar"] == expected
    assert migrated["mcpServers"]["other"] == SOCKET_MCP
    toml_server = tomllib.loads(codex.read_text())["mcp_servers"]["brainlayer"]
    assert toml_server == {"command": expected["command"], "env": server["env"], "startup_timeout_sec": 15}
    backups = list((tmp_path / "home/.local/share/brainlayer/config-backups").glob("*.bak"))
    assert sorted(p.read_bytes() for p in backups) == sorted(before.values())
    assert migrate_legacy_mcp_configs([config, codex]).changed == ()
    assert len(list(backups[0].parent.glob("*.bak"))) == 2
