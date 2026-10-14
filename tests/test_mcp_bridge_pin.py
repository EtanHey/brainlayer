"""The configured and verified transport must be the same stable keg bridge."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

PIN = "/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"


def test_setup_writes_and_verifies_one_opt_path_despite_path_shadow(tmp_path, monkeypatch):
    import brainlayer.setup as helpers
    from brainlayer.cli import app

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    shadow = tmp_path / "bin" / "brainlayer-mcp-stdio-bridge"
    shadow.parent.mkdir()
    shadow.write_text("#!/bin/sh\nexit 90\n")
    shadow.chmod(0o755)
    monkeypatch.setenv("PATH", str(shadow.parent))
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {"mcpServers": {"brainlayer": {"command": "socat", "args": ["-", "UNIX-CONNECT:/tmp/brainbar.sock"]}}}
        )
    )
    monkeypatch.setattr(helpers, "get_default_mcp_config_paths", lambda: (config,))
    monkeypatch.setattr(helpers, "ensure_spotlight_excluded_layout", lambda **kw: ())
    resolved = []
    monkeypatch.setattr(helpers, "get_current_mcp_bridge_bin", lambda: resolved.append(PIN) or PIN)
    monkeypatch.setattr(helpers, "_require_mcp_bridge", lambda command: None)
    verified = []
    monkeypatch.setattr(helpers, "verify_mcp_transport", lambda **kw: verified.append(kw["bridge_command"]) or 17)
    result = CliRunner().invoke(app, ["setup", "--env-file", str(tmp_path / "env"), "--migrate-mcp", "--verify-mcp"])
    assert result.exit_code == 0, result.stdout
    written = json.loads(config.read_text())["mcpServers"]["brainlayer"]["command"]
    assert written == PIN
    assert verified == [written]
    assert Path(written).is_absolute() and "Cellar" not in written
    assert resolved == [PIN]


@pytest.mark.parametrize(
    "command",
    [
        "brainlayer-mcp-stdio-bridge",
        "/opt/homebrew/Cellar/brainlayer/1/bin/brainlayer-mcp-stdio-bridge",
        "/scratch/.worktrees/x/brainlayer-mcp-stdio-bridge",
    ],
)
def test_doctor_reports_unpinned_bridge_and_resolved_target(tmp_path, monkeypatch, command):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    shadow = tmp_path / "bin" / "brainlayer-mcp-stdio-bridge"
    shadow.parent.mkdir()
    shadow.write_text("#!/bin/sh\n")
    shadow.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(shadow.parent))
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"memory": {"command": command}}}))
    issues = _legacy_python_mcp_config_issues([config])
    assert len(issues) == 1
    assert issues[0].code == "mcp_bridge_path_risk"
    assert issues[0].details["config_path"] == str(config)
    assert issues[0].details["resolved_target"] == str(shadow if command == shadow.name else Path(command))


def test_doctor_checks_opt_target_is_executable_inside_keg(tmp_path, monkeypatch):
    import brainlayer.mcp_socket_config as matcher
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    pin = tmp_path / "opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"
    target = tmp_path / "Cellar/brainlayer/1/bin/brainlayer-mcp-stdio-bridge"
    target.parent.mkdir(parents=True)
    target.write_text("#!/bin/sh\n")
    target.chmod(0o755)
    pin.parent.mkdir(parents=True)
    pin.symlink_to(target)
    monkeypatch.setattr(matcher, "KEG_MCP_BRIDGE_PATHS", (str(pin),))
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"brainlayer": {"command": str(pin)}}}))
    from brainlayer.setup import get_current_mcp_bridge_bin

    monkeypatch.setenv("PATH", "")
    assert get_current_mcp_bridge_bin() == str(pin)
    assert _legacy_python_mcp_config_issues([config]) == []
    config.write_text(json.dumps({"mcpServers": {"brainlayer": {"command": {"path": str(pin)}}}}))
    assert _legacy_python_mcp_config_issues([config]) == []
    from brainlayer.mcp_socket_config import socket_server_preserving

    assert socket_server_preserving({"command": {"path": "old", "args": ["old"]}}, bridge_command=str(pin)) == {
        "command": {"path": str(pin), "args": []}
    }
    pin.unlink()
    foreign = tmp_path / "foreign/brainlayer-mcp-stdio-bridge"
    foreign.parent.mkdir()
    foreign.write_text("#!/bin/sh\n")
    foreign.chmod(0o755)
    pin.symlink_to(foreign)
    issues = _legacy_python_mcp_config_issues([config])
    assert len(issues) == 1 and issues[0].details["resolved_target"] == str(foreign)
    pin.unlink()
    assert len(_legacy_python_mcp_config_issues([config])) == 1
    from brainlayer.setup import migrate_legacy_mcp_configs

    server = {"command": "socat", "args": ["-", "UNIX-CONNECT:/tmp/brainbar.sock"]}
    config.write_text(json.dumps({"mcpServers": {"brainlayer": server}}))
    before = config.read_bytes()
    with pytest.raises(FileNotFoundError):
        migrate_legacy_mcp_configs([config])
    assert config.read_bytes() == before
    config.write_text('{"mcpServers": {"other": {"command": "keep"}}}')
    assert migrate_legacy_mcp_configs([config]).changed == ()


def test_doctor_flags_any_socat_brainbar_argument_on_any_key(tmp_path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    config = tmp_path / "config.toml"
    config.write_text('[mcp_servers.memory]\ncommand="socat"\nargs=["-", "UNIX-CONNECT:/tmp/brainbar.sock,retry=2"]\n')
    issues = _legacy_python_mcp_config_issues([config])
    assert len(issues) == 1 and issues[0].code == "mcp_transport_restart_risk"
