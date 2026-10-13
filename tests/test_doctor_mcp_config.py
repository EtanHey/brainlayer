"""Doctor checks for owned MCP config health."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_doctor_does_not_crash_on_invalid_utf8_mcp_config(tmp_path: Path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    bad = tmp_path / ".mcp.json"
    bad.write_bytes(b"\xff\xfe not utf-8")
    good = tmp_path / "good.json"
    good.write_text(
        '{"mcpServers":{"brainlayer":{"command":"socat","args":["STDIO","UNIX-CONNECT:/tmp/brainbar.sock"]}}}\n',
        encoding="utf-8",
    )
    issues = _legacy_python_mcp_config_issues([bad, good])
    assert any(i.code == "mcp_config_unreadable" and i.details.get("config_path") == str(bad) for i in issues)
    assert not any(i.code == "legacy_python_mcp_entrypoint" for i in issues)


def test_doctor_reports_unparseable_json_config_as_fatal(tmp_path: Path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    bad = tmp_path / "broken.json"
    bad.write_text("{not-json\n", encoding="utf-8")
    issues = _legacy_python_mcp_config_issues([bad])
    assert len(issues) == 1
    assert issues[0].code == "mcp_config_unparseable"
    assert issues[0].severity == "fatal"
    assert "server" not in issues[0].details


def test_doctor_legacy_issue_details_exclude_full_server_dict(tmp_path: Path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        '{"mcpServers":{"brainlayer":{"command":"brainlayer-mcp","env":{"OPENAI_API_KEY":"secret"}}}}',
        encoding="utf-8",
    )
    issues = _legacy_python_mcp_config_issues([config_path])
    assert len(issues) == 1
    details = issues[0].details
    assert "env" not in str(details)
    assert details["server"]["command"] == "brainlayer-mcp"
    assert "args_preview" not in details["server"]


def test_doctor_names_raw_socat_restart_risk_without_config_secrets(tmp_path: Path):
    import json

    from brainlayer.doctor import _legacy_python_mcp_config_issues

    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "memory-alias": {
                        "command": "/opt/bin/socat",
                        "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
                        "env": {"TOKEN": "private-value"},
                    },
                    "other": {"command": "socat", "args": ["STDIO", "UNIX-CONNECT:/tmp/other.sock"]},
                }
            }
        )
    )
    issues = _legacy_python_mcp_config_issues([config])
    assert len(issues) == 1
    assert issues[0].code == "mcp_transport_restart_risk"
    assert issues[0].severity == "warning"
    assert str(config) in issues[0].message
    assert "restart" in issues[0].message.lower()
    assert "brainlayer-mcp-stdio-bridge" in issues[0].details["fix"]
    assert "private-value" not in str(issues)


def test_doctor_flags_nonowned_zed_socat_args(tmp_path: Path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    config = tmp_path / "mcp.json"
    config.write_text(
        '{"mcpServers":{"memory":{"command":{"path":"socat","args":["-","UNIX-CONNECT:/tmp/brainbar.sock"]}}}}'
    )
    issues = _legacy_python_mcp_config_issues([config])
    assert [(i.code, i.severity) for i in issues] == [("mcp_transport_restart_risk", "warning")]


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_doctor_proxy_route_is_info_without_env_values(tmp_path: Path, monkeypatch, nested, override):
    import json

    from brainlayer.doctor import _legacy_python_mcp_config_issues

    monkeypatch.setattr("brainlayer.doctor.bridge_path_risk", lambda server: None)
    env = {"MCPLAYER_BRAINLAYER_FRONT": "/tmp/private-front.sock", "TOKEN": "secret-value"}
    if override:
        env["BRAINLAYER_MCP_SOCKET"] = "/tmp/direct.sock"
    bridge = "/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"
    server = {"command": {"path": bridge, "env": env}} if nested else {"command": bridge, "env": env}
    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"brainlayer": server}}))
    issues = _legacy_python_mcp_config_issues([config])
    assert [(i.code, i.severity) for i in issues] == ([] if override else [("mcp_bridge_proxy_route", "info")])
    assert "private-front.sock" not in str(issues)
    assert "secret-value" not in str(issues)
