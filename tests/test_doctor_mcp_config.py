"""Doctor checks for owned MCP config health."""

from __future__ import annotations

from pathlib import Path


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
