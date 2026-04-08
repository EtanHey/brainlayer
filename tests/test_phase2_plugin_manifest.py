"""Phase 2 plugin manifest and hook contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_ROOT = REPO_ROOT / "extensions" / "brainlayer-plugin"
MARKETPLACE_PATH = REPO_ROOT / ".claude-plugin" / "marketplace.json"
PLUGIN_MANIFEST_PATH = PLUGIN_ROOT / ".claude-plugin" / "plugin.json"
PLUGIN_MCP_PATH = PLUGIN_ROOT / ".mcp.json"
HOOKS_PATH = PLUGIN_ROOT / "hooks" / "hooks.json"
SKILL_PATH = PLUGIN_ROOT / "skills" / "memory" / "SKILL.md"


def _load_json(path: Path) -> dict:
    assert path.exists(), f"expected file to exist: {path}"
    return json.loads(path.read_text())


def test_marketplace_catalog_lists_brainlayer_plugin():
    marketplace = _load_json(MARKETPLACE_PATH)

    assert marketplace["name"] == "brainlayer"
    plugins = marketplace["plugins"]
    assert plugins == [
        {
            "name": "brainlayer",
            "source": "./extensions/brainlayer-plugin",
        }
    ]


def test_plugin_manifest_has_brainlayer_identity():
    manifest = _load_json(PLUGIN_MANIFEST_PATH)

    assert manifest["name"] == "brainlayer"
    assert "BrainLayer" in manifest["description"]
    assert "version" in manifest


def test_plugin_mcp_config_points_at_brainbar_socket():
    mcp_config = _load_json(PLUGIN_MCP_PATH)

    assert mcp_config == {
        "mcpServers": {
            "brainlayer": {
                "command": "socat",
                "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
            }
        }
    }


def test_hooks_register_phase2_lifecycle_events():
    hooks_config = _load_json(HOOKS_PATH)["hooks"]

    assert set(hooks_config) == {"SessionStart", "PostToolUse", "PostToolUseFailure", "Stop"}
    assert hooks_config["SessionStart"][0]["hooks"][0]["type"] == "command"
    assert hooks_config["PostToolUse"][0]["hooks"][0]["async"] is True
    assert hooks_config["PostToolUseFailure"][0]["hooks"][0]["async"] is True


@pytest.mark.parametrize(
    ("script_name", "max_lines"),
    [
        ("session-start.sh", 25),
        ("tool-observe.sh", 25),
        ("tool-error.sh", 25),
        ("session-stop.sh", 25),
    ],
)
def test_shell_hooks_stay_thin(script_name: str, max_lines: int):
    script_path = PLUGIN_ROOT / "hooks" / script_name
    assert script_path.exists(), f"expected file to exist: {script_path}"
    line_count = len(script_path.read_text().splitlines())
    assert line_count <= max_lines


def test_skill_teaches_search_store_and_recall_usage():
    assert SKILL_PATH.exists(), f"expected file to exist: {SKILL_PATH}"

    skill = SKILL_PATH.read_text()
    assert "brain_search" in skill
    assert "brain_store" in skill
    assert "brain_recall" in skill
    assert "before answering" in skill.lower()
