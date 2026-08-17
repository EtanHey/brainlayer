from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOCKET_CONFIG = {
    "command": "socat",
    "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
}


def test_agent_mcp_docs_teach_socket_form_only() -> None:
    docs = [
        REPO_ROOT / "docs/mcp-config.md",
        REPO_ROOT / "docs/quickstart.md",
        REPO_ROOT / "docs/index.md",
        REPO_ROOT / "README.md",
    ]

    for path in docs:
        content = path.read_text(encoding="utf-8")
        assert '"command": "brainlayer-mcp"' not in content, str(path)
        assert "UNIX-CONNECT:/tmp/brainbar.sock" in content, str(path)


def test_agent_mcp_example_configs_prefer_socket_or_bridge() -> None:
    root_example = REPO_ROOT / ".mcp.json.example"
    plugin_config = REPO_ROOT / "extensions/brainlayer-plugin/.mcp.json"
    for path in (root_example, plugin_config):
        if not path.is_file():
            continue
        config = json.loads(path.read_text(encoding="utf-8"))
        brainlayer = config["mcpServers"]["brainlayer"]
        command = brainlayer.get("command", "")
        assert Path(str(command)).name != "brainlayer-mcp", str(path)
        serialized = json.dumps(config)
        assert '"command": "brainlayer-mcp"' not in serialized
