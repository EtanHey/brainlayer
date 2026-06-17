from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOCKET_ARGS = ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"]


def test_agent_mcp_templates_use_brainbar_socket_bridge() -> None:
    root_config = json.loads((REPO_ROOT / ".mcp.json.example").read_text(encoding="utf-8"))
    plugin_config = json.loads((REPO_ROOT / "extensions/brainlayer-plugin/.mcp.json").read_text(encoding="utf-8"))

    for config in (root_config, plugin_config):
        brainlayer = config["mcpServers"]["brainlayer"]
        assert brainlayer == {"command": "socat", "args": SOCKET_ARGS}

    for config in (root_config, plugin_config):
        serialized = json.dumps(config)
        assert "brainlayer-legacy" not in serialized
        assert "brainlayer-mcp" not in serialized


def test_agent_mcp_docs_do_not_teach_direct_brainlayer_mcp_spawn() -> None:
    docs = [
        REPO_ROOT / "docs/mcp-config.md",
        REPO_ROOT / "docs/quickstart.md",
        REPO_ROOT / "docs/index.md",
    ]

    for path in docs:
        content = path.read_text(encoding="utf-8")
        assert '"command": "brainlayer-mcp"' not in content, str(path)
        assert '"command": "socat"' in content, str(path)
        assert '"UNIX-CONNECT:/tmp/brainbar.sock"' in content, str(path)
