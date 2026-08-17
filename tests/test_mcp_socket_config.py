"""Tests for canonical BrainBar MCP socket config helpers."""

from __future__ import annotations

from pathlib import Path

SOCKET_MCP = {
    "command": "socat",
    "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
}


def test_shape_matcher_does_not_clobber_third_party_servers_via_heuristics():
    from brainlayer.mcp_socket_config import needs_socket_migration

    voicelayer_bun = {
        "command": "/Users/me/.bun/bin/bun",
        "args": ["/Users/me/.local/bin/brainlayer-mcp-stdio-bridge"],
    }
    assert not needs_socket_migration("voicelayer", voicelayer_bun)

    # Reviewer's python3 case verbatim — must not match without brainlayer server name.
    assert not needs_socket_migration(
        "voicelayer",
        {"command": "python3", "args": ["-m", "voicelayer.server", "--mem", "brainlayer.mcp"]},
    )
    assert not needs_socket_migration(
        "mytool",
        {"command": "python", "args": ["-m", "vendor.brainlayer.mcp_shim"]},
    )
    assert not needs_socket_migration("daemon", {"command": "brainlayer", "args": ["serve"]})


def test_brainlayer_stdio_bridge_is_acceptable_without_migration():
    from brainlayer.mcp_socket_config import needs_socket_migration

    assert not needs_socket_migration("brainlayer", {"command": "brainlayer-mcp-stdio-bridge"})
    assert needs_socket_migration(
        "brainlayer",
        {"command": "/Users/me/.bun/bin/bun", "args": ["/Users/me/.local/bin/brainlayer-mcp-stdio-bridge"]},
    )


def test_owned_paths_are_one_level_deep_and_skip_scratch_dirs(tmp_path: Path):
    from brainlayer.mcp_socket_config import owned_mcp_config_paths

    home = tmp_path / "home"
    gits = home / "Gits"
    (gits / "brainlayer").mkdir(parents=True)
    (gits / "brainlayer" / ".mcp.json").write_text("{}")
    nested = gits / "brainlayer" / "packages" / "nested"
    nested.mkdir(parents=True)
    (nested / ".mcp.json").write_text("{}")
    scratch = gits / "skill-creator" / "docs.local" / ".tmp" / "plugins"
    scratch.mkdir(parents=True)
    (scratch / ".mcp.json").write_text("{}")
    (gits / "other-repo").mkdir()
    (gits / "other-repo" / ".mcp.json").write_text("{}")

    paths = owned_mcp_config_paths(home=home)
    assert home / "Gits" / "brainlayer" / ".mcp.json" in paths
    assert home / "Gits" / "other-repo" / ".mcp.json" in paths
    assert not any("docs.local" in p.parts for p in paths)
    assert not any(".tmp" in p.parts for p in paths)
    assert not any(p.parent.name == "nested" for p in paths)


def test_socket_server_preserving_strips_remote_http_transport_keys():
    from brainlayer.mcp_socket_config import socket_server_preserving

    remote = {
        "type": "http",
        "url": "https://example.test/mcp",
        "headers": {"Authorization": "secret"},
        "command": "brainlayer-mcp",
        "disabled": True,
    }
    migrated = socket_server_preserving(remote)
    assert migrated["command"] == "socat"
    assert migrated["args"] == SOCKET_MCP["args"]
    assert migrated["disabled"] is True
    assert "type" not in migrated
    assert "url" not in migrated
    assert "headers" not in migrated
