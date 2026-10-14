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


@pytest.mark.parametrize("shape", ["json", "zed", "toml"])
@pytest.mark.parametrize(
    "target,args",
    [
        (target, ["STDIO", "UNIX-CONNECT:" + target + ",retry=1"])
        for target in ("/tmp/brainbar.sock", "/tmp/brainbar-dev.sock")
    ]
    + [
        ("unparsed", args)
        for args in (
            ["STDIO", "unix-connect:/tmp/brainbar.sock"],
            ["STDIO", "UNIX:/tmp/brainbar.sock"],
            ["STDIO", "UNIX-CLIENT:/tmp/brainbar.sock"],
            ["STDIO"],
            ["STDIO", "UNIX-CONNECT:"],
            ["STDIO", "UNIX-CONNECT:,retry=1"],
            ["STDIO", "UNIX-CONNECT:/tmp/a.sock", "UNIX-CONNECT:/tmp/b.sock"],
            None,
        )
    ],
)
@pytest.mark.parametrize("override", [None, "match", "conflict"])
@pytest.mark.parametrize("with_env", [False, True])
def test_socat_migration_keeps_socket_target_or_skips_conflict(tmp_path, shape, target, override, with_env, args):
    from brainlayer.mcp_stdio_bridge import config_from_env
    from brainlayer.setup import migrate_legacy_mcp_configs

    env = {"KEEP": "value", "MCPLAYER_BRAINLAYER_FRONT": "/tmp/front.sock"} if with_env else {}
    if override:
        env["BRAINLAYER_MCP_SOCKET"] = target if override == "match" else "/tmp/elsewhere.sock"
    path = tmp_path / ("config.toml" if shape == "toml" else "mcp.json")
    if shape == "toml":
        path.write_text(
            '[mcp_servers.brainlayer]\ncommand="socat"\nargs='
            + (json.dumps(args) if args is not None else "[]")
            + (
                "\n[mcp_servers.brainlayer.env]\n" + "".join(f"{k}={json.dumps(v)}\n" for k, v in env.items())
                if env
                else "\n"
            )
        )
    else:
        server = (
            {"command": {"path": "socat", "args": args, "env": env}}
            if shape == "zed"
            else {"command": "socat", "args": args, "env": env}
        )
        if not env:
            (server["command"] if shape == "zed" else server).pop("env")
        path.write_text(json.dumps({"mcpServers": {"brainlayer": server}}))
    before = path.read_bytes()
    report = migrate_legacy_mcp_configs([path])
    if override == "conflict" or target == "unparsed":
        assert report.changed == ()
        assert path.read_bytes() == before
        assert report.skipped and report.skipped[0][0] == path
        if target == "unparsed":
            assert "single non-empty UNIX-CONNECT target" in report.skipped[0][1]
        else:
            assert target in report.skipped[0][1] and "/tmp/elsewhere.sock" in report.skipped[0][1]
        assert not list((tmp_path / "home" / ".local/share/brainlayer/config-backups").glob("*"))
        if shape == "toml":
            path.write_text(
                path.read_text()
                + '\n[mcp_servers.brainbar]\ncommand="socat"\nargs=["STDIO","UNIX-CONNECT:/tmp/custom.sock,retry=1"]\n'
            )
        else:
            payload = json.loads(path.read_text())
            payload["mcpServers"]["brainbar"] = {
                "command": "socat",
                "args": ["STDIO", "UNIX-CONNECT:/tmp/custom.sock,retry=1"],
            }
            path.write_text(json.dumps(payload))
        mixed_before = tomllib.loads(path.read_text()) if shape == "toml" else json.loads(path.read_text())
        report = migrate_legacy_mcp_configs([path])
        assert report.changed == (path,) and report.skipped
        mixed_after = tomllib.loads(path.read_text()) if shape == "toml" else json.loads(path.read_text())
        key = "mcp_servers" if shape == "toml" else "mcpServers"
        assert mixed_after[key]["brainlayer"] == mixed_before[key]["brainlayer"]
        assert mixed_after[key]["brainbar"]["command"].endswith("brainlayer-mcp-stdio-bridge")
        assert mixed_after[key]["brainbar"]["env"]["BRAINLAYER_MCP_SOCKET"] == "/tmp/custom.sock"
        assert migrate_legacy_mcp_configs([path]).changed == ()
    else:
        assert report.changed == (path,) and report.skipped == ()
        payload = tomllib.loads(path.read_text()) if shape == "toml" else json.loads(path.read_text())
        server = payload["mcp_servers" if shape == "toml" else "mcpServers"]["brainlayer"]
        migrated_env = (server["command"] if shape == "zed" else server).get("env", {})
        assert config_from_env(migrated_env).socket_path == target
        assert migrated_env.get("KEEP") == ("value" if with_env else None)
        assert "MCPLAYER_BRAINLAYER_FRONT" not in migrated_env
        assert migrate_legacy_mcp_configs([path]).changed == ()


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
    server = {
        **SOCKET_MCP,
        "env": {"MCPLAYER_BRAINLAYER_FRONT": "/tmp/private-front.sock", "KEEP": "value"},
        "disabled": True,
    }
    original = {
        "mcpServers": {"brainlayer": server, "other": SOCKET_MCP},
        "projects": {"repo": {"mcpServers": {"brainbar": server}}},
    }
    config.write_text(json.dumps(original))
    codex.write_text(
        '[mcp_servers.brainlayer]\ncommand="socat"\nargs=["STDIO","UNIX-CONNECT:/tmp/brainbar.sock"]\nstartup_timeout_sec=15\n[mcp_servers.brainlayer.env]\nMCPLAYER_BRAINLAYER_FRONT="/tmp/private-front.sock"\nKEEP="value"\n[mcp_servers.other]\ncommand="keep-me"\n'
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
        "env": {"KEEP": "value"},
        "disabled": True,
    }
    assert migrated["mcpServers"]["brainlayer"] == expected
    assert migrated["projects"]["repo"]["mcpServers"]["brainbar"] == expected
    assert migrated["mcpServers"]["other"] == SOCKET_MCP
    toml_server = tomllib.loads(codex.read_text())["mcp_servers"]["brainlayer"]
    assert toml_server == {"command": expected["command"], "env": expected["env"], "startup_timeout_sec": 15}
    backups = list((tmp_path / "home/.local/share/brainlayer/config-backups").glob("*.bak"))
    assert sorted(p.read_bytes() for p in backups) == sorted(before.values())
    assert migrate_legacy_mcp_configs([config, codex]).changed == ()
    assert len(list(backups[0].parent.glob("*.bak"))) == 2


@pytest.mark.parametrize("nested", [False, True])
def test_socat_migration_preserves_direct_route_and_existing_bridge(tmp_path: Path, nested):
    from brainlayer.mcp_stdio_bridge import config_from_env
    from brainlayer.setup import migrate_legacy_mcp_configs

    bridge = "/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"
    env = {"MCPLAYER_BRAINLAYER_FRONT": "/tmp/private-front.sock", "KEEP": "MCPLAYER_BRAINLAYER_FRONT"}
    command = {"path": "/usr/bin/socat", "args": SOCKET_MCP["args"], "env": env} if nested else "socat"
    old = {"command": command, "env": env, "settings": {"keep": True}}
    if not nested:
        old["args"] = SOCKET_MCP["args"]
    existing = {"command": bridge, "env": env}
    config = tmp_path / "mcp.json"
    original = json.dumps({"mcpServers": {"brainlayer": old, "brainbar": existing, "foreign": old}})
    config.write_text(original)
    report = migrate_legacy_mcp_configs([config])
    assert report.changed == (config,)
    servers = json.loads(config.read_text())["mcpServers"]
    converted = servers["brainlayer"]
    assert converted["env"] == {"KEEP": env["KEEP"]}
    if nested:
        assert converted["command"]["env"] == converted["env"]
        assert converted["command"]["path"] == bridge
    assert config_from_env(converted["env"]).socket_path == "/tmp/brainbar.sock"
    assert servers["brainbar"] == existing
    assert servers["foreign"] == old
    assert old["env"] == env  # no mutation of the preimage's shared dict
    backups = list((tmp_path / "home/.local/share/brainlayer/config-backups").glob("*.bak"))
    assert [p.read_text() for p in backups] == [original]
    assert migrate_legacy_mcp_configs([config]).changed == ()


@pytest.mark.parametrize(
    "env_text",
    [
        'env = {MCPLAYER_BRAINLAYER_FRONT="/tmp/front.sock", KEEP="MCPLAYER_BRAINLAYER_FRONT"}\n',
        'env.MCPLAYER_BRAINLAYER_FRONT="/tmp/front.sock"\nenv.KEEP="MCPLAYER_BRAINLAYER_FRONT"\n',
        '[mcp_servers.brainlayer.env]\n"MCPLAYER_BRAINLAYER_FRONT"="/tmp/front.sock"\nKEEP="MCPLAYER_BRAINLAYER_FRONT"\n',
        '[mcp_servers.brainlayer.env]\nMCPLAYER_BRAINLAYER_FRONT="""/tmp/front.sock"""\nKEEP="MCPLAYER_BRAINLAYER_FRONT"\n',
    ],
)
def test_codex_socat_drops_only_front_env(tmp_path: Path, env_text):
    from brainlayer.setup import migrate_legacy_mcp_configs

    config = tmp_path / "config.toml"
    original = (
        '[mcp_servers.brainlayer]\ncommand="/usr/bin/socat"\nargs=["STDIO","UNIX-CONNECT:/tmp/brainbar.sock"]\n'
        + env_text
        + '[mcp_servers.brainbar]\ncommand="/opt/homebrew/opt/brainlayer/bin/brainlayer-mcp-stdio-bridge"\n'
        + '[mcp_servers.brainbar.env]\nMCPLAYER_BRAINLAYER_FRONT="/tmp/existing.sock"\n'
    )
    config.write_text(original)
    assert migrate_legacy_mcp_configs([config]).changed == (config,)
    servers = tomllib.loads(config.read_text())["mcp_servers"]
    assert servers["brainlayer"]["env"] == {"KEEP": "MCPLAYER_BRAINLAYER_FRONT"}
    assert servers["brainbar"] == tomllib.loads(original)["mcp_servers"]["brainbar"]
    backups = list((tmp_path / "home/.local/share/brainlayer/config-backups").glob("*.bak"))
    assert [p.read_text() for p in backups] == [original]
    assert migrate_legacy_mcp_configs([config]).changed == ()


def test_codex_migration_refuses_an_unsafe_text_rewrite_before_backup(tmp_path: Path):
    from brainlayer.setup import migrate_legacy_mcp_configs

    config = tmp_path / "config.toml"
    # A table-like line inside a literal must never be treated as a real env table.
    original = (
        '[mcp_servers.brainlayer]\ncommand="socat"\n'
        "note = '''keep\n[mcp_servers.brainlayer.env]\nMCPLAYER_BRAINLAYER_FRONT=literal\n'''\n"
        '[mcp_servers.brainlayer.env]\nMCPLAYER_BRAINLAYER_FRONT="/tmp/front.sock"\n'
    )
    config.write_text(original)
    report = migrate_legacy_mcp_configs([config])
    assert report.changed == ()
    assert len(report.skipped) == 1
    assert config.read_text() == original
    assert not (tmp_path / "home/.local/share/brainlayer/config-backups").exists()
