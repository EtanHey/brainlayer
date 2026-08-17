"""ONE memory server: delete Python MCP entrypoint; migrate configs to socket."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from typer.testing import CliRunner

SOCKET_MCP = {
    "command": "socat",
    "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
}

BRIDGE_VIA_BUN = {
    "command": "/Users/me/.bun/bin/bun",
    "args": ["/Users/me/.local/bin/brainlayer-mcp-stdio-bridge"],
}

CELLAR_BRIDGE_VIA_BUN = {
    "command": "/Users/me/.bun/bin/bun",
    "args": ["/opt/homebrew/Cellar/brainlayer/1.5.6/libexec/venv/bin/brainlayer-mcp-stdio-bridge"],
}


def test_pyproject_has_no_brainlayer_mcp_console_script():
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]
    assert "brainlayer-mcp" not in scripts
    assert "brainlayer-mcp-stdio-bridge" in scripts


def test_mcp_module_has_no_serve_doorway():
    module_path = Path(__file__).resolve().parents[1] / "src" / "brainlayer" / "mcp" / "__init__.py"
    source = module_path.read_text(encoding="utf-8")

    assert "def serve(" not in source
    assert "server = FastMCP(" not in source
    assert "server = Server(" not in source


def test_shape_matcher_flags_bridge_and_bun_not_only_literal_name():
    from brainlayer.mcp_socket_config import needs_socket_migration

    assert needs_socket_migration("brainlayer", {"command": "brainlayer-mcp"})
    assert not needs_socket_migration("brainlayer", {"command": "brainlayer-mcp-stdio-bridge"})
    assert needs_socket_migration("brainlayer", BRIDGE_VIA_BUN)
    assert needs_socket_migration("brainlayer", CELLAR_BRIDGE_VIA_BUN)
    assert not needs_socket_migration("brainlayer", SOCKET_MCP)
    assert not needs_socket_migration("other", {"command": "keep-me"})


def test_setup_migrates_bridge_and_bun_shapes_to_socket(tmp_path: Path):
    from brainlayer.setup import migrate_legacy_mcp_configs

    config_path = tmp_path / ".claude.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brainlayer": BRIDGE_VIA_BUN,
                    "unrelated": {"command": "other-mcp"},
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert migrate_legacy_mcp_configs([config_path]).changed == (config_path,)
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["mcpServers"]["brainlayer"] == SOCKET_MCP
    assert migrated["mcpServers"]["unrelated"] == {"command": "other-mcp"}
    assert migrate_legacy_mcp_configs([config_path]).changed == ()


def test_setup_migrates_brainlayer_mcp_json_to_socket_with_backup(tmp_path: Path, monkeypatch):
    from brainlayer.setup import migrate_legacy_mcp_configs

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config_path = home / ".claude.json"
    config_path.parent.mkdir(parents=True)
    original = {
        "theme": "dark",
        "mcpServers": {
            "brainlayer": {"command": "brainlayer-mcp"},
            "unrelated": {"command": "other-mcp"},
        },
    }
    config_path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    changed = migrate_legacy_mcp_configs([config_path])
    second = migrate_legacy_mcp_configs([config_path])

    assert changed.changed == (config_path,)
    assert second.changed == ()
    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert migrated["mcpServers"]["brainlayer"] == SOCKET_MCP
    assert migrated["mcpServers"]["unrelated"] == original["mcpServers"]["unrelated"]
    backup_dir = home / ".local" / "share" / "brainlayer" / "config-backups"
    backups = list(backup_dir.glob(".claude.json.*.bak"))
    assert len(backups) == 1
    assert json.loads(backups[0].read_text(encoding="utf-8")) == original


def test_setup_tolerates_invalid_json_without_aborting(tmp_path: Path):
    from brainlayer.setup import migrate_legacy_mcp_configs

    bad = tmp_path / "bad.mcp.json"
    good = tmp_path / "good.json"
    bad.write_text("{not-json\n", encoding="utf-8")
    good.write_text(json.dumps({"mcpServers": {"brainlayer": {"command": "brainlayer-mcp"}}}) + "\n")

    changed = migrate_legacy_mcp_configs([bad, good])
    assert changed.changed == (good,)
    assert any(path == bad for path, _reason in changed.skipped)
    assert json.loads(good.read_text())["mcpServers"]["brainlayer"] == SOCKET_MCP


def test_owned_paths_exclude_worktrees(tmp_path: Path):
    from brainlayer.mcp_socket_config import owned_mcp_config_paths

    home = tmp_path / "home"
    gits = home / "Gits"
    (gits / "brainlayer").mkdir(parents=True)
    (gits / "brainlayer" / ".mcp.json").write_text("{}")
    wt = gits / "brainlayer" / ".worktrees" / "x"
    wt.mkdir(parents=True)
    (wt / ".mcp.json").write_text("{}")

    paths = owned_mcp_config_paths(home=home)
    assert home / "Gits" / "brainlayer" / ".mcp.json" in paths
    assert not any(".worktrees" in p.parts for p in paths)


def test_setup_cli_migrates_all_four_owned_shapes_under_temp_home(tmp_path: Path, monkeypatch):
    """Real CLI integration: no mocked migrator; asserts file contents after setup."""
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    home = tmp_path / "home"
    claude = home / ".claude.json"
    cursor = home / ".cursor" / "mcp.json"
    gemini = home / ".gemini" / "settings.json"
    codex = home / ".codex" / "config.toml"
    cursor.parent.mkdir(parents=True)
    gemini.parent.mkdir(parents=True)
    codex.parent.mkdir(parents=True)

    claude.write_text(json.dumps({"mcpServers": {"brainlayer": {"command": "brainlayer-mcp"}}}) + "\n")
    cursor.write_text(json.dumps({"mcpServers": {"brainlayer": BRIDGE_VIA_BUN}}) + "\n")
    gemini.write_text(json.dumps({"mcpServers": {"brainlayer": {"command": "brainlayer-mcp"}}}) + "\n")
    codex.write_text(
        "[mcp_servers.brainlayer]\n"
        'command = "/Users/me/.bun/bin/bun"\n'
        'args = ["/Users/me/.local/bin/brainlayer-mcp-stdio-bridge"]\n'
        "\n"
        "[mcp_servers.other]\n"
        'command = "keep-me"\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        setup_helpers,
        "get_default_mcp_config_paths",
        lambda: (claude, cursor, gemini, codex),
    )
    # Avoid darwin spotlight side effects during CLI setup.
    monkeypatch.setattr(setup_helpers, "ensure_spotlight_excluded_layout", lambda **kwargs: ())

    result = CliRunner().invoke(
        app,
        ["setup", "--env-file", str(tmp_path / "brainlayer.env"), "--migrate-mcp", "--no-verify-mcp"],
    )
    assert result.exit_code == 0, result.stdout
    assert "Migrated MCP config:" in result.stdout

    assert json.loads(claude.read_text())["mcpServers"]["brainlayer"] == SOCKET_MCP
    assert json.loads(cursor.read_text())["mcpServers"]["brainlayer"] == SOCKET_MCP
    assert json.loads(gemini.read_text())["mcpServers"]["brainlayer"] == SOCKET_MCP
    codex_payload = tomllib.loads(codex.read_text(encoding="utf-8"))
    assert codex_payload["mcp_servers"]["brainlayer"]["command"] == "socat"
    assert codex_payload["mcp_servers"]["brainlayer"]["args"] == SOCKET_MCP["args"]
    assert codex_payload["mcp_servers"]["other"]["command"] == "keep-me"


def test_doctor_errors_on_codex_toml_bridge_shape(tmp_path: Path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    codex = tmp_path / "config.toml"
    codex.write_text(
        "[mcp_servers.brainlayer]\n"
        'command = "/Users/me/.bun/bin/bun"\n'
        'args = ["/Users/me/.local/bin/brainlayer-mcp-stdio-bridge"]\n',
        encoding="utf-8",
    )
    issues = _legacy_python_mcp_config_issues([codex])
    assert len(issues) == 1
    assert issues[0].severity == "fatal"
    assert issues[0].code == "legacy_python_mcp_entrypoint"
    assert "UNIX-CONNECT:/tmp/brainbar.sock" in (issues[0].details.get("fix") or "")


def test_doctor_errors_on_stale_python_mcp_entrypoint(tmp_path: Path):
    from brainlayer.doctor import _legacy_python_mcp_config_issues

    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"brainlayer": {"command": "brainlayer-mcp"}}}) + "\n",
        encoding="utf-8",
    )
    issues = _legacy_python_mcp_config_issues([config_path])
    assert len(issues) == 1
    assert issues[0].severity == "fatal"
    assert issues[0].code == "legacy_python_mcp_entrypoint"
    assert "UNIX-CONNECT:/tmp/brainbar.sock" in (issues[0].details.get("fix") or "")


def test_doctor_run_fails_on_stale_mcp_entrypoint(tmp_path: Path):
    from brainlayer.doctor import DoctorConfig, run_doctor
    from tests.test_doctor import NOW, _build_db, _hotlane_ps, _loaded_launchctl

    db_path = tmp_path / "healthy.db"
    _build_db(db_path)
    config_path = tmp_path / "mcp.json"
    config_path.write_text(
        json.dumps({"mcpServers": {"brainlayer": BRIDGE_VIA_BUN}}) + "\n",
        encoding="utf-8",
    )
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    watcher_health_path = tmp_path / "watcher-health.json"
    drain_health_path = tmp_path / "drain-health.json"
    watcher_health_path.write_text(json.dumps({"poll_count": 12}), encoding="utf-8")
    drain_health_path.write_text(json.dumps({"drained_total": 34}), encoding="utf-8")

    result = run_doctor(
        DoctorConfig(
            db_path=db_path,
            queue_dir=queue_dir,
            watcher_health_path=watcher_health_path,
            drain_health_path=drain_health_path,
            deploy_provenance_dir=tmp_path / "daemon-provenance",
            queue_movement_sample_seconds=0,
            version_check_enabled=False,
            mcp_config_paths=(config_path,),
            mcp_config_check_enabled=True,
            roundtrip_probe_enabled=False,
            deploy_drift_enabled=False,
            spotlight_check_enabled=False,
            hotlane_label="",
            watch_label="",
            drain_label="",
            enrichment_label="",
        ),
        ps_output_fn=_hotlane_ps,
        command_runner=_loaded_launchctl,
        now_fn=lambda: NOW,
    )

    assert result.ok is False
    assert result.exit_code == 1
    assert any(i.code == "legacy_python_mcp_entrypoint" and i.severity == "fatal" for i in result.issues)


def test_docs_and_readme_use_socket_form_only():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "README.md",
        "AGENTS.md",
        "docs/mcp-config.md",
        "docs/quickstart.md",
        "docs/brew-layer-conformance.md",
        "docs/architecture.md",
    ):
        content = (root / rel).read_text(encoding="utf-8")
        assert '"command": "brainlayer-mcp"' not in content, rel
        assert "brainlayer serve" not in content, rel
        assert "UNIX-CONNECT:/tmp/brainbar.sock" in content, rel
