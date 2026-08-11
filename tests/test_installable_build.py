from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path
from typing import get_type_hints

import pytest
from typer.testing import CliRunner

REPO_ROOT = Path(__file__).resolve().parents[1]


def _plist_args(name: str) -> list[str]:
    plist_path = REPO_ROOT / "scripts" / "launchd" / f"com.brainlayer.{name}.plist"
    return plistlib.loads(plist_path.read_bytes())["ProgramArguments"]


def _write_full_launchd_env(env_file: Path) -> None:
    env_file.write_text(
        "\n".join(
            [
                "GOOGLE_API_KEY=test-key",
                "BRAINLAYER_ENRICH_ENABLED=1",
                "BRAINLAYER_ENRICH_MODE=realtime",
                "BRAINLAYER_ENRICH_PROVIDER=google",
                "BRAINLAYER_ENRICH_BACKEND=gemini",
                "BRAINLAYER_ENRICH_RATE=1",
                "BRAINLAYER_ENRICH_CONCURRENCY=1",
                "BRAINLAYER_MAX_COMMIT_BATCH=1",
                "BRAINLAYER_GEMINI_SERVICE_TIER=flex",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)


def _fake_launchctl_lines(
    *behavior: str,
    post_bootstrap_output: tuple[str, ...] = ("state = running", "pid = 4242"),
) -> list[str]:
    output_commands = [f'    printf "%s\\\\n" "{line}"' for line in post_bootstrap_output]
    return [
        "#!/usr/bin/env bash",
        'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
        *behavior,
        'if [ "$1" = "print" ]; then',
        '  label="${2##*/}"',
        '  if grep -Fq "${label}.plist" "$FAKE_LAUNCHCTL_LOG"; then',
        *output_commands,
        "    exit 0",
        "  fi",
        "  exit 1",
        "fi",
        "exit 0",
        "",
    ]


def _write_fake_ps(fake_bin: Path) -> None:
    fake_ps = fake_bin / "ps"
    fake_ps.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'if [[ "$*" == *"ucomm="* ]]; then',
                '  printf "%s\\n" "${FAKE_PS_NAME:-python3}"',
                "else",
                '  printf "%s\\n" "$FAKE_PS_COMMAND"',
                "fi",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)


def _copy_packaged_launchd(launchd_dir: Path) -> None:
    shutil.copytree(REPO_ROOT / "scripts" / "launchd", launchd_dir)
    package_dir = launchd_dir.parent
    for name in ("__init__.py", "config.py", "paths.py", "spotlight.py"):
        shutil.copy2(REPO_ROOT / "src" / "brainlayer" / name, package_dir / name)


def test_brainlayer_cli_entrypoint_imports_typer_app(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from brainlayer.cli import app

    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "brainlayer" in result.stdout


def test_brainlayer_cli_exposes_transport_commands(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    from brainlayer.cli import app

    for args in (["drain", "--help"], ["p0-counter", "--help"], ["status", "--help"]):
        result = CliRunner().invoke(app, args)

        assert result.exit_code == 0, result.stdout


def test_launchd_templates_are_declared_as_package_data() -> None:
    payload = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = payload["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include["scripts/launchd"] == "brainlayer/launchd"

    plist_paths = sorted((REPO_ROOT / "scripts" / "launchd").glob("com.brainlayer.*.plist"))
    assert len(plist_paths) >= 10
    for path in plist_paths:
        plist = plistlib.loads(path.read_bytes())
        assert plist["Label"].startswith("com.brainlayer.")


def test_setup_invokes_launchd_install_script_with_env_file(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.setup import install_launchd

    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    fake_brainlayer = tmp_path / "tool" / "bin" / "brainlayer"
    fake_brainlayer.parent.mkdir(parents=True)
    fake_brainlayer.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    monkeypatch.setattr(setup_helpers.sys, "argv", [str(fake_brainlayer)])
    install_script = launchd_dir / "install.sh"
    marker = tmp_path / "called.txt"
    install_script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'printf "%s\\n" "$BRAINLAYER_ENV_FILE" "$PYTHON_BIN" "$BRAINLAYER_PYTHON" "$BRAINLAYER_BIN" "$1" > "$CALL_MARKER"',
            ]
        ),
        encoding="utf-8",
    )
    install_script.chmod(0o755)
    env_file = tmp_path / "brainlayer.env"

    install_launchd("watch", env_file=env_file, launchd_dir=launchd_dir, extra_env={"CALL_MARKER": str(marker)})

    assert marker.read_text(encoding="utf-8").splitlines() == [
        str(env_file),
        sys.executable,
        sys.executable,
        str(fake_brainlayer),
        "watch",
    ]


def test_install_launchd_times_out_with_clear_error(tmp_path: Path) -> None:
    from brainlayer.setup import install_launchd

    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    install_script = launchd_dir / "install.sh"
    install_script.write_text("#!/usr/bin/env bash\nsleep 5\n", encoding="utf-8")
    install_script.chmod(0o755)

    try:
        install_launchd("watch", launchd_dir=launchd_dir, timeout_seconds=0.01)
    except TimeoutError as exc:
        assert "launchd installer timed out after 0.01s" in str(exc)
    else:
        raise AssertionError("install_launchd did not time out")


def test_install_launchd_default_defers_to_bounded_installer_shutdown_waits(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.setup import install_launchd

    launchd_dir = tmp_path / "launchd"
    launchd_dir.mkdir()
    install_script = launchd_dir / "install.sh"
    install_script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    install_script.chmod(0o755)
    calls: list[tuple[list[str], float | None]] = []

    def record_run(command, *, env, check, timeout):  # noqa: ANN001, ARG001
        calls.append((command, timeout))

    monkeypatch.setattr(setup_helpers.subprocess, "run", record_run)

    install_launchd("all", launchd_dir=launchd_dir)

    assert calls == [([str(install_script), "all"], None)]


def test_setup_creates_complete_spotlight_excluded_layout_idempotently(tmp_path: Path) -> None:
    import brainlayer.setup as setup_helpers

    data_dir = tmp_path / "data"
    runtime_dir = tmp_path / "runtime"
    launchd_log_dir = tmp_path / "launchd-logs"
    counter_dir = tmp_path / "counter"

    first = setup_helpers.ensure_spotlight_excluded_layout(
        data_dir=data_dir,
        runtime_dir=runtime_dir,
        launchd_log_dir=launchd_log_dir,
        counter_dir=counter_dir,
    )
    second = setup_helpers.ensure_spotlight_excluded_layout(
        data_dir=data_dir,
        runtime_dir=runtime_dir,
        launchd_log_dir=launchd_log_dir,
        counter_dir=counter_dir,
    )

    roots = (data_dir, runtime_dir, launchd_log_dir, counter_dir)
    assert first == roots
    assert second == roots
    assert all((root / ".metadata_never_index").is_file() for root in roots)
    assert {path.name for path in data_dir.iterdir() if path.is_dir()} == {
        "backups",
        "chromadb",
        "chromadb.backup",
        "enrichment-scratch",
        "experiments",
        "jsonl-backups",
        "logs",
        "prompts",
        "style",
        "storage",
    }
    assert {path.name for path in runtime_dir.iterdir() if path.is_dir()} == {"logs", "quarantine", "queue"}


def test_setup_refuses_to_mark_legacy_nonempty_runtime_tree(tmp_path: Path) -> None:
    import brainlayer.setup as setup_helpers

    data_dir = tmp_path / "legacy-data"
    data_dir.mkdir()
    (data_dir / "brainlayer.db").write_bytes(b"existing production data")

    with pytest.raises(RuntimeError, match="requires the Spotlight exclusion migration runbook"):
        setup_helpers.ensure_spotlight_excluded_layout(
            data_dir=data_dir,
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not (data_dir / ".metadata_never_index").exists()


def test_setup_preflights_all_roots_before_creating_any_marker(tmp_path: Path) -> None:
    import brainlayer.setup as setup_helpers

    data_dir = tmp_path / "new-data"
    runtime_dir = tmp_path / "legacy-runtime"
    runtime_dir.mkdir()
    (runtime_dir / "queue-item.jsonl").write_text("pending\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires the Spotlight exclusion migration runbook"):
        setup_helpers.ensure_spotlight_excluded_layout(
            data_dir=data_dir,
            runtime_dir=runtime_dir,
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not data_dir.exists()
    assert not (runtime_dir / ".metadata_never_index").exists()


@pytest.mark.parametrize("invalid_kind", ["file", "dangling-symlink", "directory-symlink"])
def test_setup_preflights_invalid_root_types_before_creating_any_marker(tmp_path: Path, invalid_kind: str) -> None:
    import brainlayer.setup as setup_helpers

    data_dir = tmp_path / "new-data"
    runtime_dir = tmp_path / "invalid-runtime"
    if invalid_kind == "file":
        runtime_dir.write_text("not a directory\n", encoding="utf-8")
    elif invalid_kind == "dangling-symlink":
        runtime_dir.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    else:
        target = tmp_path / "runtime-target"
        target.mkdir()
        runtime_dir.symlink_to(target, target_is_directory=True)

    with pytest.raises(RuntimeError, match="must be a directory"):
        setup_helpers.ensure_spotlight_excluded_layout(
            data_dir=data_dir,
            runtime_dir=runtime_dir,
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not data_dir.exists()


def test_setup_default_data_root_is_not_created_when_later_root_is_invalid(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    data_dir = tmp_path / "default-data"
    runtime_dir = tmp_path / "invalid-runtime"
    runtime_dir.write_text("not a directory\n", encoding="utf-8")
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: data_dir / "brainlayer.db")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: data_dir / "brainlayer.db", raising=False)

    with pytest.raises(RuntimeError, match="must be a directory"):
        setup_helpers.ensure_spotlight_excluded_layout(
            runtime_dir=runtime_dir,
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not data_dir.exists()


def test_setup_marks_override_and_canonical_data_roots(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    override_data_dir = tmp_path / "brainlayer-override"
    canonical_data_dir = tmp_path / "canonical-data"
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: override_data_dir / "brainlayer.db")
    monkeypatch.setattr(
        setup_helpers,
        "get_canonical_db_path",
        lambda: canonical_data_dir / "brainlayer.db",
        raising=False,
    )

    roots = setup_helpers.ensure_spotlight_excluded_layout(
        runtime_dir=tmp_path / "runtime",
        launchd_log_dir=tmp_path / "launchd-logs",
        counter_dir=tmp_path / "counter",
    )

    assert roots[:2] == (override_data_dir, canonical_data_dir)
    assert (override_data_dir / ".metadata_never_index").is_file()
    assert (canonical_data_dir / ".metadata_never_index").is_file()
    assert (canonical_data_dir / "prompts").is_dir()


def test_setup_uses_selected_env_file_db_override(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    home = tmp_path / "home"
    override_data_dir = tmp_path / "brainlayer-selected"
    canonical_data_dir = tmp_path / "canonical-data"
    env_file = tmp_path / "selected.env"
    env_file.write_text(f"BRAINLAYER_DB={override_data_dir / 'brainlayer.db'}\n", encoding="utf-8")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    roots = setup_helpers.ensure_spotlight_excluded_layout(
        env_file=env_file,
        runtime_dir=home / ".brainlayer",
        launchd_log_dir=home / "Library" / "Logs" / "brainlayer",
        counter_dir=home / ".brainlayer-p0-counter",
    )

    assert roots[:2] == (override_data_dir, canonical_data_dir)
    assert (override_data_dir / ".metadata_never_index").is_file()


def test_setup_rejects_selected_env_db_file_symlink_before_creating_roots(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    target = tmp_path / "target.db"
    target.touch()
    override_data_dir = tmp_path / "brainlayer-selected"
    override_data_dir.mkdir()
    linked_db = override_data_dir / "brainlayer.db"
    linked_db.symlink_to(target)
    canonical_data_dir = tmp_path / "canonical-data"
    env_file = tmp_path / "selected.env"
    env_file.write_text(f"BRAINLAYER_DB={linked_db}\n", encoding="utf-8")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    with pytest.raises(RuntimeError, match="must not be a symbolic link"):
        setup_helpers.ensure_spotlight_excluded_layout(
            env_file=env_file,
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not canonical_data_dir.exists()


def test_setup_rejects_selected_env_runtime_path_override_before_creating_roots(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    canonical_data_dir = tmp_path / "canonical-data"
    env_file = tmp_path / "selected.env"
    env_file.write_text(f"BRAINLAYER_QUEUE_DIR={tmp_path / 'queue-override'}\n", encoding="utf-8")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    with pytest.raises(RuntimeError, match="BRAINLAYER_QUEUE_DIR"):
        setup_helpers.ensure_spotlight_excluded_layout(
            env_file=env_file,
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not canonical_data_dir.exists()


def test_spotlight_env_file_without_db_uses_canonical_not_process_override(tmp_path: Path, monkeypatch) -> None:
    from brainlayer.spotlight import ensure_spotlight_excluded_layout

    env_file = tmp_path / "selected.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    process_data_dir = tmp_path / "brainlayer-process"
    canonical_data_dir = tmp_path / "canonical-data"
    monkeypatch.setenv("BRAINLAYER_DB", str(process_data_dir / "brainlayer.db"))

    roots = ensure_spotlight_excluded_layout(
        env_file=env_file,
        runtime_dir=tmp_path / "runtime",
        launchd_log_dir=tmp_path / "launchd-logs",
        counter_dir=tmp_path / "counter",
        resolve_db_path_fn=lambda: process_data_dir / "brainlayer.db",
        get_canonical_db_path_fn=lambda: canonical_data_dir / "brainlayer.db",
        home_fn=lambda: tmp_path,
    )

    assert roots[0] == canonical_data_dir
    assert not process_data_dir.exists()


def test_setup_deduplicates_equivalent_resolved_data_roots(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    canonical_data_dir = tmp_path / "brainlayer"
    equivalent_data_dir = canonical_data_dir / ".." / "brainlayer"
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: equivalent_data_dir / "brainlayer.db")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    roots = setup_helpers.ensure_spotlight_excluded_layout(
        runtime_dir=tmp_path / "runtime",
        launchd_log_dir=tmp_path / "launchd-logs",
        counter_dir=tmp_path / "counter",
    )

    assert roots[0] == canonical_data_dir
    assert roots.count(canonical_data_dir) == 1


def test_setup_rejects_override_parent_that_is_not_dedicated_to_brainlayer(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    broad_override_dir = tmp_path / "shared-data"
    canonical_data_dir = broad_override_dir / ".local" / "share" / "brainlayer"
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: broad_override_dir / "brainlayer.db")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    with pytest.raises(RuntimeError, match="dedicated BrainLayer directory"):
        setup_helpers.ensure_spotlight_excluded_layout(
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not broad_override_dir.exists()


def test_setup_rejects_override_parent_that_overlaps_runtime_root(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    shared_root = tmp_path / "brainlayer-runtime"
    canonical_data_dir = tmp_path / "canonical-data"
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: shared_root / "brainlayer.db")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    with pytest.raises(RuntimeError, match="overlap"):
        setup_helpers.ensure_spotlight_excluded_layout(
            runtime_dir=shared_root,
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not shared_root.exists()


def test_setup_rejects_override_parent_that_is_mount_root(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    mount_root = tmp_path / "brainlayer-mounted-volume"
    canonical_data_dir = tmp_path / "canonical-data"
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: mount_root / "brainlayer.db")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")
    monkeypatch.setattr(
        setup_helpers.os.path,
        "ismount",
        lambda path: Path(path).resolve(strict=False) == mount_root.resolve(strict=False),
    )

    with pytest.raises(RuntimeError, match="dedicated BrainLayer directory"):
        setup_helpers.ensure_spotlight_excluded_layout(
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not mount_root.exists()


def test_setup_rejects_override_beneath_symlinked_parent(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    override_data_dir = linked_parent / "brainlayer-data"
    canonical_data_dir = tmp_path / "canonical-data"
    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: override_data_dir / "brainlayer.db")
    monkeypatch.setattr(setup_helpers, "get_canonical_db_path", lambda: canonical_data_dir / "brainlayer.db")

    with pytest.raises(RuntimeError, match="symbolic link"):
        setup_helpers.ensure_spotlight_excluded_layout(
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )

    assert not override_data_dir.exists()


def test_setup_rejects_relative_db_override(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    monkeypatch.setattr(setup_helpers, "resolve_db_path", lambda: Path("brainlayer-relative/brainlayer.db"))
    monkeypatch.setattr(
        setup_helpers,
        "get_canonical_db_path",
        lambda: tmp_path / "canonical-data" / "brainlayer.db",
    )

    with pytest.raises(RuntimeError, match="absolute path"):
        setup_helpers.ensure_spotlight_excluded_layout(
            runtime_dir=tmp_path / "runtime",
            launchd_log_dir=tmp_path / "launchd-logs",
            counter_dir=tmp_path / "counter",
        )


def test_setup_migrates_legacy_raw_socat_mcp_config_idempotently(tmp_path: Path) -> None:
    from brainlayer.setup import migrate_legacy_mcp_configs

    config_path = tmp_path / ".claude.json"
    original = {
        "theme": "dark",
        "mcpServers": {
            "brainlayer": {
                "command": "socat",
                "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
                "disabled": True,
                "env": {"BRAINLAYER_MCP_SOCKET": "/tmp/brainbar.sock"},
                "timeout": 30,
            },
            "unrelated": {"command": "other-mcp", "args": ["--keep"]},
        },
    }
    config_path.write_text(json.dumps(original) + "\n", encoding="utf-8")

    changed = migrate_legacy_mcp_configs(
        [config_path],
        bridge_command="/opt/homebrew/bin/brainlayer-mcp-stdio-bridge",
    )
    second_changed = migrate_legacy_mcp_configs(
        [config_path],
        bridge_command="/opt/homebrew/bin/brainlayer-mcp-stdio-bridge",
    )

    migrated = json.loads(config_path.read_text(encoding="utf-8"))
    assert changed == [config_path]
    assert second_changed == []
    assert migrated["theme"] == "dark"
    assert migrated["mcpServers"]["unrelated"] == original["mcpServers"]["unrelated"]
    assert migrated["mcpServers"]["brainlayer"] == {
        "command": "/opt/homebrew/bin/brainlayer-mcp-stdio-bridge",
        "disabled": True,
        "env": {"BRAINLAYER_MCP_SOCKET": "/tmp/brainbar.sock"},
        "timeout": 30,
    }


def test_setup_mcp_migration_refuses_unresolved_bridge_without_touching_config(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    config_path = tmp_path / ".claude.json"
    original = {
        "mcpServers": {
            "brainlayer": {
                "command": "socat",
                "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
            }
        }
    }
    original_text = json.dumps(original) + "\n"
    config_path.write_text(original_text, encoding="utf-8")
    monkeypatch.setattr(setup_helpers, "get_current_mcp_bridge_bin", lambda: None)

    with pytest.raises(FileNotFoundError, match="brainlayer-mcp-stdio-bridge was not found"):
        setup_helpers.migrate_legacy_mcp_configs([config_path])

    assert config_path.read_text(encoding="utf-8") == original_text


def test_setup_mcp_migration_preserves_original_and_cleans_temp_on_replace_failure(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers

    config_path = tmp_path / ".claude.json"
    original = {
        "mcpServers": {
            "brainlayer": {
                "command": "socat",
                "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
            }
        }
    }
    original_text = json.dumps(original) + "\n"
    config_path.write_text(original_text, encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise PermissionError("replace denied")

    monkeypatch.setattr(setup_helpers.os, "replace", fail_replace)

    with pytest.raises(PermissionError, match="replace denied"):
        setup_helpers.migrate_legacy_mcp_configs([config_path], bridge_command="brainlayer-mcp-stdio-bridge")

    assert config_path.read_text(encoding="utf-8") == original_text
    assert not (tmp_path / f".{config_path.name}.{os.getpid()}.tmp").exists()


def test_setup_mcp_migration_preserves_symlink_and_updates_target(tmp_path: Path) -> None:
    from brainlayer.setup import migrate_legacy_mcp_configs

    target = tmp_path / "dotfiles" / "claude.json"
    target.parent.mkdir()
    target.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "brainlayer": {
                        "command": "socat",
                        "args": ["STDIO", "UNIX-CONNECT:/tmp/brainbar.sock"],
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / ".claude.json"
    config_path.symlink_to(target)

    changed = migrate_legacy_mcp_configs(
        [config_path],
        bridge_command="/opt/homebrew/bin/brainlayer-mcp-stdio-bridge",
    )

    assert changed == [config_path]
    assert config_path.is_symlink()
    assert json.loads(target.read_text(encoding="utf-8"))["mcpServers"]["brainlayer"] == {
        "command": "/opt/homebrew/bin/brainlayer-mcp-stdio-bridge"
    }


def test_verify_mcp_transport_requires_initialize_tools_list_and_tool_call(tmp_path: Path) -> None:
    from brainlayer.setup import verify_mcp_transport

    bridge = tmp_path / "fake-bridge"
    call_marker = tmp_path / "tool-call-seen"
    bridge.write_text(
        f"""#!/usr/bin/env python3
import json
from pathlib import Path
import sys
for line in sys.stdin:
    request = json.loads(line)
    if request.get("id") == 1:
        print(json.dumps({{"jsonrpc": "2.0", "id": 1, "result": {{"protocolVersion": "2025-06-18"}}}}), flush=True)
    elif request.get("id") == 2:
        print(json.dumps({{"jsonrpc": "2.0", "id": 2, "result": {{"tools": [{{"name": "brain_recall"}}]}}}}), flush=True)
    elif request.get("id") == 3 and request.get("method") == "tools/call":
        assert request.get("params") == {{"name": "brain_recall", "arguments": {{"mode": "stats"}}}}
        Path({str(call_marker)!r}).write_text("seen")
        print(json.dumps({{"jsonrpc": "2.0", "id": 3, "result": {{"content": [{{"type": "text", "text": "expanded"}}]}}}}), flush=True)
""",
        encoding="utf-8",
    )
    bridge.chmod(0o755)

    assert verify_mcp_transport(bridge_command=str(bridge), timeout_seconds=2) == 1
    assert call_marker.read_text() == "seen"


def test_verify_mcp_transport_waits_for_initialize_response_before_next_message(tmp_path: Path) -> None:
    from brainlayer.setup import verify_mcp_transport

    bridge = tmp_path / "strict-bridge"
    bridge.write_text(
        """#!/usr/bin/env python3
import json
import select
import sys

initialize = json.loads(sys.stdin.readline())
if initialize.get("id") != 1 or select.select([sys.stdin], [], [], 0)[0]:
    raise SystemExit(12)
print(json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-06-18"}}), flush=True)
initialized = json.loads(sys.stdin.readline())
if initialized.get("method") != "notifications/initialized":
    raise SystemExit(13)
tools = json.loads(sys.stdin.readline())
if tools.get("id") != 2:
    raise SystemExit(14)
print(json.dumps({"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "brain_recall"}]}}), flush=True)
call = json.loads(sys.stdin.readline())
if call.get("id") != 3 or call.get("method") != "tools/call":
    raise SystemExit(15)
if call.get("params") != {"name": "brain_recall", "arguments": {"mode": "stats"}}:
    raise SystemExit(16)
print(json.dumps({"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "expanded"}]}}), flush=True)
""",
        encoding="utf-8",
    )
    bridge.chmod(0o755)

    assert verify_mcp_transport(bridge_command=str(bridge), timeout_seconds=2) == 1


def test_mcp_bridge_resolution_uses_installed_brainlayer_sibling_when_path_is_empty(
    tmp_path: Path, monkeypatch
) -> None:
    from brainlayer.setup import get_current_mcp_bridge_bin

    bin_dir = tmp_path / "Cellar" / "brainlayer" / "1.5.3" / "libexec" / "bin"
    bin_dir.mkdir(parents=True)
    brainlayer_bin = bin_dir / "brainlayer"
    bridge_bin = bin_dir / "brainlayer-mcp-stdio-bridge"
    brainlayer_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    bridge_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    brainlayer_bin.chmod(0o755)
    bridge_bin.chmod(0o755)
    monkeypatch.setattr(sys, "argv", [str(brainlayer_bin), "setup"])
    monkeypatch.setenv("PATH", "")

    assert get_current_mcp_bridge_bin() == str(bridge_bin)


def test_verify_mcp_transport_fails_loudly_when_bridge_has_no_tools(tmp_path: Path) -> None:
    from brainlayer.setup import verify_mcp_transport

    bridge = tmp_path / "dead-bridge"
    bridge.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    bridge.chmod(0o755)

    with pytest.raises(RuntimeError, match="initialize response"):
        verify_mcp_transport(bridge_command=str(bridge), timeout_seconds=2)


def test_verify_mcp_transport_times_out_on_partial_stdout_line(tmp_path: Path) -> None:
    from brainlayer.setup import verify_mcp_transport

    bridge = tmp_path / "partial-line-bridge"
    bridge.write_text(
        """#!/usr/bin/env python3
import sys
import time
sys.stdin.readline()
sys.stdout.write('{"jsonrpc":"2.0","id":1')
sys.stdout.flush()
time.sleep(10)
""",
        encoding="utf-8",
    )
    bridge.chmod(0o755)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        verify_mcp_transport(bridge_command=str(bridge), timeout_seconds=0.1)
    assert time.monotonic() - started < 1


def test_setup_command_writes_op_backed_env_without_plaintext_and_can_skip_launchd(tmp_path: Path) -> None:
    from brainlayer.cli import app

    env_file = tmp_path / "brainlayer.env"
    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--no-launchd",
            "--env-file",
            str(env_file),
            "--google-api-key-op-ref",
            "op://Private/Google AI/Gemini API key",
        ],
    )

    assert result.exit_code == 0, result.stdout
    content = env_file.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"" in content
    assert "AIza" not in content
    assert oct(env_file.stat().st_mode & 0o777) == "0o600"


def test_setup_command_excludes_runtime_layout_for_selected_env(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.cli as cli
    import brainlayer.config as config
    import brainlayer.setup as setup_helpers

    calls: list[str] = []
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    env_file = tmp_path / "brainlayer.env"
    monkeypatch.setattr(config, "get_user_env_path", lambda: env_file)
    monkeypatch.setattr(
        setup_helpers,
        "ensure_spotlight_excluded_layout",
        lambda *, env_file: calls.append(f"layout:{env_file}"),
    )
    monkeypatch.setattr(
        setup_helpers,
        "ensure_brainlayer_env",
        lambda *_args, **_kwargs: calls.append("env") or env_file,
    )

    result = CliRunner().invoke(cli.app, ["setup", "--no-launchd"])

    assert result.exit_code == 0, result.stdout
    assert calls == [f"layout:{env_file}", "env"]


def test_setup_command_skips_spotlight_layout_outside_macos(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.cli as cli
    import brainlayer.setup as setup_helpers

    calls: list[str] = []
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(
        setup_helpers,
        "ensure_spotlight_excluded_layout",
        lambda *, env_file: calls.append(f"layout:{env_file}"),
    )
    monkeypatch.setattr(
        setup_helpers,
        "ensure_brainlayer_env",
        lambda *_args, **_kwargs: calls.append("env") or tmp_path / "brainlayer.env",
    )

    result = CliRunner().invoke(cli.app, ["setup", "--no-launchd"])

    assert result.exit_code == 0, result.stdout
    assert calls == ["env"]


def test_launchd_env_runner_makes_homebrew_op_available_before_loading_env() -> None:
    runner = REPO_ROOT / "scripts" / "launchd" / "brainlayer-env-run.sh"
    content = runner.read_text(encoding="utf-8")

    path_export_index = content.index("export PATH=")
    load_index = content.index("load_simple_env_file")
    assert "/opt/homebrew/bin" in content[path_export_index:load_index]
    assert path_export_index < load_index


def test_setup_command_does_not_install_launchd_by_default(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    def fail_install(*args, **kwargs):
        raise AssertionError("launchd should be opt-in")

    monkeypatch.setattr(setup_helpers, "install_launchd", fail_install)
    env_file = tmp_path / "brainlayer.env"

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--env-file",
            str(env_file),
            "--google-api-key-op-ref",
            "op://Private/Google AI/Gemini API key",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert env_file.exists()


def test_setup_command_can_migrate_and_verify_mcp_transport(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    migrated_path = tmp_path / ".claude.json"
    calls: list[str] = []
    monkeypatch.setattr(
        setup_helpers,
        "migrate_legacy_mcp_configs",
        lambda: calls.append("migrate") or [migrated_path],
    )
    monkeypatch.setattr(
        setup_helpers,
        "verify_mcp_transport",
        lambda: calls.append("verify") or 17,
    )

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--env-file",
            str(tmp_path / "brainlayer.env"),
            "--migrate-mcp",
            "--verify-mcp",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert calls == ["migrate", "verify"]
    assert "Migrated MCP config:" in result.stdout
    assert ".claude.json" in result.stdout
    assert "MCP transport verified: 17 tools" in result.stdout


def test_setup_command_env_file_annotation_accepts_none() -> None:
    from brainlayer.cli import setup

    assert get_type_hints(setup)["env_file"] == Path | None


def test_setup_command_reports_launchd_failure_without_traceback(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    def fail_install(*args, **kwargs):
        raise FileNotFoundError("missing install.sh")

    monkeypatch.setattr(setup_helpers, "install_launchd", fail_install)
    env_file = tmp_path / "brainlayer.env"

    result = CliRunner().invoke(
        app,
        [
            "setup",
            "--launchd",
            "--env-file",
            str(env_file),
            "--google-api-key-op-ref",
            "op://Private/Google AI/Gemini API key",
        ],
    )

    assert result.exit_code == 1
    assert "BrainLayer setup failed: missing install.sh" in result.stdout
    assert "Traceback" not in result.stdout


def test_setup_command_reports_env_write_failure_without_traceback(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    def fail_env(*args, **kwargs):
        raise ValueError("empty op reference")

    monkeypatch.setattr(setup_helpers, "ensure_brainlayer_env", fail_env)

    result = CliRunner().invoke(app, ["setup", "--no-launchd", "--env-file", str(tmp_path / "brainlayer.env")])

    assert result.exit_code == 1
    assert "BrainLayer setup failed: empty op reference" in result.stdout
    assert "Traceback" not in result.stdout


def test_init_command_reports_launchd_failure_without_traceback(monkeypatch) -> None:
    import brainlayer.cli as cli
    import brainlayer.cli.wizard as wizard
    import brainlayer.setup as setup_helpers
    from brainlayer.cli import app

    class Config:
        gemini_env_file = Path("/tmp/brainlayer.env")

    monkeypatch.setattr(wizard, "run_wizard", lambda: Config())
    monkeypatch.setattr(
        setup_helpers,
        "install_launchd",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing install.sh")),
    )
    monkeypatch.setattr(cli, "install_launchd_agents", setup_helpers.install_launchd, raising=False)

    result = CliRunner().invoke(app, ["init", "--install-launchd"])

    assert result.exit_code == 1
    assert "BrainLayer init failed: missing install.sh" in result.stdout
    assert "Traceback" not in result.stdout


def test_init_command_excludes_layout_before_wizard_and_launchd(monkeypatch) -> None:
    import brainlayer.cli as cli
    import brainlayer.cli.wizard as wizard
    import brainlayer.setup as setup_helpers

    calls: list[str] = []

    class Config:
        gemini_env_file = Path("/tmp/brainlayer.env")

    monkeypatch.setattr(cli.sys, "platform", "darwin")
    monkeypatch.setattr(wizard, "get_default_env_file", lambda: Path("/tmp/brainlayer.env"))
    monkeypatch.setattr(
        setup_helpers,
        "ensure_spotlight_excluded_layout",
        lambda *, env_file: calls.append(f"layout:{env_file}"),
    )
    monkeypatch.setattr(wizard, "run_wizard", lambda: calls.append("wizard") or Config())
    monkeypatch.setattr(setup_helpers, "install_launchd", lambda *_args, **_kwargs: calls.append("launchd"))

    result = CliRunner().invoke(cli.app, ["init", "--install-launchd"])

    assert result.exit_code == 0, result.stdout
    assert calls == ["layout:/tmp/brainlayer.env", "wizard", "launchd"]


def test_config_loader_prefers_process_env_then_user_env_and_ignores_repo_root_dotenv(
    tmp_path: Path, monkeypatch
) -> None:
    import brainlayer.config as config
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text(
        "\n".join(
            [
                "BRAINLAYER_FROM_USER=ok",
                "BRAINLAYER_EXISTING=from-file",
                "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"",
            ]
        ),
        encoding="utf-8",
    )
    repo_env = tmp_path / ".env"
    repo_env.write_text("BRAINLAYER_FROM_REPO=deprecated\n", encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_EXISTING", "from-process")
    monkeypatch.delenv("BRAINLAYER_FROM_USER", raising=False)
    monkeypatch.delenv("BRAINLAYER_FROM_REPO", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def fake_run(args, **kwargs):
        assert args == ["op", "read", "op://Private/Google AI/Gemini API key"]
        return subprocess.CompletedProcess(args, 0, stdout="resolved-secret\n", stderr="")

    monkeypatch.setattr(config.subprocess, "run", fake_run)

    loaded = load_brainlayer_env(user_env, repo_env_path=repo_env)

    assert loaded == {"BRAINLAYER_FROM_USER": "ok", "GOOGLE_API_KEY": "resolved-secret"}
    assert os.environ["BRAINLAYER_FROM_USER"] == "ok"
    assert os.environ["BRAINLAYER_EXISTING"] == "from-process"
    assert "BRAINLAYER_FROM_REPO" not in os.environ
    assert os.environ["GOOGLE_API_KEY"] == "resolved-secret"


def test_config_loader_ignores_shell_substitution_that_is_not_op_read(tmp_path: Path, monkeypatch) -> None:
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text('GOOGLE_API_KEY="$(cat /tmp/key)"\n', encoding="utf-8")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert load_brainlayer_env(user_env) == {}
    assert "GOOGLE_API_KEY" not in os.environ


def test_config_loader_does_not_resolve_op_when_process_env_already_has_key(tmp_path: Path, monkeypatch) -> None:
    import brainlayer.config as config
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text("GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_API_KEY", "from-process")

    def fail_run(*args, **kwargs):
        raise AssertionError("op read should not run when process env has GOOGLE_API_KEY")

    monkeypatch.setattr(config.subprocess, "run", fail_run)

    assert load_brainlayer_env(user_env) == {}
    assert os.environ["GOOGLE_API_KEY"] == "from-process"


@pytest.mark.parametrize(
    "assignment",
    [
        r"BRAINLAYER_DB=/Volumes/brainlayer\ data/brainlayer.db",
        'BRAINLAYER_DB="/outer"inner"/brainlayer.db"',
        "export  BRAINLAYER_DB=/ignored/by/launchd/brainlayer.db",
    ],
)
def test_config_loader_rejects_db_syntax_that_launchd_parses_differently(
    tmp_path: Path, monkeypatch, assignment: str
) -> None:
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text(f"{assignment}\n", encoding="utf-8")
    monkeypatch.delenv("BRAINLAYER_DB", raising=False)

    with pytest.raises(RuntimeError, match="launchd and direct CLI"):
        load_brainlayer_env(user_env)

    assert "BRAINLAYER_DB" not in os.environ


def test_config_loader_ignores_unreadable_user_env(monkeypatch, tmp_path: Path) -> None:
    from brainlayer.config import load_brainlayer_env

    user_env = tmp_path / "brainlayer.env"
    user_env.write_text("BRAINLAYER_FROM_USER=ok\n", encoding="utf-8")

    original_read_text = Path.read_text

    def fail_read_text(path: Path, *args, **kwargs):
        if path == user_env:
            raise PermissionError("blocked")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read_text)

    assert load_brainlayer_env(user_env, repo_env_path=tmp_path / ".env") == {}


def test_launchd_repo_only_wrappers_are_not_used_by_packaged_plists() -> None:
    assert _plist_args("drain")[:5] == [
        "__BRAINLAYER_ENV_RUN__",
        "/usr/bin/env",
        "BRAINLAYER_DRAIN_EMBED=0",
        "__BRAINLAYER_BIN__",
        "drain",
    ]
    assert _plist_args("maintenance-nightly")[:3] == ["__BRAINLAYER_ENV_RUN__", "__PYTHON_BIN__", "-m"]
    assert _plist_args("maintenance-nightly")[3] == "brainlayer.maintenance"
    assert _plist_args("maintenance-weekly")[:3] == ["__BRAINLAYER_ENV_RUN__", "__PYTHON_BIN__", "-m"]
    assert _plist_args("maintenance-weekly")[3] == "brainlayer.maintenance"


def test_drain_module_supports_python_m_execution() -> None:
    content = (REPO_ROOT / "src" / "brainlayer" / "drain.py").read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in content
    assert "raise SystemExit(main())" in content


def test_launchd_installer_preflights_all_before_loading_without_google_key(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=1\n", encoding="utf-8")
    env_file.chmod(0o600)
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "did not provide GOOGLE_API_KEY" in result.stdout
    assert not launchctl_log.exists()
    assert not list((home / "Library" / "LaunchAgents").glob("com.brainlayer.*.plist"))


def test_standalone_launchd_installer_runs_spotlight_preflight_before_mkdir(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text("#!/bin/sh\nprintf 'Darwin\\n'\n", encoding="utf-8")
    fake_uname.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    fake_brainlayer = tmp_path / "brainlayer"
    fake_brainlayer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    preflight_log = tmp_path / "preflight.log"
    python_shim = tmp_path / "python-shim"
    python_shim.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$*" > "$PREFLIGHT_LOG"\nexit 73\n',
        encoding="utf-8",
    )
    python_shim.chmod(0o755)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "watch"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(fake_brainlayer),
            "PYTHON_BIN": str(python_shim),
            "PREFLIGHT_LOG": str(preflight_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 73
    assert "ensure_spotlight_excluded_layout" in preflight_log.read_text(encoding="utf-8")
    assert not (home / "Library" / "LaunchAgents").exists()
    assert not (home / ".local" / "share" / "brainlayer").exists()


def test_standalone_launchd_installer_preflights_selected_env_db(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text("#!/bin/sh\nprintf 'Darwin\\n'\n", encoding="utf-8")
    fake_uname.chmod(0o755)
    fake_launchctl = fake_bin / "launchctl"
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl.write_text("\n".join(_fake_launchctl_lines()), encoding="utf-8")
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    fake_brainlayer = tmp_path / "brainlayer"
    fake_brainlayer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    override_data_dir = tmp_path / "brainlayer-selected"
    transient_process_data_dir = tmp_path / "brainlayer-transient-process"
    env_file = tmp_path / "selected.env"
    env_file.write_text(f"BRAINLAYER_DB={override_data_dir / 'brainlayer.db'}\n", encoding="utf-8")
    run_env = os.environ.copy()
    run_env.update(
        {
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(fake_brainlayer),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_DB": str(transient_process_data_dir / "brainlayer.db"),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        }
    )

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "watch"],
        env=run_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (override_data_dir / ".metadata_never_index").is_file()
    assert not transient_process_data_dir.exists()


def test_configured_env_value_matches_launchd_grammar_and_rejects_runtime_ambiguity(
    tmp_path: Path, monkeypatch
) -> None:
    import brainlayer.config as config

    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_DB=/from/selected/brainlayer.db\n", encoding="utf-8")
    monkeypatch.setenv("BRAINLAYER_DB", "/from/process/brainlayer.db")
    assert config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file) == "/from/selected/brainlayer.db"

    env_file.write_text(
        "BRAINLAYER_DB=$(not-allowed)\n"
        "BRAINLAYER_DB=/from/first/brainlayer.db\n"
        "BRAINLAYER_DB=/from/later/brainlayer.db\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file)

    env_file.write_text(
        "BRAINLAYER_DB=$(not-allowed)\nBRAINLAYER_DB=/from/runtime/brainlayer.db\n",
        encoding="utf-8",
    )
    assert config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file) == "/from/runtime/brainlayer.db"

    env_file.write_text(
        "BRAINLAYER_DB=/from/runtime/brainlayer.db\nBRAINLAYER_DB=\"$(op read 'op://Private/Database/path')\"\n",
        encoding="utf-8",
    )

    def fail_op_read(*args, **kwargs):
        raise AssertionError("BRAINLAYER_DB command substitutions must be skipped")

    monkeypatch.setattr(config.subprocess, "run", fail_op_read)
    with pytest.raises(RuntimeError, match="command substitution"):
        config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file)

    env_file.write_text("BRAINLAYER_DB=\"$(op read 'op://Private/Database/path')\"\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="command substitution"):
        config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file)

    env_file.write_text(r"BRAINLAYER_DB=/Volumes/brainlayer\ data/brainlayer.db" + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="parse identically"):
        config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file)

    env_file.write_text('BRAINLAYER_DB="/outer"inner"/brainlayer.db"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="parse identically"):
        config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file)

    env_file.write_text('BRAINLAYER_DB="/Volumes/brainlayer data/brainlayer.db"\n', encoding="utf-8")
    assert config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file) == (
        "/Volumes/brainlayer data/brainlayer.db"
    )

    env_file.write_text(
        "BRAINLAYER_DB=/from/runtime/brainlayer.db\nexport  BRAINLAYER_DB=/ignored/by/launchd/brainlayer.db\n",
        encoding="utf-8",
    )
    assert config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file) == "/from/runtime/brainlayer.db"


def test_configured_env_value_rejects_unreadable_file(tmp_path: Path, monkeypatch, caplog) -> None:
    import brainlayer.config as config

    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_DB=/unused/brainlayer.db\n", encoding="utf-8")
    original_read_text = Path.read_text

    def fail_read(path: Path, *args, **kwargs):
        if path == env_file:
            raise PermissionError("blocked")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_read)

    with pytest.raises(RuntimeError, match="Could not read BrainLayer env file"):
        config.configured_brainlayer_env_value("BRAINLAYER_DB", env_file)
    assert "Could not read BrainLayer env file" in caplog.text


def test_launchd_installer_rejects_unknown_target_before_creating_roots(tmp_path: Path) -> None:
    fake_brainlayer = tmp_path / "brainlayer"
    fake_brainlayer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "not-a-target"],
        env={**os.environ, "HOME": str(home), "BRAINLAYER_BIN": str(fake_brainlayer)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "Usage:" in result.stdout
    assert not (home / ".local" / "share" / "brainlayer").exists()
    assert not (home / ".brainlayer").exists()


@pytest.mark.parametrize("action", ["remove", "unload"])
def test_launchd_teardown_does_not_create_runtime_roots(tmp_path: Path, action: str) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_launchctl.chmod(0o755)
    fake_brainlayer = tmp_path / "brainlayer"
    fake_brainlayer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), action],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(fake_brainlayer),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (home / ".local" / "share" / "brainlayer").exists()
    assert not (home / ".brainlayer").exists()
    assert not (home / "Library" / "Logs" / "brainlayer").exists()


def test_launchd_load_existing_unmarked_install_skips_spotlight_preflight(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text("\n".join(_fake_launchctl_lines()), encoding="utf-8")
    fake_launchctl.chmod(0o755)
    fake_brainlayer = tmp_path / "brainlayer"
    fake_brainlayer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    home = tmp_path / "home"
    launch_dir = home / "Library" / "LaunchAgents"
    launch_dir.mkdir(parents=True)
    (launch_dir / "com.brainlayer.enrichment.plist").write_bytes(
        plistlib.dumps({"Label": "com.brainlayer.enrichment", "ProgramArguments": ["/usr/bin/true"]})
    )
    legacy_data = home / ".local" / "share" / "brainlayer"
    legacy_data.mkdir(parents=True)
    (legacy_data / "brainlayer.db").touch()

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "load", "enrichment"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(fake_brainlayer),
            "PYTHON_BIN": sys.executable,
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "bootstrap" in launchctl_log.read_text(encoding="utf-8")
    assert not (legacy_data / ".metadata_never_index").exists()


def test_launchd_install_preflight_refusal_names_runbook_without_traceback(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text("#!/bin/sh\nprintf 'Darwin\\n'\n", encoding="utf-8")
    fake_uname.chmod(0o755)
    fake_brainlayer = tmp_path / "brainlayer"
    fake_brainlayer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_brainlayer.chmod(0o755)
    home = tmp_path / "home"
    legacy_data = home / ".local" / "share" / "brainlayer"
    legacy_data.mkdir(parents=True)
    (legacy_data / "brainlayer.db").touch()

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "watch"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(fake_brainlayer),
            "PYTHON_BIN": sys.executable,
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "docs/operations/spotlight-exclusion-migration.md" in result.stderr
    assert "Traceback" not in result.stderr


def test_spotlight_runbook_has_fail_closed_positive_control_and_complete_writer_fence() -> None:
    runbook = (REPO_ROOT / "docs" / "operations" / "spotlight-exclusion-migration.md").read_text(encoding="utf-8")

    assert "spotlight_positive_control_path" in runbook
    assert 'spotlight_positive_control_parent="${HOME:?}/Documents"' in runbook
    assert "/.local/share/brainlayer-spotlight-positive-control" not in runbook
    assert "SPOTLIGHT PROBE INCONCLUSIVE" in runbook
    assert "com.mcplayer.brainlayer-proxy" in runbook
    assert "com.brainlayer.gemini-loopback" in runbook
    assert "com.brainlayer.t3-ingest" in runbook
    assert "-iTCP:48123" in runbook


def test_launchd_installer_renders_brainlayer_python_override(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)
    brainlayer_python = tmp_path / "tool" / "bin" / "python"

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "backup"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": "/usr/bin/python3",
            "BRAINLAYER_PYTHON": str(brainlayer_python),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.backup-daily.plist"
    assert f"<string>{brainlayer_python}</string>" in rendered.read_text(encoding="utf-8")
    assert "__BRAINLAYER_PYTHON__" not in rendered.read_text(encoding="utf-8")


def test_packaged_launchd_installer_renders_p0_counter_console_shim(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "p0-counter"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": "/usr/bin/python3",
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.p0-counter.plist"
    plist = plistlib.loads(rendered.read_bytes())
    assert plist["ProgramArguments"] == [
        str(home / ".local" / "lib" / "brainlayer" / "brainlayer-env-run.sh"),
        sys.executable,
        "p0-counter",
    ]
    content = rendered.read_text(encoding="utf-8")
    assert "__BRAINLAYER_BIN__" not in content
    assert "__BRAINLAYER_PYTHON__" not in content
    assert "/usr/bin/python3" not in content
    assert "brainlayer.p0_longitudinal_count" not in content
    assert "site-packages/scripts/p0_longitudinal_count.py" not in content
    assert "scripts/p0_longitudinal_count.py" not in content


def test_packaged_launchd_installer_installs_tier0_watchdog_without_env_runner(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    source_script = REPO_ROOT / "scripts" / "tier0-watchdog.sh"
    assert source_script.exists(), "Tier-0 runtime script is missing"
    shutil.copy2(source_script, launchd_dir / "tier0-watchdog.sh")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)

    home = tmp_path / "home&watchdog"
    home.mkdir()
    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "tier0-watchdog"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    installed_script = home / ".local" / "lib" / "brainlayer" / "tier0-watchdog.sh"
    assert installed_script.read_bytes() == source_script.read_bytes()
    assert os.access(installed_script, os.X_OK)

    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.tier0-watchdog.plist"
    plist = plistlib.loads(rendered.read_bytes())
    assert plist["ProgramArguments"] == ["/bin/sh", str(installed_script)]
    assert plist["EnvironmentVariables"]["TIER0_STALE_SECONDS"] == "900"
    assert plist["EnvironmentVariables"]["TIER0_REPEAT_ALERT_SECONDS"] == "1800"
    assert plist["EnvironmentVariables"]["TIER0_ALERT_STATE_PATH"] == str(
        home / ".local" / "share" / "brainlayer" / "tier0-watchdog-alert-state"
    )
    rendered_content = rendered.read_text(encoding="utf-8")
    assert "__TIER0_WATCHDOG_SCRIPT__" not in rendered_content
    assert "brainlayer-env-run" not in rendered_content
    assert "python" not in rendered_content.lower()

    domain = f"gui/{os.getuid()}"
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    bootstrap_command = f"bootstrap {domain} {rendered}"
    print_command = f"print {domain}/com.brainlayer.tier0-watchdog"
    assert commands.count(bootstrap_command) == 1
    assert commands.count(print_command) == 2
    assert commands[-1] == print_command


def test_packaged_launchd_installer_installs_throughput_watchdog(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    source_script = REPO_ROOT / "scripts" / "launchd" / "throughput-watchdog.py"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)

    home = tmp_path / "home&throughput"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    env_file.write_text("BRAINLAYER_SYSTEM_ENABLED=1\n", encoding="utf-8")
    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "throughput-watchdog"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    installed_script = home / ".local" / "lib" / "brainlayer" / "throughput-watchdog.py"
    installed_env_runner = home / ".local" / "lib" / "brainlayer" / "brainlayer-env-run.sh"
    assert installed_script.read_bytes() == source_script.read_bytes()
    assert os.access(installed_script, os.X_OK)
    assert os.access(installed_env_runner, os.X_OK)

    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.throughput-watchdog.plist"
    plist = plistlib.loads(rendered.read_bytes())
    assert plist["ProgramArguments"] == [str(installed_env_runner), sys.executable, str(installed_script), "--json"]
    assert plist["EnvironmentVariables"]["HOME"] == str(home)
    assert plist["EnvironmentVariables"]["BRAINLAYER_ENV_FILE"] == str(env_file)
    assert plist["EnvironmentVariables"]["BRAINLAYER_LAUNCHD_SERVICE"] == "watch"
    assert plist["StartInterval"] == 60
    assert "__THROUGHPUT_WATCHDOG_SCRIPT__" not in rendered.read_text(encoding="utf-8")
    assert "__PYTHON_BIN__" not in rendered.read_text(encoding="utf-8")

    domain = f"gui/{os.getuid()}"
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert f"bootstrap {domain} {rendered}" in commands
    assert f"print {domain}/com.brainlayer.throughput-watchdog" in commands


def test_packaged_launchd_installer_installs_hotlane_daemon(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    source_script = REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py"
    shutil.copy2(source_script, launchd_dir / "hotlane_brainbar_daemon.py")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    _write_fake_ps(fake_bin)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)
    installed_script = home / ".local" / "lib" / "brainlayer" / "hotlane_brainbar_daemon.py"

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "BRAINLAYER_LAUNCHD_VERIFY_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_PS_COMMAND": f"{sys.executable} {installed_script} --interval 1.0",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert installed_script.read_bytes() == source_script.read_bytes()
    assert os.access(installed_script, os.X_OK)

    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.hotlane-brainbar.plist"
    plist = plistlib.loads(rendered.read_bytes())
    assert plist["ProgramArguments"][2] == str(installed_script)
    backlog_index = plist["ProgramArguments"].index("--backlog-batch")
    assert plist["ProgramArguments"][backlog_index + 1] == "4"
    assert "__HOTLANE_BRAINBAR_DAEMON__" not in rendered.read_text(encoding="utf-8")

    domain = f"gui/{os.getuid()}"
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert f"bootstrap {domain} {rendered}" in commands
    assert commands.count(f"print {domain}/com.brainlayer.hotlane-brainbar") >= 3


def test_hotlane_installer_accepts_resolved_python_interpreter(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    _write_fake_ps(fake_bin)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)
    installed_script = home / ".local" / "lib" / "brainlayer" / "hotlane_brainbar_daemon.py"
    python_shim = home / ".pyenv" / "shims" / "python3"

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": str(python_shim),
            "BRAINLAYER_PYTHON": str(python_shim),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "BRAINLAYER_LAUNCHD_VERIFY_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_PS_COMMAND": f"{sys.executable} {installed_script} --interval 1.0",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Verified running: com.brainlayer.hotlane-brainbar (pid 4242)" in result.stdout


def test_hotlane_installer_rejects_unload_timeout_before_bootstrap(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'if [ "$1" = "print" ]; then',
                '  printf "%s\\n" "state = running" "pid = 4242"',
                "fi",
                "exit 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "did not unload before replacement" in result.stderr
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert not any(command.startswith("enable ") for command in commands)
    assert not any(command.startswith("bootstrap ") for command in commands)


def test_launchd_installer_derives_unload_window_from_plist_exit_timeout(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    launchctl_count = tmp_path / "launchctl.count"
    launchctl_count.write_text("0\n", encoding="utf-8")
    bootstrapped = tmp_path / "bootstrapped"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'case "$1" in',
                "  bootout)",
                "    exit 0",
                "    ;;",
                "  bootstrap)",
                '    touch "$FAKE_BOOTSTRAPPED"',
                "    exit 0",
                "    ;;",
                "  print)",
                '    if [ -f "$FAKE_BOOTSTRAPPED" ]; then',
                '      printf "%s\\n" "state = running" "pid = 4242"',
                "      exit 0",
                "    fi",
                '    count="$(cat "$FAKE_LAUNCHCTL_COUNT")"',
                "    count=$((count + 1))",
                '    printf "%s\\n" "$count" > "$FAKE_LAUNCHCTL_COUNT"',
                '    if [ "$count" -le 25 ]; then',
                '      printf "%s\\n" "state = running" "pid = 5151"',
                "      exit 0",
                "    fi",
                "    exit 1",
                "    ;;",
                "  *)",
                "    exit 0",
                "    ;;",
                "esac",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    fake_sleep = fake_bin / "sleep"
    fake_sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_sleep.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "maintenance-weekly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_LAUNCHCTL_COUNT": str(launchctl_count),
            "FAKE_BOOTSTRAPPED": str(bootstrapped),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert int(launchctl_count.read_text(encoding="utf-8")) == 26
    assert bootstrapped.exists()


def test_launchd_installer_rejects_invalid_unload_attempt_overrides(tmp_path: Path) -> None:
    cases = [
        ("watch", "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS", "0"),
        ("watch", "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS", "-1"),
        ("watch", "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS", "invalid"),
        ("hotlane", "BRAINLAYER_HOTLANE_UNLOAD_ATTEMPTS", "0"),
    ]

    for index, (service, env_name, value) in enumerate(cases):
        case_dir = tmp_path / str(index)
        launchd_dir = case_dir / "site-packages" / "brainlayer" / "launchd"
        _copy_packaged_launchd(launchd_dir)
        if service == "hotlane":
            shutil.copy2(
                REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
                launchd_dir / "hotlane_brainbar_daemon.py",
            )

        fake_bin = case_dir / "bin"
        fake_bin.mkdir()
        launchctl_log = case_dir / "launchctl.log"
        fake_launchctl = fake_bin / "launchctl"
        fake_launchctl.write_text("\n".join(_fake_launchctl_lines()), encoding="utf-8")
        fake_launchctl.chmod(0o755)

        home = case_dir / "home"
        home.mkdir()
        env_file = home / ".config" / "brainlayer" / "brainlayer.env"
        env_file.parent.mkdir(parents=True)
        _write_full_launchd_env(env_file)

        result = subprocess.run(
            [str(launchd_dir / "install.sh"), service],
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "HOME": str(home),
                "BRAINLAYER_BIN": sys.executable,
                "PYTHON_BIN": sys.executable,
                "BRAINLAYER_PYTHON": sys.executable,
                "BRAINLAYER_ENV_FILE": str(env_file),
                env_name: value,
                "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode != 0
        assert "unload attempts must be a positive integer" in result.stderr
        assert not launchctl_log.exists()


def test_hotlane_installer_accepts_supervisor_reload_before_unload_poll(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    launchctl_state = tmp_path / "launchctl.state"
    launchctl_state.write_text("initial\n", encoding="utf-8")
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'state="$(cat "$FAKE_LAUNCHCTL_STATE")"',
                'case "$1" in',
                "  bootout)",
                "    # Simulate a supervisor reload before the first unload poll.",
                '    printf "%s\\n" "reloaded" > "$FAKE_LAUNCHCTL_STATE"',
                "    exit 0",
                "    ;;",
                "  bootstrap)",
                "    exit 5",
                "    ;;",
                "  print)",
                '    if [ "$state" = "initial" ]; then',
                '      printf "%s\\n" "state = running" "pid = 5151"',
                "    else",
                '      printf "%s\\n" "state = running" "pid = 5152"',
                "    fi",
                "    exit 0",
                "    ;;",
                "  *)",
                "    exit 0",
                "    ;;",
                "esac",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    _write_fake_ps(fake_bin)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)
    installed_script = home / ".local" / "lib" / "brainlayer" / "hotlane_brainbar_daemon.py"

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "BRAINLAYER_LAUNCHD_VERIFY_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_LAUNCHCTL_STATE": str(launchctl_state),
            "FAKE_PS_COMMAND": f"{sys.executable} {installed_script} --interval 1.0",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "reloaded before the unload poll" in result.stderr
    assert "Verified running: com.brainlayer.hotlane-brainbar (pid 5152)" in result.stdout
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert not any(command.startswith("bootstrap ") for command in commands)


def test_launchd_installer_accepts_supervisor_reload_for_healed_service(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    launchctl_state = tmp_path / "launchctl.state"
    launchctl_state.write_text("initial\n", encoding="utf-8")
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'state="$(cat "$FAKE_LAUNCHCTL_STATE")"',
                'case "$1" in',
                "  bootout)",
                '    printf "%s\\n" "reloaded" > "$FAKE_LAUNCHCTL_STATE"',
                "    exit 0",
                "    ;;",
                "  bootstrap)",
                "    exit 5",
                "    ;;",
                "  print)",
                '    if [ "$state" = "initial" ]; then',
                '      printf "%s\\n" "state = running" "pid = 5151"',
                "    else",
                '      printf "%s\\n" "state = running" "pid = 5152"',
                "    fi",
                "    exit 0",
                "    ;;",
                "  *)",
                "    exit 0",
                "    ;;",
                "esac",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "watch"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_LAUNCHCTL_STATE": str(launchctl_state),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "reloaded before the unload poll" in result.stderr
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert not any(command.startswith("bootstrap ") for command in commands)


def test_hotlane_installer_rejects_pid_change_after_failed_bootout(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    launchctl_state = tmp_path / "launchctl.state"
    launchctl_state.write_text("initial\n", encoding="utf-8")
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
                'state="$(cat "$FAKE_LAUNCHCTL_STATE")"',
                'case "$1" in',
                "  bootout)",
                '    printf "%s\\n" "restarted" > "$FAKE_LAUNCHCTL_STATE"',
                "    exit 5",
                "    ;;",
                "  print)",
                '    if [ "$state" = "initial" ]; then',
                '      printf "%s\\n" "state = running" "pid = 5151"',
                "    else",
                '      printf "%s\\n" "state = running" "pid = 5152"',
                "    fi",
                "    exit 0",
                "    ;;",
                "  *)",
                "    exit 0",
                "    ;;",
                "esac",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_LAUNCHCTL_STATE": str(launchctl_state),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "did not unload before replacement" in result.stderr
    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert not any(command.startswith("enable ") for command in commands)
    assert not any(command.startswith("bootstrap ") for command in commands)


def test_hotlane_installer_rejects_launchd_job_that_is_not_running(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines(post_bootstrap_output=("state = waiting", "last exit code = 2"))),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "BRAINLAYER_LAUNCHD_VERIFY_ATTEMPTS": "2",
            "BRAINLAYER_LAUNCHD_VERIFY_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "hotlane runtime verification failed" in result.stderr


def test_hotlane_installer_rejects_stable_disabled_env_runner_pid(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    _write_fake_ps(fake_bin)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)
    installed_script = home / ".local" / "lib" / "brainlayer" / "hotlane_brainbar_daemon.py"

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "BRAINLAYER_LAUNCHD_VERIFY_ATTEMPTS": "2",
            "BRAINLAYER_LAUNCHD_VERIFY_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_PS_NAME": "bash",
            "FAKE_PS_COMMAND": (
                f"/bin/bash {home}/.local/lib/brainlayer/brainlayer-env-run.sh {sys.executable} {installed_script}"
            ),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "does not match packaged hotlane daemon" in result.stderr


def test_hotlane_installer_accepts_supervisor_bootstrap_race_after_confirmed_unload(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    shutil.copy2(
        REPO_ROOT / "scripts" / "hotlane_brainbar_daemon.py",
        launchd_dir / "hotlane_brainbar_daemon.py",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_state = tmp_path / "launchctl.state"
    launchctl_state.write_text("loaded\n", encoding="utf-8")
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'state="$(cat "$FAKE_LAUNCHCTL_STATE")"',
                'case "$1" in',
                "  bootout)",
                '    printf "%s\\n" "unloaded" > "$FAKE_LAUNCHCTL_STATE"',
                "    exit 0",
                "    ;;",
                "  bootstrap)",
                "    # Simulate the fleet watchdog winning the race with this bootstrap.",
                '    printf "%s\\n" "loaded" > "$FAKE_LAUNCHCTL_STATE"',
                "    exit 5",
                "    ;;",
                "  print)",
                '    if [ "$state" = "loaded" ]; then',
                '      printf "%s\\n" "state = running" "pid = 5152"',
                "      exit 0",
                "    fi",
                "    exit 1",
                "    ;;",
                "  *)",
                "    exit 0",
                "    ;;",
                "esac",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    _write_fake_ps(fake_bin)

    home = tmp_path / "home"
    home.mkdir()
    env_file = home / ".config" / "brainlayer" / "brainlayer.env"
    env_file.parent.mkdir(parents=True)
    _write_full_launchd_env(env_file)
    installed_script = home / ".local" / "lib" / "brainlayer" / "hotlane_brainbar_daemon.py"

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "hotlane"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_PYTHON": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "BRAINLAYER_LAUNCHD_VERIFY_INTERVAL": "0",
            "FAKE_LAUNCHCTL_STATE": str(launchctl_state),
            "FAKE_PS_COMMAND": f"{sys.executable} {installed_script} --interval 1.0",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "bootstrap raced with another supervisor" in result.stderr
    assert "Verified running: com.brainlayer.hotlane-brainbar (pid 5152)" in result.stdout


def test_launchd_installer_renders_launchd_dir_for_maintenance_resume(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)
    launchd_dir = tmp_path / "site-packages" / "brainlayer" / "launchd"

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_LAUNCHD_DIR": str(launchd_dir),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.maintenance-nightly.plist"
    content = rendered.read_text(encoding="utf-8")
    assert f"<string>{launchd_dir}</string>" in content
    assert "__BRAINLAYER_LAUNCHD_DIR__" not in content


def test_launchd_installer_renders_homebrew_opt_symlink_instead_of_cellar_version(tmp_path: Path) -> None:
    fake_homebrew = tmp_path / "opt" / "homebrew"
    cellar_root = fake_homebrew / "Cellar" / "brainlayer" / "9.9.9"
    opt_root = fake_homebrew / "opt" / "brainlayer"
    launchd_dir = cellar_root / "libexec" / "lib" / "python3.12" / "site-packages" / "brainlayer" / "launchd"
    _copy_packaged_launchd(launchd_dir)
    opt_root.parent.mkdir(parents=True)
    opt_root.symlink_to(cellar_root)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(_fake_launchctl_lines()),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    brainlayer_bin = cellar_root / "libexec" / "bin" / "brainlayer"
    brainlayer_bin.parent.mkdir(parents=True)
    brainlayer_bin.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    brainlayer_bin.chmod(0o755)
    python_bin = cellar_root / "libexec" / "bin" / "python3"
    python_bin.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    python_bin.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": str(brainlayer_bin),
            "PYTHON_BIN": str(python_bin),
            "BRAINLAYER_PYTHON": str(python_bin),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rendered = home / "Library" / "LaunchAgents" / "com.brainlayer.maintenance-nightly.plist"
    content = rendered.read_text(encoding="utf-8")
    assert f"{fake_homebrew}/Cellar/brainlayer/9.9.9" not in content
    assert f"<string>{opt_root}/libexec/bin/python3</string>" in content
    assert f"<string>{opt_root}/libexec/lib/python3.12/site-packages</string>" in content
    assert f"<string>{opt_root}/libexec/lib/python3.12/site-packages/brainlayer/launchd</string>" in content


def test_launchd_installer_attempts_remaining_services_after_bootstrap_error(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            _fake_launchctl_lines(
                'if [ "$1" = "bootstrap" ] && [[ "$3" == *"maintenance-nightly.plist" ]]; then',
                '  printf "%s\\n" "bootstrap failed" >&2',
                "  exit 5",
                "fi",
            )
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert any(
        command.startswith("bootstrap ") and command.endswith("com.brainlayer.maintenance-nightly.plist")
        for command in commands
    )
    assert any(
        command.startswith("bootstrap ") and command.endswith("com.brainlayer.maintenance-weekly.plist")
        for command in commands
    )


def test_launchd_installer_does_not_load_services_when_env_runner_install_fails(tmp_path: Path) -> None:
    launchd_dir = tmp_path / "launchd"
    shutil.copytree(REPO_ROOT / "scripts" / "launchd", launchd_dir)
    (launchd_dir / "brainlayer-env-run.sh").unlink()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(launchd_dir / "install.sh"), "maintenance"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "brainlayer-env-run.sh" in result.stdout
    assert not launchctl_log.exists()
    assert not list((home / "Library" / "LaunchAgents").glob("com.brainlayer.*.plist"))


def test_launchd_installer_does_not_load_service_when_env_runner_copy_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_install = fake_bin / "install"
    fake_install.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "install failed" >&2',
                "exit 13",
            ]
        ),
        encoding="utf-8",
    )
    fake_install.chmod(0o755)
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "install failed" in result.stderr
    assert not launchctl_log.exists()


def test_launchd_installer_does_not_load_service_when_plist_render_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sed = fake_bin / "sed"
    fake_sed.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "sed failed" >&2',
                "exit 7",
            ]
        ),
        encoding="utf-8",
    )
    fake_sed.chmod(0o755)
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "sed failed" in result.stderr
    assert not launchctl_log.exists()


def test_launchd_enable_missing_service_is_retried_after_bootstrap(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    bootstrap_state = tmp_path / "bootstrapped"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            _fake_launchctl_lines(
                'if [ "$1" = "enable" ] && [ ! -f "$FAKE_BOOTSTRAPPED" ]; then',
                '  printf "%s\\n" "Could not find service" >&2',
                "  exit 113",
                "fi",
                'if [ "$1" = "bootstrap" ]; then',
                '  touch "$FAKE_BOOTSTRAPPED"',
                "fi",
            )
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    env_file.write_text("BRAINLAYER_ENRICH_ENABLED=0\n", encoding="utf-8")
    env_file.chmod(0o600)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "maintenance-nightly"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
            "FAKE_BOOTSTRAPPED": str(bootstrap_state),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode == 0, result.stdout + result.stderr
    assert [command.split()[0] for command in commands] == [
        "bootout",
        "print",
        "enable",
        "bootstrap",
        "enable",
        "print",
    ]


def test_launchd_all_does_not_load_backup_jobs_when_wrapper_render_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sed = fake_bin / "sed"
    fake_sed.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'last="${@: -1}"',
                'case "$last" in',
                '  */backup-daily.sh|*/jsonl-backup.sh) printf "%s\\n" "wrapper render failed" >&2; exit 7 ;;',
                "esac",
                'exec /usr/bin/sed "$@"',
            ]
        ),
        encoding="utf-8",
    )
    fake_sed.chmod(0o755)
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                'printf "%s\\n" "$*" >> "$FAKE_LAUNCHCTL_LOG"',
            ]
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert "wrapper render failed" in result.stderr
    assert not any(command.startswith("bootstrap ") and "backup-daily.plist" in command for command in commands)
    assert not any(command.startswith("bootstrap ") and "jsonl-backup.plist" in command for command in commands)


def test_launchd_all_preserves_legacy_enrich_when_replacement_batch_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            _fake_launchctl_lines(
                'if [ "$1" = "bootstrap" ] && [[ "$3" == *"com.brainlayer.enrichment.plist" ]]; then',
                '  printf "%s\\n" "replacement bootstrap failed" >&2',
                "  exit 5",
                "fi",
            )
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert "replacement bootstrap failed" in result.stderr
    assert not any(command.startswith("unload ") and "com.brainlayer.enrich.plist" in command for command in commands)


def test_launchd_all_removes_legacy_enrich_when_replacement_loads_despite_sibling_failure(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    launchctl_log = tmp_path / "launchctl.log"
    fake_launchctl = fake_bin / "launchctl"
    fake_launchctl.write_text(
        "\n".join(
            _fake_launchctl_lines(
                'if [ "$1" = "bootstrap" ] && [[ "$3" == *"com.brainlayer.repair-fts.plist" ]]; then',
                '  printf "%s\\n" "repair bootstrap failed" >&2',
                "  exit 5",
                "fi",
            )
        ),
        encoding="utf-8",
    )
    fake_launchctl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    env_file = tmp_path / "brainlayer.env"
    _write_full_launchd_env(env_file)

    result = subprocess.run(
        [str(REPO_ROOT / "scripts" / "launchd" / "install.sh"), "all"],
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": str(home),
            "BRAINLAYER_BIN": sys.executable,
            "PYTHON_BIN": sys.executable,
            "BRAINLAYER_ENV_FILE": str(env_file),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "BRAINLAYER_LAUNCHD_UNLOAD_INTERVAL": "0",
            "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    commands = launchctl_log.read_text(encoding="utf-8").splitlines()
    assert result.returncode != 0
    assert "repair bootstrap failed" in result.stderr
    assert any(command.startswith("unload ") and "com.brainlayer.enrich.plist" in command for command in commands)


def test_wheel_contains_cli_and_launchd_templates(tmp_path: Path) -> None:
    wheel_dir = tmp_path / "dist"
    pip_available = (
        subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )
    build_command = (
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-cache-dir",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(REPO_ROOT),
        ]
        if pip_available
        else ["uv", "build", "--no-cache", "--wheel", "--out-dir", str(wheel_dir)]
    )

    assert pip_available or shutil.which("uv"), "wheel build test requires either pip or uv"
    result = subprocess.run(
        build_command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    wheel = next(wheel_dir.glob("brainlayer-*.whl"))
    listing = subprocess.run(
        [sys.executable, "-m", "zipfile", "-l", str(wheel)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert "brainlayer/cli/__init__.py" in listing
    assert "brainlayer/cli_new.py" in listing
    assert "brainlayer/launchd/install.sh" in listing
    assert "brainlayer/launchd/com.brainlayer.enrichment.plist" in listing
    assert "brainlayer/launchd/com.brainlayer.tier0-watchdog.plist" in listing
    assert "brainlayer/launchd/tier0-watchdog.sh" in listing
    assert "brainlayer/launchd/com.brainlayer.throughput-watchdog.plist" in listing
    assert "brainlayer/launchd/throughput-watchdog.py" in listing
    assert "brainlayer/launchd/hotlane_brainbar_daemon.py" in listing

    extracted = tmp_path / "installed-wheel"
    with zipfile.ZipFile(wheel) as archive:
        entry_points_path = next(name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt"))
        entry_points = archive.read(entry_points_path).decode("utf-8")
        archive.extractall(extracted)
    assert "brainlayer = brainlayer.cli:app" in entry_points
    assert "brainlayer-mcp = brainlayer.mcp:serve" in entry_points
    assert "brainlayer-mcp-stdio-bridge = brainlayer.mcp_stdio_bridge:main" in entry_points
    packaged_hotlane = extracted / "brainlayer" / "launchd" / "hotlane_brainbar_daemon.py"
    help_result = subprocess.run(
        [sys.executable, str(packaged_hotlane), "--help"],
        cwd=extracted,
        env={**os.environ, "PYTHONPATH": str(extracted)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "--backlog-batch" in help_result.stdout
