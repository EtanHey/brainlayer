"""Source installer retirement; every service action is synthetic."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _dispatch(tmp_path, *args):
    source = (ROOT / "scripts/launchd/install.sh").read_text()
    dispatcher = 'case "${1:-all}" in' + source.split('case "${1:-all}" in', 1)[1]
    log = tmp_path / "actions"
    # Execute the real dispatcher, replacing all service/file operations with receipts.
    functions = """set -eu
record() { printf '%s\n' "$*" >> "$ACTION_LOG"; }
install_plist() { record install "$@"; }
install_many() { for name in "$@"; do install_plist "$name"; done; }
load_plist() { record load "$@"; }
unload_plist() { record unload "$@"; }
remove_plist() { record remove "$@"; }
"""
    for name in (
        "install_env_runner",
        "verify_config_file",
        "verify_gemini_env_file",
        "install_backup_script",
        "install_jsonl_backup_script",
        "install_tier0_watchdog",
        "install_throughput_watchdog",
        "install_fleet_watchdog",
        "launchd_install_usage",
    ):
        functions += f"{name}() {{ record {name}; }}\n"
    result = subprocess.run(
        ["bash", "-c", functions + dispatcher, "installer", *args],
        env={"PATH": os.defpath, "HOME": str(tmp_path), "ACTION_LOG": str(log), "LOG_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result, log.read_text().splitlines() if log.exists() else []


def test_enrichment_template_is_absent():
    assert not (ROOT / "scripts/launchd/com.brainlayer.enrichment.plist").exists()


@pytest.mark.parametrize("args", [(), ("all",)])
def test_install_all_keeps_local_services_without_enrichment(tmp_path, args):
    result, actions = _dispatch(tmp_path, *args)
    assert result.returncode == 0, result.stderr
    assert "install watch" in actions
    assert "install drain" in actions
    assert "install hotlane-brainbar" in actions
    assert "install backup-daily" in actions
    assert not any(action.split()[-1] in {"enrich", "enrichment"} for action in actions)
    assert "Enrichment label:" not in result.stdout


@pytest.mark.parametrize("args", [("enrich",), ("enrichment",), ("load",), ("load", "enrich"), ("load", "enrichment")])
def test_retired_install_and_load_routes_fail_without_service_actions(tmp_path, args):
    result, actions = _dispatch(tmp_path, *args)
    assert result.returncode != 0
    assert actions == []


def test_unload_requires_explicit_name_and_keeps_retired_cleanup(tmp_path):
    result, actions = _dispatch(tmp_path, "unload")
    assert result.returncode != 0
    assert actions == []
    result, actions = _dispatch(tmp_path, "unload", "enrichment")
    assert result.returncode == 0
    assert actions == ["unload enrichment"]


@pytest.mark.parametrize("name", ["enrich", "enrichment"])
def test_load_helper_rejects_stale_installed_plist_before_launchctl(tmp_path, name):
    source = (ROOT / "scripts/launchd/install.sh").read_text()
    body = "load_plist() {" + source.split("\nload_plist() {", 1)[1].split("\nunload_plist() {", 1)[0]
    plist = tmp_path / f"com.brainlayer.{name}.plist"
    plist.write_text("stale installed template")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "set -eu; launchctl() { echo UNEXPECTED; exit 42; }; " + body + '\nload_plist "$1"',
            "test",
            name,
        ],
        env={"PATH": os.defpath, "HOME": str(tmp_path), "LAUNCH_DIR": str(tmp_path)},
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "retired" in result.stderr
    assert "UNEXPECTED" not in result.stdout
    assert plist.read_text() == "stale installed template"
