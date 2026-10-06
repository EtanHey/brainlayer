"""Source installer retirement; every service action is synthetic."""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _dispatch(tmp_path, *args, fail=False, disabled=False, helper=False):
    source = (ROOT / "scripts/launchd/install.sh").read_text()
    dispatcher = 'case "${1:-all}" in' + source.split('case "${1:-all}" in', 1)[1]
    log = tmp_path / "actions"
    # Run real cleanup helpers and dispatcher; installation and launchd are synthetic.
    functions = """set -eu
record() { printf '%s\n' "$*" >> "$ACTION_LOG"; }
install_plist() { record install "$@"; }
install_many() { for name in "$@"; do install_plist "$name"; done; }
load_plist() { record load "$@"; }
launchctl() {
    record launchctl "$@"
    case "$1" in
        print) [ -f "$LAUNCH_DIR/loaded" ] ;;
        unload) [ "$FAIL_UNLOAD" = 0 ] && rm -f "$LAUNCH_DIR/loaded" ;;
        print-disabled) printf '"com.brainlayer.enrichment" => %s\n' "$DISABLED" ;;
    esac
}
"""
    for name in "install_env_runner verify_config_file verify_gemini_env_file install_backup_script install_jsonl_backup_script install_tier0_watchdog install_throughput_watchdog install_fleet_watchdog launchd_install_usage".split():
        functions += f"{name}() {{ record {name}; }}\n"
    for name in ("load_plist", "unload_plist", "job_display_name", "remove_job_wrapper", "remove_plist"):
        functions += re.search(r"(?m)^" + name + r"\(\) \{[\s\S]*?^\}", source).group() + "\n"
    if helper:
        dispatcher = 'load_plist "$1"'
    result = subprocess.run(
        ["bash", "-c", functions + dispatcher, "installer", *args],
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "ACTION_LOG": str(log),
            "LOG_DIR": str(tmp_path),
            "LAUNCH_DIR": str(tmp_path),
            "BRAINLAYER_LIB_DIR": str(tmp_path),
            "BRAINLAYER_LAUNCHD_UNLOAD_ATTEMPTS": "1",
            "FAIL_UNLOAD": str(int(fail)),
            "DISABLED": str(disabled).lower(),
        },
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
    assert not any(action.startswith(("install enrich", "load enrich")) for action in actions)


@pytest.mark.parametrize(
    "args", [("enrich",), ("enrichment",), ("load",), ("load", "enrich"), ("load", "enrichment"), ("unload",)]
)
def test_retired_install_and_load_routes_fail_without_service_actions(tmp_path, args):
    result, actions = _dispatch(tmp_path, *args)
    assert result.returncode != 0
    assert actions == []


@pytest.mark.parametrize("name", ["enrich", "enrichment"])
def test_load_helper_rejects_stale_installed_plist_before_launchctl(tmp_path, name):
    plist = tmp_path / f"com.brainlayer.{name}.plist"
    plist.write_text("stale installed template")
    result, actions = _dispatch(tmp_path, name, helper=True)
    assert result.returncode == 1 and "retired" in result.stderr
    assert actions == []
    assert plist.read_text() == "stale installed template"


@pytest.mark.parametrize(
    "action,name,disabled,present,fail",
    [
        ("all", "enrichment", False, True, False),
        ("all", "enrich", True, True, False),
        ("enrich", "enrich", False, True, False),
        ("enrichment", "enrichment", True, True, False),
        ("all", "enrichment", False, False, False),
        ("all", "enrichment", False, True, True),
    ],
)
def test_retired_installed_labels_are_removed_safely(tmp_path, action, name, disabled, present, fail):
    plist = tmp_path / f"com.brainlayer.{name}.plist"
    targets = [Path(str(plist) + suffix) for suffix in ("", ".loaded-keg", ".reload-pending")]
    targets.append(tmp_path / "BrainLayer Enrichment")
    untouched = [tmp_path / f"com.brainlayer.{n}.plist" for n in ("watch", "drain")]
    untouched.append(tmp_path / "com.brainlayer.enrichment.plist.PAUSED-old")
    for path in untouched + (targets + [tmp_path / "loaded"] if present else []):
        path.write_bytes(b"historical bytes")
    result, actions = _dispatch(tmp_path, action, fail=fail, disabled=disabled)
    calls = [a for a in actions if a.startswith("launchctl")]
    assert all(p.read_bytes() == b"historical bytes" for p in untouched)
    assert not any("watch" in a or "drain" in a or ".PAUSED-" in a for a in calls)
    if not present:
        assert calls == []
    elif fail:
        assert result.returncode != 0 and "refusing to remove" in result.stderr
        assert all(p.read_bytes() == b"historical bytes" for p in targets)
    else:
        assert any(a == f"launchctl unload {plist}" for a in calls)
        assert all(not p.exists() for p in targets)
        assert result.returncode == (0 if action == "all" else 1)
        if action != "all":
            assert "retired" in result.stderr
