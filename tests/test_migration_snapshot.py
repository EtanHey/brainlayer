"""Scratch-HOME input and state snapshots precede installer mutations."""

import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_consolidation_plan import DOMAIN, OLD, ROOT
from tests.test_consolidation_plan import setup as setup


@pytest.mark.parametrize("suffix", [".plist", ".plist.loaded-keg", ".loaded-keg", ".PLIST", ".PLIST.LOADED-KEG"])
def test_case_insensitive_scan_refuses_unknown_label(setup, suffix):
    from brainlayer.consolidation_plan import collect_legacy

    path = setup[1] / ("com.Golems.BrainLayer-reembed-monitor" + suffix)
    path.write_bytes(plistlib.dumps({"Label": path.stem}) if suffix == ".plist" else b"old-keg\n")
    with pytest.raises(ValueError, match="BrainLayer-reembed-monitor") as error:
        collect_legacy(setup[1], setup[2], set())
    assert str(path) in str(error.value)


@pytest.mark.parametrize("linked", ["symlink", "hardlink"])
def test_selected_env_link_refused_without_changing_inputs(setup, tmp_path, linked):
    from brainlayer.consolidation_plan import collect_legacy

    env = setup[2]
    raw = env.read_bytes()
    target = tmp_path / "env-target"
    target.write_bytes(raw)
    env.unlink()
    env.symlink_to(target) if linked == "symlink" else os.link(target, env)
    with pytest.raises((OSError, ValueError)):
        collect_legacy(setup[1], env, set())
    assert target.read_bytes() == raw


def test_selected_env_fifo_refused_without_waiting(setup):
    env = setup[2]
    env.unlink()
    os.mkfifo(env)
    code = (
        "import sys;from pathlib import Path;from brainlayer.consolidation_plan import collect_legacy;"
        "\ntry: collect_legacy(Path(sys.argv[1]),Path(sys.argv[2]),set())"
        "\nexcept (OSError,ValueError): print('refused')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(setup[1]), str(env)],
        capture_output=True,
        text=True,
        timeout=3,
        env=os.environ | {"PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0 and result.stdout.strip() == "refused", result.stderr


def test_shared_env_is_read_once_and_never_reopened(setup, monkeypatch):
    from brainlayer import consolidation_plan

    env = setup[2]
    env.write_text("BRAINLAYER_SYSTEM_ENABLED=1\n")
    reads = []
    original = consolidation_plan.read_owned

    def read(path):
        raw = original(path)
        if path == env:
            reads.append(path)
            env.write_text("BRAINLAYER_SYSTEM_ENABLED=0\n")
        return raw

    monkeypatch.setattr(consolidation_plan, "read_owned", read)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("env snapshot reopened"))
    records, _ = consolidation_plan.collect_legacy(setup[1], env, set())
    assert reads == [env]
    assert all(record["enabled"] for record in records.values())


def test_missing_selected_env_keeps_plist_off_fallback(setup, monkeypatch):
    from brainlayer.consolidation_plan import collect_legacy

    setup[2].unlink()
    path = setup[1] / f"{OLD}.plist"
    plist = plistlib.loads(path.read_bytes())
    plist["EnvironmentVariables"] = {"BRAINLAYER_SYSTEM_ENABLED": "disabled"}
    path.write_bytes(plistlib.dumps(plist))
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: pytest.fail("missing env reopened"))
    records, _ = collect_legacy(setup[1], setup[2], set())
    assert records[OLD]["enabled"] is False


def test_non_env_run_watchdogs_keep_effective_state_under_system_off(setup):
    from brainlayer.consolidation_plan import collect_legacy

    tier0 = "com.brainlayer.tier0-watchdog"
    (setup[1] / f"{tier0}.plist").write_bytes(
        plistlib.dumps({"Label": tier0, "ProgramArguments": ["/bin/sh", "watchdog"]})
    )
    setup[2].write_text("BRAINLAYER_SYSTEM_ENABLED=0\n")
    records, _ = collect_legacy(setup[1], setup[2], set())
    assert records[OLD]["enabled"] is False
    assert records[tier0]["enabled"] is True
    assert records["com.etanhey.brainlayer-fleet-watchdog"]["enabled"] is True


def test_snapshot_includes_archived_load_disable_and_exact_bytes(setup):
    from brainlayer.consolidation_plan import snapshot_legacy

    archived = "com.brainlayer.backup-daily"
    path = setup[1] / f"{archived}.plist"
    raw = plistlib.dumps({"Label": archived, "ProgramArguments": ["/bin/true"]})
    path.write_bytes(raw)
    calls = []

    def run(argv):
        assert argv[2] == DOMAIN or argv[2].startswith(DOMAIN + "/")
        assert argv[1] in {"print", "print-disabled"}, "snapshot must never mutate launchd"
        calls.append(argv)
        if argv[1] == "print-disabled":
            return subprocess.CompletedProcess(argv, 0, f'disabled services = {{\n"{archived}" => true\n}}', "")
        if argv[2].endswith(archived):
            return subprocess.CompletedProcess(argv, 113, "", "Could not find service")
        return subprocess.CompletedProcess(argv, 0, "state = running\npid = 4242\n", "")

    from brainlayer.migration_custody import migration_lock, write_new_file

    with migration_lock(setup[0]) as control:
        snapshot = snapshot_legacy(setup[1], setup[2], DOMAIN, run)
        journal = control / "migration.json"
        write_new_file(journal, json.dumps(snapshot).encode())
        assert journal.stat().st_mode & 0o777 == 0o600
        assert json.loads(journal.read_bytes()) == snapshot
    restored = json.loads(json.dumps(snapshot))
    assert restored["states"][archived] == {"loaded": False, "disabled": True, "pid": None}
    assert restored["states"][OLD] == {"loaded": True, "disabled": False, "pid": 4242}
    assert bytes.fromhex(restored["files"][path.name]) == raw
    assert bytes.fromhex(restored["files"][f"{OLD}.plist.loaded-keg"]) == b"previous-keg\n"
    assert len([argv for argv in calls if argv[1] == "print"]) == 3


@pytest.mark.parametrize(
    "rc,text", [(5, "permission denied"), (113, "ambiguous"), (0, "pid = invalid"), (0, "pid =\n")]
)
def test_snapshot_refuses_unmeasurable_launchd_state(setup, rc, text):
    from brainlayer.consolidation_plan import snapshot_legacy

    def run(argv):
        if argv[1] == "print-disabled":
            return subprocess.CompletedProcess(argv, 0, "disabled services = {\n}", "")
        return subprocess.CompletedProcess(argv, rc, text, "")

    with pytest.raises(RuntimeError, match="state"):
        snapshot_legacy(setup[1], setup[2], DOMAIN, run)


def test_migration_lock_excludes_a_second_installer(setup):
    from brainlayer.migration_custody import migration_lock

    with migration_lock(setup[0]):
        with pytest.raises(RuntimeError, match="busy"):
            with migration_lock(setup[0]):
                pytest.fail("second installer acquired custody")


@pytest.mark.parametrize("symlink", [False, True])
def test_snapshot_publication_preserves_an_existing_foreign_file(tmp_path, symlink):
    from brainlayer.migration_custody import write_new_file

    path, personal = tmp_path / "migration.json", tmp_path / "personal"
    personal.write_bytes(b"preserve this\n")
    path.symlink_to(personal) if symlink else path.write_bytes(personal.read_bytes())
    with pytest.raises(FileExistsError):
        write_new_file(path, b"new journal\n")
    assert path.read_bytes() == personal.read_bytes() == b"preserve this\n"


def test_snapshot_publication_refuses_symlinked_parent(tmp_path):
    from brainlayer.migration_custody import write_new_file

    actual, linked = tmp_path / "actual", tmp_path / "linked"
    actual.mkdir()
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        write_new_file(linked / "migration.json", b"new journal\n")
    assert not (actual / "migration.json").exists()
