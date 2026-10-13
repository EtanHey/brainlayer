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
    assert restored["states"][archived] == {
        "loaded": False,
        "disabled": True,
        "disabled_override": "disabled",
        "pid": None,
        "plist_present": True,
    }
    assert restored["states"][OLD] == {
        "loaded": True,
        "disabled": False,
        "disabled_override": None,
        "pid": 4242,
        "plist_present": True,
    }
    assert bytes.fromhex(restored["files"][path.name]) == raw
    assert bytes.fromhex(restored["files"][f"{OLD}.plist.loaded-keg"]) == b"previous-keg\n"
    from brainlayer.consolidation_plan import ALIASES, ARCHIVED, NEW

    assert len([argv for argv in calls if argv[1] == "print"]) == len(set(ALIASES) | ARCHIVED | set(NEW))


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


def snapshot(setup, loaded=(), overrides=""):
    from brainlayer.consolidation_plan import snapshot_legacy

    calls = []

    def run(argv):
        calls.append(argv)
        assert argv[1] in {"print", "print-disabled"} and argv[2].startswith(DOMAIN)
        text = f"disabled services = {{\n{overrides}\n}}" if argv[1] == "print-disabled" else "state = waiting"
        rc = 0 if argv[1] == "print-disabled" or argv[2].split("/")[-1] in loaded else 113
        return subprocess.CompletedProcess(argv, rc, text if rc == 0 else "", "Could not find service" if rc else "")

    return snapshot_legacy(setup[1], setup[2], DOMAIN, run), calls


@pytest.mark.parametrize("loaded", [False, True])
def test_effective_enabled_requires_loaded_but_not_pid(setup, loaded):
    from brainlayer.supervisor_jobs import load_components

    result, _ = snapshot(setup, [OLD] if loaded else [])
    assert result["components"][OLD]["enabled"] is loaded
    assert result["states"][OLD]["pid"] is None
    manifest = setup[0] / "jobs.json"
    manifest.write_text(json.dumps(result["components"]))
    manifest.chmod(0o600)
    assert load_components(manifest, setup[0] / "socket")["watch"].enabled is loaded


def test_all_owned_labels_journaled_even_loaded_without_plist(setup):
    from brainlayer.consolidation_plan import ALIASES, ARCHIVED, NEW

    orphan = "com.brainlayer.enrichment"
    result, calls = snapshot(setup, [orphan], f'"{OLD}" => false\n"{orphan}" => true')
    assert set(result["states"]) == set(ALIASES) | ARCHIVED | set(NEW)
    assert len(calls) == 1 + len(result["states"])
    assert result["states"][orphan] == {
        "loaded": True,
        "disabled": True,
        "disabled_override": "disabled",
        "pid": None,
        "plist_present": False,
    }
    assert result["states"][OLD]["disabled_override"] == "enabled"
    assert result["states"][NEW[0]]["disabled_override"] is None
    assert result["states"][NEW[0]]["loaded"] is False


@pytest.mark.parametrize("partial", [False, True])
def test_consolidated_replay_snapshots_new_bytes_and_states(setup, partial):
    from brainlayer.consolidation_plan import NEW

    for path in setup[1].iterdir():
        path.unlink()
    labels = NEW[:1] if partial else NEW
    if partial:
        (setup[1] / f"{NEW[1]}.plist.loaded-keg").write_bytes(b"partial-keg\n")
    for label in labels:
        (setup[1] / f"{label}.plist").write_bytes(plistlib.dumps({"Label": label}))
        (setup[1] / f"{label}.plist.loaded-keg").write_bytes(b"old-keg\n")
    result, _ = snapshot(setup, labels)
    assert result["components"] == {}
    for label in NEW:
        assert result["states"][label]["plist_present"] is (label in labels)
        assert result["states"][label]["loaded"] is (label in labels)
    assert {name: bytes.fromhex(raw) for name, raw in result["files"].items()} == {
        path.name: path.read_bytes() for path in setup[1].iterdir()
    }


def test_missing_env_is_not_runnable_and_has_reason(setup):
    setup[2].unlink()
    result, _ = snapshot(setup, [OLD])
    assert result["components"][OLD]["enabled"] is False
    assert "missing env file" in result["components"][OLD]["reason"]


@pytest.mark.parametrize("label", ["com.brainlayer.tier0-watchdog", "com.etanhey.brainlayer-fleet-watchdog"])
def test_missing_env_keeps_loaded_non_env_run_watchdogs_on(setup, label):
    (setup[1] / f"{label}.plist").write_bytes(
        plistlib.dumps({"Label": label, "ProgramArguments": ["/bin/true"], "KeepAlive": True})
    )
    setup[2].unlink()
    result, _ = snapshot(setup, [label])
    assert result["components"][label]["enabled"] is True


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_snapshot_publication_refuses_nonprivate_directory(tmp_path, mode):
    from brainlayer.migration_custody import write_new_file

    tmp_path.chmod(mode)
    with pytest.raises(ValueError, match="0700"):
        write_new_file(tmp_path / "migration.json", b"private env")
    assert list(tmp_path.iterdir()) == []


def test_launch_agent_publication_uses_owned_nonprivate_agents_dir(setup):
    from brainlayer.migration_custody import write_launch_agent

    path = setup[1] / "com.brainlayer.daemon.plist"
    write_launch_agent(path, b"rendered plist")
    assert path.read_bytes() == b"rendered plist" and path.stat().st_mode & 0o777 == 0o644
    with pytest.raises(FileExistsError):
        write_launch_agent(path, b"replacement")
    assert path.read_bytes() == b"rendered plist"


def test_failed_file_fsync_preserves_primary_error_when_cleanup_fsync_fails(tmp_path, monkeypatch):
    import errno
    import stat

    from brainlayer.migration_custody import write_new_file

    def fsync(fd):
        raise OSError(errno.ENOSPC if stat.S_ISDIR(os.fstat(fd).st_mode) else errno.EIO, "injected")

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError) as failure:
        write_new_file(tmp_path / "migration.json", b"private bytes")
    assert failure.value.errno == errno.EIO
    assert list(tmp_path.iterdir()) == []


def test_launch_agent_publication_refuses_foreign_name_and_writable_directory(setup):
    from brainlayer.migration_custody import write_launch_agent

    with pytest.raises(ValueError, match="non-consolidated"):
        write_launch_agent(setup[1] / "com.example.personal.plist", b"refuse")
    setup[1].chmod(0o777)
    with pytest.raises(ValueError, match="group/world writable"):
        write_launch_agent(setup[1] / "com.brainlayer.daemon.plist", b"refuse")
    assert not (setup[1] / "com.brainlayer.daemon.plist").exists()
