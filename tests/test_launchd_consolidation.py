"""Scratch-only mutation and rollback invariants."""

import importlib
import json
import os
import plistlib
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_consolidation_plan import DOMAIN, NEW, OLD, ROOT
from tests.test_consolidation_plan import setup as setup
from tests.test_supervisor_jobs import short_path as short_path


class Launchctl:
    def __init__(self, disabled=(), fail=""):
        self.loaded = {OLD, "com.etanhey.brainlayer-fleet-watchdog"}
        self.disabled, self.fail, self.calls = set(disabled), fail, []

    def __call__(self, argv):
        action, target = argv[1:3]
        assert target == DOMAIN or target.startswith(DOMAIN + "/"), argv
        self.calls.append(tuple(argv[1:]))
        label = target.rsplit("/", 1)[-1]
        rc, text = 0, ""
        if action == "print-disabled":
            text = "disabled services = {\n" + "\n".join(f'"{k}" => true' for k in self.disabled) + "\n}"
        elif action == "print":
            rc, text = (0, "state = running\npid = 4242") if label in self.loaded else (113, "Could not find service")
        elif action == "bootout":
            self.loaded.discard(label)
        elif action == "disable":
            self.disabled.add(label)
        elif action == "enable":
            self.disabled.discard(label)
        elif action == "bootstrap":
            label = plistlib.loads(Path(argv[3]).read_bytes())["Label"]
            if label == self.fail:
                self.fail, rc = "", 5
            else:
                assert label not in self.loaded, "duplicate launcher"
                self.loaded.add(label)
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, rc, text, text if rc else "")


def invoke(setup, runner, keg="keg1", **options):
    home, _, env = setup
    module = importlib.import_module("brainlayer.launchd_consolidation")
    python = home / "opt/brainlayer/libexec/venv/bin/python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    (python.parents[1] / "lib/python3.12/site-packages/brainlayer").mkdir(parents=True, exist_ok=True)
    with patch.dict(os.environ, {"BRAINLAYER_HOOK_PYTHON": str(python)}):
        return module.migrate(
            home=home,
            templates=ROOT / "scripts/launchd",
            env_file=env,
            python=python,
            keg=keg,
            domain=DOMAIN,
            run=runner,
            **({"alive": lambda pid: False, "verify": lambda: None} | options),
        )


def test_first_install_archives_files_markers_and_revivers_before_new_writers(setup):
    home, agents, _ = setup
    runner = Launchctl()
    invoke(setup, runner)
    archive = next(agents.glob(".archive-*/"))
    assert (archive / f"{OLD}.plist").is_file()
    assert (archive / f"{OLD}.plist.loaded-keg").read_text() == "previous-keg\n"
    assert not (agents / f"{OLD}.plist").exists()
    manifest = home / ".local/share/brainlayer/supervisor/jobs.json"
    assert manifest.stat().st_mode & 0o077 == 0
    assert json.loads(manifest.read_text())[OLD]["enabled"] is True
    calls = runner.calls
    fleet = "com.etanhey.brainlayer-fleet-watchdog"
    assert calls.index(("bootout", f"{DOMAIN}/{fleet}")) < calls.index(("bootout", f"{DOMAIN}/{OLD}"))
    assert not any(c[0] in {"enable", "disable"} for c in calls)
    first_bootstrap = next(i for i, c in enumerate(calls) if c[0] == "bootstrap")
    assert calls.index(("bootout", f"{DOMAIN}/{OLD}")) < first_bootstrap
    assert runner.loaded == {NEW, "com.brainlayer.maintenance"}


def test_operator_disabled_component_remains_off(setup):
    runner = Launchctl(disabled={OLD})
    runner.loaded.remove(OLD)
    invoke(setup, runner)
    manifest = json.loads((setup[0] / ".local/share/brainlayer/supervisor/jobs.json").read_text())
    assert manifest[OLD]["enabled"] is False
    assert OLD in runner.disabled
    assert not any(c[0] == "enable" and c[1].endswith(OLD) for c in runner.calls)


def test_partial_bootstrap_failure_reaps_new_then_restores_exact_old_state(setup):
    runner = Launchctl(fail="com.brainlayer.maintenance")
    before = {p.name: p.read_bytes() for p in setup[1].iterdir()}
    with pytest.raises(RuntimeError):
        invoke(setup, runner)
    assert runner.loaded == {OLD, "com.etanhey.brainlayer-fleet-watchdog"}
    assert not runner.disabled
    for name, content in before.items():
        assert (setup[1] / name).read_bytes() == content
    assert not (setup[1] / f"{NEW}.plist").exists()


def test_concurrent_migration_refuses_before_any_launchctl_mutation(setup):
    import fcntl

    control = setup[0] / ".local/share/brainlayer/supervisor"
    control.mkdir(mode=0o700, parents=True)
    lock = control / "migration.lock"
    runner = Launchctl()
    with lock.open("a") as owner:
        lock.chmod(0o600)
        fcntl.flock(owner, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="migration busy"):
            invoke(setup, runner)
    assert not runner.calls


def test_failed_exit_confirmation_never_bootstraps_an_existing_writer(setup):
    from tests.test_migration_runtime import Clock

    runner = Launchctl()
    with pytest.raises(RuntimeError):
        invoke(setup, runner, alive=lambda pid: True, clock=Clock())
    assert not any(c[0] == "bootstrap" for c in runner.calls)


def test_rollback_preserves_a_concurrently_changed_replacement_definition(setup):
    path = setup[1] / f"{NEW}.plist"

    class Changed(Launchctl):
        def __call__(self, argv):
            if argv[1] == "bootstrap" and Path(argv[3]).name == "com.brainlayer.maintenance.plist":
                path.write_bytes(b"foreign replacement data\n")
            return super().__call__(argv)

    runner = Changed(fail="com.brainlayer.maintenance")
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        invoke(setup, runner)
    assert path.read_bytes() == b"foreign replacement data\n"
    assert OLD not in runner.loaded  # Ambiguous custody cannot authorize revival.


def test_original_bytes_and_archived_states_are_journaled_before_bootout(setup):
    archived = "com.brainlayer.backup-daily"
    path = setup[1] / f"{archived}.plist"
    raw = plistlib.dumps({"Label": archived, "ProgramArguments": ["/bin/true"]})
    path.write_bytes(raw)

    class Journalled(Launchctl):
        def __call__(self, argv):
            if argv[1] in {"disable", "bootout"}:
                journal = json.loads((setup[0] / ".local/share/brainlayer/supervisor/migration.json").read_bytes())
                assert bytes.fromhex(journal["files"][path.name]) == raw
                assert journal["states"][archived]["loaded"] is False
                assert journal["states"][archived]["disabled"] is True
            return super().__call__(argv)

    invoke(setup, Journalled(disabled={archived}))


def test_disabled_new_daemon_refuses_before_stopping_running_legacy_jobs(setup):
    setup[2].write_text("BRAINLAYER_SYSTEM_ENABLED=0\n")
    runner = Launchctl()
    with pytest.raises(RuntimeError, match="configured OFF"):
        invoke(setup, runner)
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)
    assert runner.loaded == {OLD, "com.etanhey.brainlayer-fleet-watchdog"}


@pytest.mark.parametrize("kind", ["missing", "override", "plist"])
def test_non_revertible_loaded_preimage_refuses_before_mutation(setup, kind):
    from brainlayer.consolidation_plan import snapshot_legacy

    runner = Launchctl()
    missing_definition = kind == "missing"
    label = (
        "com.brainlayer.enrichment"
        if missing_definition
        else ("com.brainlayer.backup-daily" if kind == "plist" else OLD)
    )
    runner.loaded.add(label)
    if kind == "override":
        runner.disabled.add(label)
    elif kind == "plist":
        path = setup[1] / f"{label}.plist"
        path.write_bytes(plistlib.dumps({"Label": label, "ProgramArguments": ["/bin/true"], "Disabled": True}))
    preimage = snapshot_legacy(setup[1], setup[2], DOMAIN, runner)
    assert preimage["states"][label]["loaded"] is True
    assert preimage["states"][label]["plist_present"] is not missing_definition
    before = {p.name: p.read_bytes() for p in setup[1].iterdir()}
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="not revertible"):
        invoke(setup, runner)
    assert label in runner.loaded
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)
    assert {p.name: p.read_bytes() for p in setup[1].iterdir()} == before
    assert not (setup[0] / ".local/share/brainlayer/supervisor/migration.json").exists()


def test_rollback_preserves_explicit_enabled_disabled_and_absent_overrides(setup):
    class TriState(Launchctl):
        def __call__(self, argv):
            result = super().__call__(argv)
            if argv[1] == "print-disabled":
                result.stdout = result.stdout.replace("\n}", f'\n"{OLD}" => false\n}}')
            return result

    runner = TriState(disabled={"com.brainlayer.observability"}, fail="com.brainlayer.maintenance")
    with pytest.raises(RuntimeError):
        invoke(setup, runner)
    assert not any(c[0] in {"enable", "disable"} for c in runner.calls)
    journal = json.loads((setup[0] / ".local/share/brainlayer/supervisor/migration.json").read_bytes())
    assert journal["states"][OLD]["disabled_override"] == "enabled"
    assert journal["states"]["com.brainlayer.observability"]["disabled_override"] == "disabled"
    assert journal["states"]["com.etanhey.brainlayer-fleet-watchdog"]["disabled_override"] is None


def test_new_disabled_daemon_refuses_handoff_and_preserves_legacy(setup):
    runner = Launchctl(disabled={NEW})
    with pytest.raises(RuntimeError, match="disabled"):
        invoke(setup, runner)
    assert runner.loaded == {OLD, "com.etanhey.brainlayer-fleet-watchdog"}
    assert not any(c[0] == "bootout" for c in runner.calls)


def test_transaction_verifies_real_private_supervisor_and_reaps_owned_groups(short_path):
    from brainlayer.migration_runtime import pid_alive
    from brainlayer.supervisor_adapters import SupervisorCommands

    home = short_path.resolve() / "home"
    agents = home / "Library/LaunchAgents"
    agents.mkdir(parents=True)
    env = home / "brainlayer.env"
    env.write_text("BRAINLAYER_LAUNCHD_WATCH_ENABLED=1\n")
    for label in (OLD, "com.etanhey.brainlayer-fleet-watchdog"):
        (agents / f"{label}.plist").write_bytes(
            plistlib.dumps(
                {
                    "Label": label,
                    "ProgramArguments": ["/bin/sleep", "60"],
                    "KeepAlive": True,
                }
            )
        )
    control = home / ".local/share/brainlayer/supervisor"
    processes, pids = [], []

    class RealSupervisor(Launchctl):
        def __call__(self, argv):
            if argv[1] == "print" and argv[2].endswith(f"/{NEW}") and NEW in self.loaded:
                return subprocess.CompletedProcess(argv, 0, f"pid = {processes[-1].pid}", "")
            if argv[1] == "bootout" and argv[2].endswith(f"/{NEW}"):
                process = processes[-1]
                process.terminate()
                stdout, stderr = process.communicate(timeout=20)
                assert process.returncode == 0, (stdout, stderr)
            if argv[1] == "bootstrap" and Path(argv[3]).name == f"{NEW}.plist":
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "brainlayer.supervisor_jobs",
                            "--manifest",
                            str(control / "jobs.json"),
                            "--socket",
                            str(control / "control.sock"),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                )
            return super().__call__(argv)

    runner = RealSupervisor()
    body_error = None
    try:
        journal = invoke((home, agents, env), runner, verify=None, alive=lambda pid: pid != 4242 and pid_alive(pid))
        assert journal["phase"] == "migrated"
        assert journal["group_custody"][NEW] == str(control)
        states = SupervisorCommands(control / "control.sock", runner).request()
        if sys.platform == "darwin":
            assert states["brainbar-activation"]["configured"] is True
            assert states["brainbar-activation"]["interval"] == 5
        assert states["wal-checkpoint-fallback"]["interval"] == 900
        assert states["wal-checkpoint-fallback"]["pid"] is None
        pids.extend(state["pid"] for state in states.values() if not state["interval"])
        assert len(pids) == 2 and all(pid_alive(pid) for pid in pids)
        assert not (agents / f"{OLD}.plist").exists()
        invoke((home, agents, env), runner, "keg2", verify=None, alive=lambda pid: pid != 4242 and pid_alive(pid))
        updated = SupervisorCommands(control / "control.sock", runner).request()
        if sys.platform == "darwin":
            assert updated["brainbar-activation"]["configured"] is True
        replacements = [state["pid"] for state in updated.values() if not state["interval"]]
        assert len(processes) == 2 and all(pid_alive(pid) for pid in replacements)
        assert not (set(pids) & set(replacements))
        assert all(not pid_alive(pid) for pid in pids)
        pids.extend(replacements)
        assert (agents / f"{NEW}.plist.loaded-keg").read_bytes() == b"keg2\n"
    except BaseException as failure:
        body_error = failure
        raise
    finally:
        for process in processes:
            try:
                process.terminate()
                try:
                    stdout, stderr = process.communicate(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, stderr = process.communicate(timeout=5)
                if body_error is None:
                    assert process.returncode == 0, (stdout, stderr)
            except BaseException as cleanup:
                if body_error is None:
                    raise
                body_error.add_note(f"owned supervisor cleanup failed: {cleanup}")
    assert not (control / "control.sock").exists()
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.killpg(pid, 0)


def test_rollback_does_not_bootout_a_foreign_replacement_arriving_during_migration(setup):
    class Arrival(Launchctl):
        def __call__(self, argv):
            result = super().__call__(argv)
            if argv[1] == "bootout" and argv[2].endswith(OLD):
                self.loaded.add(NEW)
            return result

    runner = Arrival()
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        invoke(setup, runner)
    assert NEW in runner.loaded
    assert ("bootout", f"{DOMAIN}/{NEW}") not in runner.calls
    assert OLD not in runner.loaded


def test_same_keg_replay_does_not_mutate_launchd(setup):
    runner = Launchctl()
    invoke(setup, runner)
    runner.calls.clear()
    invoke(setup, runner)
    assert all(c[0] in {"print", "print-disabled"} for c in runner.calls)


def test_keg_change_restarts_supervisor_without_legacy_revival(setup):
    runner = Launchctl()
    original = invoke(setup, runner)
    runner.calls.clear()
    current = invoke(setup, runner, "keg2")
    assert all(current[key] == original[key] for key in ("files", "states", "archive", "group_custody"))
    assert ("bootout", f"{DOMAIN}/{NEW}") in runner.calls
    assert not any(OLD in " ".join(c) for c in runner.calls if c[0] in {"enable", "bootstrap"})
    assert (setup[1] / f"{NEW}.plist.loaded-keg").read_text().strip() == "keg2"


def test_same_keg_replay_refuses_changed_marker_before_mutation(setup):
    runner = Launchctl()
    invoke(setup, runner)
    (setup[1] / f"{NEW}.plist.loaded-keg").write_text("foreign-keg\n")
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="marker"):
        invoke(setup, runner)
    assert all(c[0] in {"print", "print-disabled"} for c in runner.calls)


def test_replay_refuses_reloaded_plistless_legacy_service(setup):
    runner = Launchctl()
    invoke(setup, runner)
    runner.loaded.add(OLD)
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="legacy service"):
        invoke(setup, runner)
    assert all(c[0] in {"print", "print-disabled"} for c in runner.calls)


def test_replay_refuses_changed_archive_before_keg_restart(setup):
    runner = Launchctl()
    journal = invoke(setup, runner)
    (Path(journal["archive"]) / f"{OLD}.plist").write_bytes(b"changed custody")
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="archive"):
        invoke(setup, runner, "keg2")
    assert not any(c[0] == "bootout" for c in runner.calls)


def test_replay_refuses_journal_pointing_outside_owned_archive(setup, tmp_path):
    runner = Launchctl()
    invoke(setup, runner)
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    journal = json.loads(receipt.read_bytes())
    journal["archive"] = str(foreign)
    receipt.write_text(json.dumps(journal))
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="archive custody"):
        invoke(setup, runner, "keg2")
    assert not any(c[0] == "bootout" for c in runner.calls)


@pytest.mark.parametrize("failed_path", ["com.brainlayer.maintenance.plist.loaded-keg", "migration.json"])
def test_keg_replay_publication_failure_restores_all_markers_and_receipt(setup, monkeypatch, failed_path):
    from brainlayer import launchd_consolidation as module

    runner = Launchctl()
    invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    original = receipt.read_bytes()
    update = module.update

    def fail_once(path, expected, content, archive, publish):
        if path.name == failed_path and (b"keg2" in content):
            raise OSError("injected publication failure")
        return update(path, expected, content, archive, publish)

    monkeypatch.setattr(module, "update", fail_once)
    with pytest.raises(RuntimeError, match="restored"):
        invoke(setup, runner, "keg2")
    assert receipt.read_bytes() == original
    for label in (NEW, "com.brainlayer.maintenance"):
        assert (setup[1] / f"{label}.plist.loaded-keg").read_bytes() == b"keg1\n"
    assert runner.loaded == {NEW, "com.brainlayer.maintenance"}
    assert not any(c[0] in {"enable", "disable"} for c in runner.calls)


def test_keg_replay_preserves_foreign_arrival_after_failed_bootstrap(setup):
    class Arrival(Launchctl):
        foreign = False

        def __call__(self, argv):
            result = super().__call__(argv)
            if self.foreign and argv[1] == "bootstrap" and result.returncode:
                self.loaded.add(NEW)
            return result

    runner = Arrival()
    invoke(setup, runner)
    runner.calls.clear()
    runner.fail, runner.foreign = NEW, True
    with pytest.raises(RuntimeError, match="rollback incomplete.*unowned"):
        invoke(setup, runner, "keg2")
    assert runner.loaded == {NEW, "com.brainlayer.maintenance"}
    assert runner.calls.count(("bootout", f"{DOMAIN}/{NEW}")) == 1


def test_keg_replay_readiness_failure_verifies_recovery_before_claiming_restore(setup):
    runner = Launchctl()
    invoke(setup, runner)
    checks = []

    def verify():
        checks.append(True)
        if len(checks) == 1:
            raise RuntimeError("new supervisor not ready")

    with pytest.raises(RuntimeError, match="restored"):
        invoke(setup, runner, "keg2", verify=verify)
    assert len(checks) == 2
    assert (setup[1] / f"{NEW}.plist.loaded-keg").read_bytes() == b"keg1\n"


def test_archive_precedes_writer_bootout_after_revivers_quiesce(setup):
    class Ordered(Launchctl):
        def __call__(self, argv):
            if argv[1] == "bootout" and argv[2].endswith(OLD):
                assert "com.etanhey.brainlayer-fleet-watchdog" not in self.loaded
                assert not (setup[1] / f"{OLD}.plist").exists()
                archive = next(setup[1].glob(".archive-*/"))
                assert (archive / f"{OLD}.plist").is_file()
            return super().__call__(argv)

    invoke(setup, Ordered())


def test_restart_journal_is_incomplete_until_verification_and_markers(setup):
    class Journalled(Launchctl):
        restarting = False

        def __call__(self, argv):
            if self.restarting and argv[1] == "bootout":
                receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
                assert json.loads(receipt.read_bytes())["phase"] == "migrating"
            return super().__call__(argv)

    runner = Journalled()
    invoke(setup, runner)
    runner.restarting = True
    invoke(setup, runner, "keg2")


def test_marker_fsync_failure_after_publication_restores_preimage(setup, monkeypatch):
    from brainlayer import launchd_consolidation as module

    runner = Launchctl()
    invoke(setup, runner)
    update = module.update
    failed = False

    def fail_after_publish(path, expected, content, archive, writer):
        nonlocal failed
        update(path, expected, content, archive, writer)
        if not failed and path.name == f"{NEW}.plist.loaded-keg" and content == b"keg2\n":
            failed = True
            raise OSError("injected post-publication fsync failure")

    monkeypatch.setattr(module, "update", fail_after_publish)
    with pytest.raises(RuntimeError, match="restored"):
        invoke(setup, runner, "keg2")
    assert failed
    for label in (NEW, "com.brainlayer.maintenance"):
        assert (setup[1] / f"{label}.plist.loaded-keg").read_bytes() == b"keg1\n"


def test_first_completion_publication_failure_rolls_back_journal_and_runtime(setup, monkeypatch):
    from brainlayer import launchd_consolidation as module

    update = module.update
    failed = False

    def fail_after_complete(path, expected, content, archive, publish):
        nonlocal failed
        update(path, expected, content, archive, publish)
        if not failed and path.name == "migration.json" and json.loads(content)["phase"] == "migrated":
            failed = True
            raise OSError("post-publication sync failure")

    monkeypatch.setattr(module, "update", fail_after_complete)
    runner = Launchctl()
    with pytest.raises(RuntimeError, match="original files/states restored"):
        invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    assert json.loads(receipt.read_bytes())["phase"] == "rolled_back"
    assert runner.loaded == {OLD, "com.etanhey.brainlayer-fleet-watchdog"}


def test_foreign_same_byte_arrival_at_exclusive_publication_is_preserved(setup, monkeypatch):
    from brainlayer import launchd_consolidation as module

    publish = module.write_launch_agent
    arrivals = []

    def arrival(path, content):
        if path.name == f"{NEW}.plist":
            path.write_bytes(content)
            arrivals.append((path, content))
        return publish(path, content)

    monkeypatch.setattr(module, "write_launch_agent", arrival)
    with pytest.raises(RuntimeError, match="rollback incomplete"):
        invoke(setup, Launchctl())
    assert len(arrivals) == 1
    path, content = arrivals[0]
    assert path.read_bytes() == content


@pytest.mark.parametrize("label", [OLD, "com.brainlayer.enrichment"])
def test_rollback_refuses_unjournaled_legacy_reload(setup, label):
    class Arrival(Launchctl):
        def __call__(self, argv):
            result = super().__call__(argv)
            if argv[1] == "bootstrap" and result.returncode:
                self.loaded.add(label)
            return result

    runner = Arrival(fail="com.brainlayer.maintenance")
    with pytest.raises(RuntimeError, match="rollback incomplete.*legacy service"):
        invoke(setup, runner)
    assert label in runner.loaded
    assert not ({NEW, "com.brainlayer.maintenance"} & runner.loaded)


def test_keg_restart_failed_exit_never_bootstraps_while_original_pid_lives(setup):
    from tests.test_migration_runtime import Clock

    class IdleCalendar(Launchctl):
        def __call__(self, argv):
            result = super().__call__(argv)
            if argv[1] == "print" and argv[2].endswith("/com.brainlayer.maintenance") and result.returncode == 0:
                result.stdout = "state = waiting"
            return result

    runner = IdleCalendar()
    invoke(setup, runner)
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="rollback incomplete.*original supervisor still lives"):
        invoke(setup, runner, "keg2", alive=lambda pid: True, clock=Clock())
    assert not any(c[0] == "bootstrap" for c in runner.calls)


def test_successful_rollback_is_retryable_without_overwriting_preimage(setup):
    runner = Launchctl(fail="com.brainlayer.maintenance")
    with pytest.raises(RuntimeError, match="original files/states restored"):
        invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    before = receipt.read_bytes()
    assert json.loads(before)["phase"] == "rolled_back"
    assert invoke(setup, runner)["phase"] == "migrated"
    backups = [p.read_bytes() for p in setup[1].glob(".archive-*/*-migration.json")]
    assert before in backups


def test_replay_recovers_only_declared_direct_publication_staging(setup):
    runner = Launchctl()
    journal = invoke(setup, runner)
    control = setup[0] / ".local/share/brainlayer/supervisor"
    paths = [control / "migration.json", control / "jobs.json", setup[1] / f"{NEW}.plist.loaded-keg"]
    stages = []
    for index, path in enumerate(paths):
        stage = path.parent / (".migration-" + f"{index + 1:032x}")
        os.link(path, stage)
        stages.append(stage)
    runner.calls.clear()
    assert invoke(setup, runner) == journal
    assert all(not path.exists() for path in stages)
    assert all(path.stat().st_nlink == 1 for path in paths)
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def test_replay_uses_live_phase_not_a_newer_named_backup(setup):
    runner = Launchctl()
    journal = invoke(setup, runner)
    archive = Path(journal["archive"])
    fake = journal | {"phase": "migrating", "keg": "foreign"}
    path = archive / ("29991231T235959Z-" + "9" * 20 + "-" + "a" * 32 + "-migration.json")
    path.write_text(json.dumps(fake))
    path.chmod(0o600)
    runner.calls.clear()
    assert invoke(setup, runner) == journal
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def test_post_bootout_second_observation_blocks_a_late_writer(setup):
    from tests.test_migration_runtime import Clock

    class Late(Launchctl):
        after_bootout = 0
        stopped_old = False

        def __call__(self, argv):
            result = super().__call__(argv)
            if argv[1] == "bootout" and argv[2].endswith("/" + OLD):
                self.stopped_old = True
            elif self.stopped_old and argv[1] == "print" and argv[2].endswith("/" + OLD):
                self.after_bootout += 1
                if self.after_bootout == 2:
                    result.returncode, result.stdout, result.stderr = 0, "pid = 4343", ""
            return result

    runner = Late()
    with pytest.raises(RuntimeError, match="rollback incomplete.*writer still lives"):
        invoke(setup, runner, alive=lambda pid: pid == 4343, clock=Clock())
    assert not any(call[0] == "bootstrap" and Path(call[2]).stem == NEW for call in runner.calls)


def test_recorded_owned_group_blocks_restart_after_daemon_pid_exit(setup):
    from brainlayer.supervisor_guard import boot_time, start_time
    from tests.test_migration_runtime import Clock

    class FastClock(Clock):
        def sleep(self, delay):
            self.now += 100

    runner = Launchctl()
    invoke(setup, runner)
    control = setup[0] / ".local/share/brainlayer/supervisor"
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"], start_new_session=True)
    try:
        groups = {str(child.pid): {"pgid": child.pid, "start_time": start_time(child.pid)}}
        path = control / "groups.json"
        path.write_text(json.dumps({"boot_time": boot_time(), "groups": groups}))
        path.chmod(0o600)
        runner.calls.clear()
        with pytest.raises(RuntimeError, match="rollback incomplete.*custody"):
            invoke(setup, runner, "keg2", clock=FastClock())
        assert child.poll() is None
        assert not any(call[0] == "bootstrap" for call in runner.calls)
    finally:
        child.terminate()
        child.wait(timeout=5)
