"""Post-success revert uses scratch HOME and an explicitly fake GUI domain."""

import importlib
import json
import plistlib
from pathlib import Path

import pytest

from tests.test_consolidation_plan import DOMAIN, NEW, OLD
from tests.test_consolidation_plan import setup as legacy_setup  # noqa: F401
from tests.test_launchd_consolidation import Launchctl, invoke

FLEET = "com.etanhey.brainlayer-fleet-watchdog"
CALENDAR = "com.brainlayer.maintenance"


@pytest.fixture
def setup(legacy_setup):  # noqa: F811 -- pytest resolves the imported fixture by name.
    # /bin/true is absent on macOS; keep the archived executable real on both CI OSes.
    for path in legacy_setup[1].glob("*.plist"):
        document = plistlib.loads(path.read_bytes())
        document["ProgramArguments"] = ["/usr/bin/true"]
        path.write_bytes(plistlib.dumps(document))
    return legacy_setup


def revert(fixture, runner, **options):
    home = fixture[0]
    return importlib.import_module("brainlayer.launchd_consolidation").revert(
        home,
        home / ".local/share/brainlayer/supervisor/migration.json",
        DOMAIN,
        runner,
        **({"alive": lambda pid: False} | options),
    )


def test_revert_restores_original_bytes_loaded_set_and_retains_journal_archive(setup):
    runner = Launchctl()
    before = {p.name: p.read_bytes() for p in setup[1].iterdir()}
    migrated = invoke(setup, runner)
    runner.calls.clear()
    result = revert(setup, runner)
    assert result["phase"] == "reverted"
    assert runner.loaded == {OLD, FLEET}
    assert {p.name: p.read_bytes() for p in setup[1].iterdir() if p.is_file()} == before
    assert all((Path(migrated["archive"]) / name).read_bytes() == raw for name, raw in before.items())
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    assert json.loads(receipt.read_bytes())["phase"] == "reverted"
    first_bootstrap = next(i for i, c in enumerate(runner.calls) if c[0] == "bootstrap")
    assert all(runner.calls.index(("bootout", f"{DOMAIN}/{label}")) < first_bootstrap for label in (NEW, CALENDAR))
    runner.calls.clear()
    assert revert(setup, runner) == result
    assert not any(c[0] in {"bootstrap", "bootout", "enable", "disable"} for c in runner.calls)


def test_revert_preserves_enabled_disabled_and_absent_overrides(setup):
    class TriState(Launchctl):
        def __call__(self, argv):
            result = super().__call__(argv)
            if argv[1] == "print-disabled":
                result.stdout = result.stdout.replace("\n}", f'\n"{OLD}" => false\n}}')
            return result

    runner = TriState(disabled={"com.brainlayer.observability"})
    invoke(setup, runner)
    result = revert(setup, runner)
    assert [result["states"][label]["disabled_override"] for label in (OLD, "com.brainlayer.observability", FLEET)] == [
        "enabled",
        "disabled",
        None,
    ]
    assert not any(c[0] in {"enable", "disable"} for c in runner.calls)


@pytest.mark.parametrize(
    "kind",
    [
        "launcher",
        "marker",
        "manifest",
        "archive",
        "destination",
        "override",
        "legacy-loaded",
        "hash",
        "phase",
        "missing-journal",
    ],
)
def test_revert_refuses_foreign_custody_before_stopping_current_writers(setup, kind):
    runner = Launchctl()
    journal = invoke(setup, runner)
    home, agents, _ = setup
    receipt = home / ".local/share/brainlayer/supervisor/migration.json"
    paths = dict(
        launcher=agents / f"{NEW}.plist",
        marker=agents / f"{NEW}.plist.loaded-keg",
        manifest=receipt.parent / "jobs.json",
        archive=Path(journal["archive"]) / f"{OLD}.plist",
        destination=agents / f"{OLD}.plist",
    )
    if kind in paths:
        paths[kind].write_bytes(b"foreign personal data")
    elif kind == "override":
        runner.disabled.add(OLD)
    elif kind == "legacy-loaded":
        runner.loaded.add(OLD)
    elif kind in {"hash", "phase"}:
        if kind == "hash":
            journal["published_sha256"][f"{NEW}.plist"] = "0" * 64
        else:
            journal["phase"] = "unknown"
        receipt.write_text(json.dumps(journal))
    else:
        receipt.unlink()
    runner.calls.clear()
    with pytest.raises((RuntimeError, ValueError, OSError)):
        revert(setup, runner)
    assert {NEW, CALENDAR}.issubset(runner.loaded)
    assert not any(c[0] in {"bootstrap", "bootout", "enable", "disable"} for c in runner.calls)
    if kind in paths:
        assert paths[kind].read_bytes() == b"foreign personal data"


def test_revert_waits_for_new_writer_exit_before_restoring_or_bootstrapping(setup):
    from tests.test_migration_runtime import Clock

    runner = Launchctl()
    invoke(setup, runner)
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="did not exit"):
        revert(setup, runner, alive=lambda pid: True, clock=Clock())
    assert not (setup[1] / f"{OLD}.plist").exists()
    assert not any(c[0] == "bootstrap" for c in runner.calls)


def test_interrupted_bootstrap_is_retryable_without_duplicate_writer(setup):
    class Interrupted(Launchctl):
        interrupt = False

        def __call__(self, argv):
            result = super().__call__(argv)
            if self.interrupt and argv[1] == "bootstrap":
                self.interrupt = False
                raise KeyboardInterrupt("after bootstrap")
            return result

    runner = Interrupted()
    invoke(setup, runner)
    runner.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        revert(setup, runner)
    assert revert(setup, runner)["phase"] == "reverted"
    assert runner.loaded == {OLD, FLEET}


def test_missing_legacy_keg_refuses_while_current_consolidation_stays_loaded(setup):
    missing = setup[0] / "Cellar/brainlayer/old/libexec/venv/bin/python"
    path = setup[1] / f"{OLD}.plist"
    document = plistlib.loads(path.read_bytes())
    document["ProgramArguments"] = [str(missing), "-m", "brainlayer", "watch"]
    path.write_bytes(plistlib.dumps(document))
    runner = Launchctl()
    invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    before = receipt.read_bytes()
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="brew cleanup.*restore the legacy keg"):
        revert(setup, runner)
    assert receipt.read_bytes() == before and runner.loaded == {NEW, CALENDAR}
    assert not any(c[0] in {"bootout", "bootstrap"} for c in runner.calls)


def test_disabled_unloaded_preimage_is_restored_without_bootstrap(setup):
    runner = Launchctl(disabled={OLD, CALENDAR})
    runner.loaded.remove(OLD)
    before = {p.name: p.read_bytes() for p in setup[1].iterdir()}
    invoke(setup, runner)
    revert(setup, runner)
    assert runner.loaded == {FLEET}
    assert {OLD, CALENDAR}.issubset(runner.disabled)
    assert {p.name: p.read_bytes() for p in setup[1].iterdir() if p.is_file()} == before
    assert not any(c[0] in {"enable", "disable"} for c in runner.calls)


def test_partial_bootstrap_failure_replays_only_the_missing_original_label(setup):
    runner = Launchctl()
    invoke(setup, runner)
    runner.fail = FLEET
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        revert(setup, runner)
    assert runner.loaded == {OLD}
    runner.calls.clear()
    assert revert(setup, runner)["phase"] == "reverted"
    assert runner.loaded == {OLD, FLEET}
    assert [Path(c[2]).stem for c in runner.calls if c[0] == "bootstrap"] == [FLEET]


def test_same_byte_concurrent_restore_is_recorded_as_foreign_and_never_retried(setup, monkeypatch):
    from brainlayer import launchd_revert as module

    runner = Launchctl()
    invoke(setup, runner)
    original = module._write_exclusive

    def race(path, raw, *args, **kwargs):
        path.write_bytes(raw)
        return original(path, raw, *args, **kwargs)

    monkeypatch.setattr(module, "_write_exclusive", race)
    with pytest.raises(RuntimeError, match="concurrent legacy publication"):
        revert(setup, runner)
    monkeypatch.setattr(module, "_write_exclusive", original)
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="reconciliation"):
        revert(setup, runner)
    assert not any(c[0] == "bootstrap" for c in runner.calls)


def test_shared_migration_lock_refuses_revert_before_launchctl(setup):
    from brainlayer.migration_custody import migration_lock

    runner = Launchctl()
    invoke(setup, runner)
    runner.calls.clear()
    with migration_lock(setup[0]), pytest.raises(RuntimeError, match="migration busy"):
        revert(setup, runner)
    assert not runner.calls


def test_calendar_custody_blocks_legacy_bootstrap_even_after_launcher_bootout(setup):
    from brainlayer.supervisor_guard import Custody
    from tests.test_migration_runtime import Clock

    runner = Launchctl()
    invoke(setup, runner)
    runner.calls.clear()
    calendar = setup[0] / ".local/share/brainlayer/supervisor/calendar"
    with Custody(calendar):
        with pytest.raises(RuntimeError, match="custody did not exit"):
            revert(setup, runner, clock=Clock())
    assert not any(c[0] == "bootstrap" for c in runner.calls)
    assert revert(setup, runner)["phase"] == "reverted"


@pytest.mark.parametrize("linked", ["journal", "archive", "launcher"])
def test_linked_custody_never_unlinks_or_boots_out(setup, linked):
    runner = Launchctl()
    result = invoke(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    source = {"journal": receipt, "archive": Path(result["archive"]), "launcher": setup[1] / f"{NEW}.plist"}[linked]
    retained = source.with_name(source.name + ".retained")
    source.rename(retained)
    source.symlink_to(retained, target_is_directory=retained.is_dir())
    runner.calls.clear()
    with pytest.raises((RuntimeError, ValueError, OSError)):
        revert(setup, runner)
    assert source.is_symlink() and retained.exists()
    assert runner.loaded == {NEW, CALENDAR}
    assert not any(c[0] in {"bootout", "bootstrap"} for c in runner.calls)


def test_rotated_new_pid_is_durable_before_bootout_and_waited_after_crash(setup, monkeypatch):
    from brainlayer import launchd_revert as module
    from tests.test_migration_runtime import Clock

    class Rotated(Launchctl):
        rotating = False
        interrupt = False

        def __call__(self, argv):
            result = super().__call__(argv)
            if self.rotating and argv[1] == "print" and argv[2].endswith("/" + NEW) and not result.returncode:
                result.stdout = "state = running\npid = 7777"
            if self.interrupt and argv[1] == "bootout" and argv[2].endswith("/" + NEW):
                self.interrupt = False
                raise KeyboardInterrupt("after rotated daemon bootout")
            return result

    runner = Rotated()
    invoke(setup, runner)
    original = module.archive_replace

    def rotation(*args, **kwargs):
        result = original(*args, **kwargs)
        runner.rotating = True
        return result

    monkeypatch.setattr(module, "archive_replace", rotation)
    runner.interrupt = True
    with pytest.raises(KeyboardInterrupt):
        revert(setup, runner)
    runner.calls.clear()
    with pytest.raises(RuntimeError, match="did not exit"):
        revert(setup, runner, alive=lambda pid: pid == 7777, clock=Clock())
    assert not any(c[0] == "bootstrap" for c in runner.calls)
    assert revert(setup, runner)["phase"] == "reverted"


def test_recovery_recognizes_hardlinked_legacy_publication_stage(setup, monkeypatch):
    import os

    from brainlayer import launchd_revert as module

    runner = Launchctl()
    invoke(setup, runner)
    original = module._write_exclusive

    def cut(path, raw, *args, **kwargs):
        stage = path.parent / (".migration-" + "a" * 32)
        stage.write_bytes(raw)
        stage.chmod(0o644)
        os.link(stage, path)
        raise KeyboardInterrupt("after legacy publication link")

    monkeypatch.setattr(module, "_write_exclusive", cut)
    with pytest.raises(KeyboardInterrupt):
        revert(setup, runner)
    monkeypatch.setattr(module, "_write_exclusive", original)
    assert revert(setup, runner)["phase"] == "reverted"
    assert not list(setup[1].glob(".migration-*"))


def test_reverted_preimage_can_be_consolidated_again_with_old_archive_retained(setup):
    runner = Launchctl()
    migrated = invoke(setup, runner)
    revert(setup, runner)
    old_archive = Path(migrated["archive"])
    original_archive = {p.name: p.read_bytes() for p in old_archive.iterdir() if p.is_file()}
    next_migration = invoke(setup, runner)
    assert next_migration["phase"] == "migrated" and runner.loaded == {NEW, CALENDAR}
    assert next_migration["archive"] != str(old_archive)
    assert {p.name: p.read_bytes() for p in old_archive.iterdir() if p.is_file()} == original_archive
