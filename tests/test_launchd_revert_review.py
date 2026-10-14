"""Legacy runtime facts may change after migration or a successful revert."""

import plistlib

import pytest

from tests.test_consolidation_plan import NEW, OLD
from tests.test_launchd_consolidation import Launchctl, invoke
from tests.test_launchd_revert import CALENDAR, FLEET, revert
from tests.test_launchd_revert import legacy_setup as legacy_setup
from tests.test_launchd_revert import setup as setup
from tests.test_migration_runtime import Clock


class Rotated(Launchctl):
    legacy_pid = 4242

    def __call__(self, argv):
        result = super().__call__(argv)
        if argv[1] == "print" and not result.returncode:
            label = argv[2].rsplit("/", 1)[-1]
            pid = 8888 if label in (NEW, CALENDAR) else self.legacy_pid
            result.stdout = "state = running" + ("\npid = " + str(pid) if pid is not None else "")
        return result


@pytest.mark.parametrize("phase", ["reverted", "reverting"])
@pytest.mark.parametrize("pid", [9999, None], ids=["respawned", "loaded-idle"])
def test_recorded_bootstrap_intent_survives_ordinary_legacy_pid_changes(setup, phase, pid):
    runner = Rotated()
    invoke(setup, runner)
    if phase == "reverting":
        runner.fail = FLEET
        with pytest.raises(RuntimeError, match="bootstrap failed"):
            revert(setup, runner)
    else:
        revert(setup, runner)
    runner.legacy_pid = pid
    runner.calls.clear()
    assert revert(setup, runner)["phase"] == "reverted"
    assert runner.loaded == {OLD, FLEET}
    bootstraps = [call for call in runner.calls if call[0] == "bootstrap"]
    assert len(bootstraps) == (1 if phase == "reverting" else 0)


@pytest.mark.parametrize("phase", ["reverted", "reverting"])
def test_already_restored_loaded_legacy_needs_no_deleted_executable(setup, phase):
    python = setup[0] / "Cellar/brainlayer/old/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    old = setup[1] / (OLD + ".plist")
    definition = plistlib.loads(old.read_bytes())
    definition["ProgramArguments"] = [str(python)]
    old.write_bytes(plistlib.dumps(definition))
    runner = Rotated()
    invoke(setup, runner)
    if phase == "reverting":
        runner.fail = FLEET
        with pytest.raises(RuntimeError):
            revert(setup, runner)
    else:
        revert(setup, runner)
    python.unlink()
    assert revert(setup, runner)["phase"] == "reverted"


@pytest.mark.parametrize("permission_error", [False, True])
def test_migration_time_pid_reuse_does_not_block_post_success_revert(setup, permission_error):
    runner = Rotated()
    invoke(setup, runner)
    probes = []

    def alive(pid):
        probes.append(pid)
        if pid == 4242:
            if permission_error:
                raise PermissionError("unrelated process owned by another UID")
            return True
        return False

    assert revert(setup, runner, alive=alive, clock=Clock())["phase"] == "reverted"
    assert 4242 not in probes  # Migration already proved this original generation exited.
    assert runner.loaded == {OLD, FLEET}


@pytest.mark.parametrize(
    "anchor",
    [
        "from brainlayer.launchd_consolidation import migrate",
        'if "com.brainlayer.watch" in state["loaded"] and "com.brainlayer.daemon" in state["loaded"]:',
    ],
)
@pytest.mark.parametrize("duplicated", [False, True])
def test_probe_construction_refuses_drift_of_its_real_import_and_overlap_anchors(anchor, duplicated):
    from tests import test_launchd_revert_crash as probe

    changed = probe.MIGRATE_PROBE + "\n" + anchor if duplicated else probe.MIGRATE_PROBE.replace(anchor, "")
    with pytest.raises(ValueError, match="probe anchor missing or duplicated"):
        probe.revert_probe(changed)
