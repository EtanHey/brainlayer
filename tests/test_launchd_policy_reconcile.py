"""Policy upgrades use fake homes and fake launchd; never activate installed jobs."""

import hashlib
import json
import plistlib
import shutil
import signal
import subprocess
import sys

import pytest

from tests.test_consolidation_plan import ROOT
from tests.test_consolidation_plan import setup as legacy_setup  # noqa: F401
from tests.test_launchd_consolidation import Launchctl, invoke
from tests.test_launchd_consolidation_recovery import PROBE, scratch
from tests.test_launchd_revert import revert
from tests.test_launchd_revert import setup as setup

HOT = "com.brainlayer.hotlane-brainbar"


@pytest.fixture
def consolidated(setup, tmp_path, monkeypatch):
    from brainlayer import launchd_consolidation as migration

    path = setup[1] / f"{HOT}.plist"
    path.write_bytes(
        plistlib.dumps(dict(Label=HOT, ProgramArguments=["/usr/bin/true"], KeepAlive=True, LowPriorityIO=True))
    )
    runner = Launchctl()
    runner.loaded.add(HOT)
    journal = invoke(setup, runner)
    templates = tmp_path / "templates"
    shutil.copytree(ROOT / "scripts/launchd", templates)
    # Isolate one policy change; retain today's other policies and original non-policy fields.
    for label, record in journal["components"].items():
        (templates / f"{label}.plist").write_bytes(plistlib.dumps(record["plist"]))
    path = templates / f"{HOT}.plist"
    document = plistlib.loads(path.read_bytes())
    document.update(LowPriorityIO=False, ProgramArguments=["must never execute"], StartInterval=99)
    path.write_bytes(plistlib.dumps(document))
    migrate = migration.migrate
    monkeypatch.setattr(migration, "migrate", lambda **kw: migrate(**(kw | {"templates": templates})))
    runner.calls.clear()
    return setup, runner, journal, templates


def test_keg_policy_upgrade_preserves_every_other_field_and_replay_is_noop(consolidated, capsys):
    setup, runner, before, _ = consolidated
    result = invoke(setup, runner, "keg2")
    expected = json.loads(bytes.fromhex(before["manifest_bytes"]))
    expected[HOT]["plist"]["LowPriorityIO"] = False
    manifest = setup[0] / ".local/share/brainlayer/supervisor/jobs.json"
    assert json.loads(manifest.read_bytes()) == expected
    assert result["manifest_bytes"] == manifest.read_bytes().hex()
    assert result["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    transition = result["policy_reconciliations"][-1]
    assert transition["old_sha256"] == before["manifest_sha256"]
    assert transition["new_sha256"] == result["manifest_sha256"]
    assert transition["diff"] == {HOT: {"LowPriorityIO": {"old": True, "new": False}}}
    assert "hotlane-brainbar: LowPriorityIO true → false" in capsys.readouterr().err
    runner.calls.clear()
    assert invoke(setup, runner, "keg2") == result
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def test_unjournaled_manifest_edit_still_refuses(consolidated):
    setup, runner, _, _ = consolidated
    manifest = setup[0] / ".local/share/brainlayer/supervisor/jobs.json"
    document = json.loads(manifest.read_bytes())
    document[HOT]["plist"]["LowPriorityIO"] = False
    manifest.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="component manifest changed"):
        invoke(setup, runner, "keg2")
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def test_revert_after_policy_reconcile_restores_legacy_bytes(consolidated):
    setup, runner, before, _ = consolidated
    invoke(setup, runner, "keg2")
    assert revert(setup, runner)["phase"] == "reverted"
    for name, raw in before["files"].items():
        assert (setup[1] / name).read_bytes() == bytes.fromhex(raw)


@pytest.mark.parametrize("failure", ["manifest-sync", "completion-sync", "readiness"])
def test_failed_policy_restart_restores_old_manifest_and_receipt(consolidated, monkeypatch, failure):
    from brainlayer import launchd_consolidation as migration

    setup, runner, before, _ = consolidated
    update, failed = migration.update, False

    def fail_sync(path, *args):
        nonlocal failed
        if failure == "completion-sync" and failed and path.name == "jobs.json":
            assert json.loads((path.parent / "migration.json").read_bytes())["phase"] == "migrating"
        update(path, *args)
        target = (
            path.name == "jobs.json"
            if failure == "manifest-sync"
            else (path.name == "migration.json" and json.loads(args[1])["phase"] == "migrated")
        )
        if target and not failed:
            failed = True
            raise OSError("injected manifest directory sync failure")

    def fail_verify():
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected readiness failure")

    if failure != "readiness":
        monkeypatch.setattr(migration, "update", fail_sync)
    with pytest.raises(RuntimeError, match="restored"):
        invoke(setup, runner, "keg2", verify=fail_verify if failure == "readiness" else lambda: None)
    control = setup[0] / ".local/share/brainlayer/supervisor"
    assert (control / "jobs.json").read_bytes() == bytes.fromhex(before["manifest_bytes"])
    assert json.loads((control / "migration.json").read_bytes()) == before


POLICY_PROBE = 'policy_phase = ""\n' + PROBE.replace(
    'templates=Path(sys.argv[4]) / "scripts/launchd"', 'templates=root / "templates"'
).replace(" + name)", ' + name + (":policy:" + policy_phase if policy_phase else ""))').replace(
    "result = migrate(",
    """
import brainlayer.launchd_consolidation as migration
original_update = migration.update
def tracked_update(path, *args):
    global policy_phase
    if path.name not in {"jobs.json", "migration.json"}:
        return original_update(path, *args)
    policy_phase = ("manifest" if path.name == "jobs.json" else
                    "intent" if json.loads(args[1])["phase"] == "migrating" else "complete")
    boundary("before:policy:" + policy_phase)
    original_update(path, *args)
    boundary("after:policy:" + policy_phase)
    policy_phase = ""
migration.update = tracked_update
result = migrate(""",
)


def test_sigkill_policy_journal_staging_rename_and_fsync_boundaries_recover(tmp_path):
    def run(root, keg, cut=0):
        return subprocess.run(
            [sys.executable, "-c", POLICY_PROBE, str(root), keg, str(cut), str(ROOT)],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def seed(root):
        scratch(root)
        shutil.copytree(ROOT / "scripts/launchd", root / "templates")
        agents = root / "home/Library/LaunchAgents"
        (agents / f"{HOT}.plist").write_bytes(
            plistlib.dumps(dict(Label=HOT, ProgramArguments=["/usr/bin/true"], KeepAlive=True, LowPriorityIO=True))
        )
        result = run(root, "keg1")
        assert result.returncode == 0, result.stderr

    baseline = tmp_path / "baseline"
    seed(baseline)
    result = run(baseline, "keg2")
    assert result.returncode == 0, result.stderr
    events = json.loads(result.stdout.splitlines()[-1])["events"]
    cuts = [
        (i, e)
        for i, e in enumerate(events, 1)
        if ":policy:" in e
        and ("os.fsync" in e or "os.replace" in e or e.startswith(("before:policy:", "after:policy:")))
    ]
    assert {e.rsplit(":", 1)[-1] for _, e in cuts} == {"intent", "manifest", "complete"}
    for phase in ("intent", "manifest", "complete"):
        for operation in ("os.fsync", "os.replace"):
            assert any(operation in event and event.endswith(":" + phase) for _, event in cuts)
    for cut, event in cuts:
        root = tmp_path / str(cut)
        seed(root)
        killed = run(root, "keg2", cut)
        assert killed.returncode == -signal.SIGKILL, (cut, event, killed.stderr)
        recovered = run(root, "keg2")
        assert recovered.returncode == 0, (cut, event, recovered.stderr)
        control = root / "home/.local/share/brainlayer/supervisor"
        journal = json.loads((control / "migration.json").read_bytes())
        raw = (control / "jobs.json").read_bytes()
        assert journal["phase"] == "migrated" and journal["keg"] == "keg2"
        assert raw.hex() == journal["manifest_bytes"]
        assert hashlib.sha256(raw).hexdigest() == journal["manifest_sha256"]
        assert json.loads(raw)[HOT]["plist"]["LowPriorityIO"] is False
        assert not json.loads((root / "launchd.json").read_bytes())["violations"]
    print(f"POLICY_CRASH_CUTS={len(cuts)}")
