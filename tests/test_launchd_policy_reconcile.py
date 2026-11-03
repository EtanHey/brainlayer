"""Policy upgrades use fake homes and fake launchd; never activate installed jobs."""

import hashlib
import json
import plistlib
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.test_consolidation_plan import ROOT
from tests.test_consolidation_plan import setup as legacy_setup  # noqa: F401
from tests.test_launchd_consolidation import Launchctl, invoke
from tests.test_launchd_consolidation_recovery import PROBE, scratch
from tests.test_launchd_revert import revert
from tests.test_launchd_revert import setup as setup

HOT = "com.brainlayer.hotlane-brainbar"
WAL = "com.brainlayer.wal-checkpoint-fallback"


def plan_on_platform(platform):
    from brainlayer.policy_reconcile import plan

    def run(previous, templates):
        with patch.object(sys, "platform", platform):
            return plan(previous, templates)

    return run


def test_maintenance_template_reconciles_process_type_to_standard(tmp_path):
    label = "com.brainlayer.maintenance"
    template = ROOT / "scripts/launchd" / f"{label}.plist"
    requested = plistlib.loads(template.read_bytes())
    assert requested["ProcessType"] == "Standard"
    assert requested["Nice"] == 10 and requested["LowPriorityIO"] is True

    previous_plist = requested | {"ProcessType": "Background"}
    old = json.dumps({label: {"enabled": True, "plist": previous_plist}}).encode()
    previous = {"manifest_bytes": old.hex(), "manifest_sha256": hashlib.sha256(old).hexdigest()}
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / f"{label}.plist").write_bytes(template.read_bytes())

    result = plan_on_platform("darwin")(previous, templates)

    assert result["diff"] == {label: {"ProcessType": {"old": "Background", "new": "Standard"}}}


@pytest.fixture
def consolidated(setup, tmp_path, monkeypatch, request):
    from brainlayer import launchd_consolidation as migration

    # These fixtures exercise launchd policy reconciliation on every CI host.
    monkeypatch.setattr(migration, "policy_plan", plan_on_platform("darwin"))
    path = setup[1] / f"{HOT}.plist"
    path.write_bytes(
        plistlib.dumps(dict(Label=HOT, ProgramArguments=["/usr/bin/true"], KeepAlive=True, LowPriorityIO=True))
    )
    runner = Launchctl()
    runner.loaded.add(HOT)
    if getattr(request, "param", None) == WAL:
        (setup[1] / f"{WAL}.plist").write_bytes(
            plistlib.dumps(
                dict(
                    Label=WAL,
                    ProgramArguments=["/usr/bin/true"],
                    KeepAlive=True,
                    Nice=17,
                    LowPriorityIO=True,
                    ProcessType="Background",
                    SoftResourceLimits={"NumberOfFiles": 1024},
                )
            )
        )
        runner.loaded.add(WAL)
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


@pytest.mark.parametrize("consolidated", [WAL], indirect=True)
def test_missing_template_keeps_pinned_policy_and_other_components_reconcile(consolidated, capsys):
    setup, runner, before, templates = consolidated
    (templates / f"{WAL}.plist").unlink()
    result = invoke(setup, runner, "keg2")
    manifest = json.loads(bytes.fromhex(result["manifest_bytes"]))
    assert result["phase"] == "migrated" and result["keg"] == "keg2"
    assert manifest[WAL] == before["components"][WAL]
    assert manifest[HOT]["plist"]["LowPriorityIO"] is False
    assert WAL not in result["policy_reconciliations"][-1]["diff"]
    assert f"{WAL}: policy unchanged: no keg template" in capsys.readouterr().err


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_template_identity_mismatch_refuses_before_restart_or_journal_change(consolidated, monkeypatch, platform):
    from brainlayer import launchd_consolidation as migration

    monkeypatch.setattr(migration, "policy_plan", plan_on_platform(platform))
    setup, runner, before, templates = consolidated
    path = templates / f"{HOT}.plist"
    document = plistlib.loads(path.read_bytes()) | {"Label": "foreign"}
    path.write_bytes(plistlib.dumps(document))
    with pytest.raises(ValueError, match="template identity changed"):
        invoke(setup, runner, "keg2")
    control = setup[0] / ".local/share/brainlayer/supervisor"
    assert json.loads((control / "migration.json").read_bytes()) == before
    assert (control / "jobs.json").read_bytes().hex() == before["manifest_bytes"]
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def test_linux_restart_preserves_pinned_policy_and_private_children_accept_it(setup, monkeypatch, capsys):
    from brainlayer import launchd_consolidation as migration
    from brainlayer import supervisor_child
    from brainlayer.supervisor_jobs import resource_policy

    runner = Launchctl()
    before = invoke(setup, runner)
    monkeypatch.setattr(migration, "policy_plan", plan_on_platform("linux"))
    result = invoke(setup, runner, "keg2")
    manifest = json.loads(bytes.fromhex(result["manifest_bytes"]))
    # Execute the same policy application that gates real supervisor children;
    # isolate it because Nice and limits must not change the pytest process.
    code = "import json,runpy,sys; apply=runpy.run_path(sys.argv[1])['apply_policy']; sys.platform='linux'; apply(json.loads(sys.argv[2]))"
    for record in manifest.values():
        child = subprocess.run(
            (sys.executable, "-c", code, supervisor_child.__file__, json.dumps(resource_policy(record["plist"]))),
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert child.returncode == 0, child.stderr
    assert result["phase"] == "migrated" and result["keg"] == "keg2"
    assert result["manifest_bytes"] == before["manifest_bytes"]
    assert "policy_reconciliations" not in result
    messages = capsys.readouterr().err
    for label in manifest:
        assert f"{label}: policy unchanged: non-Darwin platform linux" in messages


def test_unjournaled_manifest_edit_still_refuses(consolidated):
    setup, runner, _, _ = consolidated
    manifest = setup[0] / ".local/share/brainlayer/supervisor/jobs.json"
    document = json.loads(manifest.read_bytes())
    document[HOT]["plist"]["LowPriorityIO"] = False
    manifest.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match="component manifest changed"):
        invoke(setup, runner, "keg2")
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def _install_old_maintenance_policy(setup, runner):

    result = invoke(setup, runner)
    home, agents, _ = setup
    control = home / ".local/share/brainlayer/supervisor"
    journal_path = control / "migration.json"
    journal = json.loads(journal_path.read_bytes())
    label = "com.brainlayer.maintenance"
    name = f"{label}.plist"
    old = plistlib.loads(bytes.fromhex(journal["published_files"][name]))
    old["ProcessType"] = "Background"
    raw = plistlib.dumps(old, sort_keys=False)
    (agents / name).write_bytes(raw)
    journal["published_files"][name] = raw.hex()
    journal["published_sha256"][name] = hashlib.sha256(raw).hexdigest()
    journal["configuration_sha256"][label] = hashlib.sha256(raw).hexdigest()
    journal_path.write_text(json.dumps(journal))
    runner.calls.clear()
    assert result["phase"] == "migrated"
    return journal


def test_same_keg_reconciles_launcher_policy_only_and_is_idempotent(setup, capsys):
    from brainlayer.launchd_consolidation import NEW

    runner = Launchctl()
    before = _install_old_maintenance_policy(setup, runner)
    result = invoke(setup, runner)
    home, agents, _ = setup
    label = "com.brainlayer.maintenance"
    name = f"{label}.plist"
    updated = plistlib.loads((agents / name).read_bytes())
    assert updated["ProcessType"] == "Standard"
    changed_actions = [
        (
            call[0],
            call[1].rsplit("/", 1)[-1] if call[0] == "bootout" else plistlib.loads(Path(call[2]).read_bytes())["Label"],
        )
        for call in runner.calls
        if call[0] in {"bootout", "bootstrap"}
    ]
    assert changed_actions == [("bootout", label), ("bootstrap", label)]
    assert runner.loaded == {*NEW, label}
    assert result["phase"] == "migrated" and result["keg"] == "keg1"
    assert result["configuration_sha256"][label] == hashlib.sha256((agents / name).read_bytes()).hexdigest()
    assert result["published_sha256"][name] == result["configuration_sha256"][label]
    assert result["launcher_policy_reconciliations"][-1]["diff"] == {
        label: {"ProcessType": {"old": "Background", "new": "Standard"}}
    }
    assert f"{label}: ProcessType Background → Standard" in capsys.readouterr().err
    runner.calls.clear()
    assert invoke(setup, runner) == result
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)
    assert before["keg"] == result["keg"]


def test_keg_upgrade_applies_launcher_policy_then_restarts_daemon(setup, capsys):
    runner = Launchctl()
    _install_old_maintenance_policy(setup, runner)

    result = invoke(setup, runner, "keg2")

    home, agents, _ = setup
    label = "com.brainlayer.maintenance"
    launcher = agents / f"{label}.plist"
    assert plistlib.loads(launcher.read_bytes())["ProcessType"] == "Standard"
    assert result["phase"] == "migrated" and result["keg"] == "keg2"
    assert result["configuration_sha256"][label] == hashlib.sha256(launcher.read_bytes()).hexdigest()
    assert result["published_sha256"][launcher.name] == result["configuration_sha256"][label]
    assert result["launcher_policy_reconciliations"][-1]["diff"] == {
        label: {"ProcessType": {"old": "Background", "new": "Standard"}}
    }
    assert f"{label}: ProcessType Background → Standard" in capsys.readouterr().err
    assert (agents / "com.brainlayer.daemon.plist.loaded-keg").read_bytes() == b"keg2\n"


def test_live_calendar_refuses_launcher_policy_without_changes(setup):
    runner = Launchctl()
    _install_old_maintenance_policy(setup, runner)
    home, agents, _ = setup
    control = home / ".local/share/brainlayer/supervisor"
    receipt = control / "migration.json"
    before = receipt.read_bytes()
    launcher = agents / "com.brainlayer.maintenance.plist"
    launcher_bytes = launcher.read_bytes()
    runner.calls.clear()

    with pytest.raises(RuntimeError, match="busy, retry later"):
        invoke(setup, runner, alive=lambda _pid: True)

    assert receipt.read_bytes() == before
    assert launcher.read_bytes() == launcher_bytes
    assert runner.loaded == {"com.brainlayer.daemon", "com.brainlayer.maintenance"}
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


def test_non_policy_launcher_change_still_requires_explicit_replan(setup, monkeypatch):
    from brainlayer import launchd_consolidation as migration

    runner = Launchctl()
    _install_old_maintenance_policy(setup, runner)
    receipt = setup[0] / ".local/share/brainlayer/supervisor/migration.json"
    original = receipt.read_bytes()
    render = migration.rendered_launchers

    def changed(*args, **kwargs):
        prepared = render(*args, **kwargs)
        label = "com.brainlayer.maintenance"
        document = plistlib.loads(prepared[label])
        document["ProgramArguments"] = ["/foreign/command"]
        return prepared | {label: plistlib.dumps(document, sort_keys=False)}

    monkeypatch.setattr(migration, "rendered_launchers", changed)
    with pytest.raises(RuntimeError, match="explicit replan required"):
        invoke(setup, runner)
    assert receipt.read_bytes() == original
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


@pytest.mark.parametrize(
    "lock_relative",
    [
        ".local/share/brainlayer/supervisor/maintenance.lock",
        ".local/share/brainlayer/.maintenance.lock",
        ".local/share/brainlayer/backups/.backup.lock",
    ],
)
def test_busy_maintenance_refuses_launcher_restart_without_changes(setup, lock_relative):
    import fcntl

    runner = Launchctl()
    _install_old_maintenance_policy(setup, runner)
    home, agents, _ = setup
    control = home / ".local/share/brainlayer/supervisor"
    lock_path = home / lock_relative
    lock_path.parent.mkdir(mode=0o700, exist_ok=True)
    lock_path.touch(mode=0o600)
    receipt = control / "migration.json"
    before = receipt.read_bytes()
    launcher = agents / "com.brainlayer.maintenance.plist"
    launcher_bytes = launcher.read_bytes()
    runner.calls.clear()
    with lock_path.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match="busy; retry later"):
            invoke(setup, runner)
    assert receipt.read_bytes() == before
    assert launcher.read_bytes() == launcher_bytes
    assert runner.loaded == {"com.brainlayer.daemon", "com.brainlayer.maintenance"}
    assert all(call[0] in {"print", "print-disabled"} for call in runner.calls)


@pytest.mark.parametrize("cut", ["bootout", "publish", "bootstrap"])
def test_launcher_policy_interruption_recovers_forward(setup, monkeypatch, cut):
    from brainlayer import launchd_consolidation as migration

    class Interrupted(Launchctl):
        fired = False
        armed = False

        def __call__(self, argv):
            result = super().__call__(argv)
            action = argv[1]
            if (
                self.armed
                and not self.fired
                and action == cut
                and (action != "bootstrap" or plistlib.loads(Path(argv[3]).read_bytes())["Label"])
            ):
                self.fired = True
                raise KeyboardInterrupt("injected launcher policy crash")
            return result

    runner = Interrupted()
    _install_old_maintenance_policy(setup, runner)
    runner.armed = True
    if cut == "publish":
        publish = migration.update
        fired = False

        def interrupt_publish(path, *args):
            nonlocal fired
            result = publish(path, *args)
            if not fired and path.name == "com.brainlayer.maintenance.plist":
                fired = True
                raise KeyboardInterrupt("injected launcher policy crash")
            return result

        monkeypatch.setattr(migration, "update", interrupt_publish)
    with pytest.raises(KeyboardInterrupt, match="injected launcher policy crash"):
        invoke(setup, runner)
    monkeypatch.undo()
    result = invoke(setup, runner)
    home, agents, _ = setup
    launcher = agents / "com.brainlayer.maintenance.plist"
    assert plistlib.loads(launcher.read_bytes())["ProcessType"] == "Standard"
    assert result["phase"] == "migrated"
    assert result["published_sha256"][launcher.name] == hashlib.sha256(launcher.read_bytes()).hexdigest()
    assert runner.loaded == {"com.brainlayer.daemon", "com.brainlayer.maintenance"}


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


POLICY_PROBE = 'from unittest.mock import patch\npolicy_phase = ""\n' + PROBE.replace(
    'templates=Path(sys.argv[4]) / "scripts/launchd"', 'templates=root / "templates"'
).replace(" + name)", ' + name + (":policy:" + policy_phase if policy_phase else ""))').replace(
    "result = migrate(",
    """
import brainlayer.launchd_consolidation as migration
original_plan = migration.policy_plan
def darwin_plan(previous, templates):
    with patch.object(sys, "platform", "darwin"):
        return original_plan(previous, templates)
migration.policy_plan = darwin_plan
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
