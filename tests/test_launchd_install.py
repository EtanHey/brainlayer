"""Installer dispatch/preflight uses scratch HOME and gui/424242 only."""

import json
import os
import subprocess
import sys
from contextlib import nullcontext

import pytest

from tests.test_consolidation_plan import DOMAIN, ROOT
from tests.test_consolidation_plan import setup as setup


@pytest.fixture
def inputs(request, monkeypatch):
    home, agents, env = request.getfixturevalue("setup")
    python = home / "opt/brainlayer/libexec/venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/sh\nexit 0\n")
    python.chmod(0o755)
    (python.parents[1] / "lib/python3.13/site-packages/brainlayer").mkdir(parents=True)
    monkeypatch.setenv("BRAINLAYER_HOOK_PYTHON", str(python))
    return dict(
        home=home,
        templates=ROOT / "scripts/launchd",
        env_file=env,
        python=python,
        keg="keg1",
        domain=DOMAIN,
        run=lambda argv, **kw: subprocess.run(argv, **kw),
    )


def test_installer_prepares_named_launchers_and_calendar_helpers_before_migrate(inputs, monkeypatch):
    from brainlayer import launchd_consolidation, launchd_install

    def migrate(**options):
        assert options["domain"] == DOMAIN
        assert options["keg"] == "keg1"
        lib = inputs["home"] / ".local/lib/brainlayer"
        for name in (
            "BrainLayer Daemon",
            "BrainLayer Maintenance",
            "brainlayer-env-run.sh",
            "backup-daily.sh",
            "jsonl-backup.sh",
        ):
            assert (lib / name).stat().st_mode & 0o111
            assert subprocess.run(["/bin/bash", "-n", str(lib / name)]).returncode == 0
        return {"phase": "complete"}

    monkeypatch.setattr(launchd_consolidation, "migrate", migrate)
    assert launchd_install.install(**inputs)["phase"] == "complete"
    lib = inputs["home"] / ".local/lib/brainlayer"
    before = {p.name: p.stat().st_mtime_ns for p in lib.iterdir()}
    launchd_install.install(**inputs)
    assert before == {p.name: p.stat().st_mtime_ns for p in lib.iterdir()}


def test_shared_runner_change_is_refused_before_writing_any_helper(inputs, monkeypatch):
    from brainlayer import launchd_consolidation, launchd_install

    lib = inputs["home"] / ".local/lib/brainlayer"
    lib.mkdir(parents=True)
    runner = lib / "brainlayer-env-run.sh"
    runner.write_bytes(b"foreign runner\n")
    runner.chmod(0o755)
    monkeypatch.setattr(launchd_consolidation, "migrate", lambda **kw: pytest.fail("mutated"))
    with pytest.raises(RuntimeError, match="helper"):
        launchd_install.install(**inputs)
    assert list(lib.iterdir()) == [runner]
    assert runner.read_bytes() == b"foreign runner\n"


@pytest.mark.parametrize(
    "cause", ["supervisor custody identity refused", "launchctl bootstrap failed for gui/424242: 5"]
)
def test_main_reports_keg_restart_cause_after_successful_rollback(inputs, monkeypatch, capsys, cause):
    from brainlayer import launchd_install

    def fail(**options):
        try:
            raise RuntimeError(cause)
        except RuntimeError as failure:
            raise RuntimeError("keg restart failed; markers/journal and supervisor readiness restored") from failure

    monkeypatch.setattr(launchd_install, "_install", fail)
    with pytest.raises(SystemExit) as stopped:
        launchd_install.main(
            [
                "consolidate",
                "--home",
                str(inputs["home"]),
                "--domain",
                DOMAIN,
                "--templates",
                str(inputs["templates"]),
                "--env-file",
                str(inputs["env_file"]),
                "--python",
                str(inputs["python"]),
                "--keg",
                "../Cellar/brainlayer/1.5.58",
            ]
        )
    assert stopped.value.code == 1
    error = capsys.readouterr().err
    assert "markers/journal and supervisor readiness restored" in error
    assert f"Caused by: {cause}" in error
    assert "Traceback" not in error


def test_rollback_reports_nested_causes_once_even_with_a_cycle(inputs, monkeypatch, capsys):
    from brainlayer import launchd_consolidation, launchd_install

    outer = RuntimeError("rollback incomplete")
    middle = RuntimeError("readiness failed")
    root = RuntimeError("control socket unavailable")
    outer.__cause__, middle.__cause__, root.__cause__ = middle, root, outer

    def fail(**options):
        raise outer

    monkeypatch.setattr(launchd_consolidation, "revert", fail)
    with pytest.raises(SystemExit) as stopped:
        launchd_install.main(["consolidation-rollback", "--home", str(inputs["home"]), "--domain", DOMAIN])
    assert stopped.value.code == 1
    assert capsys.readouterr().err.splitlines() == [
        "rollback incomplete",
        "Caused by: readiness failed",
        "Caused by: control socket unavailable",
    ]


def test_rollback_calls_pr_c_with_journal_path_without_forward_preflight(inputs, monkeypatch):
    from brainlayer import launchd_consolidation, launchd_install

    calls = []
    monkeypatch.setattr(launchd_consolidation, "revert", lambda **kw: calls.append(kw) or {"phase": "reverted"})
    launchd_install.main(["consolidation-rollback", "--home", str(inputs["home"]), "--domain", DOMAIN])
    assert calls[0]["home"] == inputs["home"]
    assert calls[0]["journal"] == inputs["home"] / ".local/share/brainlayer/supervisor/migration.json"
    assert calls[0]["domain"] == DOMAIN
    assert not (inputs["home"] / ".local/lib/brainlayer").exists()


@pytest.mark.parametrize("action", ["watch", "all", "load", "unload", "remove", "fleet-watchdog-resume"])
@pytest.mark.parametrize("phase", ["migrating", "migrated", "reverting", "complete", "unknown", "missing"])
def test_legacy_shell_paths_refuse_when_supervisor_journal_exists(tmp_path, action, phase, request):
    home = tmp_path / "home"
    journal = home / ".local/share/brainlayer/supervisor/migration.json"
    if phase == "missing":
        from tests.test_launchd_consolidation import Launchctl, invoke

        fixture = request.getfixturevalue("setup")
        runner = Launchctl()
        migrated = invoke(fixture, runner)
        journal.unlink()
        watch = fixture[1] / "com.brainlayer.watch.plist"
        watch.write_bytes((watch.parent / os.path.basename(migrated["archive"]) / watch.name).read_bytes())
        assert runner.loaded == {"com.brainlayer.daemon", "com.brainlayer.maintenance"}
    else:
        journal.parent.mkdir(mode=0o700, parents=True)
        journal.write_text(json.dumps({"phase": phase}))
        journal.chmod(0o600)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uname").write_text("#!/bin/sh\necho Linux\n")
    (fake / "uname").chmod(0o755)
    calls = tmp_path / "legacy-calls"
    launchctl = fake / "launchctl"
    launchctl.write_text(f'#!/bin/sh\necho "$*" >> "{calls}"\necho FORBIDDEN_LAUNCHCTL >&2\nexit 91\n')
    launchctl.chmod(0o755)
    env = os.environ | {
        "HOME": str(home),
        "BRAINLAYER_BIN": "/usr/bin/true",
        "PATH": str(fake) + ":/usr/bin:/bin",
        "PYTHON_BIN": sys.executable,
    }
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/launchd/install.sh"), action, "watch"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "supervisor" in result.stderr
    assert "FORBIDDEN_LAUNCHCTL" not in result.stderr
    assert not calls.exists()
    if phase == "missing":
        assert "jobs.json" in result.stderr and "manual recovery" in result.stderr
    else:
        assert not (home / "Library/LaunchAgents").exists()


@pytest.mark.parametrize("failed", [False, True, KeyboardInterrupt])
def test_wiring_uses_real_transaction_for_replay_restart_and_failure(inputs, monkeypatch, capsys, failed):
    from brainlayer import launchd_consolidation, launchd_install
    from tests.test_launchd_consolidation import Launchctl

    migrate = launchd_consolidation.migrate
    monkeypatch.setattr(
        launchd_consolidation, "migrate", lambda **kw: migrate(**kw, alive=lambda pid: False, verify=lambda: None)
    )
    for path in (inputs["home"] / "Library/LaunchAgents").glob("*.plist"):
        raw = path.read_bytes().replace(b"/bin/true", b"/usr/bin/true")
        path.write_bytes(raw)
    runner = Launchctl(fail="com.brainlayer.maintenance" if failed else "")
    inputs["run"] = runner
    if failed is KeyboardInterrupt:

        def interrupt(argv):
            result = runner(argv)
            if argv[1] == "bootstrap" and not getattr(runner, "interrupted", False):
                runner.interrupted = True
                raise KeyboardInterrupt
            return result

        inputs["run"] = interrupt
    if failed:
        with pytest.raises(KeyboardInterrupt if failed is KeyboardInterrupt else RuntimeError):
            launchd_install.install(**inputs)
        assert runner.loaded == {"com.brainlayer.watch", "com.etanhey.brainlayer-fleet-watchdog"}
        assert not any(c[0] == "enable" for c in runner.calls)
        runner.calls.clear()
        receipt = inputs["home"] / ".local/share/brainlayer/supervisor/migration.json"
        before = receipt.read_bytes()
        monkeypatch.setattr(launchd_install, "command", runner)
        launchd_install.main(["consolidation-rollback", "--home", str(inputs["home"]), "--domain", DOMAIN])
        assert capsys.readouterr().out.strip() == "nothing to roll back; legacy jobs own the install"
        assert not runner.calls and receipt.read_bytes() == before
        return
    launchd_install.install(**inputs)
    runner.calls.clear()
    launchd_install.install(**inputs)
    assert not any(c[0] in {"bootout", "bootstrap"} for c in runner.calls)
    inputs["keg"] = "keg2"
    launchd_install.install(**inputs)
    assert ("bootout", DOMAIN + "/com.brainlayer.daemon") in runner.calls
    assert runner.loaded == {"com.brainlayer.daemon", "com.brainlayer.maintenance"}
    revert = launchd_consolidation.revert
    monkeypatch.setattr(launchd_consolidation, "revert", lambda **kw: revert(**kw, alive=lambda pid: False))
    monkeypatch.setattr(launchd_install, "command", runner)
    launchd_install.main(["consolidation-rollback", "--home", str(inputs["home"]), "--domain", DOMAIN])
    receipt = inputs["home"] / ".local/share/brainlayer/supervisor/migration.json"
    assert json.loads(receipt.read_text())["phase"] == "reverted"
    assert runner.loaded == {"com.brainlayer.watch", "com.etanhey.brainlayer-fleet-watchdog"}


def test_broken_shell_helper_refused_before_mutation(inputs, monkeypatch, tmp_path):
    import shutil

    from brainlayer import launchd_consolidation, launchd_install

    templates = tmp_path / "templates"
    shutil.copytree(inputs["templates"], templates)
    (templates / "backup-daily.sh").write_bytes(b"if ( broken\n")
    inputs["templates"] = templates
    monkeypatch.setattr(launchd_consolidation, "migrate", lambda **kw: pytest.fail("mutated"))
    with pytest.raises(subprocess.CalledProcessError):
        launchd_install.install(**inputs)
    assert not (inputs["home"] / ".local/lib/brainlayer").exists()


@pytest.mark.parametrize("phase", ["reverted", "rolled_back"])
@pytest.mark.parametrize("busy", [False, True])
def test_legacy_shell_path_reopens_after_journaled_return(tmp_path, phase, busy):
    from brainlayer.migration_custody import migration_lock

    home = tmp_path / "home"
    journal = home / ".local/share/brainlayer/supervisor/migration.json"
    journal.parent.mkdir(mode=0o700, parents=True)
    journal.write_text(json.dumps({"phase": phase}))
    journal.chmod(0o600)
    fake = tmp_path / "bin"
    fake.mkdir()
    launchctl = fake / "launchctl"
    launchctl.write_text(
        '#!/bin/sh\ncase "$2" in gui/424242/*) exit 113;; */home/Library/LaunchAgents/*) exit 1;; *) echo FOREIGN_DOMAIN >&2; exit 91;; esac\n'
    )
    launchctl.chmod(0o755)
    harness = tmp_path / "install.sh"
    harness.write_text((ROOT / "scripts/launchd/install.sh").read_text().replace("gui/$UID", "gui/424242"))
    with migration_lock(home) if busy else nullcontext():
        result = subprocess.run(
            ["/bin/bash", str(harness), "unload", "watch"],
            env=os.environ
            | {
                "HOME": str(home),
                "BRAINLAYER_BIN": "/usr/bin/true",
                "PYTHON_BIN": sys.executable,
                "PATH": str(fake) + ":/usr/bin:/bin",
            },
            capture_output=True,
            text=True,
        )
    if busy:
        assert result.returncode != 0 and "migration busy" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
    assert journal.exists()


@pytest.mark.parametrize("action,gate", [("consolidate", 0), ("consolidation-rollback", 0), ("consolidate", 1)])
def test_shell_dispatch_uses_keg_interpreter_only_after_signatures(tmp_path, action, gate):
    import shutil

    keg = tmp_path / "Cellar/brainlayer/9.9.9"
    scripts = keg / "libexec/venv/lib/python3.13/site-packages/brainlayer/launchd"
    scripts.mkdir(parents=True)
    shutil.copy(ROOT / "scripts/launchd/install.sh", scripts)
    (keg / "bin").mkdir()
    (keg / "libexec/venv/bin").mkdir()
    opt = tmp_path / "opt/brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(keg)
    python = keg / "libexec/venv/bin/python"
    python.write_text(
        f'#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["DISPATCH"]).write_text(json.dumps(sys.argv[1:]))\n'
    )
    binary = keg / "bin/brainlayer"
    binary.write_text("#!/bin/sh\nexit 0\n")
    verify = scripts / "release-verify-signatures.sh"
    verify.write_text(f"#!/bin/sh\nexit {gate}\n")
    for path in (python, binary, verify):
        path.chmod(0o755)
    dispatch = tmp_path / "dispatch.json"
    result = subprocess.run(
        ["/bin/bash", str(scripts / "install.sh"), action],
        env=os.environ
        | {"HOME": str(tmp_path / "home"), "DISPATCH": str(dispatch), "BRAINLAYER_LAUNCHD_DOMAIN": DOMAIN},
        capture_output=True,
        text=True,
    )
    assert result.returncode == gate, result.stderr
    if gate:
        assert not dispatch.exists()
    else:
        args = json.loads(dispatch.read_text())
        assert args[:3] == ["-m", "brainlayer.launchd_install", action]
        assert args[args.index("--domain") + 1] == DOMAIN
        if action == "consolidate":
            assert args[args.index("--python") + 1] == str(opt / "libexec/venv/bin/python")
            assert args[args.index("--keg") + 1] == str(keg)


def test_legacy_install_blocks_concurrent_consolidation_during_preflight(tmp_path):
    import shutil

    home, fake = tmp_path / "home", tmp_path / "bin"
    fake.mkdir()
    templates = tmp_path / "launchd"
    shutil.copytree(ROOT / "scripts/launchd", templates)
    harness = templates / "install.sh"
    harness.write_text(harness.read_text().replace("gui/$UID", "gui/424242"))
    envfile = home / ".config/brainlayer/brainlayer.env"
    envfile.parent.mkdir(parents=True)
    envfile.write_text("BRAINLAYER_LAUNCHD_WATCH_ENABLED=1\n")
    journal = home / ".local/share/brainlayer/supervisor/migration.json"
    contender = tmp_path / "contender.py"
    contender.write_text(
        f"import fcntl,os\nfrom pathlib import Path\nfd=os.open({str(home)!r},os.O_RDONLY|os.O_DIRECTORY)\ntry: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: raise SystemExit('CONCURRENT_MIGRATION_REFUSED')\npath=Path({str(journal)!r}); path.parent.mkdir(mode=0o700,parents=True,exist_ok=True); path.write_text('{{\"phase\":\"migrated\"}}'); path.chmod(0o600)\n"
    )
    (fake / "uname").write_text(f'#!/bin/sh\n/usr/bin/python3 "{contender}"\necho Linux\n')
    (fake / "launchctl").write_text("#!/bin/sh\necho FORBIDDEN_LAUNCHCTL >&2\nexit 91\n")
    for path in fake.iterdir():
        path.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(harness), "watch"],
        env=os.environ
        | {
            "HOME": str(home),
            "PATH": str(fake) + ":/usr/bin:/bin",
            "BRAINLAYER_BIN": "/usr/bin/true",
            "PYTHON_BIN": sys.executable,
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "CONCURRENT_MIGRATION_REFUSED" in result.stderr
    assert not journal.exists()


@pytest.mark.parametrize(
    "archive,blocked",
    [
        (None, False),
        (".archive-2026-10-12", False),
        ("resource-guard-archive", False),
        (".archive-20261014T074500Z-abcdef12", True),
    ],
)
def test_rollback_without_transaction_is_noop(inputs, monkeypatch, capsys, archive, blocked):
    from brainlayer import launchd_install

    if archive:
        folder = inputs["home"] / "Library/LaunchAgents" / archive
        folder.mkdir()
        (folder / "watch.plist.bak").write_bytes(b"operator backup")
    monkeypatch.setattr(launchd_install, "command", lambda argv: pytest.fail("unexpected launchctl"))
    args = ["consolidation-rollback", "--home", str(inputs["home"]), "--domain", DOMAIN]
    with pytest.raises(SystemExit, match="1") if blocked else nullcontext():
        launchd_install.main(args)
    output = capsys.readouterr()
    expected = "nothing to roll back; legacy jobs own the install"
    assert archive in output.err if blocked else output.out.strip() == expected
    assert not (inputs["home"] / ".local/share/brainlayer/supervisor/migration.json").exists()
    if archive:
        assert (folder / "watch.plist.bak").read_bytes() == b"operator backup"


def test_archive_pattern_matches_pr_b_in_bash_and_python():
    pattern = r"\.archive-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}"
    assert f"BRAINLAYER_ARCHIVE_PATTERN='^{pattern}$'" in (ROOT / "scripts/launchd/install.sh").read_text()
    for module, variable in [("launchd_revert", "p"), ("launchd_consolidation", "archive")]:
        assert f're.fullmatch(r"{pattern}", {variable}.name)' in (ROOT / f"src/brainlayer/{module}.py").read_text()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit, RuntimeError, ValueError, OSError])
@pytest.mark.parametrize("action", ["consolidate", "consolidation-rollback"])
@pytest.mark.parametrize("boundary", ["before_lock", "before_handoff", "mutating"])
def test_interrupt_reports_truthful_recovery(inputs, monkeypatch, capsys, interruption, action, boundary):
    from contextlib import contextmanager

    from brainlayer import launchd_consolidation, launchd_install

    @contextmanager
    def lock(home):
        if boundary == "before_lock":
            raise interruption("operator refusal")
        yield

    def interrupted(**kw):
        if boundary == "mutating":
            receipt = inputs["home"] / ".local/share/brainlayer/supervisor/migration.json"
            receipt.parent.mkdir(mode=0o700, parents=True)
            receipt.write_text(json.dumps({"phase": "reverting" if action.endswith("rollback") else "migrating"}))
        raise interruption("operator refusal")

    monkeypatch.setattr(launchd_install, "installer_lock", lock)
    monkeypatch.setattr(launchd_install, "_install", interrupted)
    monkeypatch.setattr(launchd_consolidation, "revert", interrupted)
    args = [action, "--home", str(inputs["home"]), "--domain", DOMAIN]
    if action == "consolidate":
        args += [
            "--templates",
            str(inputs["templates"]),
            "--env-file",
            str(inputs["env_file"]),
            "--python",
            str(inputs["python"]),
            "--keg",
            inputs["keg"],
        ]
    with pytest.raises(SystemExit) as stopped:
        launchd_install.main(args)
    assert stopped.value.code != 0
    out = capsys.readouterr()
    if issubclass(interruption, Exception):
        assert out.err.strip() == "operator refusal"
    elif boundary == "before_lock":
        assert out.err.strip() == f"{action} interrupted before any changes; nothing was stopped; re-run {action}"
    else:
        assert "to finish" in out.err and "jobs are stopped" not in out.err
    assert not out.out and "Traceback" not in out.err


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_command_settles_and_reaps_owned_child(monkeypatch, interruption):
    from brainlayer import launchd_install

    communicate, owned = subprocess.Popen.communicate, []

    def interrupt_once(child, *args, **kw):
        if not owned:
            owned.append(child)
            raise interruption()
        return communicate(child, *args, **kw)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_once)
    with pytest.raises(interruption):
        launchd_install.command([sys.executable, "-c", 'import time; time.sleep(.05); print("settled")'])
    assert owned[0].returncode == 0
    with pytest.raises(ChildProcessError):
        os.waitpid(owned[0].pid, os.WNOHANG)
