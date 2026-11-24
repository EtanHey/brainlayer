"""Exercise the shipped shell with an invented keg and command recorder, never brew."""

import os
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(os.environ.get("BRAINBAR_UPDATE_SCRIPT", Path(__file__).parents[1] / "scripts/brainbar-update.sh"))


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


@pytest.fixture
def install(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    app = tmp_path / "Applications/BrainBar.app"
    app.mkdir(parents=True)
    (app / "identity").write_text("old")
    root = tmp_path / "state"
    keg = tmp_path / "Cellar/brainlayer/2.0.0/lib/brainlayer/launchd"
    keg.mkdir(parents=True)
    log = tmp_path / "commands"

    def command(path, body):
        path.write_text("#!/bin/bash\nset -e\n" + body)
        path.chmod(0o700)

    command(
        commands / "brew",
        """
echo "brew $*" >> "$TEST_LOG"
if [ "$1" = info ]; then
  if [ "$3" = --formula ]; then echo '{"formulae":[{"versions":{"stable":"2.0.0"}}]}'
  elif [ "$FAIL_STEP" = mismatch ]; then echo '{"casks":[{"version":"1.9.9"}]}'
  else echo '{"casks":[{"version":"2.0.0"}]}' ; fi
elif [ "$1" = update ]; then
  if [ "$FAIL_STEP" = late-busy ]; then touch "$TEST_PREFIX/busy"; fi
  exit 0
elif [ "$1" = --prefix ]; then echo "$TEST_PREFIX"
elif [ "$2" = --cask ]; then
  if [ "$1" = upgrade ] && [ -f "$TEST_PREFIX/cask-recorded" ]; then exit 0; fi
  touch "$TEST_PREFIX/cask-recorded"
  mkdir -p "$BRAINBAR_UPDATE_APP"
  echo new > "$BRAINBAR_UPDATE_APP/identity"
  [ "$FAIL_STEP" != cask ]
else [ "$FAIL_STEP" != formula ]; fi
""",
    )
    command(
        commands / "pgrep",
        'if [ "$FAIL_STEP" = busy ] || [ -f "$TEST_PREFIX/busy" ]; then echo "123 backup_daily"; fi; exit 0\n',
    )
    command(
        commands / "plist",
        'if [ -d "$BRAINBAR_UPDATE_APP" ] && [ "$(cat "$BRAINBAR_UPDATE_APP/identity")" = old ]; then echo 1.0.0; elif [ "$FAIL_STEP" = health ]; then echo 1.9.9; else echo 2.0.0; fi\n',
    )
    command(keg / "release-verify-signatures.sh", 'echo signatures >> "$TEST_LOG"; [ "$FAIL_STEP" != signatures ]\n')
    command(keg / "install.sh", 'echo consolidate >> "$TEST_LOG"; [ "$FAIL_STEP" != consolidate ]\n')
    command(
        commands / "brainlayer",
        """
echo activate >> "$TEST_LOG"
[ -d "$BRAINBAR_UPDATE_APP" ] || exit 1
if [ "$FAIL_STEP" = activate ] && [ "$(cat "$BRAINBAR_UPDATE_APP/identity")" != old ]; then exit 1; fi
""",
    )
    env = dict(
        os.environ,
        BRAINBAR_UPDATE_PATH=f"{commands}:/usr/bin:/bin",
        BRAINBAR_UPDATE_APP=str(app),
        BRAINBAR_UPDATE_ROOT=str(root),
        BRAINBAR_UPDATE_PLIST=str(commands / "plist"),
        TEST_LOG=str(log),
        TEST_PREFIX=str(tmp_path),
        FAIL_STEP="",
    )

    # The installer calls the canonical prefix/bin/brainlayer.
    def run(step="", *, jobs=None, operation="test-operation"):
        env["FAIL_STEP"] = step
        if jobs is not None:
            env["BRAINBAR_UPDATE_JOBS"] = str(jobs)
        result = subprocess.run(["bash", str(SCRIPT), "2.0.0", operation], env=env, capture_output=True, text=True)
        return result, log.read_text() if log.exists() else "", root, app

    return run


@pytest.mark.parametrize("step", ["formula", "signatures", "consolidate"])
def test_failure_before_cask_preserves_old_app(install, step):
    result, log, root, app = install(step)
    assert result.returncode != 0, result.stderr
    assert "--cask etanhey/layers/brainbar\n" not in log.replace("--json=v2 --cask", "metadata")
    assert "activate" not in log
    assert (app / "identity").read_text() == "old"
    assert not (root / "update.lock").exists()


@pytest.mark.parametrize("step", ["cask", "activate", "health"])
def test_swap_failure_restores_and_verifies_old_app(install, step):
    result, log, root, app = install(step)
    assert result.returncode != 0
    assert (app / "identity").read_text() == "old"
    assert "activate" in log
    assert "failed" in (root / "update-state").read_text()


def test_success_order(install):
    result, log, root, app = install()
    assert result.returncode == 0, result.stderr
    assert (
        log.index("upgrade --formula")
        < log.index("signatures")
        < log.index("consolidate")
        < log.index("reinstall --cask")
        < log.index("activate")
    )
    assert "success" in (root / "update-state").read_text()
    assert (app / "identity").read_text().strip() == "new"


def test_busy_never_mutates(install):
    result, log, root, app = install("busy")
    assert result.returncode == 0
    assert "upgrade" not in log
    assert "waiting" in (root / "update-state").read_text()
    assert not (root / "update.lock").exists()


def test_unsettled_attempt_refuses_retry(install):
    _, _, root, _ = install("busy")
    (root / "update-state").write_text("old-id\ncask\n")
    result, log, _, _ = install()
    assert result.returncode == 75
    assert "upgrade" not in log


def test_concurrent_lock_refuses(install):
    _, _, root, _ = install("busy")
    (root / "update.lock").mkdir()
    result, log, _, _ = install()
    assert result.returncode == 75
    assert "upgrade" not in log


def test_cancel_before_mutation(install):
    _, _, root, app = install("busy")
    (root / "cancel-test-operation").touch()
    result, log, _, _ = install()
    assert result.returncode == 0
    assert "upgrade" not in log
    assert "cancelled" in (root / "update-state").read_text()
    assert (app / "identity").read_text() == "old"


def test_dead_owner_never_retries_unsettled_mutation(install):
    _, _, root, _ = install("busy")
    lock = root / "update.lock"
    lock.mkdir()
    (lock / "pid").write_text("2147483647")
    (lock / "operation").write_text("dead-operation")
    (root / "update-state").write_text("dead-operation\ncask\n")
    result, log, _, _ = install()
    assert result.returncode == 75
    assert "upgrade" not in log
    assert (root / "update-state").read_text() == "dead-operation\ncask\n"


def test_mismatched_tap_refuses_before_mutation(install):
    result, log, root, app = install("mismatch")
    assert result.returncode != 0
    assert "upgrade" not in log
    assert (app / "identity").read_text() == "old"
    assert not (root / "update.lock").exists()


def test_retry_after_restoration_reinstalls_recorded_cask(install):
    result, _, _, app = install("activate")
    assert result.returncode != 0
    assert (app / "identity").read_text() == "old"
    result, log, _, app = install(operation="retry-operation")
    assert result.returncode == 0, result.stderr
    assert "reinstall --cask" in log
    assert (app / "identity").read_text().strip() == "new"


def test_second_preflight_catches_work_started_during_update(install):
    result, log, root, _ = install("late-busy")
    assert result.returncode == 0, result.stderr
    assert "upgrade" not in log
    assert "waiting" in (root / "update-state").read_text()


def test_update_script_is_executable():
    assert SCRIPT.stat().st_mode & 0o111 == 0o111


def test_update_script_git_mode():
    result = subprocess.check_output(
        ["git", "ls-files", "--stage", "scripts/brainbar-update.sh"], text=True, env=_clean_git_env()
    )
    assert result.startswith("100755 ")


def test_update_script_packaged_mode(tmp_path):
    root = Path(__file__).parents[1]
    configured = os.environ.get("BRAINBAR_UPDATE_WHEEL")
    if not configured:
        pytest.skip("Set BRAINBAR_UPDATE_WHEEL to the built release wheel for packaging proof")
    wheel = Path(configured).resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        entry = archive.getinfo("brainlayer/launchd/brainbar-update.sh")
        assert (entry.external_attr >> 16) & 0o111 == 0o111
    package = tmp_path / "installed"
    result = subprocess.run(
        ["uv", "pip", "install", "--offline", "--no-deps", "--target", str(package), str(wheel)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert os.access(package / "brainlayer/launchd/brainbar-update.sh", os.X_OK)


@pytest.mark.parametrize("status", ["idle", "busy", "stale", "missing"])
def test_job_snapshot_preflight(install, tmp_path, status):
    import json
    import time

    jobs = tmp_path / "jobs.json"
    if status != "missing":
        jobs.write_text(
            json.dumps(
                {"at": time.time() - (20 if status == "stale" else 0), "busy": "backup" if status == "busy" else ""}
            )
        )
    result, log, root, _ = install(jobs=jobs)
    assert result.returncode == 0, result.stderr
    if status == "idle":
        assert "success" in (root / "update-state").read_text()
    else:
        assert "upgrade" not in log
        assert "waiting" in (root / "update-state").read_text()


def test_dead_owner_with_settled_state_can_retry(install):
    _, _, root, _ = install("busy")
    lock = root / "update.lock"
    lock.mkdir()
    (lock / "pid").write_text("2147483647")
    (lock / "operation").write_text("dead-operation")
    result, _, _, app = install()
    assert result.returncode == 0, result.stderr
    assert (app / "identity").read_text().strip() == "new"


def test_prune_failure_does_not_prevent_restore(install):
    _, _, root, app = install("busy")
    previous = root / "Backups.noindex/BrainBar.failed-previous.app"
    previous.mkdir()
    (root / "last-failed").write_text(str(previous))
    commands = app.parents[1] / "bin"
    ruby = commands / "ruby"
    ruby.write_text(
        '#!/bin/bash\nif [ "$1" = -rfileutils ]; then echo prune >> "$TEST_LOG"; exit 1; fi\nexec /usr/bin/ruby "$@"\n'
    )
    ruby.chmod(0o700)
    result, log, _, app = install("activate")
    assert result.returncode != 0
    assert (app / "identity").read_text() == "old"
    assert log.index("activate", log.index("activate") + 1) < log.index("prune")
    assert "failed" in (root / "update-state").read_text()


def test_reaper_blocks_new_acquisition_during_owner_read(install):
    import time

    _, _, root, app = install("busy")
    lock = root / "update.lock"
    lock.mkdir()
    (lock / "pid").write_text("2147483647")
    (lock / "operation").write_text("dead")
    commands = app.parents[1] / "bin"
    cat = commands / "cat"
    cat.write_text(
        '#!/bin/bash\nif [ "$1" = "$BRAINBAR_UPDATE_ROOT/update.lock/pid" ]; then touch "$BRAINBAR_UPDATE_ROOT/reading"; while [ ! -f "$BRAINBAR_UPDATE_ROOT/continue" ]; do sleep 0.02; done; fi\nexec /bin/cat "$@"\n'
    )
    cat.chmod(0o700)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor() as pool:
        first = pool.submit(install, "busy")
        try:
            deadline = time.monotonic() + 10
            while not (root / "reading").exists():
                assert time.monotonic() < deadline
                time.sleep(0.02)
            # Model the exiting owner releasing the old mkdir while the reaper is suspended.
            (lock / "pid").unlink()
            (lock / "operation").unlink()
            lock.rmdir()
            second, log, _, _ = install(operation="contender")
            assert second.returncode == 75, second.stderr
            assert "upgrade" not in log
        finally:
            (root / "continue").touch()
        first.result(timeout=10)
