"""Exercise the shipped shell with an invented keg and command recorder, never brew."""

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/brainbar-update.sh"


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
elif [ "$1" = update ]; then exit 0
elif [ "$1" = --prefix ]; then echo "$TEST_PREFIX"
elif [ "$2" = --cask ]; then
  mkdir -p "$BRAINBAR_UPDATE_APP"
  echo new > "$BRAINBAR_UPDATE_APP/identity"
  [ "$FAIL_STEP" != cask ]
else [ "$FAIL_STEP" != formula ]; fi
""",
    )
    command(commands / "pgrep", '[ "$FAIL_STEP" = busy ] && echo "123 backup_daily"; exit 0\n')
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
    def run(step=""):
        env["FAIL_STEP"] = step
        result = subprocess.run(
            ["bash", str(SCRIPT), "2.0.0", "test-operation"], env=env, capture_output=True, text=True
        )
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
        < log.index("upgrade --cask")
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
