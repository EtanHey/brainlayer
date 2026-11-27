"""Exercise the shipped shell with an invented keg and command recorder, never brew."""

import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(os.environ.get("BRAINBAR_UPDATE_SCRIPT", Path(__file__).parents[1] / "scripts/brainbar-update.sh"))


# Shared by the fakes. A BrainBar process is one whose argv[0] is an executable inside the bundle,
# as launchd starts it. The fake app ignores SIGTERM (SIG_IGN survives exec), like an app that
# Homebrew's AppleScript quit cannot stop. Execute a copied binary at its real bundle path
# instead of spoofing argv[0], so both Linux and macOS process scans see that path.
APP_ALIVE = """
app_alive() {
  /bin/ps -axo command= | P="$BRAINBAR_UPDATE_APP/Contents/MacOS/" /usr/bin/awk 'index($0, ENVIRON["P"]) == 1 {f = 1} END {exit !f}'
}
start_app() {
  nohup /bin/bash -c 'trap "" TERM; exec "$0" 300' "$1/Contents/MacOS/BrainBar" >/dev/null 2>&1 &
  echo $!
}
"""


def _wait_for_command(proc: subprocess.Popen, prefix: str) -> subprocess.Popen:
    # Linux's per-pid ps output can truncate long pytest bundle paths to the
    # terminal width. Read argv directly there; macOS needs full-width ps.
    deadline = time.monotonic() + 5
    while True:
        assert proc.poll() is None, "fixture process exited before becoming ready"
        if sys.platform == "linux":
            command = Path(f"/proc/{proc.pid}/cmdline").read_bytes().split(b"\0", 1)[0].decode()
        else:
            command = subprocess.run(
                ["ps", "-ww", "-o", "command=", "-p", str(proc.pid)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        if command.startswith(prefix):
            break
        assert time.monotonic() < deadline, "fixture process never showed its command line"
        time.sleep(0.02)
    # Readiness must also be visible to the exact process scan the updater uses.
    scan = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True)
    assert any(
        fields[0] == str(proc.pid) and fields[1].startswith(prefix)
        for line in scan.stdout.splitlines()
        if len(fields := line.split(maxsplit=1)) == 2
    ), "ready fixture is absent from the updater's process scan"
    return proc


def _start_stubborn_app(app: Path) -> subprocess.Popen:
    exe = str(app / "Contents/MacOS/BrainBar")
    # Its own session, so cleanup can reap the whole process group.
    return subprocess.Popen(
        [exe, "300"],
        start_new_session=True,
        preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_IGN),
    )


def _reap(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=10)


def _kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _clean_git_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


@pytest.fixture
def install(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()
    app = tmp_path / "Applications/BrainBar.app"
    app.mkdir(parents=True)
    executable = app / "Contents/MacOS/BrainBar"
    executable.parent.mkdir(parents=True)
    sleep = shutil.which("sleep")
    assert sleep is not None, "the fixture needs the system sleep executable"
    # copy2 also copies macOS protected system flags, which unprivileged fixtures cannot set.
    fixture_sleep = tmp_path / "fixture-sleep"
    shutil.copy(sleep, fixture_sleep)
    if sys.platform == "darwin":
        # A relocated Apple system binary needs a fresh signature before it can execute.
        subprocess.run(["/usr/bin/codesign", "--force", "--sign", "-", str(fixture_sleep)], check=True)
    shutil.copy(fixture_sleep, executable)
    (app / "identity").write_text("old")
    root = tmp_path / "state"
    keg = tmp_path / "Cellar/brainlayer/2.0.0/lib/brainlayer/launchd"
    keg.mkdir(parents=True)
    log = tmp_path / "commands"

    agents = tmp_path / "LaunchAgents"
    agents.mkdir()
    for label in ("brainbar", "brainbar-daemon"):
        (agents / f"com.brainlayer.{label}.plist").write_text("old")
        (tmp_path / f"loaded-com.brainlayer.{label}").touch()

    def command(path, body):
        path.write_text("#!/bin/bash\nset -e\n" + APP_ALIVE + body)
        path.chmod(0o700)

    command(
        commands / "brew",
        """
echo "brew $*" >> "$TEST_LOG"
[ "$*" != "upgrade --formula etanhey/layers/brainlayer" ] || echo "formula-ppid $PPID" >> "$TEST_LOG"
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
  # Homebrew's uninstall: unload both labels, try an AppleScript quit (which a launchd job is
  # never allowed to send), trash both plists, then install and let postflight rewrite them.
  if app_alive; then echo ALIVE_AT_CASK >> "$TEST_LOG"; echo "Application did not quit" >&2; fi
  rm -f "${BRAINBAR_UPDATE_AGENTS:?}"/com.brainlayer.brainbar*.plist
  mkdir -p "$BRAINBAR_UPDATE_APP/Contents/MacOS"
  /bin/cp "$TEST_SLEEP" "$BRAINBAR_UPDATE_APP/Contents/MacOS/BrainBar"
  echo new > "$BRAINBAR_UPDATE_APP/identity"
  [ "$FAIL_STEP" != cask ]
  if [ "$FAIL_STEP" != no-plists ] && [ "$FAIL_STEP" != restore-copy ]; then
    for label in brainbar brainbar-daemon; do echo cask > "$BRAINBAR_UPDATE_AGENTS/com.brainlayer.$label.plist"; done
  fi
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
    # Every system tool the installer may reach is faked: no test may touch real launchd or /Applications.
    command(
        commands / "launchctl",
        """
echo "launchctl $*" >> "$TEST_LOG"
label=${2##*/}
case "$1" in
  print) [ -f "$TEST_PREFIX/loaded-$label" ] || exit 113 ;;
  bootout) [ "$FAIL_STEP" != bootout ] || exit 5; rm -f "$TEST_PREFIX/loaded-$label" ;;
esac
""",
    )
    command(
        commands / "mv",
        'if [ "$1" = "$BRAINBAR_UPDATE_APP" ] && app_alive; then echo MOVED_RUNNING >> "$TEST_LOG"; fi\nexec /bin/mv "$@"\n',
    )
    command(
        commands / "ps",
        """
count=$(( $(/bin/cat "$TEST_PREFIX/ps-count" 2>/dev/null || echo 0) + 1 ))
echo "$count" > "$TEST_PREFIX/ps-count"
if [ -n "${PS_FAIL_ON:-}" ] && [ "$count" -ge "$PS_FAIL_ON" ]; then echo "ps: injected failure" >&2; exit 1; fi
exec /bin/ps "$@"
""",
    )
    command(
        commands / "cp",
        """
for last; do :; done
case "$last" in "$BRAINBAR_UPDATE_AGENTS"/*) [ "$FAIL_STEP" != restore-copy ] || exit 1 ;; esac
exec /bin/cp "$@"
""",
    )
    command(
        commands / "codesign",
        'echo "codesign $*" >> "$TEST_LOG"\n[ "$1" != -dv ] || printf "Identifier=com.brainlayer.brainbar\\nTeamIdentifier=TEAM1\\n" >&2\n',
    )
    command(
        commands / "spctl",
        '[ "$FAIL_STEP" != gatekeeper ] || exit 3\nprintf "%s: accepted\\nsource=Notarized Developer ID\\n" "${@: -1}"\n',
    )
    command(
        commands / "xattr",
        """
echo "xattr $*" >> "$TEST_LOG"
if [ "$1" = -p ]; then
  if [ "$(cat "$3/identity")" = old ]; then echo '03c1;6b05b29c;;old'; else echo '0381;6b05b29d;;new'; fi
elif [ "$1" = -w ]; then exit 0
else echo com.apple.quarantine; fi
""",
    )
    command(
        keg / "release-verify-signatures.sh",
        """
echo signatures >> "$TEST_LOG"
echo "signatures-ppid $PPID" >> "$TEST_LOG"
if ! /usr/bin/ruby -e 'exit(File.open(ARGV[0]).flock(File::LOCK_EX | File::LOCK_NB) ? 1 : 0)' "$BRAINBAR_UPDATE_ROOT/update-custody.lock"; then
  echo custody-free >> "$TEST_LOG"
fi
[ "$FAIL_STEP" != signatures ]
""",
    )
    command(keg / "install.sh", 'echo consolidate >> "$TEST_LOG"; [ "$FAIL_STEP" != consolidate ]\n')
    command(
        commands / "brainlayer",
        """
echo activate >> "$TEST_LOG"
[ -d "$BRAINBAR_UPDATE_APP" ] || exit 1
for label in brainbar brainbar-daemon; do [ -f "$BRAINBAR_UPDATE_AGENTS/com.brainlayer.$label.plist" ] || exit 1; done
touch "$TEST_PREFIX/loaded-com.brainlayer.brainbar" "$TEST_PREFIX/loaded-com.brainlayer.brainbar-daemon"
if [ "$FAIL_STEP" = activate ] && [ "$(cat "$BRAINBAR_UPDATE_APP/identity")" != old ]; then
  # The new build started and then failed verification; rollback has to stop it again.
  if [ -n "$TEST_STUBBORN_NEW_APP" ]; then start_app "$BRAINBAR_UPDATE_APP" > "$TEST_PREFIX/new-app.pid"; fi
  exit 1
fi
""",
    )
    env = dict(
        os.environ,
        BRAINBAR_UPDATE_PATH=f"{commands}:/usr/bin:/bin",
        BRAINBAR_UPDATE_APP=str(app),
        BRAINBAR_UPDATE_ROOT=str(root),
        BRAINBAR_UPDATE_PLIST=str(commands / "plist"),
        BRAINBAR_UPDATE_AGENTS=str(agents),
        BRAINBAR_UPDATE_STOP_TIMEOUT="2",
        TEST_SLEEP=str(fixture_sleep),
        TEST_LOG=str(log),
        TEST_PREFIX=str(tmp_path),
        FAIL_STEP="",
    )

    # The installer calls the canonical prefix/bin/brainlayer.
    def run(step="", *, jobs=None, operation="test-operation", **extra):
        env["FAIL_STEP"] = step
        env.update(extra)
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


@pytest.mark.parametrize("previous", [None, "waiting", "success", "failed", "cancelled"])
def test_shared_activation_custody_waits_then_retry_succeeds_without_another_click(install, previous):
    import fcntl

    _, _, root, app = install("busy")
    if previous is None:
        (root / "update-state").unlink()
    else:
        owner = "activation-wait" if previous == "waiting" else "finished-operation"
        (root / "update-state").write_text(f"{owner}\n{previous}\n\n")
    # Fake auto-activator: the same shared OS lock held across its child activation.
    with (root / "update-custody.lock").open("r+") as activation:
        fcntl.flock(activation, fcntl.LOCK_SH)
        result, log, _, _ = install(operation="activation-wait")
        assert result.returncode == 0, result.stderr
        assert (root / "update-state").read_text().splitlines()[:2] == ["activation-wait", "waiting"]
        assert "upgrade" not in log
        assert (app / "identity").read_text() == "old"
    # BrainBarUpdates.tick retries .waiting with the same operation and copied helper.
    result, _, _, _ = install(operation="activation-wait")
    assert result.returncode == 0, result.stderr
    assert (root / "update-state").read_text().splitlines()[:2] == ["activation-wait", "success"]
    assert (app / "identity").read_text().strip() == "new"


@pytest.mark.parametrize("phase", ["preflight", "cask", "recovery-required", "waiting"])
def test_shared_activation_custody_preserves_other_operations_nonterminal_state(install, phase):
    import fcntl

    _, _, root, _ = install("busy")
    previous = f"other-operation\n{phase}\nKeep this receipt\n"
    (root / "update-state").write_text(previous)
    with (root / "update-custody.lock").open("r+") as activation:
        fcntl.flock(activation, fcntl.LOCK_SH)
        result, log, _, _ = install(operation="contender")
    assert result.returncode == 75
    assert (root / "update-state").read_text() == previous
    assert "upgrade" not in log


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
    assert log.index("\nactivate\n", log.index("\nactivate\n") + 1) < log.index("\nprune\n")
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


def test_custody_survives_supervisor_death_until_installer_exits(install):
    import signal
    import time

    _, _, root, app = install("busy")
    commands = app.parents[1] / "bin"
    brew = commands / "brew"
    original = brew.read_text()
    brew.write_text(
        original.replace(
            'echo "brew $*"',
            'if [ "$1" = update ]; then touch "$BRAINBAR_UPDATE_ROOT/blocked"; while [ ! -f "$BRAINBAR_UPDATE_ROOT/release" ]; do sleep 0.02; done; fi\necho "brew $*"',
            1,
        )
    )
    env = dict(
        os.environ,
        BRAINBAR_UPDATE_PATH=f"{commands}:/usr/bin:/bin",
        BRAINBAR_UPDATE_APP=str(app),
        BRAINBAR_UPDATE_ROOT=str(root),
        BRAINBAR_UPDATE_PLIST=str(commands / "plist"),
        BRAINBAR_UPDATE_AGENTS=str(app.parents[1] / "LaunchAgents"),
        BRAINBAR_UPDATE_STOP_TIMEOUT="2",
        TEST_SLEEP=str(app.parents[1] / "fixture-sleep"),
        TEST_LOG=str(app.parents[1] / "commands"),
        TEST_PREFIX=str(app.parents[1]),
        FAIL_STEP="",
    )
    supervisor = subprocess.Popen(
        ["bash", str(SCRIPT), "2.0.0", "held-attempt"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    child = None
    try:
        deadline = time.monotonic() + 10
        while not (root / "blocked").exists():
            assert supervisor.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        child = int((root / "update.lock/pid").read_text())
        supervisor.kill()  # Kill only Ruby; the installer remains blocked in fake brew.
        supervisor.wait(timeout=10)
        os.kill(child, 0)
        previous = (root / "update-state").read_bytes()
        second, _, _, _ = install(operation="contender")
        assert (root / "update-state").read_bytes() == previous
        assert second.returncode == 75
        assert "Another update owns custody" in second.stderr
    finally:
        (root / "release").touch()
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=10)
        deadline = time.monotonic() + 10
        while child is not None:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                os.kill(child, signal.SIGKILL)
                pytest.fail("Installer did not exit after fake brew was released")
            time.sleep(0.02)
        # Confirm the inherited open-file description is released when the child exits.
        import fcntl

        with (root / "update-custody.lock").open("r+") as lock:
            deadline = time.monotonic() + 10
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(lock, fcntl.LOCK_UN)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        if child is not None:
                            os.kill(child, signal.SIGKILL)
                        raise
                    time.sleep(0.02)
    third, _, _, _ = install(operation="after-child")
    assert third.returncode == 0, third.stderr


# --- 24 Nov 2026 regression: the helper must stop BrainBar itself, before touching the bundle ---


def test_running_app_that_ignores_quit_is_stopped_before_bundle_moves(install, tmp_path):
    """The M1 failure: Homebrew's quit cannot stop the app from a launchd job, so the helper must."""
    app_path = tmp_path / "Applications/BrainBar.app"
    proc = _start_stubborn_app(app_path)
    try:
        _wait_for_command(proc, str(app_path / "Contents/MacOS/BrainBar"))
        result, log, root, app = install()
        assert "MOVED_RUNNING" not in log, "the running bundle was moved"
        assert "ALIVE_AT_CASK" not in log, "BrainBar was still running when Homebrew reinstalled the cask"
        assert proc.poll() is not None, "the old BrainBar process survived the update"
        assert result.returncode == 0, result.stderr
        assert "success" in (root / "update-state").read_text()
    finally:
        _reap(proc)


def test_process_that_only_names_the_bundle_is_left_alone(install, tmp_path):
    """Only a process whose executable is in the bundle is BrainBar; `tail <bundle file>` is not."""
    exe = tmp_path / "Applications/BrainBar.app/Contents/MacOS/BrainBar"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text("")
    proc = subprocess.Popen(["/usr/bin/tail", "-f", str(exe)], start_new_session=True)
    try:
        _wait_for_command(proc, "/usr/bin/tail")
        result, log, root, app = install()
        assert proc.poll() is None, "a process that merely names a bundle path was killed"
        assert result.returncode == 0, result.stderr
        assert "success" in (root / "update-state").read_text()
    finally:
        _reap(proc)


@pytest.mark.parametrize("failing_scan", ["1", "2"])
def test_failed_process_scan_is_recovery_required_and_moves_nothing(install, tmp_path, failing_scan):
    """An unknown process state (the first scan, or the wait loop after SIGTERM) never moves the bundle."""
    app_path = tmp_path / "Applications/BrainBar.app"
    proc = _start_stubborn_app(app_path)
    try:
        _wait_for_command(proc, str(app_path / "Contents/MacOS/BrainBar"))
        result, log, root, app = install(PS_FAIL_ON=failing_scan)
        assert result.returncode != 0
        assert "MOVED_RUNNING" not in log and "reinstall --cask" not in log
        assert (app / "identity").read_text() == "old"
        assert "activate\n" not in log, "unknown state must not be re-activated blindly"
        state = (root / "update-state").read_text()
        assert "recovery-required" in state and "state unknown" in state
    finally:
        _reap(proc)


def test_failed_plist_restore_is_recovery_required_with_exact_commands(install):
    result, log, root, app = install("restore-copy")
    assert result.returncode != 0
    assert (app / "identity").read_text() == "old"
    assert "activate\n" not in log, "activation cannot recreate missing plists"
    state = (root / "update-state").read_text()
    assert "recovery-required" in state
    assert "Backups.noindex/LaunchAgents-test-operation/com.brainlayer.brainbar.plist" in state
    assert "brainbar activate --verify --timeout 90" in state


def test_services_boot_out_before_cask_and_activation_runs_once(install, tmp_path):
    result, log, root, app = install()
    assert result.returncode == 0, result.stderr
    uid = os.getuid()
    assert (
        log.index(f"launchctl bootout gui/{uid}/com.brainlayer.brainbar\n")
        < log.index("reinstall --cask")
        < log.index("activate")
    )
    assert log.index(f"launchctl bootout gui/{uid}/com.brainlayer.brainbar-daemon\n") < log.index("reinstall --cask")
    assert log.count("activate\n") == 1


def test_failed_bootout_leaves_bundle_and_restarts_old_app(install):
    result, log, root, app = install("bootout")
    assert result.returncode != 0
    assert "reinstall --cask" not in log
    assert (app / "identity").read_text() == "old"
    assert log.count("activate\n") == 1, "the untouched old app must be brought back"
    assert "failed" in (root / "update-state").read_text()


def test_trashed_plists_are_restored_before_activation(install):
    result, log, root, app = install("no-plists")
    assert result.returncode == 0, result.stderr
    agents = app.parents[1] / "LaunchAgents"
    assert (agents / "com.brainlayer.brainbar.plist").read_text() == "old"
    assert (agents / "com.brainlayer.brainbar-daemon.plist").read_text() == "old"
    assert "success" in (root / "update-state").read_text()


def test_approval_is_carried_after_security_checks(install):
    result, log, root, app = install()
    assert result.returncode == 0, result.stderr
    # Old app 0x03c1 is user-approved (0x40); the new quarantine 0x0381 gains the flag, provenance kept.
    assert "xattr -w com.apple.quarantine 03c1;6b05b29d;;new" in log
    assert log.index("codesign --verify --deep --strict") < log.index("xattr -w") < log.index("activate")


def test_gatekeeper_rejection_rolls_back(install):
    result, log, root, app = install("gatekeeper")
    assert result.returncode != 0
    assert (app / "identity").read_text() == "old"
    assert "failed" in (root / "update-state").read_text()


def test_rollback_stops_the_new_app_before_moving_it(install, tmp_path):
    result, log, root, app = install("activate", TEST_STUBBORN_NEW_APP="1")
    new_pid = int((tmp_path / "new-app.pid").read_text())
    try:
        assert result.returncode != 0
        assert "MOVED_RUNNING" not in log, "rollback moved the running new bundle"
        alive = subprocess.run(["ps", "-o", "stat=", "-p", str(new_pid)], capture_output=True, text=True).stdout.strip()
        assert not alive or alive.startswith("Z"), "the new BrainBar process survived rollback"
        assert (app / "identity").read_text() == "old"
        rollback = log[log.index("activate") :]
        assert "launchctl bootout" in rollback and rollback.index("launchctl bootout") < rollback.index("activate", 1)
        assert "failed" in (root / "update-state").read_text()
    finally:
        _kill(new_pid)


def test_hands_over_to_the_new_kegs_script_with_custody(install, tmp_path):
    keg = tmp_path / "Cellar/brainlayer/2.0.0/lib/brainlayer/launchd"
    newer = keg / "brainbar-update.sh"
    newer.write_text(SCRIPT.read_text().replace("\nshift\n", '\nshift\necho "new-script $*" >> "$TEST_LOG"\n', 1))
    newer.chmod(0o755)
    result, log, root, app = install()
    assert result.returncode == 0, result.stderr
    assert "new-script --handoff 2.0.0 test-operation" in log
    formula_pid = log.split("formula-ppid ")[1].split()[0]
    assert log.split("signatures-ppid ")[1].split()[0] == formula_pid, "the handoff must exec in place"
    assert "custody-free" not in log, "custody must stay held across the handoff"
    assert log.count("upgrade --formula") == 1
    assert (root / "update-state").read_text().startswith("test-operation\nsuccess\n")
    assert not (root / "update.lock").exists()


def test_handoff_refuses_without_live_custody(install, tmp_path):
    _, _, root, _ = install("busy")
    result = subprocess.run(
        ["bash", str(SCRIPT), "--custody-held", "--handoff", "2.0.0", "test-operation"],
        env=dict(os.environ, BRAINBAR_UPDATE_ROOT=str(root), BRAINBAR_UPDATE_PATH="/usr/bin:/bin"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 75
    assert "waiting" in (root / "update-state").read_text()
