"""#1075: run_with_deadline must not exit while its private suite group is alive.

Every fixture process is spawned here: the wrapper (in its own session, so the
test signals only a group it owns, like the heavy-suite helper), a leader and a
grandchild. Liveness is a FIFO both fixtures hold open for writing: the read end
sees EOF exactly when neither is alive, zombie or not. No sleeps decide outcomes;
every timer only bounds a failure.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform not in ("linux", "darwin"), reason="requires POSIX process inspection")

RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "ci" / "run_with_deadline.py"
BOUND = 30.0

# argv: fifo piddir role mode [ready_fd]; leader mode is "<gc mode>+<leader mode>" or "<leader mode>".
# "stubborn" ignores INT/TERM/HUP, "exit" leaves once the grandchild is ready, "plain" ignores nothing.
FIXTURE = r"""
import os, signal, subprocess, sys, time
fifo, piddir, role, mode = sys.argv[1:5]
gc_mode, own_mode = mode.split("+") if "+" in mode else ("plain", mode)
if own_mode == "stubborn":
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)
start = subprocess.check_output(["/bin/ps", "-o", "lstart=", "-p", str(os.getpid())], text=True).strip()
with open(os.path.join(piddir, role), "w") as f:
    f.write(f"{os.getpid()} {start}")
keep = os.open(fifo, os.O_WRONLY)
if role == "gc":
    ready = int(sys.argv[5]); os.write(ready, b"1"); os.close(ready)
else:
    r, w = os.pipe()
    subprocess.Popen([sys.executable, "-c", __CODE__, fifo, piddir, "gc", gc_mode, str(w)], pass_fds=(w,))
    os.close(w)
    assert os.read(r, 1) == b"1"
    print("READY", flush=True)
    if own_mode == "exit":
        sys.exit(0)
while True:
    time.sleep(3600)
"""

# Runs the real wrapper via runpy with recording/forcing shims on the OS boundary.
HARNESS = r"""
import os, runpy, signal, subprocess, sys
force, path = os.environ["FORCE"], sys.argv[1]
real_popen, real_killpg, real_waitpid = subprocess.Popen, os.killpg, os.waitpid
real_waitid = getattr(os, "waitid", None)
if force == "no-waitid" and real_waitid is not None:
    del os.waitid
leaders, fired = [], []
def log(*a): print("EVENT", *a, file=sys.stderr, flush=True)
def popen(*a, **k):
    if force == "spawn":
        os.kill(os.getpid(), signal.SIGTERM)  # lands after handlers, before the child exists
    p = real_popen(*a, **k); leaders.append(p.pid); return p
def waitid(idtype, pid, options):
    res = real_waitid(idtype, pid, options)
    if force == "exited" and res is not None and not fired:
        fired.append(1); os.kill(os.getpid(), signal.SIGTERM)  # leader exited, not yet reaped
    return res
def killpg(pgid, sig):
    log("killpg", pgid, int(sig)); real_killpg(pgid, sig)
    if force == "repeat" and sig != signal.SIGKILL and not fired:
        fired.append(1); os.kill(os.getpid(), signal.SIGTERM)  # second signal mid-grace
def waitpid(pid, options):
    res = real_waitpid(pid, options)
    if res[0] == pid and pid in leaders:
        log("reap", pid)
    return res
subprocess.Popen, os.killpg, os.waitpid = popen, killpg, waitpid
if real_waitid is not None and force != "no-waitid":
    os.waitid = waitid
sys.argv = [path] + sys.argv[2:]
runpy.run_path(path, run_name="__main__")
"""


def _leader_code() -> str:
    return FIXTURE.replace("__CODE__", repr(FIXTURE))


class Suite:
    def __init__(self, tmp_path: Path) -> None:
        self.piddir = tmp_path / "pids"
        self.piddir.mkdir()
        self.fifo = tmp_path / "alive.fifo"
        os.mkfifo(self.fifo)
        self.rfd = os.open(self.fifo, os.O_RDONLY | os.O_NONBLOCK)
        self.wrapper: subprocess.Popen | None = None

    def start(self, mode: str, *, seconds: float = 300, grace: float = 5, force: str | None = None):
        argv = ["--seconds", str(seconds), "--grace", str(grace), "--label", "fixture", "--"]
        argv += [sys.executable, "-c", _leader_code(), str(self.fifo), str(self.piddir), "leader", mode]
        head = [sys.executable, str(RUNNER)] if force is None else [sys.executable, "-c", HARNESS, str(RUNNER)]
        env = {**os.environ, "FORCE": force or ""}
        self.wrapper = subprocess.Popen(
            head + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env=env
        )
        return self.wrapper

    def wait_ready(self) -> None:
        assert self.wrapper is not None and self.wrapper.stdout is not None
        deadline, seen = time.monotonic() + BOUND, b""
        while b"READY" not in seen:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"fixture never became ready: {seen!r}"
            if select.select([self.wrapper.stdout], [], [], remaining)[0]:
                chunk = os.read(self.wrapper.stdout.fileno(), 4096)
                assert chunk, f"wrapper closed stdout before READY: {seen!r}"
                seen += chunk

    def finish(self) -> tuple[int, str]:
        assert self.wrapper is not None
        self.out, err = self.wrapper.communicate(timeout=BOUND)
        return self.wrapper.returncode, err.decode(errors="replace")

    def no_fixture_alive(self, timeout: float = 10.0) -> bool:
        """EOF on the FIFO: no fixture holds it open any more."""
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.read(self.rfd, 4096) == b"":
                    return True
            except BlockingIOError:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            select.select([self.rfd], [], [], remaining)

    def cleanup(self) -> None:
        if self.wrapper is not None and self.wrapper.poll() is None:
            os.killpg(self.wrapper.pid, signal.SIGKILL)  # our own unreaped child's group
            self.wrapper.wait()
        for record in self.piddir.iterdir():
            try:
                pid, start = record.read_text().split(" ", 1)
                stat = subprocess.check_output(["/bin/ps", "-o", "lstart=", "-p", pid], text=True).strip()
            except (OSError, ValueError, subprocess.CalledProcessError):
                continue
            if stat == start:  # same process, not a reused PID
                os.kill(int(pid), signal.SIGKILL)
        os.close(self.rfd)


def _assert_exited_with_group_gone(suite: Suite, rc: int, err: str, expected: int) -> None:
    gone = suite.no_fixture_alive()
    assert (rc, gone) == (expected, True), (
        f"rc={rc} (want {expected}); suite group gone={gone} -- a live group behind an exited "
        f"wrapper is the heavy-suite lock released over a running suite\n{err}"
    )


@pytest.fixture
def suite(tmp_path: Path):
    s = Suite(tmp_path)
    try:
        yield s
    finally:
        s.cleanup()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_signal_to_wrapper_group_reaches_the_private_suite_group(suite: Suite, signum: int) -> None:
    suite.start("plain")
    suite.wait_ready()
    os.killpg(suite.wrapper.pid, signum)  # what the heavy-suite helper does on Ctrl-C/TERM/HUP
    rc, err = suite.finish()
    _assert_exited_with_group_gone(suite, rc, err, 128 + signum)


def test_descendant_surviving_the_forward_is_killed_before_the_wrapper_exits(suite: Suite) -> None:
    suite.start("stubborn+plain")  # leader dies on TERM, grandchild ignores it
    suite.wait_ready()
    os.killpg(suite.wrapper.pid, signal.SIGTERM)
    rc, err = suite.finish()
    _assert_exited_with_group_gone(suite, rc, err, 128 + signal.SIGTERM)


def test_group_ignoring_the_signal_is_killed_after_the_bounded_grace(suite: Suite) -> None:
    suite.start("stubborn+stubborn", grace=0.5)
    suite.wait_ready()
    os.killpg(suite.wrapper.pid, signal.SIGTERM)
    rc, err = suite.finish()
    _assert_exited_with_group_gone(suite, rc, err, 128 + signal.SIGTERM)


def test_deadline_kills_a_descendant_that_survives_sigterm(suite: Suite) -> None:
    suite.start("stubborn+plain", seconds=2, grace=0.5)
    rc, err = suite.finish()
    assert b"READY" in suite.out, "deadline fired before the grandchild ignored TERM: proves nothing"
    assert "exceeded" in err
    _assert_exited_with_group_gone(suite, rc, err, 124)


def _events(err: str) -> list[tuple[str, ...]]:
    return [tuple(line.split()[1:]) for line in err.splitlines() if line.startswith("EVENT ")]


def _assert_group_signalled_only_before_reap(err: str) -> None:
    events = _events(err)
    kills = [i for i, e in enumerate(events) if e[0] == "killpg"]
    assert kills, events
    leader = events[kills[0]][1]
    # The no-waitid observer spawns ps, whose reap does not unpin the suite pgid.
    reaps = [i for i, e in enumerate(events) if e[0] == "reap" and e[1] == leader]
    assert reaps and kills, events
    assert max(kills) < reaps[0], f"killpg after the leader was reaped (PID-reuse hazard): {events}"
    assert events[kills[-1]][2] == str(int(signal.SIGKILL)), events


def test_forced_signal_during_spawn_is_forwarded_after_the_child_exists(suite: Suite) -> None:
    suite.start("plain", force="spawn")
    rc, err = suite.finish()
    _assert_exited_with_group_gone(suite, rc, err, 128 + signal.SIGTERM)
    _assert_group_signalled_only_before_reap(err)


@pytest.mark.skipif(not hasattr(os, "waitid"), reason="race injection requires waitid")
def test_forced_signal_after_leader_exit_still_kills_the_pinned_group(suite: Suite) -> None:
    suite.start("stubborn+exit", force="exited")  # leader exits; grandchild ignores TERM
    rc, err = suite.finish()
    _assert_exited_with_group_gone(suite, rc, err, 128 + signal.SIGTERM)
    _assert_group_signalled_only_before_reap(err)


def test_forced_repeated_signal_escalates_without_waiting_out_the_grace(suite: Suite) -> None:
    suite.start("stubborn+stubborn", grace=10 * BOUND, force="repeat")
    suite.wait_ready()
    os.killpg(suite.wrapper.pid, signal.SIGTERM)
    rc, err = suite.finish()  # bounded by BOUND, far below the grace: escalation must happen
    _assert_exited_with_group_gone(suite, rc, err, 128 + signal.SIGTERM)
    _assert_group_signalled_only_before_reap(err)


def test_deadline_without_waitid_finishes_before_default_grace(suite: Suite) -> None:
    suite.start("plain", seconds=1, force="no-waitid")
    suite.wait_ready()
    suite.wrapper.communicate(timeout=4)
    assert suite.wrapper.returncode == 124
    assert suite.no_fixture_alive()
