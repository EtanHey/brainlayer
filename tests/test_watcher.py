"""R3: the watcher must not burn a core doing nothing.

Measured on the M4 (2026-09-04): one `_discover_jsonl_files()` pass over the real
12,742-file corpus takes 1.034s, while the installed LaunchAgent polled at `--poll 1.0`.
The poll interval was shorter than a single discovery pass, so the loop never slept --
roughly 100% of one core, permanently, with zero files changed.

Two constraints from the R3 brief are enforced here:
  1. event-driven, or batched at >= 30s -- no tight polling
  2. a hard CPU budget with no idle burn

The fix is an (mtime, size) gate: a file whose stat is unchanged since the poll that
last drained it costs nothing. The gate must never stall a file that is genuinely
behind, which is what most of these tests pin down.
"""

import math
import os

from brainlayer.watcher import JSONLWatcher, WatchRoot


def _watcher(tmp_path, **kwargs):
    src = tmp_path / "projects"
    src.mkdir(exist_ok=True)
    return JSONLWatcher(
        watch_roots=[WatchRoot("claude", src)],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda items: None,
        health_path=tmp_path / "health.json",
        **kwargs,
    )


def test_default_poll_interval_is_batched_at_30s_or_more(tmp_path):
    """Constraint 1: no tight polling. The default must not be sub-30s."""
    watcher = _watcher(tmp_path)
    assert watcher.poll_interval_s >= 30.0, (
        f"default poll interval is {watcher.poll_interval_s}s; the brief requires "
        "event-driven or >=30s batching, and a discovery pass alone measures ~1.03s"
    )


def test_unchanged_file_is_skipped_on_the_next_poll(tmp_path):
    """A file whose (mtime, size) has not moved must not be re-read."""
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    path = str(f)

    watcher.poll_once()
    assert path in watcher._observed_file_stats, "first poll must record the file's stat"

    # Nothing changed on disk -- the second poll must be able to skip it.
    watcher._discover_jsonl_files()
    assert watcher._can_skip_unchanged(path) is True


def test_changed_file_is_not_skipped(tmp_path):
    """An append moves size, so the gate must let the file through."""
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    path = str(f)

    watcher.poll_once()
    with f.open("a") as handle:
        handle.write('{"type":"user","message":{"role":"user","content":"more"}}\n')

    watcher._discover_jsonl_files()
    assert watcher._can_skip_unchanged(path) is False


def test_never_skips_a_file_that_was_never_read(tmp_path):
    """No tailer yet means the file has never been drained -- always read it."""
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    path = str(f)

    watcher._discover_jsonl_files()
    watcher._observed_file_stats = dict(watcher._current_file_stats)
    assert watcher._can_skip_unchanged(path) is False


def test_never_skips_a_file_the_tailer_is_still_behind_on(tmp_path):
    """A read capped by max_lines_per_file leaves the tailer behind at an unchanged stat.

    Skipping on stat alone would stall that file forever, so offset must be checked too.
    """
    watcher = _watcher(tmp_path, max_lines_per_file=1)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text(
        '{"type":"user","message":{"role":"user","content":"one"}}\n'
        '{"type":"user","message":{"role":"user","content":"two"}}\n'
        '{"type":"user","message":{"role":"user","content":"three"}}\n'
    )
    path = str(f)

    watcher.poll_once()
    tailer = watcher._tailers.get(path)
    assert tailer is not None
    assert tailer.offset < f.stat().st_size, "cap should have left the tailer behind"

    watcher._discover_jsonl_files()
    assert watcher._can_skip_unchanged(path) is False, (
        "a file the tailer has not finished draining must never be skipped"
    )


def test_max_offset_lag_reuses_discovery_stats(tmp_path, monkeypatch):
    """The lag probe must not re-stat every file that discovery just stat'd.

    `_write_health_snapshot` runs `_max_offset_lag_bytes(files)` on every poll. Calling
    `os.path.getsize` there is a second full pass over the whole corpus (~12,700 files on
    the real machine) for sizes discovery already collected a moment earlier.
    """
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    path = str(f)

    watcher._discover_jsonl_files()

    calls = []
    real_getsize = __import__("os").path.getsize

    def counting_getsize(p):
        calls.append(p)
        return real_getsize(p)

    monkeypatch.setattr("brainlayer.watcher.os.path.getsize", counting_getsize)
    lag = watcher._max_offset_lag_bytes([path])

    assert lag == f.stat().st_size, "lag must still be correct for an unread file"
    assert calls == [], f"re-stat'd files discovery already sized: {calls}"


def test_max_offset_lag_falls_back_to_stat_when_not_discovered(tmp_path):
    """A file absent from the discovery cache must still be measured, not skipped."""
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text("x" * 500)

    watcher._current_file_stats = {}
    assert watcher._max_offset_lag_bytes([str(f)]) == 500


def test_incomplete_prune_is_retried_on_a_timer_not_every_poll(tmp_path, monkeypatch):
    """An incomplete prune must not re-scan the whole registry on every poll.

    Measured on the real registry (21,529 entries, 8,744 of them orphaned):
    `prune_missing_files` takes 10-12s, and after the first pass it prunes ZERO and still
    reports `last_prune_complete is False` -- because entries whose parent directory no
    longer exists can never satisfy the live-parent guard. With the prune gated only on
    that flag, a 10s full-filesystem scan ran on every single poll, forever, achieving
    nothing: ~33% of one core at a 30s interval, which is the bulk of the measured 37.75%
    idle burn. The guard is correct; retrying it every poll is not.
    """
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    (src / "a.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')

    calls = []

    def counting_prune(roots, active_files=None):
        calls.append(1)
        watcher.registry._last_prune_complete = False  # orphans present: never completes
        return 0

    monkeypatch.setattr(watcher.registry, "prune_missing_files", counting_prune)

    watcher.poll_once()
    assert len(calls) == 1, "first poll should attempt the prune"

    watcher.poll_once()
    watcher.poll_once()
    assert len(calls) == 1, (
        f"an incomplete prune was retried on every poll ({len(calls)} scans in 3 polls); it must back off onto a timer"
    )


def test_incomplete_prune_is_retried_once_the_interval_elapses(tmp_path, monkeypatch):
    """Backing off must not mean giving up -- a returning volume still gets pruned."""
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    (src / "a.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')

    calls = []

    def counting_prune(roots, active_files=None):
        calls.append(1)
        watcher.registry._last_prune_complete = False
        return 0

    monkeypatch.setattr(watcher.registry, "prune_missing_files", counting_prune)

    watcher.poll_once()
    assert len(calls) == 1

    # pretend the retry interval has elapsed
    watcher._last_offset_prune_attempt -= watcher.offset_prune_retry_interval_s + 1
    watcher.poll_once()
    assert len(calls) == 2, "prune must be retried once its interval elapses"


def test_denylist_is_evaluated_once_per_file_per_poll(tmp_path, monkeypatch):
    """`is_denylisted` is not cheap -- it must not be re-run 3x per file per poll.

    Measured warm on the real corpus: one sweep over 12,796 files costs 0.82s, and
    `poll_once` evaluated it at three separate sites (the live-files set, the tailer sweep,
    and the main loop) for ~2.5s of every poll. Re-evaluating per poll is deliberate so a
    changed denylist is picked up; re-evaluating three times within one poll is waste.
    """
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    for name in ("a.jsonl", "b.jsonl"):
        (src / name).write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')

    calls = []
    import brainlayer.watcher as watcher_module

    real = watcher_module.is_denylisted

    def counting(path, **kwargs):
        calls.append(str(path))
        return real(path, **kwargs)

    monkeypatch.setattr(watcher_module, "is_denylisted", counting)

    watcher.poll_once()

    from collections import Counter

    worst = Counter(calls).most_common(1)
    assert worst, "expected at least one denylist evaluation"
    path, count = worst[0]
    assert count == 1, f"{path} was denylist-checked {count}x in a single poll; expected 1"


def test_same_path_replacement_with_identical_mtime_and_size_is_not_skipped(tmp_path):
    """A file replaced at the same path must never be skipped, even if stat looks identical.

    The skip gate fingerprinted only (mtime, size). Rotation is detected inside
    `_ensure_tailer`/`read_new_lines`, which a skip `continue`s past -- so a file swapped
    at the same path with the same size and the same mtime was skipped on every poll,
    forever, and its new content was never ingested. Silent: no error, no log, no alarm.

    The pre-existing replacement tests in test_jsonl_watcher.py all change size, so none of
    them cover this. Same failure shape as the prune bug this PR fixes -- a condition that
    can never flip -- except this one loses data instead of burning CPU.
    """
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    original = '{"type":"user","message":{"role":"user","content":"AAA"}}\n'
    f.write_text(original)
    path = str(f)

    watcher.poll_once()
    first_inode = f.stat().st_ino
    stat_before = f.stat()

    # Replace at the same path: new inode, identical size, identical mtime.
    replacement = '{"type":"user","message":{"role":"user","content":"BBB"}}\n'
    assert len(replacement) == len(original), "test requires a byte-identical length"
    swap = src / "swap.tmp"
    swap.write_text(replacement)
    os.replace(swap, f)
    os.utime(f, (stat_before.st_atime, stat_before.st_mtime))

    assert f.stat().st_ino != first_inode, "replacement must produce a new inode"
    assert f.stat().st_size == stat_before.st_size
    assert f.stat().st_mtime == stat_before.st_mtime

    watcher._discover_jsonl_files()
    assert watcher._can_skip_unchanged(path) is False, (
        "a same-path replacement was skipped: its new content can never be ingested"
    )


def test_tailer_ahead_of_file_is_not_skipped(tmp_path):
    """Refuse the skip unless the tailer is exactly at EOF.

    `offset >= size` also accepts `offset > size` -- a tailer that believes it read more
    bytes than the file holds. That is the signature of a truncation/rewind, and skipping
    there bypasses `check_rewind`, which AGENTS.md documents as the checkpoint-restore path
    that soft-archives reverted chunks. Only `offset == size` is a safe skip.
    """
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    f = src / "a.jsonl"
    f.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    path = str(f)

    watcher.poll_once()
    tailer = watcher._tailers.get(path)
    assert tailer is not None

    watcher._discover_jsonl_files()
    watcher._observed_file_stats = dict(watcher._current_file_stats)
    tailer.offset = watcher._current_file_stats[path][1] + 500  # ahead of EOF

    assert watcher._can_skip_unchanged(path) is False, "a tailer ahead of EOF was skipped, bypassing rewind detection"


def test_prune_retry_interval_rejects_nan_and_inf(monkeypatch):
    """nan/inf pass both `ValueError` and `<= 0`, and disable the retry timer forever.

    `monotonic() - attempt >= nan` and `>= inf` are never true, so an incomplete prune
    would retry only when parent dirs change -- never on the timer.
    """
    from brainlayer.watcher import _watch_offset_prune_retry_interval_s

    for raw in ("nan", "inf", "-inf", "NaN", "Infinity"):
        monkeypatch.setenv("BRAINLAYER_WATCHER_OFFSET_PRUNE_RETRY_S", raw)
        value = _watch_offset_prune_retry_interval_s()
        assert math.isfinite(value) and value > 0, f"{raw!r} produced non-finite interval {value}"


def test_deployed_poll_interval_is_batched_at_30s_or_more():
    """The constructor default is not what production runs -- the plists pass --poll.

    Asserting only `JSONLWatcher(...).poll_interval_s` would still pass if a plist said
    `--poll 1`, which is exactly the configuration this PR exists to remove. Check the
    surfaces that actually deploy: the CLI option default and both plists.
    """
    import inspect
    import plistlib
    from pathlib import Path

    from brainlayer.cli import watch

    default = inspect.signature(watch).parameters["poll_interval"].default
    cli_default = getattr(default, "default", default)
    assert float(cli_default) >= 30.0, f"CLI --poll default is {cli_default}s"

    repo_root = Path(__file__).resolve().parent.parent
    plists = [
        repo_root / "scripts" / "launchd" / "com.brainlayer.watch.plist",
        repo_root / "launchd" / "com.brainlayer.watch.plist",
    ]
    for plist_path in plists:
        assert plist_path.exists(), f"missing {plist_path}"
        args = plistlib.loads(plist_path.read_bytes())["ProgramArguments"]
        assert "--poll" in args, f"{plist_path.name} does not pass --poll"
        value = float(args[args.index("--poll") + 1])
        assert value >= 30.0, f"{plist_path.name} polls every {value}s"


# --- F2: the >=30s constraint, enforced where it can actually be violated -------------
#
# The three assertions above check the CLI option default and both repo plists -- values
# that live in this tree and cannot change behind the code's back. What reaches the poll
# loop is the `--poll` *argument*, and until this fix nothing checked it:
# cli/__init__.py took `poll_interval: float`, watcher.py stored it verbatim, and
# `start()` waited on it. The installed LaunchAgent still passes `--poll 1.0` (installed
# Sep 1, unchanged by this PR), so the violating input is a real artifact on this machine,
# not a hypothetical.


def test_sub_30s_poll_argument_is_clamped_loudly(caplog):
    """Every value that would defeat the R3 floor is clamped, and says so."""
    import logging as logging_module

    from brainlayer.watcher import MIN_WATCH_POLL_INTERVAL_S, enforce_min_poll_interval

    # 1.0 is verbatim what the installed LaunchAgent passes today. 0 and the negatives
    # mean "never sleep"; nan and inf are the shapes that survive a bare `< 30` bounds
    # check -- nan compares False against everything, inf parks Event.wait() forever.
    violations = [1.0, 0.05, 0.0, -5.0, float("nan"), float("inf"), float("-inf")]
    for requested in violations:
        caplog.clear()
        with caplog.at_level(logging_module.WARNING, logger="brainlayer.watcher"):
            effective = enforce_min_poll_interval(requested)
        assert effective == MIN_WATCH_POLL_INTERVAL_S, f"--poll {requested!r} produced {effective!r}"
        assert math.isfinite(effective) and effective >= 30.0
        assert any(record.levelno >= logging_module.WARNING for record in caplog.records), (
            f"--poll {requested!r} was clamped silently; a silent clamp is a new false-green"
        )
        assert "install.sh" in caplog.text, "the warning must name the fix at the source"


def test_conforming_poll_argument_passes_through_unchanged(caplog):
    """The other direction: a legal value is not touched and does not warn."""
    import logging as logging_module

    from brainlayer.watcher import enforce_min_poll_interval

    for requested in (30.0, 30.5, 60.0, 900.0):
        caplog.clear()
        with caplog.at_level(logging_module.WARNING, logger="brainlayer.watcher"):
            effective = enforce_min_poll_interval(requested)
        assert effective == requested, f"legal --poll {requested} was rewritten to {effective}"
        assert not caplog.records, f"legal --poll {requested} warned: {caplog.text}"


def _run_watch_command(tmp_path, monkeypatch, poll_interval: float) -> dict:
    """Drive the real `watch` command and return the kwargs it handed JSONLWatcher.

    Everything with a side effect outside the tmp tree is stubbed: no DB is opened
    (arbitrated=True leaves VectorStore unconstructed), no signal handler is installed,
    and start() returns instead of blocking forever on the poll loop.
    """
    import signal as signal_module
    import types

    import brainlayer.deploy_drift as deploy_drift
    import brainlayer.parent_death as parent_death
    import brainlayer.watcher as watcher_module
    from brainlayer.cli import watch

    monkeypatch.setenv("BRAINLAYER_DB", str(tmp_path / "watch-cli.db"))
    monkeypatch.setenv("BRAINLAYER_ARBITRATED", "1")
    monkeypatch.setattr(signal_module, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(parent_death, "install_parent_death_watcher", lambda *_a, **_kw: None)
    monkeypatch.setattr(deploy_drift, "record_launch_from_environment", lambda *_a, **_kw: None)

    built: dict = {}

    class _StubWatcher:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.indexer = types.SimpleNamespace(total_flushed=0)

        def start(self):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(watcher_module, "JSONLWatcher", _StubWatcher)

    source = tmp_path / "projects"
    source.mkdir()
    watch(source=[source], poll_interval=poll_interval, batch_size=10, flush_interval=500)

    assert built, "watch() never constructed a watcher"
    return built


def test_watch_command_clamps_the_installed_plists_poll_argument(tmp_path, monkeypatch):
    """The wiring, not just the helper.

    A validator that exists but is never called is the same defect one level down, so this
    drives the real command with `--poll 1.0` -- the argument in the deployed
    ~/Library/LaunchAgents/com.brainlayer.watch.plist -- and asserts the value that reaches
    JSONLWatcher is the floor.
    """
    built = _run_watch_command(tmp_path, monkeypatch, 1.0)
    assert built["poll_interval_s"] == 30.0, (
        f"watch() handed the poll loop poll_interval_s={built['poll_interval_s']}; "
        "--poll 1.0 is what the installed LaunchAgent passes and it must not survive the CLI"
    )


def test_watch_command_leaves_a_conforming_poll_argument_alone(tmp_path, monkeypatch):
    """Same path, legal input: the clamp must not rewrite an operator's deliberate value."""
    built = _run_watch_command(tmp_path, monkeypatch, 120.0)
    assert built["poll_interval_s"] == 120.0, f"legal --poll 120 became {built['poll_interval_s']}"
