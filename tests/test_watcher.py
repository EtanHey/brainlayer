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
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import brainlayer.watcher as watcher_module
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


# ── Per-poll costs that scale with the corpus, not with new bytes (2026-09-05) ──────────
#
# Measured on the M4 with the real corpus (12,125 files on disk, 4,994 tracked, 16 GB DB),
# running as the LaunchAgent does -- `ProcessType=Background`, which macOS schedules onto the
# four efficiency cores. One steady-state poll_once cost 4.4-5.2 CPU-seconds there (1.0-1.2s
# on a performance core), of which: discovery walk ~1.6s, denylist glob matching ~1.8s, the
# health probe's COUNT over every realtime_watcher row ~1.3s. The pipeline itself (classify,
# chunk, scrub, insert) cost 0.9s across the whole 10-minute soak. The tests below pin the
# shape of the three fixes; the soak in the PR body is the proof they were enough.


def _iso(epoch: int, offset_hours: int = 0) -> str:
    """created_at as the watcher writes it, optionally in a non-UTC offset."""
    tz = timezone(timedelta(hours=offset_hours))
    stamp = datetime.fromtimestamp(epoch, tz).isoformat(timespec="milliseconds")  # ...T09:01:00.000-05:00
    return stamp.replace("+00:00", "Z")


def _probe_watcher(tmp_path, db_path):
    return _watcher(tmp_path, db_path=db_path)


def test_db_realtime_insert_probe_walks_indexed_ranges_not_the_source_index(tmp_path):
    """The health probe must be bounded by the window, not by the size of the table.

    `COUNT(*) ... WHERE source = 'realtime_watcher' AND COALESCE(ingested_at, strftime(created_at)) >= ?`
    can only use idx_chunks_source: it visits every one of the 451,000 realtime rows on the
    real DB (0.33s, every poll) to count the ~50 written in the last minute. Splitting the
    COALESCE into its two branches lets each use a range index -- ingested_at for current
    rows, created_at for the legacy NULL-ingested_at rows -- and the same window then costs
    3ms. The `+column` markers stop the planner from falling back to the source index.
    Semantics are unchanged: the counted set is identical to the COALESCE form, including a
    legacy row whose created_at carries a non-UTC offset.
    """
    db_path = tmp_path / "brainlayer.db"
    window = 1_700_000_000
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, source TEXT, ingested_at INTEGER, created_at TEXT)")
        conn.execute("CREATE INDEX idx_chunks_source ON chunks(source)")
        conn.execute("CREATE INDEX idx_chunks_ingested_at ON chunks(ingested_at)")
        conn.execute("CREATE INDEX idx_chunks_created ON chunks(created_at)")  # the name vector_store.py creates
        conn.executemany(
            "INSERT INTO chunks (source, ingested_at, created_at) VALUES (?, ?, ?)",
            [
                ("realtime_watcher", window + 5, _iso(0)),  # counted: ingested in window, created_at irrelevant
                ("realtime_watcher", window - 5, _iso(window + 999)),  # not counted: ingested before the window
                ("realtime_watcher", None, _iso(window + 60)),  # counted: legacy row created in the window
                ("realtime_watcher", None, _iso(window + 60, offset_hours=-5)),  # counted: same instant, -05:00
                ("realtime_watcher", None, _iso(window - 60)),  # not counted: legacy row before the window
                ("import", window + 5, _iso(window + 5)),  # not counted: other source
                ("import", None, _iso(window + 5)),  # not counted: other source, legacy
            ],
        )
        conn.commit()

    watcher = _probe_watcher(tmp_path, db_path)
    watcher._health_window_started_epoch = window
    with sqlite3.connect(db_path) as conn:
        # The invariant is query-level: the split probe counts exactly what the original
        # COALESCE statement counted, on the same rows, for the same window.
        coalesce_count = conn.execute(
            """
            SELECT COUNT(*) FROM chunks
            WHERE source = 'realtime_watcher'
              AND COALESCE(ingested_at, CAST(strftime('%s', created_at) AS INTEGER)) >= ?
            """,
            (window,),
        ).fetchone()[0]
    assert coalesce_count == 3, "fixture sanity: the COALESCE form counts the three rows in the window"
    assert watcher._db_realtime_inserts_since_window_start() == coalesce_count

    with sqlite3.connect(db_path) as conn:
        statements = watcher_module.realtime_insert_probe_statements(window)
        assert len(statements) == 2, "one indexed statement per COALESCE branch"
        for sql, params in statements:
            plan = " ".join(str(row[3]) for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params))
            assert "idx_chunks_source" not in plan, f"probe still walks the source index: {plan}"
            assert "idx_chunks_ingested_at" in plan or "idx_chunks_created" in plan, f"probe is unindexed: {plan}"


def test_db_realtime_insert_probe_trusts_liveness_evidence_without_scanning_chunks(tmp_path):
    """When the drain has written liveness rows for this window, the chunks table is not touched.

    The liveness count already won whenever it was non-zero; the chunk scan was paid first
    and then thrown away. Here there is no chunks table at all, so a probe that still
    scanned it would come back None (db_probe_failed) instead of the liveness count.
    """
    db_path = tmp_path / "brainlayer.db"
    window = 1_700_000_000
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE watcher_liveness_events (id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT, ingested_at INTEGER)"
        )
        conn.executemany(
            "INSERT INTO watcher_liveness_events (chunk_id, ingested_at) VALUES (?, ?)",
            [("a", window + 1), ("b", window + 2), ("c", window - 1)],
        )
        conn.commit()

    watcher = _probe_watcher(tmp_path, db_path)
    watcher._health_window_started_epoch = window
    assert watcher._db_realtime_inserts_since_window_start() == 2


def test_discovery_matches_glob_semantics_without_a_pathlib_glob_per_root(tmp_path, monkeypatch):
    """Discovery must find exactly what `base.glob('**/*.jsonl')` + `is_file()` found, cheaper.

    The pathlib walk cost ~2.4 million pathlib calls per poll over the real corpus (three
    syscalls and several Path objects per file). An os.scandir walk keeps the semantics --
    directory symlinks are not followed, file symlinks are, dotfiles match, the suffix match
    is case-sensitive, the recorded stat is the target's -- at one stat per file.
    """
    root = tmp_path / "projects"
    (root / "p1" / "s1" / "subagents").mkdir(parents=True)
    (root / "p2").mkdir()
    for rel in ("p1/a.jsonl", "p1/s1/b.jsonl", "p1/s1/subagents/agent-1.jsonl", "p2/c.jsonl", "p2/.hidden.jsonl"):
        (root / rel).write_text('{"id":"x"}\n')
    (root / "p2" / "notes.txt").write_text("not a transcript\n")
    (root / "p2" / "UPPER.JSONL").write_text('{"id":"upper"}\n')
    (root / "p2" / "link-to-p1").symlink_to(root / "p1", target_is_directory=True)
    (root / "p2" / "link-file.jsonl").symlink_to(root / "p1" / "a.jsonl")
    (root / "p2" / "dangling.jsonl").symlink_to(root / "missing.jsonl")
    (root / "stray.jsonl").write_text('{"id":"top-level, not under a project dir"}\n')

    expected = sorted(
        str(f) for base in root.iterdir() if base.is_dir() for f in base.glob("**/*.jsonl") if f.is_file()
    )
    assert str(root / "p2" / "link-file.jsonl") in expected and str(root / "p2" / ".hidden.jsonl") in expected
    assert not any("link-to-p1" in path for path in expected)

    def no_glob(self, pattern, *args, **kwargs):
        raise AssertionError(f"discovery must not fall back to pathlib glob for {pattern!r}")

    monkeypatch.setattr(Path, "glob", no_glob)
    watcher = _watcher(tmp_path)
    found = watcher._discover_jsonl_files()

    assert sorted(found) == expected
    for path in found:
        st = os.stat(path)  # follows symlinks, exactly as Path.stat() did
        assert watcher._current_file_stats[path] == (st.st_mtime, st.st_size, st.st_ino)
        assert watcher._file_providers[path] == "claude"
    mtimes = [watcher._current_file_stats[path][0] for path in found]
    assert mtimes == sorted(mtimes, reverse=True), "newest files first, as before"


def test_discovery_does_not_descend_into_denylisted_subtrees(tmp_path, monkeypatch):
    """A directory every file of which is denylisted is not opened at all.

    Real numbers behind this: the deployed `~/.cursor/**/agent-transcripts/**` pattern denies
    4,302 files spread over 4,315 directories; discovery still scandir'd every one of them
    on every poll (and, for that root, through pathlib's two-`**` glob), then asked the
    denylist about each file. Skipping the subtree at the `agent-transcripts` directory is
    the difference between ~5,100 and ~470 directory reads per poll.
    """
    root = tmp_path / "cursor-projects"
    kept = root / "repo" / "notes"
    denied = root / "repo" / "agent-transcripts" / "session-1"
    kept.mkdir(parents=True)
    denied.mkdir(parents=True)
    (kept / "keep.jsonl").write_text('{"id":"keep"}\n')
    (denied / "agent.jsonl").write_text('{"id":"denied"}\n')
    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", f"{root}/**/agent-transcripts/**")
    import brainlayer.ingest_denylist as denylist_module

    denylist_module.clear_pattern_match_cache()

    opened: list[str] = []
    real_scandir = os.scandir

    def recording_scandir(path=".", *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(watcher_module.os, "scandir", recording_scandir)

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("cursor-agent-transcripts", root, "**/*.jsonl")],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda items: None,
        health_path=tmp_path / "health.json",
    )
    found = watcher._discover_jsonl_files()

    assert found == [str(kept / "keep.jsonl")]
    assert str(root / "repo" / "agent-transcripts") not in opened, "the denylisted subtree was still walked"
    assert str(denied) not in opened
    assert str(kept) in opened


def test_discovery_walks_non_default_patterns_with_scandir_and_prunes_them_too(tmp_path, monkeypatch):
    """The cursor root's `**/agent-transcripts/**/*.jsonl` pattern gets the same walk and the same pruning.

    pathlib's glob was the fallback for any non-default pattern, and that fallback cannot
    skip a subtree. The scandir walk matches file paths against the pattern with the same
    `**`-aware part matcher the denylist uses, so every root is walked once, cheaply, and a
    denylisted subtree is skipped regardless of the root's own pattern.
    """
    root = tmp_path / "cursor-projects"
    transcript = root / "repo" / "agent-transcripts" / "session-1" / "session-1.jsonl"
    stray = root / "repo" / "other" / "stray.jsonl"
    denied = root / "denied-repo" / "agent-transcripts" / "session-2" / "session-2.jsonl"
    for path in (transcript, stray, denied):
        path.parent.mkdir(parents=True)
        path.write_text('{"id":"x"}\n')
    monkeypatch.setenv("BRAINLAYER_INGEST_DENYLIST", f"{root}/denied-repo/**")
    import brainlayer.ingest_denylist as denylist_module

    denylist_module.clear_pattern_match_cache()

    def no_glob(self, pattern, *args, **kwargs):
        raise AssertionError(f"discovery must not fall back to pathlib glob for {pattern!r}")

    monkeypatch.setattr(Path, "glob", no_glob)
    opened: list[str] = []
    real_scandir = os.scandir

    def recording_scandir(path=".", *args, **kwargs):
        opened.append(str(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(watcher_module.os, "scandir", recording_scandir)

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("cursor-agent-transcripts", root, "**/agent-transcripts/**/*.jsonl")],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda items: None,
        health_path=tmp_path / "health.json",
    )
    found = watcher._discover_jsonl_files()

    assert found == [str(transcript)], "only files under an agent-transcripts directory match the pattern"
    assert str(root / "denied-repo") not in opened, "the denylisted repo subtree must be pruned, not walked"
    assert watcher.provider_for_file(str(transcript)) == "cursor-agent-transcripts"


def test_new_parent_dir_with_no_stale_registry_entries_does_not_retrigger_prune(tmp_path, monkeypatch):
    """A brand-new directory holding only live files is not evidence that anything can be pruned.

    The re-trigger existed for a returning volume: entries the registry could not evaluate
    because their root was unmounted become prunable once a live file appears beside them.
    But it fired on ANY change to the set of parent directories -- and on a live machine a
    new session directory appears every few minutes. Each firing is the full registry scan
    (15,619 entries; 1.4-2s on a performance core, 5.6-8s on the efficiency cores the
    LaunchAgent runs on) and it can never complete, because 8,744 entries are orphans. In the
    600s soak of the otherwise-fixed watcher it fired inside the first two minutes and cost
    more than eight steady polls. The trigger now asks the only question that matters: does
    the registry hold an entry under one of the NEW directories that is not among the live
    files? If not, the timer is the retry.
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

    # A new session directory with a new live transcript -- the everyday case.
    new_dir = tmp_path / "projects" / "q"
    new_dir.mkdir()
    (new_dir / "b.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    watcher.poll_once()
    assert len(calls) == 1, "a new directory with only live files must not re-run the registry scan"

    # A new directory the registry already has a now-missing entry under -- the returning-volume
    # case the re-trigger exists for -- still fires immediately, without waiting for the timer.
    returned = tmp_path / "projects" / "r"
    watcher.registry.set(str(returned / "gone.jsonl"), 10, 1)
    returned.mkdir()
    (returned / "live.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    watcher.poll_once()
    assert len(calls) == 2, "a new directory holding a stale registry entry must trigger the prune"


def test_prune_retriggers_when_a_returning_volume_has_live_files_only_in_a_subdirectory(tmp_path, monkeypatch):
    """The stale entry's parent need not itself hold the live file -- an ancestor is enough.

    Round-1 review of #781 (Cursor, medium): `has_stale_entries_under` was direct-parent only,
    so a volume that remounts with live files only in a subdirectory of the stale entry's
    parent waited for the 900 s timer. `prune_missing_files` would already have pruned it --
    `_has_live_parent_evidence` accepts any ancestor of a live file -- so the trigger and the
    prune disagreed on what counts as evidence. They agree now: a stale entry anywhere under a
    NEW directory re-triggers the scan.
    """
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

    returned = tmp_path / "projects" / "r"
    watcher.registry.set(str(returned / "gone.jsonl"), 10, 1)
    (returned / "sub").mkdir(parents=True)
    (returned / "sub" / "live.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    watcher.poll_once()
    assert len(calls) == 2, "a stale entry under an ancestor of the new directory must trigger the prune"


def test_parent_set_shrinkage_waits_for_the_prune_timer(tmp_path, monkeypatch):
    """Losing the last live file in a directory does not re-trigger the scan. Deliberate.

    Round-1 review of #781 (Cursor, medium) asked for this trade to be explicit: before cut 5
    any change to the parent-dir set re-ran the registry scan, shrinkage included. Shrinkage
    can never make an orphan prunable -- `prune_missing_files` requires live evidence in the
    entry's directory tree, and the directory just lost its last live file -- so the scan
    would find nothing it could not find at the next 900 s tick. The orphan therefore waits
    at most one timer interval. That is the trade, and this test pins it.
    """
    watcher = _watcher(tmp_path)
    src = tmp_path / "projects" / "p"
    src.mkdir()
    (src / "a.jsonl").write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')
    other = tmp_path / "projects" / "q"
    other.mkdir()
    doomed = other / "b.jsonl"
    doomed.write_text('{"type":"user","message":{"role":"user","content":"hi"}}\n')

    calls = []

    def counting_prune(roots, active_files=None):
        calls.append(1)
        watcher.registry._last_prune_complete = False
        return 0

    monkeypatch.setattr(watcher.registry, "prune_missing_files", counting_prune)
    watcher.poll_once()
    assert len(calls) == 1

    doomed.unlink()  # the parent-dir set shrinks by one
    watcher.poll_once()
    assert len(calls) == 1, "shrinkage must wait for the timer; the scan could prune nothing here"


def test_discovery_does_not_enter_a_directory_symlink_even_for_a_literal_pattern_segment(tmp_path):
    """Pins the one stated narrowing vs pathlib.glob.

    Round-1 review of #781 (Cursor, low): pathlib.glob would step into a directory symlink whose
    name matched a literal segment (`agent-transcripts`); the scandir walk never follows
    directory symlinks, so transcripts reachable only through such a symlink are not
    discovered. Default `**/*.jsonl` roots are unaffected (pathlib skipped symlink recursion
    for `**` too). Real files under a real `agent-transcripts` directory are still found.
    """
    root = tmp_path / "cursor-projects"
    real = root / "repo-a" / "agent-transcripts" / "s1"
    real.mkdir(parents=True)
    (real / "s1.jsonl").write_text('{"id":"real"}\n')
    elsewhere = tmp_path / "outside" / "transcripts" / "s2"
    elsewhere.mkdir(parents=True)
    (elsewhere / "s2.jsonl").write_text('{"id":"via-symlink"}\n')
    (root / "repo-b").mkdir()
    (root / "repo-b" / "agent-transcripts").symlink_to(elsewhere.parent, target_is_directory=True)

    pattern = "**/agent-transcripts/**/*.jsonl"
    assert any("s2.jsonl" in str(f) for f in root.glob(pattern)), "pathlib.glob does enter the symlink"

    watcher = JSONLWatcher(
        watch_roots=[WatchRoot("cursor-agent-transcripts", root, pattern)],
        registry_path=tmp_path / "offsets.json",
        on_flush=lambda items: None,
        health_path=tmp_path / "health.json",
    )
    found = watcher._discover_jsonl_files()
    assert found == [str(real / "s1.jsonl")], "the scandir walk does not follow the directory symlink -- by design"
