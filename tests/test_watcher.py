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
