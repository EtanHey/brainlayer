import datetime as dt
import time

import pytest

from brainlayer.backup_sleep import SleepTracker


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_default_clock_selection_detects_suspend_with_platform_clock_ids(monkeypatch, platform):
    for name in ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC_RAW", "CLOCK_UPTIME_RAW", "CLOCK_MONOTONIC"):
        monkeypatch.delattr(time, name, raising=False)
    inclusive, awake = (101, 102)
    names = (
        ("CLOCK_MONOTONIC_RAW", "CLOCK_UPTIME_RAW") if platform == "darwin" else ("CLOCK_BOOTTIME", "CLOCK_MONOTONIC")
    )
    for name, value in zip(names, (inclusive, awake), strict=True):
        monkeypatch.setattr(time, name, value, raising=False)
    values = {inclusive: 0, awake: 0}
    calls = []

    def read(clock):
        calls.append(clock)
        return values[clock]

    monkeypatch.setattr(time, "clock_gettime", read)
    tracker = SleepTracker()
    values[inclusive] = 21600
    result = tracker.tick(prefix="upload", action="upload resumed after wake")
    assert result["upload_slept_seconds"] == 21600
    assert calls == [inclusive, awake, inclusive, awake]


def test_unavailable_platform_clocks_have_a_clear_failure(monkeypatch):
    monkeypatch.delattr(time, "clock_gettime")
    with pytest.raises(RuntimeError, match="sleep observation requires.*Darwin.*Linux"):
        SleepTracker()


def test_darkwake_fragments_merge_only_in_the_note():
    from brainlayer.backup_sleep import sleep_note

    intervals = [
        {"from": "2026-11-22T07:09:00+02:00", "to": "2026-11-22T08:09:00+02:00", "seconds": 3600},
        {"from": "2026-11-22T08:20:00+02:00", "to": "2026-11-22T09:20:00+02:00", "seconds": 3600},
        {"from": "2026-11-22T09:31:00+02:00", "to": "2026-11-22T13:20:00+02:00", "seconds": 13740},
    ]
    original = [dict(item) for item in intervals]
    assert sleep_note(intervals, action="uploaded the rest after wake") == (
        "paused 5 h 49 min while the Mac slept (07:09-13:20); uploaded the rest after wake"
    )
    assert intervals == original


def test_note_keeps_separate_sleeps_and_labels_midnight_dates():
    from brainlayer.backup_sleep import sleep_note

    intervals = [
        {"from": "2026-11-22T23:00:00+02:00", "to": "2026-11-23T01:00:00+02:00", "seconds": 7200},
        {"from": "2026-11-23T01:21:00+02:00", "to": "2026-11-23T02:21:00+02:00", "seconds": 3600},
    ]
    assert "(2026-11-22 23:00-2026-11-23 01:00, 01:21-02:21)" in sleep_note(
        intervals, action="backup continued after wake"
    )


def clock_pair():
    values = {"inclusive": 0, "awake": 0, "wall": dt.datetime.fromisoformat("2026-11-22T07:09:00+02:00")}
    tracker = SleepTracker(
        inclusive=lambda: values["inclusive"], awake=lambda: values["awake"], wall=lambda: values["wall"]
    )
    return values, tracker


def test_six_hour_sleep_has_one_interval_with_correct_bounds():
    values, tracker = clock_pair()
    values.update(inclusive=6 * 3600, wall=values["wall"] + dt.timedelta(hours=6))
    result = tracker.tick(prefix="upload", action="uploaded the rest after wake")
    assert result["upload_wall_seconds"] == result["upload_slept_seconds"] == 21600
    assert result["sleep_intervals"] == [
        {"from": "2026-11-22T07:09:00+02:00", "to": "2026-11-22T13:09:00+02:00", "seconds": 21600, "bounds": "inferred"}
    ]
    assert result["sleep_note"] == "paused 6 h 0 min while the Mac slept (07:09-13:09); uploaded the rest after wake"


def test_wall_clock_jump_is_never_detected_as_sleep():
    values, tracker = clock_pair()
    values.update(inclusive=30, awake=30, wall=values["wall"] + dt.timedelta(hours=6))
    result = tracker.tick(prefix="upload", action="uploaded the rest after wake")
    assert result["upload_wall_seconds"] == 30
    assert result["upload_slept_seconds"] == 0
    assert result["sleep_intervals"] == []
    assert "sleep_note" not in result


def test_no_sleep_omits_note_and_ticks_do_not_double_count():
    values, tracker = clock_pair()
    values.update(inclusive=21600, wall=values["wall"] + dt.timedelta(hours=6))
    tracker.tick(prefix="snapshot", action="snapshot continued after wake")
    values.update(inclusive=21630, awake=30, wall=values["wall"] + dt.timedelta(seconds=30))
    result = tracker.tick(prefix="snapshot", action="snapshot continued after wake")
    assert result["snapshot_slept_seconds"] == 21600
    assert len(result["sleep_intervals"]) == 1
    _, fresh = clock_pair()
    assert "sleep_note" not in fresh.tick(prefix="snapshot", action="snapshot continued after wake")
