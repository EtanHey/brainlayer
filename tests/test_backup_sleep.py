import datetime as dt

from brainlayer.backup_sleep import SleepTracker


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
        {"from": "2026-11-22T07:09:00+02:00", "to": "2026-11-22T13:09:00+02:00", "seconds": 21600}
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
