from datetime import UTC, datetime

import pytest

import brainlayer.backfill as backfill
from brainlayer.backfill import WindowedFlush, is_legacy_excluded_path, parse_backfill_window, window_registry_suffix
from brainlayer.watcher import normalize_provider_entry
from brainlayer.watcher_bridge import FlushWatermarks


def _entry(timestamp: str, offset: int) -> dict:
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "_source_file": "/tmp/session.jsonl",
        "_line_end_offset": offset,
    }


def test_windowed_flush_filters_half_open_interval_and_confirms_scanned_offsets():
    received = []

    def downstream(entries):
        received.extend(entries)
        return FlushWatermarks(
            {"/tmp/session.jsonl": entries[-1]["_line_end_offset"]},
            inserted=len(entries),
        )

    windowed = WindowedFlush(
        downstream,
        since=datetime(2026, 7, 10, tzinfo=UTC),
        until=datetime(2026, 7, 16, tzinfo=UTC),
    )

    result = windowed(
        [
            _entry("2026-07-09T23:59:59Z", 100),
            _entry("2026-07-10T00:00:00Z", 200),
            _entry("2026-07-15T23:59:59Z", 300),
            _entry("2026-07-16T00:00:00Z", 400),
        ]
    )

    assert [entry["_line_end_offset"] for entry in received] == [200, 300]
    assert result == {"/tmp/session.jsonl": 400}
    assert result.inserted == 2
    assert result.skipped == 2
    assert windowed.scanned_entries == 4
    assert windowed.matched_entries == 2
    assert windowed.inserted_chunks == 2


def test_windowed_flush_excludes_invalid_timestamps_but_confirms_them():
    windowed = WindowedFlush(
        lambda entries: FlushWatermarks(inserted=len(entries)),
        since=datetime(2026, 7, 10, tzinfo=UTC),
        until=datetime(2026, 7, 16, tzinfo=UTC),
    )

    result = windowed([_entry("not-a-date", 50)])

    assert result == {"/tmp/session.jsonl": 50}
    assert windowed.matched_entries == 0


def test_windowed_flush_excludes_overflowing_timezone_timestamps():
    windowed = WindowedFlush(
        lambda entries: FlushWatermarks(inserted=len(entries)),
        since=datetime(2026, 7, 10, tzinfo=UTC),
        until=datetime(2026, 7, 16, tzinfo=UTC),
    )

    result = windowed([_entry("0001-01-01T00:00:00+23:59", 50)])

    assert result == {"/tmp/session.jsonl": 50}
    assert windowed.matched_entries == 0


def test_windowed_flush_rejects_timestamps_synthesized_during_normalization():
    normalized = normalize_provider_entry(
        {"role": "user", "content": "historical undated transcript"},
        "codex",
    )
    assert normalized is not None
    normalized.update(
        {
            "_source_file": "/tmp/session.jsonl",
            "_line_end_offset": 50,
        }
    )
    windowed = WindowedFlush(
        lambda entries: FlushWatermarks(inserted=len(entries)),
        since=datetime(2020, 1, 1, tzinfo=UTC),
        until=datetime(2100, 1, 1, tzinfo=UTC),
    )

    result = windowed([normalized])

    assert result == {"/tmp/session.jsonl": 50}
    assert windowed.matched_entries == 0


def test_windowed_flush_does_not_confirm_offsets_when_downstream_returns_none():
    windowed = WindowedFlush(
        lambda _entries: None,
        since=datetime(2026, 7, 10, tzinfo=UTC),
        until=datetime(2026, 7, 16, tzinfo=UTC),
    )

    result = windowed([_entry("2026-07-12T00:00:00Z", 50)])

    assert result is None
    assert windowed.scanned_entries == 1
    assert windowed.matched_entries == 1
    assert windowed.inserted_chunks == 0


def test_windowed_flush_stops_before_unconfirmed_matched_entry():
    windowed = WindowedFlush(
        lambda _entries: FlushWatermarks(inserted=0),
        since=datetime(2026, 7, 10, tzinfo=UTC),
        until=datetime(2026, 7, 16, tzinfo=UTC),
    )

    result = windowed(
        [
            _entry("2026-07-09T23:59:59Z", 100),
            _entry("2026-07-12T00:00:00Z", 200),
            _entry("2026-07-16T00:00:00Z", 300),
        ]
    )

    assert result == {"/tmp/session.jsonl": 100}


def test_windowed_flush_advances_exclusions_after_confirmed_match_until_next_match():
    windowed = WindowedFlush(
        lambda _entries: FlushWatermarks({"/tmp/session.jsonl": 200}, inserted=1),
        since=datetime(2026, 7, 10, tzinfo=UTC),
        until=datetime(2026, 7, 16, tzinfo=UTC),
    )

    result = windowed(
        [
            _entry("2026-07-12T00:00:00Z", 200),
            _entry("2026-07-16T00:00:00Z", 300),
            _entry("2026-07-13T00:00:00Z", 400),
            _entry("2026-07-16T00:00:00Z", 500),
        ]
    )

    assert result == {"/tmp/session.jsonl": 300}


def test_normalize_provider_entry_uses_valid_created_at_for_canonical_entry():
    normalized = normalize_provider_entry(
        {
            "type": "assistant",
            "timestamp": {"invalid": True},
            "created_at": "2026-07-12T12:00:00Z",
            "message": {"role": "assistant", "content": "durable decision"},
        },
        "claude",
    )

    assert normalized is not None
    assert normalized["timestamp"] == "2026-07-12T12:00:00Z"
    assert normalized["_timestamp_synthesized"] is False


def test_normalize_provider_entry_falls_back_from_malformed_timestamp_to_created_at():
    normalized = normalize_provider_entry(
        {
            "type": "assistant",
            "timestamp": "not-an-iso-timestamp",
            "created_at": "2026-07-12T12:00:00Z",
            "message": {"role": "assistant", "content": "durable decision"},
        },
        "claude",
    )

    assert normalized is not None
    assert normalized["timestamp"] == "2026-07-12T12:00:00Z"
    assert normalized["_timestamp_synthesized"] is False


def test_window_registry_suffix_preserves_fractional_seconds_without_changing_whole_seconds():
    whole_since = datetime(2026, 7, 10, tzinfo=UTC)
    whole_until = datetime(2026, 7, 16, tzinfo=UTC)
    first = window_registry_suffix(
        whole_since.replace(microsecond=100_000),
        whole_until.replace(microsecond=200_000),
    )
    second = window_registry_suffix(
        whole_since.replace(microsecond=300_000),
        whole_until.replace(microsecond=400_000),
    )

    assert window_registry_suffix(whole_since, whole_until) == "20260710T000000Z-20260716T000000Z"
    assert first != second


def test_parse_backfill_window_is_utc_and_rejects_empty_or_reversed_ranges():
    since, until = parse_backfill_window("2026-07-10", "2026-07-16")

    assert since == datetime(2026, 7, 10, tzinfo=UTC)
    assert until == datetime(2026, 7, 16, tzinfo=UTC)

    for invalid in ((None, "2026-07-16"), ("2026-07-16", None), ("2026-07-16", "2026-07-10")):
        try:
            parse_backfill_window(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid window: {invalid}")


def test_legacy_excluded_path_selects_only_roots_blocked_by_old_policy(tmp_path):
    assert is_legacy_excluded_path(tmp_path / ".codex" / "sessions" / "worker.jsonl")
    assert is_legacy_excluded_path(
        tmp_path / ".cursor" / "projects" / "repo" / "agent-transcripts" / "session" / "worker.jsonl"
    )
    assert is_legacy_excluded_path(tmp_path / ".gemini" / "sessions" / "worker.jsonl")
    assert is_legacy_excluded_path(
        tmp_path / ".claude" / "projects" / "repo" / "session" / "subagents" / "agent-worker.jsonl"
    )
    assert not is_legacy_excluded_path(tmp_path / ".claude" / "projects" / "repo" / "direct.jsonl")
    assert not is_legacy_excluded_path(tmp_path / ".cursor" / "projects" / "repo" / "state.jsonl")
    assert not is_legacy_excluded_path(tmp_path / ".codex" / "archive" / "sessions" / "worker.jsonl")
    assert not is_legacy_excluded_path(
        tmp_path / ".claude" / "cache" / "projects" / "repo" / "subagents" / "worker.jsonl"
    )


def test_backfill_run_lock_rejects_concurrent_registry_owner(tmp_path):
    registry = tmp_path / "offsets.json"

    with backfill.backfill_run_lock(registry):
        with pytest.raises(backfill.BackfillAlreadyRunning):
            with backfill.backfill_run_lock(registry):
                raise AssertionError("concurrent backfill unexpectedly acquired the run lock")
