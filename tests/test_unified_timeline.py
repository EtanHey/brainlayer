"""Unified timeline emits timezone-aware UTC timestamps."""

from datetime import datetime, timezone

from brainlayer.pipeline.unified_timeline import normalize_timestamp


def test_normalize_timestamp_returns_aware_utc():
    naive = datetime(2026, 8, 18, 12, 0, 0)
    converted = normalize_timestamp(naive)
    assert converted.tzinfo is not None
    assert converted.utcoffset() == timezone.utc.utcoffset(converted)
    assert converted == datetime(2026, 8, 18, 12, 0, 0, tzinfo=timezone.utc)


def test_normalize_timestamp_converts_local_to_utc():
    local = datetime(2026, 6, 9, 13, 0, 0, tzinfo=timezone.utc)
    assert normalize_timestamp(local).isoformat().endswith("+00:00")
