"""Re-export surface and canonical timestamp conversion."""

from brainlayer.chunk_origin_wipe import live_canonical_db_path as wipe_live_path
from brainlayer.timestamp_iso import is_iso_utc, live_canonical_db_path, normalize_timestamp


def test_live_canonical_db_path_is_reexported():
    assert live_canonical_db_path is wipe_live_path


def test_normalize_timestamp_appends_z_to_naive_iso():
    converted = normalize_timestamp("2026-07-14T11:45:26.994441")
    assert converted == "2026-07-14T11:45:26.994441Z"
    assert is_iso_utc(converted)


def test_normalize_timestamp_returns_none_for_garbage_iso():
    assert normalize_timestamp("2026-05-28T~12:35:00Z") is None
    assert not is_iso_utc("2026-05-28T~12:35:00Z")


def test_normalize_timestamp_converts_non_utc_offset_to_z():
    converted = normalize_timestamp("2026-06-09T13:37:11+03:00")
    assert converted == "2026-06-09T10:37:11Z"
    assert is_iso_utc(converted)
    assert not is_iso_utc("2026-06-09T13:37:11+03:00")
