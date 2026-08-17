"""Rewind archive writes archived_at only. Behavioral coverage: test_rewind_batch_archival.py."""

from brainlayer.drain import _apply_rewind_archive


def test_drain_rewind_archive_helper_exists():
    assert callable(_apply_rewind_archive)
