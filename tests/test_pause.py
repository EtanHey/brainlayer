"""Shared pause interpretation keeps strict control and legacy readers consistent."""

import json
from datetime import UTC, datetime

import pytest

from brainlayer.pause import pause_applies_to_label, pause_sentinel_state


@pytest.mark.parametrize("payload", [{}, {"reason": "done"}, {"labels": []}])
def test_valid_label_free_pause_never_holds_a_writer(tmp_path, payload):
    path = tmp_path / "pause"
    path.write_text(json.dumps(payload))
    for strict in (False, True):
        observed, _, _ = pause_sentinel_state(path, strict=strict)
        assert not pause_applies_to_label(observed, "com.brainlayer.watch")


@pytest.mark.parametrize("raw", [b"{", b"null", b"\xff"])
def test_strict_control_refuses_corrupt_pause_and_legacy_reader_remains_compatible(tmp_path, raw):
    path = tmp_path / "pause"
    path.write_bytes(raw)
    assert pause_sentinel_state(path) == ({}, False, False)
    with pytest.raises((ValueError, UnicodeDecodeError)):
        pause_sentinel_state(path, strict=True)


def test_missing_pause_is_clear_but_dangling_link_is_unreadable(tmp_path):
    path = tmp_path / "pause"
    assert pause_sentinel_state(path, strict=True) == ({}, False, False)
    path.symlink_to(tmp_path / "missing")
    with pytest.raises(OSError, match="unreadable"):
        pause_sentinel_state(path, strict=True)


def test_expiry_and_label_scope_are_shared(tmp_path):
    path = tmp_path / "pause"
    path.write_text(json.dumps({"labels": ["watch"], "expires_at": "2026-10-12T00:00:00Z"}))
    for strict in (False, True):
        payload, active, stale = pause_sentinel_state(path, datetime(2026, 10, 13, tzinfo=UTC), strict=strict)
        assert not active and stale
        assert pause_applies_to_label(payload, "watch") and not pause_applies_to_label(payload, "drain")
