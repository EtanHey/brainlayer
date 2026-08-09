"""The DEFERRED store receipt must read as SUCCESS so agents stop re-storing (Etan, 2026-08-09)."""

from brainlayer.mcp._format import format_store_result


def test_deferred_store_receipt_reads_as_success_and_forbids_restore() -> None:
    msg = format_store_result("brainbar-abc123", queued=True)
    assert "STORED (deferred)" in msg
    assert "DB busy" in msg
    assert "brainbar-abc123" in msg
    assert "durably queued" in msg
    assert "the drain persists it automatically" in msg
    assert "Do NOT re-store" in msg
    assert "fallback copy" in msg
    assert "DEFERRED:" not in msg  # old failure-sounding prefix retired


def test_deferred_store_receipt_carries_non_busy_reason() -> None:
    msg = format_store_result("brainbar-abc123", queued=True, queued_reason="INTERACTIVE_PRIORITY")
    assert "STORED (deferred): INTERACTIVE_PRIORITY" in msg
    assert "DB busy" not in msg
    assert "Do NOT re-store" in msg


def test_non_queued_store_receipt_unchanged() -> None:
    msg = format_store_result("brainbar-abc123")
    assert msg == "✔ Stored → brainbar-abc123"


def test_deferred_receipt_for_legacy_queue_schedules_replay(monkeypatch, tmp_path) -> None:
    """A queued_for_replay receipt must arm the background replay (codex P1, PR #693)."""
    from brainlayer.mcp import store_handler

    calls = []
    monkeypatch.setattr(store_handler, "_schedule_pending_store_replay", lambda: calls.append(True))

    store_handler._deferred_store_receipt("c1", tmp_path / "pending-stores.jsonl")
    assert calls == [True]

    store_handler._deferred_store_receipt("c2", tmp_path / "mcp-123.jsonl")
    assert calls == [True]  # drain-daemon path must NOT arm the replay


def test_schema_mismatch_deferral_names_its_reason(tmp_path) -> None:
    """Schema-fingerprint deferrals must not claim DB busy (codex P2, PR #693)."""
    msg = format_store_result("brainbar-x", queued=True, queued_reason="SCHEMA_FINGERPRINT_MISMATCH")
    assert "STORED (deferred): SCHEMA_FINGERPRINT_MISMATCH" in msg
    assert "DB busy" not in msg
