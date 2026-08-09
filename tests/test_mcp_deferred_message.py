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
