"""The DEFERRED store receipt must read as SUCCESS so agents stop re-storing (Etan, 2026-08-09)."""

from brainlayer.mcp._format import format_store_result


def test_deferred_store_receipt_reads_as_success_and_forbids_restore() -> None:
    msg = format_store_result("brainbar-abc123", queued=True)
    assert "STORED (deferred)" in msg
    assert "brainbar-abc123" in msg
    assert "Do NOT re-store" in msg
    assert "fallback copy" in msg
    assert "DEFERRED:" not in msg  # old failure-sounding prefix retired


def test_non_queued_store_receipt_unchanged() -> None:
    msg = format_store_result("brainbar-abc123")
    assert msg.startswith("✔ Stored → ")
