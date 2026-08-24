"""Incident 2026-08-24: one unbounded query pegged a core for ~7 hours.

`brainbar_hybrid_helper` pid 1216 sat at 100% CPU with 46m12s of CPU burned.
Proof it was wedged rather than loaded: SIGTERM it, BrainBar respawns against the
SAME db and the SAME client traffic, and the fresh process measures 0% CPU.

`serve_forever()` handles one connection at a time and `_CONNECTION_TIMEOUT_SECONDS`
bounds only the SOCKET -- nothing bounds request PROCESSING. So a single pathological
query blocks every subsequent hybrid search for as long as it runs, which is forever.

These tests pin a hard request deadline that fails LOUDLY. It must not fail quietly:
returning empty results on timeout would silently degrade retrieval, which BrainLayer
forbids.
"""

from __future__ import annotations

import threading

from brainlayer.brainbar_hybrid_helper import HybridSearchHelper


def _helper(tmp_path, **kwargs):
    return HybridSearchHelper(socket_path=tmp_path / "helper.sock", db_path=tmp_path / "brain.db", **kwargs)


def test_request_deadline_is_configurable_and_has_a_default(tmp_path):
    helper = _helper(tmp_path)
    assert helper.request_deadline_seconds > 0, "a helper with no deadline can peg a core forever"

    tuned = _helper(tmp_path, request_deadline_seconds=12.5)
    assert tuned.request_deadline_seconds == 12.5


def test_slow_request_trips_the_deadline(tmp_path):
    """A request that outruns the deadline must trigger the abort hook."""
    tripped = threading.Event()
    helper = _helper(tmp_path, request_deadline_seconds=0.05, on_deadline_expired=lambda _info: tripped.set())

    started = threading.Event()

    def slow_request(_request):
        started.set()
        tripped.wait(timeout=5.0)
        return {"ok": True}

    helper._handle_request = slow_request  # type: ignore[method-assign]
    helper._dispatch_with_deadline({"method": "brain_search", "arguments": {"query": "x"}})

    assert started.is_set()
    assert tripped.is_set(), "deadline expired but the abort hook never fired"


def test_fast_request_does_not_trip_the_deadline(tmp_path):
    """The common case must be untouched -- no false positives."""
    tripped = threading.Event()
    helper = _helper(tmp_path, request_deadline_seconds=5.0, on_deadline_expired=lambda _info: tripped.set())

    helper._handle_request = lambda _request: {"ok": True, "text": "fast"}  # type: ignore[method-assign]
    response = helper._dispatch_with_deadline({"method": "brain_search", "arguments": {"query": "x"}})

    assert response == {"ok": True, "text": "fast"}
    assert not tripped.is_set(), "a fast request must never trip the deadline"


def test_deadline_hook_receives_enough_to_diagnose_without_the_raw_query(tmp_path):
    """When a query wedges the helper we must be able to characterise it afterwards.

    The 2026-08-24 wedge was unattributable because the process died with no record of
    what it was running. But the record must be a DESCRIPTOR, not the text: the helper's
    stderr lands in brainbar.err.log, so the raw query would become a second plaintext
    copy of personal memory (PR #735 review, codex P2).
    """
    seen: list[dict] = []
    helper = _helper(tmp_path, request_deadline_seconds=0.05, on_deadline_expired=seen.append)

    done = threading.Event()

    def slow_request(_request):
        done.wait(timeout=5.0)
        return {"ok": True}

    helper._handle_request = slow_request  # type: ignore[method-assign]
    helper._dispatch_with_deadline({"method": "brain_search", "arguments": {"query": "the wedging query"}})
    done.set()

    assert seen, "abort hook never fired"
    rendered = str(seen[0])
    assert "the wedging query" not in rendered, f"raw query must not reach the hook: {rendered}"
    assert "brain_search" in rendered, "must still identify the method"
    assert "query_sha256" in rendered, "must still carry a correlatable fingerprint"
    assert seen[0].get("elapsed_seconds") is not None


def test_deadline_log_does_not_leak_the_raw_query(caplog):
    """Review finding (codex P2, PR #735): the helper inherits BrainBar's stderr, which
    launchd redirects to brainbar.err.log. Logging the full request would write a
    plaintext copy of potentially personal or secret-bearing memory to a second file.

    The hook still needs enough to DIAGNOSE the wedge, so a bounded descriptor is kept.
    """
    import logging

    from brainlayer.brainbar_hybrid_helper import _describe_request_for_log

    secret = "my private therapy notes about SECRET_TOKEN_abc123"
    described = _describe_request_for_log(
        {"method": "brain_search", "arguments": {"query": secret, "project": "brainlayer", "num_results": 5}}
    )
    rendered = str(described)

    assert secret not in rendered, f"raw query leaked into the log payload: {rendered}"
    assert "SECRET_TOKEN_abc123" not in rendered
    # still diagnosable
    assert "brain_search" in rendered
    assert "query_chars" in rendered, "must keep enough shape to diagnose the wedge"
    assert "query_sha256" in rendered, "a stable fingerprint lets repeat wedges be correlated"
    # non-sensitive structural fields are useful and safe
    assert "brainlayer" in rendered or "project" in rendered
    del caplog, logging
