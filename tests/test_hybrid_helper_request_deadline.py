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


def test_deadline_hook_receives_the_query_for_diagnosis(tmp_path):
    """When a query wedges the helper we must be able to name it afterwards.

    The 2026-08-24 wedge was unattributable because the process died with no record
    of what it was running.
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
    assert "the wedging query" in str(seen[0]), f"hook must carry the query text: {seen[0]}"
    assert seen[0].get("elapsed_seconds") is not None
