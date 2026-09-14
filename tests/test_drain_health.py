from __future__ import annotations

import json
import logging

import pytest

from brainlayer.drain import _configure_daemon_logging, run_daemon

# Repair (c): rewind archive writes archived_at only. See also test_rewind_batch_archival.py.


def test_run_daemon_writes_progress_heartbeat(tmp_path):
    health_path = tmp_path / "drain-health.json"
    drained_values = iter([2, 0])

    run_daemon(
        interval=0,
        batch_size=10,
        health_path=health_path,
        drain_once_fn=lambda **_kwargs: next(drained_values),
        sleep_fn=lambda _seconds: None,
        max_cycles=2,
        # Explicit no-op, not inherited hermeticity. The default seam runs the real filesystem sweep,
        # and `BRAINLAYER_FALLBACK_GITS_ROOT` is an ABSOLUTE path that bypasses conftest's HOME
        # remapping -- with it set, this heartbeat test could write production fallback markers.
        replay_fallbacks_fn=lambda: None,
    )

    payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert payload["drain_cycles"] == 2
    assert payload["drained_total"] == 2
    assert payload["updated_at"]


def test_drain_daemon_rotates_oversized_error_log_at_start(tmp_path):
    log_path = tmp_path / "drain.err.log"
    log_path.write_text("x" * 256, encoding="utf-8")

    handler = _configure_daemon_logging(
        log_path=log_path,
        max_bytes=128,
        backup_count=2,
        configure_root=False,
    )
    try:
        handler.emit(logging.LogRecord("brainlayer.drain", logging.WARNING, __file__, 1, "after rotation", (), None))
    finally:
        handler.close()

    assert log_path.read_text(encoding="utf-8").strip() == "after rotation"
    assert log_path.with_name("drain.err.log.1").stat().st_size == 256


def _run_test_daemon(tmp_path, drain_once_fn, **kwargs):
    run_daemon(
        batch_size=10,
        health_path=tmp_path / "drain-health.json",
        drain_once_fn=drain_once_fn,
        replay_fallbacks_fn=lambda: None,
        **kwargs,
    )


def test_run_daemon_survives_failed_cycle_and_rate_limits_errors(tmp_path, caplog):
    outcomes = iter([RuntimeError("one bad cycle"), 2, RuntimeError("still failing")])

    def flaky_drain(**_kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with caplog.at_level(logging.ERROR, logger="brainlayer.drain"):
        _run_test_daemon(
            tmp_path,
            flaky_drain,
            interval=0.25,
            sleep_fn=lambda _seconds: None,
            max_cycles=3,
        )

    payload = json.loads((tmp_path / "drain-health.json").read_text(encoding="utf-8"))
    assert (payload["drain_cycles"], payload["drained_total"]) == (3, 2)
    error_records = [record for record in caplog.records if "drain cycle failed" in record.message]
    assert len(error_records) == 1
    assert error_records[0].exc_info is not None


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit])
def test_run_daemon_does_not_swallow_process_control(signal, tmp_path):
    def stop_drain(**_kwargs):
        raise signal()

    with pytest.raises(signal):
        _run_test_daemon(
            tmp_path,
            stop_drain,
            interval=0,
            sleep_fn=lambda _seconds: None,
            max_cycles=1,
        )
