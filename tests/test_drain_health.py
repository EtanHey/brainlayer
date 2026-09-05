from __future__ import annotations

import json

from brainlayer.drain import run_daemon

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
