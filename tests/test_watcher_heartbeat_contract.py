from copy import deepcopy

import pytest

from scripts.watcher_heartbeat_contract import validate_completion, validate_stores


def test_serialized_completion_requires_advancing_aware_clocks():
    completed = dict(
        poll_count=3,
        actual_poll_count=3,
        previous_poll_count=2,
        updated_at="2026-10-07T10:00:02+00:00",
        previous_updated_at="2026-10-07T10:00:00+00:00",
        completion_started_at="2026-10-07T10:00:01+00:00",
        completion_finished_at="2026-10-07T10:00:03+00:00",
    )
    validate_completion(completed, 2)
    for key, value in [("poll_count", 2), ("updated_at", "2026-10-07T10:00:02")]:
        changed = deepcopy(completed)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_completion(changed, 2)


def test_discarded_independent_stores_are_not_evidence():
    with pytest.raises(ValueError):
        validate_stores([])
