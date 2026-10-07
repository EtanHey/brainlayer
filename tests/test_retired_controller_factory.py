"""Retirement closes the last SDK factory before any legacy entrypoint can write."""

from unittest.mock import MagicMock

import pytest


def test_controller_sdk_factory_is_absent():
    from brainlayer import enrichment_controller

    assert not hasattr(enrichment_controller, "_get_gemini_client")


@pytest.mark.parametrize(
    "name", ["enrich_realtime", "enrich_single", "run_enrich_supervisor", "call_gemini_for_extraction"]
)
def test_controller_legacy_entrypoints_retire_before_using_arguments(name):
    from brainlayer import enrichment_controller

    store = MagicMock(side_effect=AssertionError("retired producer used an argument"))
    args = (store, "synthetic") if name == "enrich_single" else (store,)
    with pytest.raises(RuntimeError, match="enrichment has been retired"):
        getattr(enrichment_controller, name)(
            *args, **({"max_cycles": 1, "sleep_fn": lambda _: None} if name == "run_enrich_supervisor" else {})
        )
    assert store.mock_calls == []
