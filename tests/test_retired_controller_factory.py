"""The retired controller host is absent; local replay remains importable."""

import importlib.util

import pytest

pytestmark = pytest.mark.retired_enrichment


def test_retired_controller_host_is_absent():
    assert importlib.util.find_spec("brainlayer.enrichment_controller") is None
    from brainlayer.enrichment_replay import _apply_enrichment, _apply_enrichment_impl

    assert callable(_apply_enrichment)
    assert callable(_apply_enrichment_impl)
