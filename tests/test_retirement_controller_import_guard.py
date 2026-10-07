"""The local-path backstop rejects retired imports through both Python loaders."""

import builtins
import importlib

import pytest

from tests.retirement_helpers import forbid_enrichment_controller


@pytest.mark.parametrize("loader", ["module", "fromlist", "dynamic"])
def test_local_path_import_backstop_rejects_controller(monkeypatch, loader):
    forbid_enrichment_controller(monkeypatch)
    with pytest.raises(AssertionError, match="retired enrichment controller"):
        if loader == "module":
            builtins.__import__("brainlayer.enrichment_controller")
        elif loader == "fromlist":
            builtins.__import__("brainlayer", fromlist=("enrichment_controller",))
        else:
            importlib.import_module("brainlayer.enrichment_controller")
    assert importlib.import_module("brainlayer.enrichment_replay")._apply_enrichment is not None
