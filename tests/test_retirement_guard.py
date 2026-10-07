"""The early fixture's import boundary is independent of installed SDKs."""

import importlib
import json

import pytest

from scripts import retirement_guard as guard


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("RETIREMENT_EVENTS", str(events))
    monkeypatch.setenv("HOME", str(tmp_path))
    return events


@pytest.mark.parametrize(
    "name", ["google.genai", "brainlayer.enrichment_controller", "brainlayer.pipeline.groq", "openai.chat"]
)
def test_forbidden_import_records_even_if_caught(ledger, name):
    with pytest.raises(guard.Attempt):
        guard.check_import(name)
    assert json.loads(ledger.read_text())["kind"] == "model_or_retired_import"


def test_relative_import_identity_and_safe_local_import(ledger):
    with pytest.raises(guard.Attempt):
        guard.check_import(importlib.util.resolve_name(".enrichment_controller", "brainlayer"))
    guard.check_import("brainlayer.enrichment_replay")
    guard.check_import("google.auth")


def test_database_boundary_rejects_uri_escape(ledger):
    with pytest.raises(guard.Attempt, match="outside_fixture_db"):
        guard.check_db("file:/nonfixture/canonical.db?mode=ro")
    guard.check_db(ledger.parent / "fixture.db")
    guard.check_db(":memory:")
