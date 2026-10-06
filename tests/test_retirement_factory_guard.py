"""Caught cloud construction after a retirement response must still fail pytest."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("surface", ["cli", "mcp"])
@pytest.mark.parametrize(
    "attempt",
    [
        "controller._get_gemini_client()",
        "enrichment.call_llm('synthetic prompt')",
        "Client(api_key='synthetic-key')",
        "AsyncClient(api_client=None)",
        "httpx.Client()",
        "httpx.AsyncClient()",
    ],
)
def test_caught_cloud_attempt_fails_retirement_test(surface, attempt):
    """Run the real R1 tests with the reviewer's swallowed-exception mutation."""
    target = (
        "tests/test_cli_enrich.py::test_retired_enrich_is_hidden_and_does_not_resolve_database"
        if surface == "cli"
        else "tests/test_enrichment_controller.py::test_brain_enrich_handler_stats_is_retired_without_reading_metadata"
    )
    probe = f"""
import pytest
import httpx

class Mutation:
    def pytest_collection_finish(self, session):
        # Conftest isolates config before these imports. Capture SDK aliases
        # before the guard is armed, and make an absent guard network-safe.
        global Client, AsyncClient, controller, enrichment
        from google.genai.client import Client, AsyncClient
        from brainlayer import enrichment_controller as controller
        from brainlayer.pipeline import enrichment
        enrichment.GROQ_API_KEY = ''
        enrichment.ENRICH_BACKEND = 'groq'

    def pytest_runtest_call(self, item):
        monkeypatch = item.funcargs['monkeypatch']
        if {surface!r} == 'cli':
            from brainlayer import cli as owner
            symbol = 'rprint'
        else:
            from brainlayer.mcp import enrich_handler as owner
            symbol = '_error_result'
        original = getattr(owner, symbol)
        def after_response(*args, **kwargs):
            response = original(*args, **kwargs)
            try:
                {attempt}
            except Exception:
                pass
            return response
        monkeypatch.setattr(owner, symbol, after_response)

raise SystemExit(pytest.main([{target!r}, '-q'], plugins=[Mutation()]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert "Cloud construction/send attempted in retirement test" in output, output
