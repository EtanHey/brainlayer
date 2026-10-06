"""Caught cloud construction after a retirement response must still fail pytest."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_brain_enrich_handler_cold_bootstrap():
    """Import the retired handler only after its factory guard has initialized."""
    from brainlayer.mcp.enrich_handler import _brain_enrich

    result = await _brain_enrich(stats=True)
    assert result.is_error is True
    assert "Enrichment has been retired" in result.content[0].text


@pytest.mark.parametrize(
    "target",
    [
        "tests/test_cli_enrich.py::test_retired_enrich_is_hidden_and_does_not_resolve_database",
        "tests/test_retirement_factory_guard.py::test_brain_enrich_handler_cold_bootstrap",
    ],
)
def test_retirement_guard_works_without_collection_import_side_effects(target):
    """A standalone retired test must arm after synthetic runtime isolation."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
        "precreated_httpx.send(httpx.Request('POST', 'http://retirement.invalid'))",
        "send_precreated_async()",
        "precreated_requests.send(requests.Request('POST', 'http://retirement.invalid').prepare())",
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
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor

class Mutation:
    def pytest_collection_finish(self, session):
        # Conftest isolates config before these imports. Capture SDK aliases
        # before the guard is armed, and make an absent guard network-safe.
        global Client, AsyncClient, controller, enrichment, precreated_httpx, precreated_requests, send_precreated_async
        from google.genai.client import Client, AsyncClient
        from brainlayer import enrichment_controller as controller
        from brainlayer.pipeline import enrichment
        enrichment.GROQ_API_KEY = ''
        enrichment.ENRICH_BACKEND = 'groq'
        if {"precreated" in attempt!r}:
            transport = httpx.MockTransport(lambda req: httpx.Response(200, request=req))
            precreated_httpx = httpx.Client(transport=transport, trust_env=False)
            precreated_async = httpx.AsyncClient(transport=transport, trust_env=False)
            precreated_requests = requests.Session()
            precreated_requests.trust_env = False
            class OfflineAdapter(requests.adapters.BaseAdapter):
                def send(self, request, **kwargs):
                    response = requests.Response()
                    response.status_code = 200
                    response._content = b'{{}}'
                    response.request = request
                    return response
                def close(self):
                    pass
            precreated_requests.mount('http://', OfflineAdapter())
            def send_precreated_async():
                # Also works when the MCP test already owns an event loop.
                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(lambda: asyncio.run(precreated_async.send(
                        httpx.Request('POST', 'http://retirement.invalid')))).result()
            self.clients = (precreated_httpx, precreated_async, precreated_requests)

    def pytest_sessionfinish(self, session, exitstatus):
        if hasattr(self, 'clients'):
            self.clients[0].close()
            asyncio.run(self.clients[1].aclose())
            self.clients[2].close()

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
