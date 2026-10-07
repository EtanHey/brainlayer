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


@pytest.mark.parametrize("surface", ["cli", "mcp", "palette", "marked"])
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
        "http.client.HTTPConnection('127.0.0.1', 9).request('GET', '/')",
        "urllib3.PoolManager(retries=False).request('GET', 'http://127.0.0.1:9/')",
        "requests.adapters.HTTPAdapter().send(requests.Request('GET', 'http://127.0.0.1:9/').prepare())",
    ],
)
def test_caught_cloud_attempt_fails_retirement_test(surface, attempt):
    """Run the real R1 tests with the reviewer's swallowed-exception mutation."""
    target = (
        "tests/test_retirement_factory_guard.py::test_brain_enrich_handler_cold_bootstrap"
        if surface in {"marked", "unmarked"}
        else "tests/test_cli_enrich.py::test_retired_enrich_is_hidden_and_does_not_resolve_database"
        if surface == "cli"
        else "tests/test_mcp_palette.py::test_retired_enrich_is_neither_advertised_nor_dispatched[core]"
        if surface == "palette"
        else "tests/test_retirement_factory_guard.py::test_brain_enrich_handler_cold_bootstrap"
    )
    from importlib import import_module

    def optional_module(name):
        try:
            return import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and (name == exc.name or name.startswith(exc.name + ".")):
                return None
            raise

    targets = {
        "controller._get_gemini_client()": ("brainlayer.enrichment_controller", "_get_gemini_client"),
        "enrichment.call_llm('synthetic prompt')": ("brainlayer.pipeline.enrichment", "call_llm"),
        "Client(api_key='synthetic-key')": ("google.genai.client", "Client"),
        "AsyncClient(api_client=None)": ("google.genai.client", "AsyncClient"),
    }
    if attempt in targets:
        module_name, symbol = targets[attempt]
        module = optional_module(module_name)
        if module is None or not hasattr(module, symbol):
            pytest.skip(f"Retired optional mutation target absent: {module_name}.{symbol}")

    probe = f"""
import pytest
import importlib
import httpx
import requests
import asyncio
import http.client
import urllib3
from concurrent.futures import ThreadPoolExecutor

class Mutation:
    def pytest_collection_finish(self, session):
        if {surface!r} in ('marked', 'unmarked'):
            for item in session.items:
                item.name = 'test_retirement_marker_target'
                if {surface!r} == 'marked':
                    item.add_marker('retired_enrichment')
        # Conftest isolates config before these imports. Capture SDK aliases
        # before the guard is armed, and make an absent guard network-safe.
        global Client, AsyncClient, controller, enrichment, precreated_httpx, precreated_requests, send_precreated_async
        def optional(name):
            try:
                return importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if exc.name and (name == exc.name or name.startswith(exc.name + '.')):
                    return None
                raise
        sdk = optional('google.genai.client')
        Client = sdk.Client if sdk else None
        AsyncClient = sdk.AsyncClient if sdk else None
        controller = optional('brainlayer.enrichment_controller')
        enrichment = optional('brainlayer.pipeline.enrichment')
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
        elif {surface!r} == 'palette':
            import brainlayer.mcp as owner
            symbol = '_error_result'
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

raise SystemExit(pytest.main([{target!r}, '-q', '-s'], plugins=[Mutation()]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == (0 if surface == "unmarked" else 1), output
    expected = "UNMARKED_OK" if surface == "unmarked" else "Cloud construction/send attempted in retirement test"
    assert expected in output, output


def test_unmarked_retirement_target_keeps_offline_transport():
    test_caught_cloud_attempt_fails_retirement_test(
        "unmarked",
        "assert precreated_httpx.send(httpx.Request('POST', 'http://retirement.invalid')).status_code == 200;"
        " print('UNMARKED_OK')",
    )


@pytest.mark.parametrize(
    "missing", ["google.genai.client", "brainlayer.enrichment_controller", "brainlayer.pipeline.enrichment"]
)
@pytest.mark.parametrize("attempt", [False, True])
def test_guard_arms_with_patch_target_missing(missing, attempt):
    """An absent optional sender cannot disarm the network backstop."""
    probe = f"""
import pytest
import socket
import sys
class MissingTarget:
    def pytest_collection_finish(self, session):
        sys.modules[{missing!r}] = None
    def pytest_runtest_call(self, item):
        if {attempt!r}:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.connect(('127.0.0.1', 9))
                except Exception:
                    pass
raise SystemExit(pytest.main([
    'tests/test_retirement_factory_guard.py::test_brain_enrich_handler_cold_bootstrap', '-q'
], plugins=[MissingTarget()]))
"""
    completed = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    output = completed.stdout + completed.stderr
    assert completed.returncode == (1 if attempt else 0), output
    if attempt:
        assert "Cloud construction/send attempted in retirement test" in output, output


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
@pytest.mark.parametrize("family", ["AF_INET", "AF_INET6"])
def test_socket_backstop_catches_both_inet_families(method, family):
    probe = f"""
import pytest
import socket
class InetAttempt:
    def pytest_runtest_call(self, item):
        with socket.socket(socket.{family}, socket.SOCK_STREAM) as sock:
            try:
                sock.{method}({("127.0.0.1", 9) if family == "AF_INET" else ("::1", 9)!r})
            except Exception:
                pass
raise SystemExit(pytest.main([
    'tests/test_retirement_factory_guard.py::test_brain_enrich_handler_cold_bootstrap', '-q'
], plugins=[InetAttempt()]))
"""
    completed = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert "Cloud construction/send attempted in retirement test" in output, output


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_brain_enrich_handler_unix_socket_backstop_passthrough(method):
    """Private AF_UNIX traffic retains the existing socket guard contract."""
    import socket
    import tempfile

    with tempfile.TemporaryDirectory(prefix="bl-guard-") as scratch:
        address = str(Path(scratch) / "private.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(address)
            server.listen(1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                result = getattr(client, method)(address)
                assert result in (None, 0)
                with server.accept()[0] as accepted:
                    client.sendall(b"synthetic")
                    assert accepted.recv(9) == b"synthetic"
