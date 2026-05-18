"""Shared test fixtures for BrainLayer tests."""

import os
import uuid

import pytest


def pytest_configure(config):
    """Register custom pytest marks."""
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring a live production DB (skipped in CI if DB absent)",
    )


@pytest.fixture
def eval_project() -> str:
    """Return a unique project name for each eval test case.

    Prevents cross-case data contamination when eval tests seed brain_store
    chunks. Each test invocation gets its own project namespace.
    """
    return f"eval-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def disable_live_gemini_for_unit_tests(monkeypatch, request):
    """Keep unit tests from making live Gemini calls through local shell env."""
    if request.node.get_closest_marker("live"):
        return

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENERATIVE_AI_API_KEY", raising=False)


@pytest.fixture
def test_user() -> str:
    """Username for path-based tests.

    Defaults to 'janedev' (safe for CI/commits).
    Set BRAINLAYER_TEST_USER to your real username for local filesystem tests.
    """
    return os.environ.get("BRAINLAYER_TEST_USER", "janedev")
