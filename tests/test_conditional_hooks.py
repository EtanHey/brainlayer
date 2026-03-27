"""Tests for BrainLayer hook conditional activation (CC 2.1.85).

Validates the should_activate() gate in all three hook scripts:
- brainlayer-session-start.py
- brainlayer-prompt-search.py
- brainbar-stop-index.py

Env vars tested:
- BRAINLAYER_HOOKS_DISABLED=1 → skip all hooks
- CLAUDE_NON_INTERACTIVE=1 → skip (--print mode)
- BRAINLAYER_HOOKS_LIGHT=1 → reduced results for workers
"""

import importlib.util
import os
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def load_hook_module(filename):
    """Import a hook script as a module."""
    spec = importlib.util.spec_from_file_location(
        filename.replace("-", "_").replace(".py", ""),
        HOOKS_DIR / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def session_start():
    return load_hook_module("brainlayer-session-start.py")


@pytest.fixture
def prompt_search():
    return load_hook_module("brainlayer-prompt-search.py")


@pytest.fixture
def stop_index():
    return load_hook_module("brainbar-stop-index.py")


@pytest.fixture(autouse=True)
def clean_env():
    """Ensure relevant env vars are unset before each test."""
    env_vars = [
        "BRAINLAYER_HOOKS_DISABLED",
        "CLAUDE_NON_INTERACTIVE",
        "BRAINLAYER_HOOKS_LIGHT",
    ]
    saved = {k: os.environ.pop(k, None) for k in env_vars}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


class TestSessionStartConditional:
    def test_default_activates(self, session_start):
        hook_input = {"session_id": "abc123", "cwd": "/tmp/test"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is True
        assert light is False

    def test_disabled_env_var(self, session_start):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is False

    def test_non_interactive_skips(self, session_start):
        os.environ["CLAUDE_NON_INTERACTIVE"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is False

    def test_light_mode(self, session_start):
        os.environ["BRAINLAYER_HOOKS_LIGHT"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is True
        assert light is True

    def test_disabled_takes_precedence_over_light(self, session_start):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        os.environ["BRAINLAYER_HOOKS_LIGHT"] = "1"
        hook_input = {"session_id": "abc123"}
        activate, light = session_start.should_activate(hook_input)
        assert activate is False

    def test_empty_input(self, session_start):
        activate, light = session_start.should_activate({})
        assert activate is True
        assert light is False


class TestPromptSearchConditional:
    def test_default_activates(self, prompt_search):
        activate, light = prompt_search.should_activate()
        assert activate is True
        assert light is False

    def test_disabled_env_var(self, prompt_search):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        activate, light = prompt_search.should_activate()
        assert activate is False

    def test_non_interactive_skips(self, prompt_search):
        os.environ["CLAUDE_NON_INTERACTIVE"] = "1"
        activate, light = prompt_search.should_activate()
        assert activate is False

    def test_light_mode_reduces_results(self, prompt_search):
        os.environ["BRAINLAYER_HOOKS_LIGHT"] = "1"
        activate, light = prompt_search.should_activate()
        assert activate is True
        assert light is True


class TestStopIndexConditional:
    def test_default_activates(self, stop_index):
        assert stop_index.should_activate() is True

    def test_disabled_env_var(self, stop_index):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "1"
        assert stop_index.should_activate() is False

    def test_non_interactive_skips(self, stop_index):
        os.environ["CLAUDE_NON_INTERACTIVE"] = "1"
        assert stop_index.should_activate() is False

    def test_disabled_value_must_be_1(self, stop_index):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "0"
        assert stop_index.should_activate() is True

    def test_disabled_value_true_not_recognized(self, stop_index):
        os.environ["BRAINLAYER_HOOKS_DISABLED"] = "true"
        assert stop_index.should_activate() is True


