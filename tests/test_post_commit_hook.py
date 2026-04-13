"""Tests for hooks/post-commit.py — ensures the hook survives non-UTF8
subprocess output from git (commit messages with non-UTF8 encoding,
legacy-imported commit metadata, or non-UTF8 file paths).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

HOOKS_DIR = Path(__file__).parent.parent / "hooks"


def _load_hook():
    spec = importlib.util.spec_from_file_location(
        "post_commit_hook",
        HOOKS_DIR / "post-commit.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def captured_calls(monkeypatch):
    """Stub out brainlayer.* modules so the hook can import them without
    loading the real embedding model, and capture store_memory() calls.
    """
    calls = []

    store_mod = types.ModuleType("brainlayer.store")

    def fake_store_memory(**kwargs):
        calls.append(kwargs)

    store_mod.store_memory = fake_store_memory

    vs_mod = types.ModuleType("brainlayer.vector_store")

    class _FakeStore:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    vs_mod.VectorStore = _FakeStore

    embed_mod = types.ModuleType("brainlayer.embeddings")

    class _FakeModel:
        def embed_query(self, text):
            return [0.0] * 8

    embed_mod.get_embedding_model = lambda: _FakeModel()

    paths_mod = types.ModuleType("brainlayer.paths")
    paths_mod.DEFAULT_DB_PATH = "/tmp/fake-brainlayer.db"

    monkeypatch.setitem(sys.modules, "brainlayer.store", store_mod)
    monkeypatch.setitem(sys.modules, "brainlayer.vector_store", vs_mod)
    monkeypatch.setitem(sys.modules, "brainlayer.embeddings", embed_mod)
    monkeypatch.setitem(sys.modules, "brainlayer.paths", paths_mod)

    return calls


def _make_fake_check_output(
    commit_msg_bytes,
    file_list_bytes=b"src/a.py\nsrc/b.py\n",
    toplevel_bytes=b"/tmp/repo\n",
):
    def fake(cmd, **kwargs):
        if "rev-parse" in cmd and "HEAD" in cmd:
            return b"deadbeefcafe1234\n"
        if "log" in cmd and "--pretty=%B" in cmd:
            return commit_msg_bytes
        if "diff-tree" in cmd:
            return file_list_bytes
        if "--show-toplevel" in cmd:
            return toplevel_bytes
        raise AssertionError(f"unexpected subprocess call: {cmd}")

    return fake


def test_post_commit_handles_non_utf8_commit_message(captured_calls, monkeypatch):
    """Discriminating test: non-UTF8 bytes in commit message must not crash
    the hook. Fails before the fix (UnicodeDecodeError escapes the outer
    try/except), passes after.
    """
    hook_mod = _load_hook()
    monkeypatch.setattr(
        hook_mod.subprocess,
        "check_output",
        _make_fake_check_output(commit_msg_bytes=b"Latin-1 commit \xff\xfe body\n"),
    )

    hook_mod.main()  # must not raise

    assert len(captured_calls) == 1
    assert "\ufffd" in captured_calls[0]["content"]


def test_post_commit_handles_non_utf8_file_path(captured_calls, monkeypatch):
    hook_mod = _load_hook()
    monkeypatch.setattr(
        hook_mod.subprocess,
        "check_output",
        _make_fake_check_output(
            commit_msg_bytes=b"fix: something\n",
            file_list_bytes=b"src/\xff\xfe.py\nsrc/b.py\n",
        ),
    )

    hook_mod.main()  # must not raise

    assert len(captured_calls) == 1
    # U+FFFD must appear in the Files portion (from the non-UTF8 filename)
    assert "\ufffd" in captured_calls[0]["content"]


def test_post_commit_preserves_valid_utf8(captured_calls, monkeypatch):
    """Sanity: valid UTF-8 (including non-ASCII) flows through unchanged."""
    hook_mod = _load_hook()
    monkeypatch.setattr(
        hook_mod.subprocess,
        "check_output",
        _make_fake_check_output(
            commit_msg_bytes="fix: 日本語 commit\n".encode("utf-8"),
        ),
    )

    hook_mod.main()

    assert len(captured_calls) == 1
    content = captured_calls[0]["content"]
    assert "\ufffd" not in content
    assert "日本語" in content
