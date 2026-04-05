"""Tests for prompt-search truncation formatting."""

import importlib.util
from pathlib import Path

HOOK_PATH = Path(__file__).parent.parent / "hooks" / "brainlayer-prompt-search.py"


def load_hook_module():
    spec = importlib.util.spec_from_file_location("brainlayer_prompt_search", HOOK_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_truncate_short_text():
    hook = load_hook_module()

    text = "Short text that should stay untouched."

    assert hook.truncate(text, max_chars=200) == text


def test_truncate_sentence_boundary():
    hook = load_hook_module()

    text = ("A" * 170) + ". Next sentence continues with more words beyond the limit."

    result = hook.truncate(text, max_chars=200)

    assert result == ("A" * 170) + "...."
    assert "Next sentence" not in result


def test_truncate_word_boundary_fallback():
    hook = load_hook_module()

    text = ("word " * 60).strip()

    result = hook.truncate(text, max_chars=200)

    assert result.endswith("...")
    assert result[:-3].endswith("word")
    assert len(result) <= 203


def test_truncate_newlines_collapsed():
    hook = load_hook_module()

    text = "Line one\n\nLine two"

    assert hook.truncate(text, max_chars=200) == "Line one | Line two"
