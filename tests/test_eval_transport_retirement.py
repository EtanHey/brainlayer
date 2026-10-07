"""Offline evaluation survives removal of the built-in remote model transports."""

import ast
import importlib
from pathlib import Path

import pytest

RETIRED = {
    "abcde_enrich_runner": ("make_http_chat_fn", "DEFAULT_BASE_URL", "DEFAULT_MODEL"),
}


@pytest.mark.parametrize("name", RETIRED)
def test_eval_has_no_builtin_remote_sender(name):
    module = importlib.import_module("brainlayer.eval." + name)
    for symbol in RETIRED[name]:
        assert not hasattr(module, symbol), (name, symbol)
    imports = []
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        item.startswith(("requests", "httpx", "google", "groq", "openai", "brainlayer.enrichment_controller"))
        for item in imports
    ), imports
