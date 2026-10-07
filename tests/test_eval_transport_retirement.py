"""Offline evaluation survives removal of the built-in remote model transports."""

import ast
import importlib
from pathlib import Path

import pytest

RETIRED = {
    "abcde_enrich_runner": ("make_http_chat_fn", "DEFAULT_BASE_URL", "DEFAULT_MODEL"),
    "enrichment_quality_benchmark": ("run_gemini_flex_sample",),
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


def test_flex_cli_is_removed_before_inputs_or_models(tmp_path, monkeypatch, capsys):
    from brainlayer.eval import enrichment_quality_benchmark as benchmark

    monkeypatch.setattr(benchmark, "get_db_path", lambda: tmp_path / "never-opened.db")
    with pytest.raises(SystemExit) as error:
        benchmark.main(["flex-sample", "--help"])
    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    assert not (tmp_path / "never-opened.db").exists()
    with pytest.raises(SystemExit) as help_exit:
        benchmark.main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "flex-sample" not in help_text
    assert "grade" in help_text and "local" in help_text
