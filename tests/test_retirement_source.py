"""R11 source absence applies to lazy/relative imports, not only import time."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.retirement_source import MODEL_URL, python_findings, source_scan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "host",
    [
        "openrouter.ai",
        "api.deepseek.com",
        "api.together.xyz",
        "api.fireworks.ai",
        "api.perplexity.ai",
        "bedrock-runtime.us-east-1.amazonaws.com",
        "bedrock-runtime.cn-north-1.amazonaws.com.cn",
        "fixture.openai.azure.com",
        "api-inference.huggingface.co",
        "router.huggingface.co",
        "fixture.us-east-1.aws.endpoints.huggingface.cloud",
    ],
)
@pytest.mark.parametrize("ending", ["/model", ":443/model", "?model=fixture", "#fixture"])
def test_model_provider_rest_hosts_are_forbidden(host, ending):
    assert MODEL_URL.search(f"https://{host}{ending}")


@pytest.mark.parametrize("host", ["api.deepseek.com.example.invalid", "openrouter.ai.invalid", "huggingface.co"])
def test_model_url_gate_requires_an_exact_model_provider_host(host):
    assert not MODEL_URL.search(f"https://{host}/fixture")


@pytest.mark.parametrize(
    "module,code",
    [
        ("brainlayer.replay", "from .enrichment_controller import X"),
        ("brainlayer.replay", "from . import enrichment_controller as ec"),
        ("brainlayer.replay", "from .enrichment_controller import *"),
        ("brainlayer.pipeline.replay", "from ..enrichment_controller import X"),
        ("brainlayer.pipeline.__init__", "from . import enrichment"),
        ("brainlayer.pipeline.replay", "from .groq import validate_groq_model"),
        ("brainlayer.replay", "from .pipeline import enrichment as e"),
        ("brainlayer.replay", "import brainlayer.pipeline.groq as g"),
        ("brainlayer.replay", "def replay():\n    from .enrichment_controller import RATE_LIMITS"),
        ("brainlayer.replay", "def replay():\n    from brainlayer.enrichment_controller import RATE_LIMITS"),
        (
            "brainlayer.replay",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .enrichment_controller import X",
        ),
        (
            "brainlayer.replay",
            "from importlib import import_module as load\nload('.enrichment_controller', 'brainlayer')",
        ),
        ("brainlayer.replay", "import importlib as i\ni.import_module('.groq', package='brainlayer.pipeline')"),
        ("brainlayer.replay", "__import__('brainlayer.pipeline.enrichment')"),
        ("brainlayer.replay", "from brainlayer import pipeline as p\np.enrichment.call_llm('fixture')"),
        ("brainlayer.replay", "from google import genai as g\ng.Client()"),
        ("brainlayer.replay", "client.models.generate_content('fixture')"),
        ("brainlayer.replay", "client.chat.completions.create()"),
        ("brainlayer.replay", "from google import genai\ndef other():\n    import math as genai\ngenai.Client()"),
    ],
)
def test_forbidden_import_and_send_forms(module, code):
    assert python_findings(code, module)


@pytest.mark.parametrize(
    "code",
    [
        "from .enrichment_replay import _apply_enrichment",
        "from .pipeline.enrichment_results import parse_enrichment",
        "from .pipeline.enrichment_prompts import build_external_prompt",
        "from .pipeline.enrichment_tiers import HIGH_VALUE_TYPES",
        "from google.auth import credentials",
        "from googleapiclient.discovery import build",
        "import requests",
        '"""Historical from .enrichment_controller import X"""',
    ],
)
def test_retained_imports(code):
    assert not python_findings(code, "brainlayer.replay")


def test_no_src_module_references_retired_hosts_in_any_import_form():
    report = source_scan(ROOT)
    assert not report["errors"], report["errors"]
    assert not report["findings"], report["findings"]


def test_scan_is_red_on_parse_error_and_empty_inventory(tmp_path):
    path = tmp_path / "src/brainlayer/bad.py"
    path.parent.mkdir(parents=True)
    path.write_text("from .enrichment_controller import")
    report = source_scan(tmp_path)
    assert report["errors"]
    assert source_scan(tmp_path / "missing")["errors"]


def test_native_model_sender_restoration_is_detected(tmp_path):
    path = tmp_path / "brain-bar/Sources/BrainBar/Mutant.swift"
    path.parent.mkdir(parents=True)
    path.write_text('client.generateContent("fixture")')
    assert source_scan(tmp_path)["findings"]


@pytest.fixture(params=["js", "ts", "mjs"])
def es_source_fixture(tmp_path, request):
    for relative in ("src/brainlayer/__init__.py", "brain-bar/Sources/Probe.swift"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    path = tmp_path / f"dashboard/probe.{request.param}"
    path.parent.mkdir()
    return tmp_path, path


def scan_es_fixture(root, path, code):
    path.write_text(code)
    syntax = root / "syntax.mjs"
    syntax.write_text(code + "\n")
    checked = subprocess.run(
        ["node", "--check", str(syntax)],
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key not in {"NODE_OPTIONS", "NODE_PATH"}},
    )
    if receipt := os.environ.get("BRAINLAYER_ES_SYNTAX_RECEIPT"):
        with Path(receipt).open("a") as stream:
            stream.write(
                json.dumps(
                    {"code": code, "sha256": hashlib.sha256(code.encode()).hexdigest(), "exit": checked.returncode}
                )
                + "\n"
            )
    assert checked.returncode == 0, checked.stderr
    report = source_scan(root)
    assert not report["errors"], report["errors"]
    return report


@pytest.mark.parametrize("sdk", ["openai", "@google/genai", "@google/generative-ai", "@anthropic-ai/sdk"])
@pytest.mark.parametrize(
    "form",
    [
        "import('{sdk}')",
        'await import("{sdk}")',
        "await import \t( \t'{sdk}' \t)",
        "import(\n'{sdk}'\n)",
        "`text ${{await import('{sdk}')}}`",
        "await import/*local note*/('{sdk}')",
        "await import(/*local note*/'{sdk}')",
        "await import('{sdk}'/*local note*/)",
        "await import(//local note\n'{sdk}')",
        "await import('{sdk}',)",
        "await import/*x*/(/*x*/'{sdk}'/*x*/,/*x*/)",
        "await import('{sdk}', {{with: {{type: 'json'}}}})",
    ],
)
def test_dynamic_es_model_sdk_import_is_detected(es_source_fixture, sdk, form):
    root, path = es_source_fixture
    report = scan_es_fixture(root, path, form.format(sdk=sdk))
    assert not report["errors"], report["errors"]
    assert report["findings"] == [
        {"path": path.relative_to(root).as_posix(), "line": 1, "target": "model transport syntax"}
    ]


@pytest.mark.parametrize(
    "code",
    [
        "const help = \"await import('openai')\";",
        "const help = 'await import(\"@anthropic-ai/sdk\")';",
        "const help = `await import('openai')`;",
        "const help = `await import('openai') ${local}`;",
        "const help = `${\"await import('openai')\"}`;",
        "const help = `\\${await import('openai')}`;",
        "// await import('openai')",
        '/*\nawait import("@anthropic-ai/sdk")\n*/',
        "import helper from './local-helper';",
        "await import('./local-helper');",
        "require('./local-helper');",
        "const example = /import('openai')/;",
        "/import('openai')/.test(text);",
        "const example = /[/'\"]import('openai')/g;",
        r"const example = /import('openai')\//g;",
        "if (flag) /import('openai')/.test(text);",
        "if (flag) {} /import('openai')/.test(text);",
        "const example = `value ${/import('openai')/.source}`;",
        "function f() { return /import('openai')/; }",
        "function f() {} /import('openai')/.test(text);",
        "class Local {} /import('openai')/.test(text);",
        "for (const local of /import('openai')/) {}",
        "await import(/*local note*/'./local-helper',);",
    ],
)
def test_harmless_es_strings_comments_and_local_imports_are_preserved(es_source_fixture, code):
    root, path = es_source_fixture
    report = scan_es_fixture(root, path, code)
    assert not report["errors"] and not report["findings"], report


@pytest.mark.parametrize("sdk", ["openai", "@google/genai", "@google/generative-ai", "@anthropic-ai/sdk"])
def test_static_es_model_sdk_control_is_armed(es_source_fixture, sdk):
    root, path = es_source_fixture
    report = scan_es_fixture(root, path, f"import sdk from '{sdk}';")
    assert not report["errors"], report["errors"]
    assert report["findings"] == [
        {"path": path.relative_to(root).as_posix(), "line": 1, "target": "model transport syntax"}
    ]


@pytest.mark.parametrize("sdk", ["openai", "@google/genai", "@google/generative-ai", "@anthropic-ai/sdk"])
@pytest.mark.parametrize(
    "form,line",
    [
        ("await import('{sdk}');\nconst example = /import('openai')/;", 1),
        ("const example = /import('openai')/;\nawait import/*x*/('{sdk}',);", 2),
        ("const example = /import('openai')/;\nconst n = 1 / (await import('{sdk}')) / 2;", 2),
        ("const example = /import('openai')/;\nconst n = f() / (await import('{sdk}')) / 2;", 2),
        ("const example = /import('openai')/;\nconst n = ({{n: 1}}).n / (await import('{sdk}')) / 2;", 2),
        ("const example = /import('openai')/;\nconst n = function() {{}} / (await import('{sdk}')) / 2;", 2),
        ("const example = /import('openai')/;\nconst n = class {{}} / (await import('{sdk}')) / 2;", 2),
        ("const example = /import('openai')/;\nconst n = object.return / (await import('{sdk}')) / 2;", 2),
        ("const example = /import('openai')/;\nconst of = 1; const n = of / (await import('{sdk}')) / 2;", 2),
        ("if (flag) /import('openai')/.test(text);\nawait import('{sdk}');", 2),
        ("if (flag) {{}} /import('openai')/.test(text);\nawait import('{sdk}');", 2),
        ("`x ${{/import('openai')/.source}}\n${{await import/*x*/('{sdk}',)}}`", 2),
    ],
)
def test_executable_imports_around_regex_are_not_masked(es_source_fixture, sdk, form, line):
    root, path = es_source_fixture
    report = scan_es_fixture(root, path, form.format(sdk=sdk))
    assert report["findings"] == [
        {"path": path.relative_to(root).as_posix(), "line": line, "target": "model transport syntax"}
    ]
