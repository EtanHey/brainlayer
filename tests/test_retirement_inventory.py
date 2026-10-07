"""An unclassified generic sender cannot hide behind a novel endpoint/alias."""

import json

import pytest

from scripts.retirement_inventory import check_policy, fingerprint, inventory, python_sites


@pytest.mark.parametrize(
    "code",
    [
        "import requests\ndef send():\n    requests.post(endpoint, json=payload)",
        "import requests as r\ndef send():\n    r.post(endpoint, json=payload)",
        "from subprocess import run as execute\ndef send():\n    execute(command)",
        "from importlib import import_module as load\ndef send():\n    load(variable_module)",
        "def send(client):\n    client.request('POST', endpoint, json=payload)",
        "import httpx",
    ],
)
def test_generic_senders_and_new_imports_are_inventoried(code):
    assert python_sites(code, "src/brainlayer/sender.py")


def test_changing_redirect_or_proxy_policy_invalidates_admission(tmp_path):
    source = tmp_path / "src/brainlayer/sender.py"
    source.parent.mkdir(parents=True)
    source.write_text("def send(client):\n    client.trust_env=False\n    client.post(endpoint, allow_redirects=False)")
    sites = inventory(tmp_path)
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sites": {
                    key: {"site": value, "non_cloud_purpose": "loopback-only fixture"} for key, value in sites.items()
                },
            }
        )
    )
    assert not check_policy(tmp_path)["unclassified"]
    source.write_text(source.read_text().replace("allow_redirects=False", "allow_redirects=True"))
    assert check_policy(tmp_path)["unclassified"]


def test_new_native_source_cannot_bypass_inventory(tmp_path):
    source = tmp_path / "brain-bar/Sources/BrainBar/Sender.swift"
    source.parent.mkdir(parents=True)
    source.write_text("URLSession.shared.dataTask(with: endpoint)")
    sites = inventory(tmp_path)
    assert len(sites) == 1 and next(iter(sites.values()))["kind"] == "non_python_source"
    assert fingerprint(sites) != fingerprint({})


@pytest.mark.parametrize("code", ["import httpx", "import requests\nrequests.post('https://novel.invalid/model')"])
def test_new_transport_module_requires_classification(tmp_path, code):
    source = tmp_path / "src/brainlayer/new_sender.py"
    source.parent.mkdir(parents=True)
    source.write_text(code)
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_text(json.dumps({"schema_version": 1, "sites": {}}))
    assert check_policy(tmp_path)["unclassified"]


def test_call_binding_ignores_enclosing_body_and_plain_imports():
    code = "import json\ndef send(client):\n    client.post(endpoint)"
    before = python_sites(code, "sender.py")
    after = python_sites(code + "\ndef harmless():\n    return 1", "sender.py")
    assert before == after and len(before) == 1
    assert "body_sha256" not in before[0]


def test_non_transport_native_file_is_not_bound(tmp_path):
    source = tmp_path / "brain-bar/Sources/BrainBar/Local.swift"
    source.parent.mkdir(parents=True)
    source.write_text("let harmless = 1")
    assert not inventory(tmp_path)


@pytest.mark.parametrize(
    "path,edit",
    [
        ("chunk_write.py", lambda s: s + "\ndef _harmless_helper():\n    return 1\n"),
        ("paths.py", lambda s: s.replace('"""', '"""Harmless docstring edit. ', 1)),
    ],
)
def test_opus_harmless_edits_leave_zero_unclassified(tmp_path, path, edit):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "src/brainlayer" / path
    source.parent.mkdir(parents=True)
    source.write_text(edit((root / "src/brainlayer" / path).read_text()))
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_bytes((root / "scripts/retirement_policy.json").read_bytes())
    assert check_policy(tmp_path)["unclassified"] == []
