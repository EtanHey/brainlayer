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
        "import new_vendor_sdk",
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
    source.write_text(source.read_text().replace("trust_env=False", "trust_env=True"))
    assert check_policy(tmp_path)["unclassified"]


def test_new_native_source_cannot_bypass_inventory(tmp_path):
    source = tmp_path / "brain-bar/Sources/BrainBar/Sender.swift"
    source.parent.mkdir(parents=True)
    source.write_text("customVendor.transmit(payload)")
    sites = inventory(tmp_path)
    assert len(sites) == 1 and next(iter(sites.values()))["kind"] == "non_python_source"
    assert fingerprint(sites) != fingerprint({})


def test_no_import_python_module_with_custom_method_requires_classification(tmp_path):
    source = tmp_path / "src/brainlayer/new_sender.py"
    source.parent.mkdir(parents=True)
    source.write_text("def invoke(client, payload):\n    return client.custom_model_call(payload)")
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_text(json.dumps({"schema_version": 1, "sites": {}}))
    assert check_policy(tmp_path)["unclassified"]
