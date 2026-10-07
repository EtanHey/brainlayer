"""An admitted import cannot hide a reachable sender behind a callable binding."""

import json

import pytest

from scripts.retirement_inventory import check_policy, inventory
from scripts.retirement_policy_generate import render


@pytest.mark.parametrize(
    "imports,body",
    [
        ("import requests", "transmit = requests.post\n    return transmit(endpoint, json=payload)"),
        ("import requests", "transmit: object = requests.post\n    return transmit(endpoint, json=payload)"),
        ("import requests as r", "first = r.post\n    second = first\n    return second(endpoint)"),
        ("from requests import post", "transmit = post\n    return transmit(endpoint)"),
        ("from subprocess import run as execute", "transmit = execute\n    return transmit(command)"),
        ("from importlib import import_module as load", "transmit = load\n    return transmit(variable_module)"),
        ("import requests", "return requests.post"),
        ("import requests", "transports = [requests.post]\n    return transports[0](endpoint)"),
        ("import requests", "transmit = getattr(requests, 'post')\n    return transmit(endpoint)"),
        ("", "transmit = client.get\n    return transmit(endpoint)"),
        ("", "transmit = eval\n    return transmit(variable_code)"),
        ("import socket", "factory = socket.socket\n    return factory()"),
    ],
)
def test_callable_alias_is_unknown_after_import_already_admitted(tmp_path, imports, body):
    source = tmp_path / "src/brainlayer/backup_daily.py"
    source.parent.mkdir(parents=True)
    source.write_text(imports + "\n")
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sites": {
                    key: {"site": site, "non_cloud_purpose": "Existing synthetic import"}
                    for key, site in inventory(tmp_path).items()
                },
            }
        )
    )
    assert not check_policy(tmp_path)["unclassified"]
    source.write_text(imports + "\ndef restored(endpoint, payload, client=None):\n    " + body + "\n")
    assert check_policy(tmp_path)["unclassified"]
    original = policy.read_bytes()
    with pytest.raises(ValueError, match="New sites require"):
        render(tmp_path)
    assert policy.read_bytes() == original


def test_alias_call_arguments_are_bound_even_if_binding_is_admitted(tmp_path):
    source = tmp_path / "src/brainlayer/sender.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "import requests\ndef send(endpoint):\n    transmit = requests.post\n    return transmit(endpoint, allow_redirects=False)"
    )
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sites": {
                    key: {"site": site, "non_cloud_purpose": "Synthetic local transport"}
                    for key, site in inventory(tmp_path).items()
                },
            }
        )
    )
    source.write_text(source.read_text().replace("allow_redirects=False", "allow_redirects=True"))
    unknown = check_policy(tmp_path)["unclassified"]
    assert any(site["kind"] == "generic_send" and site["target"] == "requests.post" for site in unknown)


def test_subprocess_result_data_does_not_become_a_transport_alias():
    from scripts.retirement_inventory import python_sites

    code = "import subprocess\ndef read():\n    result = subprocess.run(command)\n    return result.stdout.strip()"
    sites = python_sites(code, "local.py")
    assert [site["kind"] for site in sites] == ["import_binding", "child_process"]


def test_constant_getattr_and_constructed_client_calls_are_bound():
    from scripts.retirement_inventory import python_sites

    code = "import requests\ndef send(endpoint, client):\n    transmit = getattr(client, 'post')\n    session = requests.Session()\n    return session.get(endpoint)"
    sites = python_sites(code, "sender.py")
    assert any(site["kind"] == "callable_reference" for site in sites)
    assert any(site["kind"] == "generic_send" and site["target"] == "requests.Session().get" for site in sites)
