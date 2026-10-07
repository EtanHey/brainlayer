"""Published registry metadata must not offer retired model activation knobs."""

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_registry_offers_storage_configuration_without_model_credentials():
    server = json.loads((ROOT / "server.json").read_text())
    variables = {item["name"] for pkg in server["packages"] for item in pkg.get("environmentVariables", [])}
    assert "BRAINLAYER_DB" in variables
    assert not any("ENRICH" in name or name in {"GROQ_API_KEY", "GOOGLE_API_KEY"} for name in variables)
    assert "no api keys" not in server["description"].lower(), "Installed hotlane credential gate remains"


def test_package_and_registry_describe_retained_local_memory():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    server = json.loads((ROOT / "server.json").read_text())
    for description in (project["description"], server["description"]):
        assert "enrich" not in description.lower()
        assert "sqlite" in description.lower()
        assert "search" in description.lower()
    assert not any("enrich" in keyword.lower() for keyword in project["keywords"])
    assert not any("enrich" in classifier.lower() for classifier in project["classifiers"])
