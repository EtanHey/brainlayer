"""Policy regeneration never assigns a purpose to an unreviewed site."""

import json

import pytest

from scripts.retirement_inventory import inventory
from scripts.retirement_policy_generate import main, render


def fixture(tmp_path):
    source = tmp_path / "src/brainlayer/sender.py"
    source.parent.mkdir(parents=True)
    source.write_text("import httpx")
    policy = tmp_path / "scripts/retirement_policy.json"
    policy.parent.mkdir()
    policy.write_text(json.dumps({"schema_version": 1, "sites": {}}))
    return source, policy


def test_generator_refuses_missing_purpose_without_writing(tmp_path):
    _, policy = fixture(tmp_path)
    original = policy.read_bytes()
    assert main(["--root", str(tmp_path), "--write"]) == 1
    assert policy.read_bytes() == original
    with pytest.raises(ValueError, match="New sites require"):
        render(tmp_path)


def test_carry_forward_is_deterministic_and_call_changes_need_new_purpose(tmp_path):
    source, policy = fixture(tmp_path)
    purposes = {key: "Private HTTP transport control, no cloud model" for key in inventory(tmp_path)}
    additions = tmp_path / "purposes.json"
    additions.write_text(json.dumps(purposes))
    assert main(["--root", str(tmp_path), "--write", "--purposes", str(additions)]) == 0
    original = policy.read_bytes()
    assert main(["--root", str(tmp_path), "--check"]) == 0
    assert main(["--root", str(tmp_path), "--write"]) == 0
    assert policy.read_bytes() == original
    source.write_text(source.read_text() + "\nhttpx.post(endpoint)")
    assert main(["--root", str(tmp_path), "--check"]) == 1
    assert main(["--root", str(tmp_path), "--write"]) == 1
    assert policy.read_bytes() == original
