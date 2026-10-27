"""Keep scanner-shaped credentials out of source and synthetic fixtures."""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOOGLE_KEY = re.compile(rb"AI" + rb"za[0-9A-Za-z_-]{35}")
TAILSCALE_LITERAL = re.compile(rb"ts" + rb"key-[0-9A-Za-z_-]{20,}")


def _literal_locations(paths, pattern):
    return [
        f"{path.relative_to(ROOT)}:{number}"
        for path in paths
        for number, line in enumerate(path.read_bytes().splitlines(), 1)
        if pattern.search(line)
    ]


def test_no_google_key_literals_in_scripts_or_src():
    paths = (
        path
        for directory in ("scripts", "src")
        for path in (ROOT / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    findings = _literal_locations(paths, GOOGLE_KEY)
    if findings:
        pytest.fail("Google key-shaped literals at " + ", ".join(findings), pytrace=False)


def test_google_literal_guard_reports_only_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    synthetic = b"AI" + b"za" + b"0" * 35
    (scripts / "example.py").write_bytes(b"# example\nKEY = '" + synthetic + b"'\n")

    with pytest.raises(pytest.fail.Exception) as failure:
        test_no_google_key_literals_in_scripts_or_src()

    assert str(failure.value) == "Google key-shaped literals at scripts/example.py:2"


def test_synthetic_tailscale_examples_are_assembled_at_runtime():
    paths = [
        ROOT / "tests/fixtures/secret_scrub/golden.json",
        ROOT / "tests/test_secret_scrub_token_families.py",
    ]
    findings = _literal_locations(paths, TAILSCALE_LITERAL)
    if findings:
        pytest.fail("Scanner-shaped test literals at " + ", ".join(findings), pytrace=False)
