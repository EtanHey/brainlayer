"""The Python scrubber matches the shared golden fixture.

``brain-bar/Tests/BrainBarTests/SecretScrubberGoldenTests.swift`` asserts the SAME
file against BrainBar's Swift port, so a rule change in one implementation
without the other fails one of the two suites.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from brainlayer.pipeline.secret_scrub import scrub_secrets

GOLDEN = Path(__file__).parent / "fixtures" / "secret_scrub" / "golden.json"

# The fixture writes any run of 16+ identical characters as {{c*N}}, so no
# token-shaped literal is committed. The Swift harness expands it the same way.
_RUN = re.compile(r"\{\{([A-Za-z0-9])\*(\d+)\}\}")


def _expand(text: str) -> str:
    return _RUN.sub(lambda match: match.group(1) * int(match.group(2)), text)


CASES = [
    {**case, "input": _expand(case["input"]), "expected_text": _expand(case["expected_text"])}
    for case in json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
]


def test_golden_fixture_covers_every_provider_family():
    from brainlayer.pipeline import secret_scrub

    families = {pattern.provider for pattern in secret_scrub._PROVIDER_PATTERNS} | {"assignment"}
    covered = {provider for case in CASES for provider in case["expected_providers"]}

    assert families <= covered, f"golden fixture misses: {sorted(families - covered)}"


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_python_scrubber_matches_golden(case):
    result = scrub_secrets(case["input"])

    assert result.text == case["expected_text"]
    assert [redaction.provider for redaction in result.redactions] == case["expected_providers"]
    assert len(result.quarantine) == case["expected_quarantine_count"]
