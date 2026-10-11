"""Old enrichment URLs explain retirement and still point to preserved history."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", ["docs/enrichment.md", "docs/enrichment-runbook.md"])
def test_retirement_guides_cannot_run_or_resume_model_jobs(relative):
    text = (ROOT / relative).read_text()
    assert "retired" in text.lower()
    assert "CHANGELOG.md" in text
    assert not re.search(r"brainlayer\s+enrich|cloud_backfill\.py|auto-enrich\.sh", text)
    assert "historical" in text.lower()
    assert "local" in text.lower()


def test_runbook_preserves_installed_google_credential_gate_documentation():
    text = (ROOT / "docs/enrichment-runbook.md").read_text()
    for retained in ("GOOGLE_API_KEY", "BRAINLAYER_REQUIRE_GOOGLE_API_KEY", "78", "configuration.md"):
        assert retained in text


def test_site_navigation_labels_retired_guides():
    text = (ROOT / "mkdocs.yml").read_text()
    assert not re.search(r"- Enrichment(?: Runbook)?:", text)
    assert "Retired Enrichment" in text
    assert "enrichment-runbook.md" in text
