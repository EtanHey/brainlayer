"""Current entry guides cannot advertise enrichment producers or activation."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("relative", ["README.md", "docs/index.md", "docs/quickstart.md"])
def test_current_entry_guides_retire_model_production_and_keep_local_memory(relative):
    text = (ROOT / relative).read_text()
    assert not re.search(r"brainlayer\s+enrich(?:-sessions)?\b", text)
    assert not re.search(r"pip install[^\n]*brainlayer\[cloud\]", text)
    assert not re.search(r"\*\*(?:15-field )?Enrichment\*\*", text, re.I)
    assert "retired" in text.lower()
    assert "enrichment.md" in text
    for retained in ("brain_search", "brainlayer index", "SQLite", "knowledge graph"):
        assert retained.lower() in text.lower(), (relative, retained)
