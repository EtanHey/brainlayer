"""Contract tests for the stale index regression fixture."""

import json
from pathlib import Path


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "stale_index_query.json"


def test_stale_index_fixture_exists_and_has_expected_shape():
    payload = json.loads(FIXTURE_PATH.read_text())

    assert payload["query"]["match"] == "apple AND machine"
    assert payload["query"]["expected_ids"]
    assert payload["chunks"]
    assert len(payload["sample_text"]["baseline_embedding"]) == 1024
    assert payload["sample_text"]["min_cosine_similarity"] >= 0.999
