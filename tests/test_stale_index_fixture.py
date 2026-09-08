"""Contract tests for the stale index regression fixture."""

import json
import math
from pathlib import Path

import pytest

import brainlayer.embeddings as embeddings

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "stale_index_query.json"


def test_stale_index_fixture_exists_and_has_expected_shape():
    payload = json.loads(FIXTURE_PATH.read_text())

    assert payload["query"]["match"] == "apple AND machine"
    assert payload["query"]["expected_ids"]
    assert payload["chunks"]
    assert len(payload["sample_text"]["baseline_embedding"]) == 1024
    assert payload["sample_text"]["min_cosine_similarity"] >= 0.999


@pytest.mark.embedding_model
def test_stale_index_sample_embedding_matches_baseline():
    payload = json.loads(FIXTURE_PATH.read_text())
    sample = payload["sample_text"]
    live_embedding = embeddings.get_embedding_model().embed_query(sample["text"])
    baseline_embedding = sample["baseline_embedding"]

    assert len(live_embedding) == len(baseline_embedding)
    dot_product = math.fsum(left * right for left, right in zip(live_embedding, baseline_embedding))
    live_norm = math.sqrt(math.fsum(value * value for value in live_embedding))
    baseline_norm = math.sqrt(math.fsum(value * value for value in baseline_embedding))

    assert dot_product / (live_norm * baseline_norm) > sample["min_cosine_similarity"]
