"""Deepchecks regression for stale-index embedding drift."""

from __future__ import annotations

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift

from tests.regression._stale_index_fixture import (
    baseline_embedding_rows,
    cosine_similarity,
    current_embedding_rows,
    load_fixture,
)


def _embedding_frame(rows: list[list[float]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("embedding fixture rows must be non-empty")
    width = len(rows[0])
    columns = [f"dim_{index:04d}" for index in range(width)]
    return pd.DataFrame(rows, columns=columns)


def test_fixture_embeddings_pass_deepchecks_and_cosine_threshold() -> None:
    fixture = load_fixture()
    baseline_rows = baseline_embedding_rows()
    current_rows = current_embedding_rows()
    min_cosine_similarity = fixture["sample_text"]["min_cosine_similarity"]

    assert len(baseline_rows) == len(current_rows)
    for baseline_row, current_row in zip(baseline_rows, current_rows):
        assert cosine_similarity(current_row, baseline_row) > min_cosine_similarity

    baseline_frame = _embedding_frame(baseline_rows)
    current_frame = _embedding_frame(current_rows)
    drift_check = FeatureDrift(min_samples=len(baseline_rows))
    # With five rows, Deepchecks' KS-based numeric drift bottoms out around 0.2
    # even when the distributions are effectively unchanged across platforms.
    drift_check.add_condition_drift_score_less_than(max_allowed_numeric_score=0.21)
    result = drift_check.run(
        train_dataset=Dataset(baseline_frame, cat_features=[]),
        test_dataset=Dataset(current_frame, cat_features=[]),
        with_display=False,
    )

    assert result.passed_conditions()
