"""Tests for Phase 4a metrics.py — pure-Python, no daemon needed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from metrics import (  # noqa: E402
    _is_relevant,
    aggregate_category,
    compare_to_baseline,
    compute_all,
    ndcg_at_n,
    recall_at_n,
)


# ──────────────────────────────────────────────────────────────────────────────
# _is_relevant
# ──────────────────────────────────────────────────────────────────────────────


def test_is_relevant_with_expected_entity_match():
    chunk = {"chunk_id": "rt-sagitstern-abc123", "score": 0.45}
    meta = {"expected_entity": "Sagit-Stern"}
    assert _is_relevant(chunk, meta) is True


def test_is_relevant_with_expected_entity_no_match():
    chunk = {"chunk_id": "rt-michal-xyz", "score": 0.0}
    meta = {"expected_entity": "Sagit-Stern"}
    assert _is_relevant(chunk, meta) is False


def test_is_relevant_with_score_range_pass():
    chunk = {"chunk_id": "rt-anything", "score": 0.40}
    meta = {"expected_score_range": [0.30, 0.50]}
    assert _is_relevant(chunk, meta) is True


def test_is_relevant_with_score_range_fail():
    chunk = {"chunk_id": "rt-anything", "score": 0.20}
    meta = {"expected_score_range": [0.30, 0.50]}
    assert _is_relevant(chunk, meta) is False


def test_is_relevant_default_falls_back_to_score_gt_zero():
    # score > 0 with no explicit expectation → relevant
    assert _is_relevant({"chunk_id": "x", "score": 0.5}, {}) is True
    # score == 0.0 means no signal — NOT relevant (we require strictly positive)
    assert _is_relevant({"chunk_id": "x", "score": 0.0}, {}) is False
    # None/missing chunk_id with no signal → not relevant
    assert _is_relevant({"chunk_id": "", "score": None}, {}) is False


# ──────────────────────────────────────────────────────────────────────────────
# recall_at_n
# ──────────────────────────────────────────────────────────────────────────────


def test_recall_at_20_relevant_in_top_20():
    chunks = [
        {"chunk_id": "rt-sagitstern-abc", "score": 0.45},
        {"chunk_id": "other", "score": 0.3},
    ]
    meta = {"expected_entity": "Sagit-Stern"}
    assert recall_at_n(chunks, meta, n=20) == 1.0


def test_recall_at_20_relevant_not_in_top_20():
    chunks = [{"chunk_id": "unrelated-1"}, {"chunk_id": "unrelated-2"}]
    meta = {"expected_entity": "Sagit-Stern"}
    assert recall_at_n(chunks, meta, n=20) == 0.0


def test_recall_at_n_min_required_partial():
    chunks = [
        {"chunk_id": "rt-michal-1", "score": 0.5},
        {"chunk_id": "rt-michal-2", "score": 0.4},
    ]
    meta = {"expected_entity": "Michal", "expected_min_recall_at_20": 3}
    # Only 2 relevant out of 3 required → 2/3
    assert recall_at_n(chunks, meta, n=20) == pytest.approx(2 / 3, abs=0.01)


def test_recall_at_n_truncates_to_top_n():
    chunks = [{"chunk_id": f"unrelated-{i}"} for i in range(19)]
    chunks.append({"chunk_id": "rt-sagit-found", "score": 0.5})
    meta = {"expected_entity": "Sagit"}
    # Top 20 includes the relevant chunk
    assert recall_at_n(chunks, meta, n=20) == 1.0
    # Top 5 does NOT include
    chunks_top5_first = [{"chunk_id": f"rt-other-{i}", "score": 0.1} for i in range(20)]
    chunks_top5_first[19] = {"chunk_id": "rt-sagit-found", "score": 0.5}
    meta_strict = {"expected_entity": "Sagit", "expected_min_recall_at_5": 1}
    # When asking recall@5 with min_required from "expected_min_recall_at_5" — no relevant in top 5
    assert recall_at_n(chunks_top5_first, meta_strict, n=5) == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ndcg_at_n
# ──────────────────────────────────────────────────────────────────────────────


def test_ndcg_at_10_ideal_ranking():
    # Top 3 all relevant, ranks 4-10 not relevant
    chunks = [{"chunk_id": "rt-sagit-1", "score": 1.0}] * 3 + [
        {"chunk_id": "other", "score": 0.0}
    ] * 7
    meta = {"expected_entity": "Sagit"}
    ndcg = ndcg_at_n(chunks, meta, n=10)
    assert ndcg == pytest.approx(1.0, abs=0.05), f"ideal ranking should ndcg=1.0, got {ndcg}"


def test_ndcg_at_10_no_relevant():
    chunks = [{"chunk_id": f"other-{i}", "score": 0.1} for i in range(10)]
    meta = {"expected_entity": "Sagit"}
    assert ndcg_at_n(chunks, meta, n=10) == 0.0


def test_ndcg_at_10_partial():
    # Only chunk at rank 5 is relevant — DCG less than ideal
    chunks = [{"chunk_id": f"other-{i}", "score": 0.1} for i in range(4)]
    chunks.append({"chunk_id": "rt-sagit", "score": 0.7})
    chunks.extend({"chunk_id": f"other-{i}", "score": 0.1} for i in range(5, 10))
    meta = {"expected_entity": "Sagit"}
    ndcg = ndcg_at_n(chunks, meta, n=10)
    assert 0.0 < ndcg < 1.0


# ──────────────────────────────────────────────────────────────────────────────
# aggregate_category
# ──────────────────────────────────────────────────────────────────────────────


def test_aggregate_category_empty():
    out = aggregate_category([])
    assert out["n"] == 0
    assert out["recall_at_20_mean"] == 0.0


def test_aggregate_category_basic():
    results = [
        {
            "id": "heb-02",
            "category": "hebrew",
            "expected_entity": "Sagit-Stern",
            "top_5_chunk_ids": ["rt-sagitstern-a", "other-b"],
            "top_5_scores": [0.5, 0.3],
            "latency_ms": 100.0,
        },
        {
            "id": "heb-03",
            "category": "hebrew",
            "expected_entity": "Sagit-Stern",
            "top_5_chunk_ids": ["other-c", "other-d"],
            "top_5_scores": [0.1, 0.1],
            "latency_ms": 200.0,
        },
    ]
    out = aggregate_category(results)
    assert out["n"] == 2
    # First has relevant, second doesn't → recall mean = 0.5
    assert out["recall_at_20_mean"] == pytest.approx(0.5, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# compute_all
# ──────────────────────────────────────────────────────────────────────────────


def test_compute_all_groups_by_category():
    summary = {
        "results": [
            {"id": "heb-1", "category": "hebrew", "top_5_chunk_ids": [], "top_5_scores": [], "latency_ms": 10},
            {"id": "hlt-1", "category": "health", "top_5_chunk_ids": [], "top_5_scores": [], "latency_ms": 20},
            {"id": "hlt-2", "category": "health", "top_5_chunk_ids": [], "top_5_scores": [], "latency_ms": 30},
        ]
    }
    out = compute_all(summary)
    assert out["n_queries"] == 3
    assert "hebrew" in out["per_category"]
    assert "health" in out["per_category"]
    assert out["per_category"]["health"]["n"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# compare_to_baseline
# ──────────────────────────────────────────────────────────────────────────────


def test_compare_to_baseline_pass():
    current = {
        "overall": {"recall_at_20_mean": 0.92, "ndcg_at_10_mean": 0.90, "latency_p50_ms": 100},
        "per_category": {"hebrew": {"recall_at_20_mean": 0.88, "ndcg_at_10_mean": 0.85}},
    }
    baseline = {
        "per_category": {"hebrew": {"recall_at_20_mean": 0.87}},
    }
    thresholds = {
        "aggregate": {"recall_at_20_minimum_per_category": 0.85, "ndcg_at_10_minimum": 0.85, "no_category_regression_percent": -5},
        "per_category_minimum": {"hebrew": 0.85},
    }
    verdict = compare_to_baseline(current, baseline, thresholds)
    assert verdict["passed"] is True
    assert verdict["n_failures"] == 0


def test_compare_to_baseline_aggregate_ndcg_fail():
    current = {
        "overall": {"recall_at_20_mean": 0.90, "ndcg_at_10_mean": 0.80},
        "per_category": {"hebrew": {"recall_at_20_mean": 0.90, "ndcg_at_10_mean": 0.80}},
    }
    baseline = {"per_category": {"hebrew": {"recall_at_20_mean": 0.85}}}
    thresholds = {
        "aggregate": {"ndcg_at_10_minimum": 0.85, "recall_at_20_minimum_per_category": 0.85, "no_category_regression_percent": -5},
        "per_category_minimum": {},
    }
    verdict = compare_to_baseline(current, baseline, thresholds)
    assert verdict["passed"] is False
    assert any(f["check"] == "aggregate_ndcg_at_10" for f in verdict["failures"])


def test_compare_to_baseline_category_regression_fail():
    current = {
        "overall": {"recall_at_20_mean": 0.90, "ndcg_at_10_mean": 0.90},
        "per_category": {"hebrew": {"recall_at_20_mean": 0.70, "ndcg_at_10_mean": 0.85}},  # regressed from 0.90 (-22%)
    }
    baseline = {"per_category": {"hebrew": {"recall_at_20_mean": 0.90}}}
    thresholds = {
        "aggregate": {"ndcg_at_10_minimum": 0.85, "recall_at_20_minimum_per_category": 0.65, "no_category_regression_percent": -5},
        "per_category_minimum": {"hebrew": 0.65},  # category min satisfied so we ONLY catch regression
    }
    verdict = compare_to_baseline(current, baseline, thresholds)
    assert verdict["passed"] is False
    # Should have a regression failure
    assert any(f["check"] == "category_regression" for f in verdict["failures"])
