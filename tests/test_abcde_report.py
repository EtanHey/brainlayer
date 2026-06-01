"""Tests for deterministic ABCDE enrichment report aggregation."""

from __future__ import annotations

import json

from scripts.abcde_report import build_report


def _row(variant_id: str, enrichment: dict, *, status: str = "ok", error: str | None = None) -> dict:
    return {
        "chunk_id": f"{variant_id}-1",
        "variant_id": variant_id,
        "chunk_text": "BrainLayer shipped ExperimentStore with Python and SQLite.",
        "enrichment": enrichment,
        "model": "gemini-2.5-flash-lite",
        "prompt_hash": "hash",
        "generation": {
            "status": status,
            "error": error,
            "usage": {
                "completion_tokens": 100,
                "cost_in_usd_ticks": 1_000_000,
                "prompt_tokens": 200,
                "total_tokens": 300,
            },
        },
    }


def _schema_complete_enrichment() -> dict:
    return {
        "summary": "ExperimentStore isolated eval storage for BrainLayer.",
        "key_facts": ["ExperimentStore shipped", "SQLite storage isolated"],
        "tags": ["brainlayer", "sqlite", "eval"],
        "importance": 8,
        "intent": "implementing",
        "primary_symbols": ["ExperimentStore"],
        "resolved_query": "How was ExperimentStore isolated?",
        "resolved_queries": [
            "How was ExperimentStore isolated?",
            "BrainLayer ExperimentStore SQLite eval storage",
            "ExperimentStore isolated eval storage for BrainLayer using SQLite.",
        ],
        "epistemic_level": "validated",
        "version_scope": None,
        "debt_impact": "resolution",
        "external_deps": [],
        "entities": [{"name": "BrainLayer", "type": "project", "relation": "owns ExperimentStore"}],
        "sentiment_label": "satisfaction",
        "sentiment_score": 0.5,
        "sentiment_signals": ["shipped"],
    }


def test_build_report_aggregates_per_variant_from_raw_enrichment(tmp_path):
    input_path = tmp_path / "abcde.jsonl"
    rows = [
        _row("A", _schema_complete_enrichment()),
        _row("A", {}, status="error", error="invalid_json"),
        _row(
            "B",
            {
                **_schema_complete_enrichment(),
                "summary": "This chunk describes ExperimentStore isolation.",
                "entities": [{"name": "Imaginary Vendor", "type": "company", "relation": None}],
            },
        ),
    ]
    input_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")

    report = build_report(input_path)

    assert report.variants["A"].n == 2
    assert report.variants["A"].ok_n == 1
    assert report.variants["A"].gen_error_n == 1
    assert report.variants["A"].error_reasons == {"invalid_json": 1}
    assert report.variants["A"].schema_pass_pct == 100.0
    assert report.variants["A"].mean_resolved_queries == 3.0
    assert report.variants["A"].total_cost_usd == 0.002

    assert report.variants["B"].banned_pattern_pct == 100.0
    assert report.variants["B"].mean_unsupported_entities == 1.0
    assert "| Variant | label | n | ok | gen errors |" in report.table
