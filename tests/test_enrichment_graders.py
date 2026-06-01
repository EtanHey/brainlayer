import json
import math


def _candidate(**overrides):
    payload = {
        "summary": (
            "BrainLayer PR #423 added src/brainlayer/eval/enrichment_gold.py on June 1, 2026 "
            "to sample enrichment gold records from a read-only snapshot."
        ),
        "key_facts": [
            "PR #423",
            "src/brainlayer/eval/enrichment_gold.py",
            "June 1, 2026",
        ],
        "tags": ["brainlayer", "eval-harness", "testing"],
        "importance": 8,
        "intent": "implementing",
        "primary_symbols": ["sample_enrichment_gold"],
        "resolved_query": "How does BrainLayer sample enrichment gold records?",
        "resolved_queries": [
            "How does BrainLayer sample enrichment gold records?",
            "BrainLayer enrichment gold PR #423 sample_enrichment_gold",
            "BrainLayer PR #423 added sample_enrichment_gold for offline gold-set sampling.",
        ],
        "epistemic_level": "validated",
        "version_scope": "PR #423",
        "debt_impact": "none",
        "external_deps": ["sqlite"],
        "entities": [
            {"name": "BrainLayer", "type": "project", "relation": "memory layer"},
            {"name": "SQLite", "type": "technology", "relation": "snapshot store"},
        ],
        "sentiment_label": "neutral",
        "sentiment_score": 0.0,
        "sentiment_signals": [],
    }
    payload.update(overrides)
    return payload


def test_schema_gate_accepts_current_enrichment_shape_from_mapping_and_json():
    from brainlayer.eval.enrichment_graders import validate_schema_gate

    mapping_result = validate_schema_gate(_candidate())
    json_result = validate_schema_gate(json.dumps(_candidate()))

    assert mapping_result.passed is True
    assert mapping_result.errors == []
    assert mapping_result.value["importance"] == 8
    assert json_result.passed is True


def test_schema_gate_disqualifies_malformed_json_missing_keys_and_bad_ranges():
    from brainlayer.eval.enrichment_graders import validate_schema_gate

    assert validate_schema_gate("{not json").passed is False

    missing = _candidate()
    missing.pop("resolved_queries")
    missing_result = validate_schema_gate(missing)
    assert missing_result.passed is False
    assert "missing required key: resolved_queries" in missing_result.errors

    invalid = _candidate(
        importance=8.5,
        tags=["too-few"],
        resolved_queries=["only one"],
        intent="summarizing",
        epistemic_level="guessed",
        debt_impact="low",
        sentiment_label="happy",
        sentiment_score=1.5,
    )
    result = validate_schema_gate(invalid)
    assert result.passed is False
    assert "importance must be an integer from 1 to 10" in result.errors
    assert "tags must contain 3 to 7 strings" in result.errors
    assert "resolved_queries must contain exactly 3 strings" in result.errors
    assert "intent has invalid enum value: summarizing" in result.errors
    assert "epistemic_level has invalid enum value: guessed" in result.errors
    assert "debt_impact has invalid enum value: low" in result.errors
    assert "sentiment_label has invalid enum value: happy" in result.errors
    assert "sentiment_score must be a number in [-1, 1]" in result.errors


def test_banned_pattern_hard_fails_meta_descriptions():
    from brainlayer.eval.enrichment_graders import find_banned_summary_pattern, grade_candidate

    payload = _candidate(summary="The user is discussing BrainLayer PR #423 and eval harness work.")
    gold = {"must_capture_facts": ["PR #423"], "gold_entities": [], "gold_importance": 8}

    assert (
        find_banned_summary_pattern(payload["summary"])
        == "the user/assistant is asking/instructing/discussing/explaining"
    )
    grade = grade_candidate(payload, gold, chunk_text="BrainLayer PR #423 added eval harness work.")
    assert grade.disqualified is True
    assert grade.overall_score == 0.0
    assert grade.banned_pattern_hit is True


def test_key_facts_recall_matches_gold_values_in_summary_or_key_facts():
    from brainlayer.eval.enrichment_graders import score_key_facts_recall

    result = score_key_facts_recall(
        _candidate(key_facts=["PR #423", "June 1, 2026"]),
        {"must_capture_facts": ["PR #423", "$5.00", "src/brainlayer/eval/enrichment_gold.py", "June 1, 2026"]},
    )

    assert result.matched == ("PR #423", "src/brainlayer/eval/enrichment_gold.py", "June 1, 2026")
    assert result.missed == ("$5.00",)
    assert result.recall == 0.75


def test_entity_metrics_report_name_and_type_strict_scores_plus_hallucinations():
    from brainlayer.eval.enrichment_graders import score_entities

    candidate = _candidate(
        entities=[
            {"name": "BrainLayer", "type": "project", "relation": "memory layer"},
            {"name": "SQLite", "type": "tool", "relation": "mismatched type"},
            {"name": "Nonexistent Vendor", "type": "company", "relation": "hallucinated"},
        ]
    )
    gold = {
        "gold_entities": [
            {"name": "brain layer", "type": "project", "relation": "memory layer"},
            {"name": "SQLite", "type": "technology", "relation": "snapshot store"},
            {"name": "Etan Heyman", "type": "person", "relation": "owner"},
        ]
    }
    chunk_text = "Etan Heyman uses BrainLayer with SQLite for the enrichment eval harness."

    result = score_entities(candidate, gold, chunk_text)

    assert result.name_precision == 2 / 3
    assert result.name_recall == 2 / 3
    assert result.name_f1 == 2 / 3
    assert result.type_strict_precision == 1 / 3
    assert result.type_strict_recall == 1 / 3
    assert result.type_strict_f1 == 1 / 3
    assert result.hallucinated_entities == ("Nonexistent Vendor",)


def test_importance_calibration_reports_mae_spearman_and_band_accuracy():
    from brainlayer.eval.enrichment_graders import score_importance_calibration

    result = score_importance_calibration([2, 5, 9, 10], [2, 6, 8, 10])

    assert result.mae == 0.5
    assert result.spearman_rho == 1.0
    assert result.band_accuracy == 1.0

    inverted = score_importance_calibration([10, 9, 5, 2], [2, 6, 8, 10])
    assert inverted.spearman_rho == -1.0
    assert inverted.band_accuracy == 0.0


def test_meta_research_forced_two_check_is_literal_and_deterministic():
    from brainlayer.eval.enrichment_graders import check_meta_research_forced_importance

    chunk_text = 'Tool output: brain_search("enrichment eval harness") returned prior research.'

    assert check_meta_research_forced_importance(chunk_text, _candidate(importance=2)).passed is True
    failed = check_meta_research_forced_importance(chunk_text, _candidate(importance=4))
    assert failed.passed is False
    assert failed.expected_importance == 2
    assert failed.actual_importance == 4
    assert (
        check_meta_research_forced_importance("Plain text mentioning brain search", _candidate(importance=8)).passed
        is True
    )


def test_grade_candidate_combines_pure_deterministic_metrics():
    from brainlayer.eval.enrichment_graders import grade_candidate

    gold = {
        "must_capture_facts": ["PR #423", "src/brainlayer/eval/enrichment_gold.py", "June 1, 2026"],
        "gold_entities": [
            {"name": "BrainLayer", "type": "project", "relation": "memory layer"},
            {"name": "SQLite", "type": "technology", "relation": "snapshot store"},
        ],
        "gold_importance": 8,
    }
    chunk_text = (
        "BrainLayer PR #423 added src/brainlayer/eval/enrichment_gold.py on June 1, 2026. "
        "The sampler reads SQLite snapshots offline."
    )

    grade = grade_candidate(_candidate(), gold, chunk_text=chunk_text)

    assert grade.disqualified is False
    assert grade.schema.passed is True
    assert grade.key_facts.recall == 1.0
    assert grade.entities.name_f1 == 1.0
    assert grade.importance.mae == 0.0
    assert grade.meta_research.passed is True
    assert 0.0 < grade.overall_score <= 1.0
    assert math.isclose(grade.overall_score, 1.0)
