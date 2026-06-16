from brainlayer.eval.enrichment_llm_judge import (
    FIELD_NAMES,
    aggregate_judgments,
    build_pair_requests,
    parse_judge_response,
)


def _selection_row(chunk_id: str = "chunk-1") -> dict:
    return {
        "chunk_id": chunk_id,
        "chunk_text": "DECISION: use Gemini Batch for the backlog after tags are fixed.",
        "pure_metadata": {"project": "brainlayer"},
    }


def _output_row(chunk_id: str, backend: str, summary: str, tags: list[str]) -> dict:
    return {
        "chunk_id": chunk_id,
        "backend": backend,
        "model": backend,
        "enrichment": {
            "summary": summary,
            "tags": tags,
            "importance": 8,
            "intent": "deciding",
            "epistemic_level": "validated",
            "debt_impact": "none",
            "sentiment_label": "neutral",
            "sentiment_score": 0,
        },
    }


def test_build_pair_requests_blinds_local_and_flex_candidates():
    requests = build_pair_requests(
        [_selection_row("c1"), _selection_row("c2")],
        {
            "c1": _output_row("c1", "local-mlx", "local c1", ["local"]),
            "c2": _output_row("c2", "local-mlx", "local c2", ["local"]),
        },
        {
            "c1": _output_row("c1", "gemini-flex", "flex c1", ["flex"]),
            "c2": _output_row("c2", "gemini-flex", "flex c2", ["flex"]),
        },
        seed=7,
    )

    assert len(requests) == 2
    assert {requests[0]["candidates"]["A"]["system"], requests[0]["candidates"]["B"]["system"]} == {"local", "flex"}
    assert requests[0]["source"]["chunk_text"].startswith("DECISION:")
    assert requests[0]["fields"] == list(FIELD_NAMES)


def test_parse_judge_response_extracts_strict_json_from_wrapped_text():
    response = """
    ```json
    {"field_scores":{"summary":{"A":5,"B":3,"winner":"A","reason":"A is specific"},
    "tags":{"A":4,"B":4,"winner":"tie","reason":"both useful"},
    "importance":{"A":5,"B":5,"winner":"tie","reason":"same"},
    "intent":{"A":5,"B":4,"winner":"A","reason":"A matches deciding"},
    "epistemic_level":{"A":4,"B":4,"winner":"tie","reason":"same"},
    "debt_impact":{"A":5,"B":5,"winner":"tie","reason":"same"},
    "sentiment":{"A":5,"B":5,"winner":"tie","reason":"same"}},
    "overall":{"A":4.7,"B":4.3,"winner":"A","reason":"A slightly better"}}
    ```
    """

    parsed = parse_judge_response(response)

    assert parsed["field_scores"]["summary"]["winner"] == "A"
    assert parsed["overall"]["winner"] == "A"


def test_aggregate_judgments_maps_blinded_winners_back_to_systems():
    requests = build_pair_requests(
        [_selection_row("c1")],
        {"c1": _output_row("c1", "local-mlx", "local", ["local"])},
        {"c1": _output_row("c1", "gemini-flex", "flex", ["flex"])},
        seed=1,
    )
    label_to_system = requests[0]["label_to_system"]
    winning_label = next(label for label, system in label_to_system.items() if system == "local")
    losing_label = "B" if winning_label == "A" else "A"
    judgment = {
        "chunk_id": "c1",
        "label_to_system": label_to_system,
        "field_scores": {
            field: {winning_label: 5, losing_label: 3, "winner": winning_label, "reason": "local better"}
            for field in FIELD_NAMES
        },
        "overall": {winning_label: 5, losing_label: 3, "winner": winning_label, "reason": "local better"},
    }

    summary = aggregate_judgments([judgment])

    assert summary["overall"]["winner_counts"]["local"] == 1
    assert summary["fields"]["summary"]["winner_counts"]["local"] == 1
    assert summary["fields"]["summary"]["mean_scores"]["local"] == 5
