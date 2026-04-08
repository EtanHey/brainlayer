"""Tests for enrichment entity schema — Gemini response schema and prompt wording.

Verifies that:
1. GEMINI_RESPONSE_SCHEMA includes a required 'entities' array
2. Entity items have 'name' (string) and 'type' (enum of 6 values)
3. ENRICHMENT_PROMPT mentions entities with non-code-only instructions
4. parse_enrichment correctly extracts and validates entities
5. parse_enrichment rejects entities with invalid types
6. parse_enrichment rejects entities with missing fields
"""

# ── Schema tests ──────────────────────────────────────────────────────────────


def test_gemini_response_schema_has_entities_field():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA

    props = GEMINI_RESPONSE_SCHEMA["properties"]
    assert "entities" in props, "GEMINI_RESPONSE_SCHEMA must have an 'entities' property"


def test_gemini_response_schema_entities_is_required():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA

    required = GEMINI_RESPONSE_SCHEMA.get("required", [])
    assert "entities" in required, "'entities' must be in the required fields"


def test_gemini_response_schema_entities_is_array_of_objects():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA

    entities_schema = GEMINI_RESPONSE_SCHEMA["properties"]["entities"]
    assert entities_schema["type"] == "array"
    items = entities_schema["items"]
    assert items["type"] == "object"
    assert "name" in items["properties"]
    assert "type" in items["properties"]


def test_gemini_response_schema_entity_type_enum():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA

    entity_type = GEMINI_RESPONSE_SCHEMA["properties"]["entities"]["items"]["properties"]["type"]
    expected_types = {"person", "company", "project", "technology", "tool", "concept"}
    assert set(entity_type["enum"]) == expected_types


def test_gemini_response_schema_entity_items_require_name_and_type():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA

    items = GEMINI_RESPONSE_SCHEMA["properties"]["entities"]["items"]
    assert "required" in items
    assert "name" in items["required"]
    assert "type" in items["required"]


def test_build_gemini_config_includes_response_schema():
    from brainlayer.enrichment_controller import GEMINI_RESPONSE_SCHEMA, _build_gemini_config

    config = _build_gemini_config()
    assert "response_schema" in config
    assert config["response_schema"] is GEMINI_RESPONSE_SCHEMA


# ── Prompt tests ──────────────────────────────────────────────────────────────


def test_enrichment_prompt_mentions_entities_field():
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    assert '"entities"' in ENRICHMENT_PROMPT, "ENRICHMENT_PROMPT must include an 'entities' field in the JSON example"


def test_enrichment_prompt_says_non_code_entities():
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    assert "non-code entities" in ENRICHMENT_PROMPT.lower(), (
        "ENRICHMENT_PROMPT must instruct to extract non-code entities only"
    )


def test_enrichment_prompt_excludes_code_symbols():
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    lower = ENRICHMENT_PROMPT.lower()
    assert "variable names" in lower, "Prompt must mention excluding variable names"
    assert "function names" in lower, "Prompt must mention excluding function names"
    assert "file paths" in lower, "Prompt must mention excluding file paths"
    assert "code symbols" in lower, "Prompt must mention excluding code symbols"


def test_enrichment_prompt_lists_entity_types():
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    lower = ENRICHMENT_PROMPT.lower()
    for etype in ["person", "company", "project", "technology", "tool", "concept"]:
        assert etype in lower, f"Prompt must mention entity type '{etype}'"


def test_enrichment_prompt_adds_meta_research_and_short_chunk_guidance():
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    assert "META-RESEARCH DETECTION" in ENRICHMENT_PROMPT
    assert '"meta-research"' in ENRICHMENT_PROMPT
    assert "brain_search" in ENRICHMENT_PROMPT
    assert "brain_entity" in ENRICHMENT_PROMPT
    assert "SHORT/CONVERSATIONAL CHUNK" in ENRICHMENT_PROMPT
    assert "ACTIONABLE ITEMS" in ENRICHMENT_PROMPT
    assert "COMMITMENTS" in ENRICHMENT_PROMPT


def test_enrichment_prompt_keeps_epistemic_debt_and_sentiment_rubrics():
    from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT

    lower = ENRICHMENT_PROMPT.lower()
    assert "epistemic rubric" in lower
    assert "debt impact rubric" in lower
    assert "sentiment rubric" in lower


# ── parse_enrichment entity handling ──────────────────────────────────────────


def test_parse_enrichment_extracts_valid_entities():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Discussion about React migration",
            "tags": ["react", "migration"],
            "importance": 7,
            "intent": "designing",
            "entities": [
                {"name": "React", "type": "technology"},
                {"name": "Vercel", "type": "company"},
            ],
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert "entities" in result
    assert len(result["entities"]) == 2
    assert result["entities"][0] == {"name": "React", "type": "technology"}
    assert result["entities"][1] == {"name": "Vercel", "type": "company"}


def test_parse_enrichment_rejects_invalid_entity_type():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": [
                {"name": "myVariable", "type": "variable"},  # invalid type
                {"name": "React", "type": "technology"},  # valid
            ],
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    # Only the valid entity should survive
    assert len(result.get("entities", [])) == 1
    assert result["entities"][0]["name"] == "React"


def test_parse_enrichment_rejects_entities_missing_name():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": [
                {"type": "technology"},  # missing name
                {"name": "", "type": "tool"},  # empty name
            ],
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert "entities" not in result  # no valid entities -> field omitted


def test_parse_enrichment_handles_missing_entities_gracefully():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    # entities is optional — result should still be valid without it
    assert "entities" not in result


def test_parse_enrichment_caps_entities_at_20():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": [{"name": f"Entity{i}", "type": "concept"} for i in range(30)],
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert len(result["entities"]) == 20


def test_parse_enrichment_normalizes_entity_type_case():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": [
                {"name": "React", "type": "TECHNOLOGY"},
                {"name": "Google", "type": "Company"},
            ],
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert result["entities"][0]["type"] == "technology"
    assert result["entities"][1]["type"] == "company"


def test_parse_enrichment_strips_entity_name_whitespace():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": [
                {"name": "  React  ", "type": "technology"},
            ],
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert result["entities"][0]["name"] == "React"


def test_parse_enrichment_handles_null_entities():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": None,
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert "entities" not in result


def test_parse_enrichment_handles_non_list_entities():
    import json

    from brainlayer.pipeline.enrichment import parse_enrichment

    raw = json.dumps(
        {
            "summary": "Some summary",
            "tags": ["test"],
            "importance": 5,
            "intent": "debugging",
            "entities": "not a list",
        }
    )
    result = parse_enrichment(raw)
    assert result is not None
    assert "entities" not in result
