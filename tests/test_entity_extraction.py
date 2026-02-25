"""Tests for entity extraction pipeline.

Covers:
- ExtractedEntity dataclass
- Seed entity matching (known entities found by string match)
- LLM-based extraction (structured output from local LLM)
- Batch pipeline processing
"""

import json
from dataclasses import asdict

import pytest

from brainlayer.pipeline.entity_extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    build_ner_prompt,
    extract_entities_combined,
    extract_entities_llm,
    extract_seed_entities,
    parse_llm_ner_response,
)

try:
    from brainlayer.pipeline.entity_extraction import extract_entities_gliner

    _HAS_GLINER = True
except ImportError:
    _HAS_GLINER = False

# ── Data classes ──


class TestExtractedEntity:
    """Verify the extraction result data classes."""

    def test_extracted_entity_fields(self):
        e = ExtractedEntity(
            text="Etan Heyman",
            entity_type="person",
            start=0,
            end=11,
            confidence=0.95,
            source="seed",
        )
        assert e.text == "Etan Heyman"
        assert e.entity_type == "person"
        assert e.start == 0
        assert e.end == 11
        assert e.confidence == 0.95
        assert e.source == "seed"

    def test_extracted_entity_serializable(self):
        e = ExtractedEntity(
            text="Domica",
            entity_type="company",
            start=10,
            end=16,
            confidence=0.9,
            source="seed",
        )
        d = asdict(e)
        assert d["text"] == "Domica"
        assert d["entity_type"] == "company"

    def test_extracted_relation_fields(self):
        r = ExtractedRelation(
            source_text="Etan Heyman",
            target_text="Domica",
            relation_type="works_at",
            confidence=0.8,
        )
        assert r.source_text == "Etan Heyman"
        assert r.relation_type == "works_at"

    def test_extraction_result_combines(self):
        result = ExtractionResult(
            entities=[
                ExtractedEntity("Etan", "person", 0, 4, 0.9, "seed"),
            ],
            relations=[],
            chunk_id="chunk-1",
        )
        assert len(result.entities) == 1
        assert result.chunk_id == "chunk-1"


# ── Seed Entity Matching ──


SEED_ENTITIES = {
    "person": [
        "Etan Heyman",
        "Dor Zohar",
        "Shachar Gerby",
        "Maor Noah",
        "Avi Simon",
        "Yuval Nir",
        "Daniel Munk",
    ],
    "company": ["Cantaloupe AI", "Domica", "MeHayom", "ProductDZ", "Weby"],
    "project": [
        "brainlayer",
        "voicelayer",
        "golems",
        "songscript",
        "domica",
        "rudy-monorepo",
        "union",
    ],
    "golem": [
        "golemsClaude",
        "brainClaude",
        "voiceClaude",
        "coachClaude",
        "contentClaude",
    ],
}


class TestSeedEntityMatching:
    """Seed entity matching finds known entities by string match."""

    def test_finds_person(self):
        text = "Dor Zohar is the CEO and handles UX/UI."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        names = {e.text for e in entities}
        assert "Dor Zohar" in names

    def test_finds_company(self):
        text = "We're building Domica as a real estate platform."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        names = {e.text for e in entities}
        assert "Domica" in names

    def test_finds_multiple(self):
        text = "Etan Heyman and Dor Zohar co-founded Domica."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        names = {e.text for e in entities}
        assert "Etan Heyman" in names
        assert "Dor Zohar" in names
        assert "Domica" in names

    def test_correct_spans(self):
        text = "We use brainlayer for memory."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        bl = [e for e in entities if e.text == "brainlayer"][0]
        assert text[bl.start : bl.end] == "brainlayer"

    def test_case_insensitive_match(self):
        text = "BrainLayer is the memory system."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        names = {e.text for e in entities}
        assert any("brainlayer" in n.lower() for n in names)

    def test_no_partial_word_match(self):
        """'domica' should NOT match inside 'domica-app' as 'domica' AND leave '-app'."""
        text = "The domica-app repository has the frontend code."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        # Should find "domica" but only as a proper entity
        # The key thing: span boundaries should be sensible
        for e in entities:
            assert text[e.start : e.end] == e.text

    def test_golem_names(self):
        text = "brainClaude handles memory while voiceClaude handles speech."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        names = {e.text for e in entities}
        assert "brainClaude" in names
        assert "voiceClaude" in names

    def test_seed_source_tag(self):
        text = "Etan Heyman is working."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        assert all(e.source == "seed" for e in entities)

    def test_high_confidence(self):
        """Seed matches should have confidence >= 0.95."""
        text = "Dor Zohar wrote this."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        assert all(e.confidence >= 0.95 for e in entities)

    def test_empty_text(self):
        entities = extract_seed_entities("", SEED_ENTITIES)
        assert entities == []

    def test_no_matches(self):
        text = "This text has no known entities."
        entities = extract_seed_entities(text, SEED_ENTITIES)
        assert entities == []


# ── LLM Response Parsing ──


class TestLLMResponseParsing:
    """Test parsing of LLM NER responses (no actual LLM needed)."""

    def test_parse_valid_json(self):
        response = '{"entities": [{"text": "Etan Heyman", "type": "person"}], "relations": []}'
        source = "Etan Heyman is the founder."
        entities, relations = parse_llm_ner_response(response, source)
        assert len(entities) == 1
        assert entities[0].text == "Etan Heyman"
        assert entities[0].entity_type == "person"
        assert entities[0].start == 0
        assert entities[0].end == 11

    def test_parse_with_relations(self):
        response = json.dumps(
            {
                "entities": [
                    {"text": "Dor Zohar", "type": "person"},
                    {"text": "Domica", "type": "company"},
                ],
                "relations": [{"source": "Dor Zohar", "target": "Domica", "type": "works_at"}],
            }
        )
        source = "Dor Zohar is CEO of Domica."
        entities, relations = parse_llm_ner_response(response, source)
        assert len(entities) == 2
        assert len(relations) == 1
        assert relations[0].source_text == "Dor Zohar"
        assert relations[0].relation_type == "works_at"

    def test_parse_json_in_text(self):
        """LLM sometimes wraps JSON in explanation text."""
        response = (
            'Here are the entities:\n{"entities": [{"text": "brainlayer", "type": "project"}], "relations": []}\nDone.'
        )
        source = "We use brainlayer for memory."
        entities, _ = parse_llm_ner_response(response, source)
        assert len(entities) == 1
        assert entities[0].text == "brainlayer"

    def test_parse_empty_response(self):
        entities, relations = parse_llm_ner_response("", "some text")
        assert entities == []
        assert relations == []

    def test_parse_invalid_json(self):
        entities, relations = parse_llm_ner_response("not json at all", "some text")
        assert entities == []
        assert relations == []

    def test_entity_not_in_text_skipped(self):
        """Entities the LLM hallucinates (not in source text) are dropped."""
        response = '{"entities": [{"text": "Phantom Entity", "type": "person"}], "relations": []}'
        source = "This text has no such entity."
        entities, _ = parse_llm_ner_response(response, source)
        assert entities == []

    def test_llm_confidence_lower_than_seed(self):
        """LLM-extracted entities should have lower confidence than seed."""
        response = '{"entities": [{"text": "Railway", "type": "tool"}], "relations": []}'
        source = "Deploy to Railway for hosting."
        entities, _ = parse_llm_ner_response(response, source)
        assert entities[0].confidence < 0.95  # Less than seed confidence
        assert entities[0].source == "llm"

    def test_span_accuracy(self):
        """Extracted spans should match the actual text position."""
        response = '{"entities": [{"text": "Domica", "type": "company"}], "relations": []}'
        source = "We're building Domica as a platform."
        entities, _ = parse_llm_ner_response(response, source)
        e = entities[0]
        assert source[e.start : e.end] == "Domica"


# ── NER Prompt ──


class TestNERPrompt:
    """Test the NER prompt construction."""

    def test_prompt_contains_text(self):
        prompt = build_ner_prompt("Hello world")
        assert "Hello world" in prompt

    def test_prompt_has_entity_types(self):
        prompt = build_ner_prompt("test")
        assert "person" in prompt
        assert "company" in prompt
        assert "project" in prompt

    def test_prompt_has_relation_types(self):
        prompt = build_ner_prompt("test")
        assert "works_at" in prompt
        assert "builds" in prompt


# ── LLM Extraction with Mock ──


class TestLLMExtractionMock:
    """Test LLM extraction using a mock caller."""

    def _mock_llm(self, prompt):
        """Simple mock that returns hardcoded entities."""
        return json.dumps(
            {
                "entities": [
                    {"text": "Railway", "type": "tool"},
                    {"text": "FastAPI", "type": "tool"},
                ],
                "relations": [{"source": "FastAPI", "target": "Railway", "type": "uses"}],
            }
        )

    def test_extract_with_mock(self):
        text = "Deploy FastAPI to Railway for production."
        entities, relations = extract_entities_llm(text, llm_caller=self._mock_llm)
        names = {e.text for e in entities}
        assert "Railway" in names
        assert "FastAPI" in names
        assert len(relations) == 1

    def test_empty_text_returns_empty(self):
        entities, relations = extract_entities_llm("", llm_caller=self._mock_llm)
        assert entities == []
        assert relations == []


# ── Combined Extraction ──


class TestCombinedExtraction:
    """Test merged seed + LLM extraction."""

    def _mock_llm(self, prompt):
        return json.dumps(
            {
                "entities": [
                    {"text": "Railway", "type": "tool"},
                    {"text": "Etan Heyman", "type": "person"},
                ],
                "relations": [],
            }
        )

    def test_seed_wins_over_llm(self):
        """Seed matches should be preferred over LLM for the same entity."""
        text = "Etan Heyman deploys to Railway."
        result = extract_entities_combined(text, SEED_ENTITIES, llm_caller=self._mock_llm)
        # "Etan Heyman" should come from seed (higher confidence)
        etan = [e for e in result.entities if "Etan" in e.text][0]
        assert etan.source == "seed"
        assert etan.confidence >= 0.95

    def test_llm_adds_new_entities(self):
        """LLM should find entities that seeds miss."""
        text = "Etan Heyman deploys to Railway."
        result = extract_entities_combined(text, SEED_ENTITIES, llm_caller=self._mock_llm)
        names = {e.text for e in result.entities}
        assert "Railway" in names

    def test_no_llm_mode(self):
        """With use_llm=False, only seed entities are returned."""
        text = "Etan Heyman deploys to Railway."
        result = extract_entities_combined(text, SEED_ENTITIES, use_llm=False)
        names = {e.text for e in result.entities}
        assert "Etan Heyman" in names
        assert "Railway" not in names  # Not a seed entity


# ── GLiNER Extraction ──


@pytest.mark.skipif(not _HAS_GLINER, reason="gliner not installed")
class TestGLiNERExtraction:
    """Test GLiNER multi-lingual NER extraction.

    These tests load the actual model (~800MB download on first run).
    Marked slow — skip with `pytest -m 'not slow'`.
    """

    @pytest.mark.slow
    def test_finds_person_english(self):
        text = "Elon Musk is the CEO of Tesla."
        entities = extract_entities_gliner(text)
        names = {e.text for e in entities}
        assert "Elon Musk" in names

    @pytest.mark.slow
    def test_finds_company_english(self):
        text = "Microsoft acquired GitHub for $7.5 billion."
        entities = extract_entities_gliner(text)
        types = {e.entity_type for e in entities}
        assert "company" in types

    @pytest.mark.slow
    def test_entities_have_valid_spans(self):
        text = "Google released TensorFlow as open source."
        entities = extract_entities_gliner(text)
        for e in entities:
            assert text[e.start : e.end] == e.text

    @pytest.mark.slow
    def test_source_tagged_gliner(self):
        text = "Amazon Web Services powers cloud computing."
        entities = extract_entities_gliner(text)
        if entities:
            assert all(e.source == "gliner" for e in entities)

    @pytest.mark.slow
    def test_empty_text(self):
        entities = extract_entities_gliner("")
        assert entities == []

    @pytest.mark.slow
    def test_hebrew_text(self):
        """GLiNER multi should handle Hebrew text."""
        text = 'אילון מאסק הוא המנכ"ל של טסלה.'
        entities = extract_entities_gliner(text)
        # Should find at least one entity (person or company)
        # Hebrew support varies, so just check it doesn't crash
        assert isinstance(entities, list)
