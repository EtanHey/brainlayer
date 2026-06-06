"""Tests for closed KG extractor entity type gates."""

from brainlayer.pipeline.entity_extraction import ExtractedEntity, ExtractionResult, parse_llm_ner_response
from brainlayer.pipeline.kg_extraction import process_extraction_result
from brainlayer.pipeline.kg_extraction_groq import parse_multi_chunk_response
from brainlayer.vector_store import VectorStore


def test_parse_llm_ner_response_normalizes_closed_entity_types():
    """LLM entity types should be normalized before becoming ExtractedEntity rows."""
    source = "OpenAI published the Huberman Lab newsletter."
    response = """
    {
      "entities": [
        {"text": "OpenAI", "type": "organization"},
        {"text": "Huberman Lab newsletter", "type": "Source", "entity_subtype": "newsletter"}
      ],
      "relations": []
    }
    """

    entities, _ = parse_llm_ner_response(response, source)

    assert [(entity.text, entity.entity_type, entity.entity_subtype) for entity in entities] == [
        ("OpenAI", "company", None),
        ("Huberman Lab newsletter", "source", "newsletter"),
    ]


def test_parse_llm_ner_response_coerces_unknown_type_to_topic():
    """Unknown model-emitted entity types should not pass through raw."""
    source = "NovelNoise is part of graph RAG."
    response = '{"entities": [{"text": "NovelNoise", "type": "artifact"}], "relations": []}'

    entities, _ = parse_llm_ner_response(response, source)

    assert len(entities) == 1
    assert entities[0].entity_type == "topic"


def test_parse_llm_ner_response_applies_source_heuristics():
    """Source-like names should override bad model types and carry subtype."""
    source = "I learned it from youtube.com/@t3dotgg and Huberman Lab Podcast."
    response = """
    {
      "entities": [
        {"text": "youtube.com/@t3dotgg", "type": "project"},
        {"text": "Huberman Lab Podcast", "type": "company"}
      ],
      "relations": []
    }
    """

    entities, _ = parse_llm_ner_response(response, source)

    assert [(entity.text, entity.entity_type, entity.entity_subtype) for entity in entities] == [
        ("youtube.com/@t3dotgg", "source", "channel"),
        ("Huberman Lab Podcast", "source", "podcast"),
    ]


def test_parse_multi_chunk_response_normalizes_entity_types():
    """Groq batch output should use the same closed type gate."""
    response = """
    {
      "chunks": [
        {
          "chunk_id": "chunk-1",
          "entities": [
            {"text": "OpenAI", "type": "organization"},
            {"text": "youtube.com/@t3dotgg", "type": "project"}
          ],
          "relations": []
        }
      ]
    }
    """

    chunks = parse_multi_chunk_response(response)

    assert chunks[0]["entities"] == [
        {"text": "OpenAI", "type": "company"},
        {"text": "youtube.com/@t3dotgg", "type": "source", "entity_subtype": "channel"},
    ]


def test_process_extraction_result_persists_source_subtype(tmp_path):
    """The extraction-to-KG write path should not drop Source facets."""
    store = VectorStore(tmp_path / "kg.db")
    try:
        result = ExtractionResult(
            entities=[ExtractedEntity("youtube.com/@t3dotgg", "source", 0, 19, 0.8, "llm", "channel")],
            relations=[],
            chunk_id="",
        )

        process_extraction_result(store, result)

        row = (
            store._read_cursor()
            .execute(
                "SELECT entity_type, entity_subtype FROM kg_entities WHERE name = ?",
                ("youtube.com/@t3dotgg",),
            )
            .fetchone()
        )
        assert row is not None
        assert row[0] == "source"
        assert row[1] == "channel"
    finally:
        store.close()
