"""Tests for KG rebuild adapters."""

from scripts.kg_rebuild import extracted_entity_from_groq_payload


def test_groq_rebuild_entity_payload_preserves_source_subtype():
    entity = extracted_entity_from_groq_payload(
        {"text": "youtube.com/@t3dotgg", "type": "source", "entity_subtype": "channel"},
        "Watch youtube.com/@t3dotgg for the update.",
    )

    assert entity is not None
    assert entity.entity_type == "source"
    assert entity.entity_subtype == "channel"
    assert entity.start == 6
