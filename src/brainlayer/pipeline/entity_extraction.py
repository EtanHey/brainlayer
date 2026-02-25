"""Entity extraction from text chunks.

Extracts named entities and relationships using multiple strategies:
1. Seed entity matching (known entities by string match)
2. LLM-based extraction (structured output from local LLM)
3. ML model extraction (GLiNER/DictaBERT — future)

Each strategy returns ExtractedEntity/ExtractedRelation with confidence scores
and source provenance.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExtractedEntity:
    """A named entity extracted from text."""

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    source: str  # "seed", "llm", "gliner", "dictabert"


@dataclass
class ExtractedRelation:
    """A relationship between two extracted entities."""

    source_text: str
    target_text: str
    relation_type: str
    confidence: float
    properties: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Combined extraction output for a single chunk."""

    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    chunk_id: str
    metadata: dict = field(default_factory=dict)


def extract_seed_entities(
    text: str,
    seed_entities: dict[str, list[str]],
) -> list[ExtractedEntity]:
    """Find known seed entities in text by string matching.

    Case-insensitive matching with word boundary awareness.
    Returns entities with exact character spans.
    """
    if not text:
        return []

    results: list[ExtractedEntity] = []
    text_lower = text.lower()

    for entity_type, names in seed_entities.items():
        for name in names:
            name_lower = name.lower()
            # Find all occurrences (case-insensitive)
            start = 0
            while True:
                idx = text_lower.find(name_lower, start)
                if idx == -1:
                    break

                # Use the original text's casing for the match
                matched_text = text[idx : idx + len(name)]

                results.append(
                    ExtractedEntity(
                        text=matched_text,
                        entity_type=entity_type,
                        start=idx,
                        end=idx + len(name),
                        confidence=0.95,
                        source="seed",
                    )
                )

                start = idx + len(name)

    # Sort by start position, deduplicate overlapping spans
    results.sort(key=lambda e: (e.start, -len(e.text)))
    return _deduplicate_overlaps(results)


def _deduplicate_overlaps(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Remove overlapping entities, keeping the longest match."""
    if not entities:
        return []

    deduped: list[ExtractedEntity] = []
    for entity in entities:
        # Check if this overlaps with any already-kept entity
        overlaps = False
        for kept in deduped:
            if entity.start < kept.end and kept.start < entity.end:
                overlaps = True
                break
        if not overlaps:
            deduped.append(entity)

    return deduped


# ── LLM-based extraction ──

_NER_PROMPT_TEMPLATE = """Extract named entities and relationships from this text.

Entity types: person, company, project, golem, tool, topic
Relation types: works_at, owns, builds, uses, client_of, mentioned_in

Return JSON only, no explanation:
{{"entities": [{{"text": "exact text from input", "type": "entity_type"}}], "relations": [{{"source": "entity text", "target": "entity text", "type": "relation_type"}}]}}

If no entities found, return: {{"entities": [], "relations": []}}

Text:
{text}"""


def build_ner_prompt(text: str) -> str:
    """Build the NER extraction prompt for a text chunk."""
    return _NER_PROMPT_TEMPLATE.format(text=text)


def parse_llm_ner_response(response: str, source_text: str) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Parse LLM NER response into entities and relations with spans.

    The LLM returns entity names and types. We locate the actual character
    spans by searching in the source text. This is more reliable than asking
    the LLM for exact offsets.
    """
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []

    if not response:
        return entities, relations

    # Extract JSON from response
    parsed = _extract_json(response)
    if not parsed:
        return entities, relations

    source_lower = source_text.lower()

    # Parse entities
    for raw_entity in parsed.get("entities", []):
        text = raw_entity.get("text", "")
        etype = raw_entity.get("type", "")
        if not text or not etype:
            continue

        # Find span in source text (case-insensitive)
        idx = source_lower.find(text.lower())
        if idx == -1:
            continue

        entities.append(
            ExtractedEntity(
                text=source_text[idx : idx + len(text)],
                entity_type=etype,
                start=idx,
                end=idx + len(text),
                confidence=0.8,
                source="llm",
            )
        )

    # Parse relations
    for raw_rel in parsed.get("relations", []):
        source = raw_rel.get("source", "")
        target = raw_rel.get("target", "")
        rtype = raw_rel.get("type", "")
        if not source or not target or not rtype:
            continue

        relations.append(
            ExtractedRelation(
                source_text=source,
                target_text=target,
                relation_type=rtype,
                confidence=0.7,
                properties=raw_rel.get("properties", {}),
            )
        )

    # Deduplicate overlapping entity spans
    entities.sort(key=lambda e: (e.start, -len(e.text)))
    entities = _deduplicate_overlaps(entities)

    return entities, relations


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Extract JSON object from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON in the response
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def extract_entities_llm(
    text: str,
    llm_caller: Optional[Any] = None,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Extract entities using LLM (Ollama/MLX).

    Args:
        text: Source text to extract from.
        llm_caller: Callable(prompt) -> str. If None, uses enrichment.call_llm.

    Returns:
        Tuple of (entities, relations).
    """
    if not text.strip():
        return [], []

    prompt = build_ner_prompt(text)

    if llm_caller is None:
        from .enrichment import call_llm

        llm_caller = call_llm

    response = llm_caller(prompt)
    if not response:
        return [], []

    return parse_llm_ner_response(response, text)


# ── GLiNER-based extraction ──

# Lazy-loaded GLiNER model singleton
_gliner_model = None

# Map BrainLayer entity types to GLiNER labels
_GLINER_LABELS = [
    "person",
    "company",
    "organization",
    "project",
    "software tool",
    "programming language",
    "framework",
    "technology",
    "product",
]

# Map GLiNER labels back to our entity types
_GLINER_TYPE_MAP = {
    "person": "person",
    "company": "company",
    "organization": "company",
    "project": "project",
    "software tool": "tool",
    "programming language": "tool",
    "framework": "tool",
    "technology": "tool",
    "product": "project",
}


def _get_gliner_model():
    """Load GLiNER multi model (lazy singleton)."""
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER

        _gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    return _gliner_model


def extract_entities_gliner(
    text: str,
    labels: Optional[list[str]] = None,
    threshold: float = 0.3,
) -> list[ExtractedEntity]:
    """Extract entities using GLiNER multi-lingual model.

    Supports English and Hebrew in a single model (mdeberta backbone).
    """
    if not text.strip():
        return []

    model = _get_gliner_model()
    if labels is None:
        labels = _GLINER_LABELS

    predictions = model.predict_entities(text, labels, threshold=threshold)

    entities = []
    for pred in predictions:
        entity_type = _GLINER_TYPE_MAP.get(pred["label"], pred["label"])
        entities.append(
            ExtractedEntity(
                text=pred["text"],
                entity_type=entity_type,
                start=pred["start"],
                end=pred["end"],
                confidence=float(pred["score"]),
                source="gliner",
            )
        )

    entities.sort(key=lambda e: (e.start, -len(e.text)))
    return _deduplicate_overlaps(entities)


def extract_entities_combined(
    text: str,
    seed_entities: dict[str, list[str]],
    llm_caller: Optional[Any] = None,
    use_llm: bool = True,
    use_gliner: bool = False,
) -> ExtractionResult:
    """Run all extraction strategies and merge results.

    Priority: seed matches (highest confidence) > GLiNER > LLM extraction.
    Deduplicates overlapping spans, keeping the highest-confidence match.
    """
    all_entities: list[ExtractedEntity] = []
    all_relations: list[ExtractedRelation] = []

    # 1. Seed entity matching (fast, high confidence)
    seed_results = extract_seed_entities(text, seed_entities)
    all_entities.extend(seed_results)

    # 2. GLiNER extraction (fast ML, good confidence)
    if use_gliner:
        gliner_entities = extract_entities_gliner(text)
        all_entities.extend(gliner_entities)

    # 3. LLM-based extraction (slower, moderate confidence)
    if use_llm:
        llm_entities, llm_relations = extract_entities_llm(text, llm_caller)
        all_entities.extend(llm_entities)
        all_relations.extend(llm_relations)

    # Deduplicate: sort by confidence (desc), then by start position
    all_entities.sort(key=lambda e: (-e.confidence, e.start))

    # Keep highest-confidence for overlapping spans
    final_entities: list[ExtractedEntity] = []
    for entity in all_entities:
        overlaps = False
        for kept in final_entities:
            if entity.start < kept.end and kept.start < entity.end:
                overlaps = True
                break
        if not overlaps:
            final_entities.append(entity)

    final_entities.sort(key=lambda e: e.start)

    return ExtractionResult(
        entities=final_entities,
        relations=all_relations,
        chunk_id="",
    )
