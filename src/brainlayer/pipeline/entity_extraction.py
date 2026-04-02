"""Entity extraction from text chunks.

Extracts named entities and relationships using multiple strategies:
1. Seed entity matching (known entities by string match)
2. LLM-based extraction (structured output from local LLM)
3. ML model extraction (GLiNER/DictaBERT — future)

Each strategy returns ExtractedEntity/ExtractedRelation with confidence scores
and source provenance.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


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

_NER_PROMPT_TEMPLATE = """Extract named entities and relationships from this developer conversation text.

Entity types: person, agent, company, project, tool, technology, topic
- person: Human names (First Last). NOT repos/tools/agents.
- agent: AI agents (*Claude, *Golem, Ralph). NOT humans.
- company: Businesses. project: Code repos/apps. tool/technology: Dev tools, languages, frameworks.

Relation types (direction: source → target):
- works_at: person → company. owns: person → project/company. builds: person/agent → project.
- uses: entity → tool/technology. client_of: A → B (B serves A). affiliated_with: person → company.
- coaches: agent → person. related_to: generic fallback.

Return JSON only:
{{"entities": [{{"text": "exact text from input", "type": "entity_type"}}], "relations": [{{"source": "entity text", "target": "entity text", "type": "relation_type", "fact": "natural language sentence"}}]}}

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

        fact = raw_rel.get("fact")
        props = raw_rel.get("properties") or {}
        if fact and "fact" not in props:
            props["fact"] = fact

        relations.append(
            ExtractedRelation(
                source_text=source,
                target_text=target,
                relation_type=rtype,
                confidence=0.7,
                properties=props,
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

    try:
        response = llm_caller(prompt)
    except Exception:
        logger.exception("LLM caller failed in extract_entities_llm")
        return [], []

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


# ── Tag-based extraction ──

# Known technology tags → technology entities
KNOWN_TECH_TAGS = {
    "python",
    "typescript",
    "javascript",
    "react",
    "nextjs",
    "fastapi",
    "sqlite",
    "docker",
    "railway",
    "vercel",
    "supabase",
    "convex",
    "tailwind",
    "playwright",
    "bun",
    "nodejs",
    "rust",
    "go",
    "redis",
    "postgresql",
    "mongodb",
    "graphql",
    "prisma",
    "drizzle",
    "trpc",
    "astro",
    "svelte",
    "vue",
    "angular",
    "express",
    "flask",
    "django",
    "pytorch",
    "tensorflow",
    "langchain",
    "chromadb",
    "pinecone",
    "openai",
    "anthropic",
    "groq",
    "ollama",
    "mlx",
    "huggingface",
    "kubernetes",
    "terraform",
    "aws",
    "gcp",
    "azure",
    "cloudflare",
    "github",
    "gitlab",
    "linear",
    "figma",
    "notion",
    "obsidian",
    "turborepo",
    "nx",
    "pnpm",
    "yarn",
    "npm",
    "vite",
    "webpack",
    "jest",
    "vitest",
    "pytest",
    "cypress",
    "selenium",
    "spacy",
    "gliner",
    "apsw",
    "tree-sitter",
    "ruff",
}

# Known project tags → project entities
KNOWN_PROJECT_TAGS = {
    "brainlayer",
    "voicelayer",
    "golems",
    "songscript",
    "domica",
    "rudy-monorepo",
    "union",
    "6pm",
    "6pm-mini",
    "soltome",
    "orchestrator",
    "golem-profiles",
}


def extract_entities_from_tags(
    tags: list[str],
    known_tech: Optional[set[str]] = None,
    known_projects: Optional[set[str]] = None,
) -> list[ExtractedEntity]:
    """Extract entities from enrichment tags without API calls.

    Maps known tags to entity types. Returns ExtractedEntity objects
    with source="tag" and span positions of -1 (not text-based).
    """
    if known_tech is None:
        known_tech = KNOWN_TECH_TAGS
    if known_projects is None:
        known_projects = KNOWN_PROJECT_TAGS

    entities = []
    seen_norms: set[str] = set()
    norm_projects = {p.lower().replace("-", "").replace("_", "").replace(".", ""): p for p in known_projects}
    norm_tech = {t.lower().replace("-", "").replace("_", "").replace(".", ""): t for t in known_tech}

    for tag in tags:
        if not isinstance(tag, str):
            continue
        tag_norm = tag.lower().replace("-", "").replace("_", "").replace(".", "")
        if tag_norm in seen_norms:
            continue
        seen_norms.add(tag_norm)
        # Check projects first (higher priority)
        if tag_norm in norm_projects:
            entities.append(
                ExtractedEntity(
                    text=norm_projects[tag_norm],
                    entity_type="project",
                    start=-1,
                    end=-1,
                    confidence=0.85,
                    source="tag",
                )
            )
        elif tag_norm in norm_tech:
            entities.append(
                ExtractedEntity(
                    text=norm_tech[tag_norm],
                    entity_type="technology",
                    start=-1,
                    end=-1,
                    confidence=0.80,
                    source="tag",
                )
            )
    return entities


def extract_cooccurrence_relations(entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
    """Infer co-occurrence relations between entities of different types.

    Two entities in the same text that have different types are assumed to be
    related (e.g., a project uses a technology). This is a low-cost heuristic
    that runs without any LLM, producing edges for the knowledge graph.

    Only cross-type pairs are linked — same-type pairs (project-project) are
    skipped as too noisy.
    """
    relations: list[ExtractedRelation] = []
    seen: set[tuple[str, str]] = set()

    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            if a.entity_type == b.entity_type:
                continue
            pair = (a.text, b.text) if a.text < b.text else (b.text, a.text)
            if pair in seen:
                continue
            seen.add(pair)
            confidence = min(a.confidence, b.confidence) * 0.7
            relations.append(
                ExtractedRelation(
                    source_text=a.text,
                    target_text=b.text,
                    relation_type="co_occurs_with",
                    confidence=confidence,
                )
            )

    return relations


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

    # 4. Co-occurrence relations (always runs — no LLM needed)
    cooccurrence = extract_cooccurrence_relations(final_entities)
    all_relations.extend(cooccurrence)

    return ExtractionResult(
        entities=final_entities,
        relations=all_relations,
        chunk_id="",
    )
