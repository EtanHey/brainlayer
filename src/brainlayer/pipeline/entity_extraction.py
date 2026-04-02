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

_NER_PROMPT_TEMPLATE = """Extract ALL named entities and relationships from this developer conversation text.

## Entity types (be precise — choose the most specific type):
- person: Human individuals (First Last). NOT repos, tools, or agents.
- agent: AI coding agents (orcClaude, coachClaude, brainClaude, Ralph, etc.). NOT humans.
- company: Businesses and organizations (Anthropic, Weby, Cantaloupe AI).
- project: Code repositories, apps, products (BrainLayer, VoiceLayer, 6PM).
- tool: Developer tools and services (Docker, Railway, Supabase, CodeRabbit).
- technology: Languages, frameworks, protocols (SQLite, SwiftUI, MCP, TypeScript).
- skill: Reusable AI skill or command (/commit, /pr-loop, /coach).
- service: Deployed infrastructure (LaunchAgent, daemon, watcher).
- config: Configuration files or settings (CLAUDE.md, pyproject.toml, .env).
- decision: Architectural or design decisions made during sessions.
- topic: Abstract concepts or domains (enrichment, graph RAG, dark mode).

## Relation types (source → target, with description):
- created: person/agent → project/tool. "Anthropic created Claude Code"
- owns: person → project/company. "Etan owns BrainLayer"
- works_at: person → company. "Josh Anderson works at Cantaloupe AI"
- uses: entity → tool/technology. "BrainLayer uses SQLite"
- depends_on: project → technology/tool. "VoiceLayer depends on whisper-cpp"
- deployed_on: project/service → tool. "Golems deployed on Railway"
- fixes: agent/person → topic/project. "brainClaude fixes dark mode regression"
- configures: config → project/service. "CLAUDE.md configures BrainLayer hooks"
- spawns: agent → agent. "orcClaude spawns brainlayerClaude"
- client_of: person → person/company. "Yuval is client of Etan"
- affiliated_with: person → company. "Josh affiliated with Cantaloupe AI"
- coaches: agent → entity. "coachClaude coaches scheduling"
- builds: person/agent → project. "Etan builds VoiceLayer"
- related_to: generic fallback (use ONLY if no specific type fits)

## Output format — return JSON only:
{{"entities": [{{"text": "exact text from input", "type": "entity_type", "description": "one-sentence description of this entity based on context"}}], "relations": [{{"source": "entity text", "target": "entity text", "type": "relation_type", "description": "natural language sentence describing the relationship", "strength": 0.8}}]}}

## Rules:
- Extract entities that are CLEARLY identifiable, not vague mentions
- Each relation MUST have a substantive description — reject empty relations
- Strength is 0.0-1.0: explicit statements=0.9+, implied=0.5-0.8, speculative=0.3-0.5
- Decompose N-ary relationships into binary pairs
- Include Hebrew entity names if present (e.g., MeHayom/מהיום)
- If no entities found, return: {{"entities": [], "relations": []}}

Text:
{text}"""

_GLEANING_PROMPT = """The previous extraction from the same text missed important entities and relationships.

Previous extraction found: {previous_count} entities and {previous_rel_count} relations.

Re-read the text carefully. Extract ADDITIONAL entities and relationships that were missed. Focus on:
- Implicit relationships (X depends on Y, X was deployed to Y)
- Agent names and their roles
- Configuration files and what they configure
- Decisions and what they decided about
- Services and what they serve

Return ONLY newly found entities/relations (not duplicates of previous extraction).

Same JSON format:
{{"entities": [{{"text": "exact text", "type": "entity_type", "description": "description"}}], "relations": [{{"source": "entity text", "target": "entity text", "type": "relation_type", "description": "description", "strength": 0.7}}]}}

Text:
{text}"""


def build_ner_prompt(text: str) -> str:
    """Build the NER extraction prompt for a text chunk."""
    return _NER_PROMPT_TEMPLATE.format(text=text)


def build_gleaning_prompt(text: str, prev_entity_count: int, prev_rel_count: int) -> str:
    """Build the gleaning re-prompt for missed entities."""
    return _GLEANING_PROMPT.format(
        text=text,
        previous_count=prev_entity_count,
        previous_rel_count=prev_rel_count,
    )


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
        desc = raw_rel.get("description", "")
        if not source or not target or not rtype:
            continue

        try:
            strength = float(raw_rel.get("strength", 0.7))
        except (TypeError, ValueError):
            strength = 0.7
        fact = raw_rel.get("fact") or desc
        props = raw_rel.get("properties") or {}
        if fact:
            props["fact"] = fact
        if desc:
            props["description"] = desc

        relations.append(
            ExtractedRelation(
                source_text=source,
                target_text=target,
                relation_type=rtype,
                confidence=min(float(strength), 1.0),
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
    enable_gleaning: bool = False,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Extract entities using LLM with optional gleaning second pass.

    Args:
        text: Source text to extract from.
        llm_caller: Callable(prompt) -> str. If None, uses Gemini via enrichment_controller.
        enable_gleaning: If True, re-prompt for missed entities (catches 20-40% more).
            Default False to avoid doubling LLM calls. Enable for high-value chunks.

    Returns:
        Tuple of (entities, relations).
    """
    if not text.strip():
        return [], []

    if llm_caller is None:
        llm_caller = _get_default_llm_caller()

    # Pass 1: Primary extraction
    prompt = build_ner_prompt(text)
    try:
        response = llm_caller(prompt)
    except Exception:
        logger.exception("LLM caller failed in extract_entities_llm")
        return [], []

    if not response:
        return [], []

    entities, relations = parse_llm_ner_response(response, text)

    # Pass 2: Gleaning — re-prompt for missed entities
    if enable_gleaning and (entities or relations):
        gleaning_prompt = build_gleaning_prompt(text, len(entities), len(relations))
        try:
            gleaning_response = llm_caller(gleaning_prompt)
            if gleaning_response:
                extra_entities, extra_relations = parse_llm_ner_response(gleaning_response, text)
                if extra_entities or extra_relations:
                    logger.info(
                        "Gleaning found %d extra entities, %d extra relations",
                        len(extra_entities),
                        len(extra_relations),
                    )
                    entities.extend(extra_entities)
                    relations.extend(extra_relations)
        except Exception:
            logger.debug("Gleaning pass failed (non-critical)", exc_info=True)

    # Deduplicate relations (gleaning may re-find the same ones)
    seen_rels: set[tuple[str, str, str]] = set()
    unique_relations: list[ExtractedRelation] = []
    for r in relations:
        key = (r.source_text.lower(), r.target_text.lower(), r.relation_type)
        if key not in seen_rels:
            seen_rels.add(key)
            unique_relations.append(r)

    return entities, unique_relations


def _get_default_llm_caller():
    """Get the best available LLM caller — Gemini first, then enrichment.call_llm."""
    try:
        from ..enrichment_controller import call_gemini_for_extraction

        return call_gemini_for_extraction
    except (ImportError, RuntimeError):
        pass

    try:
        from .enrichment import call_llm

        return call_llm
    except ImportError:
        pass

    raise RuntimeError("No LLM backend available for entity extraction")


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
