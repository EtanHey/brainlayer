"""Tests for KG entity quality improvements (A7).

Covers:
- Entity type coercion (agent pattern, known projects/tech)
- Relation direction validation and correction
- Self-referential relation filtering
- Relation type normalization to canonical types
- Fact field propagation
- Entity importance computation from chunk links
"""

from brainlayer.pipeline.entity_extraction import ExtractedEntity, ExtractedRelation, ExtractionResult

# ── Entity Type Coercion ──────────────────────────────────────────


class TestEntityTypeCoercion:
    """Extracted entities should have their types corrected by validation rules."""

    def test_claude_suffix_forces_agent_type(self):
        """Names ending in 'Claude' should be coerced to type 'agent'."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="coachClaude", entity_type="person", start=0, end=11, confidence=0.75, source="llm"
                ),
            ],
            relations=[],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.entities[0].entity_type == "agent"

    def test_golem_suffix_forces_agent_type(self):
        """Names ending in 'Golem' should be coerced to type 'agent'."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="recruiterGolem", entity_type="person", start=0, end=14, confidence=0.75, source="llm"
                ),
            ],
            relations=[],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.entities[0].entity_type == "agent"

    def test_ralph_forces_agent_type(self):
        """'Ralph' is a known agent (autonomous executor)."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="Ralph", entity_type="person", start=0, end=5, confidence=0.75, source="llm"),
            ],
            relations=[],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.entities[0].entity_type == "agent"

    def test_known_project_tag_forces_project_type(self):
        """Names matching KNOWN_PROJECT_TAGS should be coerced to 'project'."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="brainlayer", entity_type="person", start=0, end=10, confidence=0.75, source="llm"
                ),
            ],
            relations=[],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.entities[0].entity_type == "project"

    def test_known_tech_tag_forces_technology_type(self):
        """Names matching KNOWN_TECH_TAGS should be coerced to 'technology'."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="SQLite", entity_type="company", start=0, end=6, confidence=0.75, source="llm"),
            ],
            relations=[],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.entities[0].entity_type == "technology"

    def test_correct_types_not_changed(self):
        """Entities with correct types should pass through unchanged."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan Heyman", entity_type="person", start=0, end=11, confidence=0.95, source="seed"
                ),
            ],
            relations=[],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.entities[0].entity_type == "person"
        assert validated.entities[0].text == "Etan Heyman"


# ── Relation Direction Validation ─────────────────────────────────


class TestRelationDirectionValidation:
    """Relations with wrong direction should be swapped or dropped."""

    def test_works_at_wrong_direction_swapped(self):
        """company works_at person should be swapped to person works_at company."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Cantaloupe AI", entity_type="company", start=0, end=13, confidence=0.9, source="seed"
                ),
                ExtractedEntity(
                    text="Etan Heyman", entity_type="person", start=20, end=31, confidence=0.9, source="seed"
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Cantaloupe AI",
                    target_text="Etan Heyman",
                    relation_type="works_at",
                    confidence=0.7,
                ),
            ],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert len(validated.relations) == 1
        rel = validated.relations[0]
        # Should be swapped: person works_at company
        assert rel.source_text == "Etan Heyman"
        assert rel.target_text == "Cantaloupe AI"

    def test_owns_wrong_direction_swapped(self):
        """project owns person should be swapped to person owns project."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="golems", entity_type="project", start=0, end=6, confidence=0.9, source="seed"),
                ExtractedEntity(
                    text="Etan Heyman", entity_type="person", start=10, end=21, confidence=0.9, source="seed"
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="golems",
                    target_text="Etan Heyman",
                    relation_type="owns",
                    confidence=0.7,
                ),
            ],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        rel = validated.relations[0]
        assert rel.source_text == "Etan Heyman"
        assert rel.target_text == "golems"

    def test_self_referential_relation_dropped(self):
        """Relations where source == target should be removed."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="brainClaude", entity_type="agent", start=0, end=11, confidence=0.9, source="seed"
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="brainClaude",
                    target_text="brainClaude",
                    relation_type="maintains",
                    confidence=0.7,
                ),
            ],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert len(validated.relations) == 0

    def test_valid_relation_preserved(self):
        """Correctly directed relations should pass through unchanged."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan Heyman", entity_type="person", start=0, end=11, confidence=0.9, source="seed"
                ),
                ExtractedEntity(text="golems", entity_type="project", start=20, end=26, confidence=0.9, source="seed"),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Etan Heyman",
                    target_text="golems",
                    relation_type="owns",
                    confidence=0.8,
                ),
            ],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert len(validated.relations) == 1
        assert validated.relations[0].source_text == "Etan Heyman"
        assert validated.relations[0].target_text == "golems"


# ── Relation Type Normalization ───────────────────────────────────


class TestRelationTypeNormalization:
    """Ad-hoc relation types should be normalized to canonical types."""

    def test_ad_hoc_type_normalized(self):
        """Non-canonical relation types should map to 'related_to'."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="Ollama", entity_type="technology", start=0, end=6, confidence=0.8, source="llm"),
                ExtractedEntity(
                    text="brainClaude", entity_type="agent", start=10, end=21, confidence=0.8, source="llm"
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Ollama",
                    target_text="brainClaude",
                    relation_type="enriches",
                    confidence=0.7,
                ),
            ],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.relations[0].relation_type == "related_to"

    def test_canonical_type_preserved(self):
        """Canonical relation types should not be changed."""
        from brainlayer.pipeline.kg_extraction import validate_extraction_result

        result = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Etan Heyman", entity_type="person", start=0, end=11, confidence=0.9, source="seed"
                ),
                ExtractedEntity(text="golems", entity_type="project", start=20, end=26, confidence=0.9, source="seed"),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Etan Heyman",
                    target_text="golems",
                    relation_type="builds",
                    confidence=0.8,
                ),
            ],
            chunk_id="test-chunk",
        )
        validated = validate_extraction_result(result)
        assert validated.relations[0].relation_type == "builds"


# ── Fact Field Propagation ────────────────────────────────────────


class TestFactFieldPropagation:
    """The fact field on ExtractedRelation should flow through to kg_relations."""

    def test_extracted_relation_has_fact_field(self):
        """ExtractedRelation should support a fact attribute."""
        rel = ExtractedRelation(
            source_text="Yuval Nir",
            target_text="Etan Heyman",
            relation_type="client_of",
            confidence=0.8,
            properties={"fact": "Yuval Nir is a client of Etan Heyman"},
        )
        # fact should be accessible (either as attribute or via properties)
        fact = getattr(rel, "fact", None) or rel.properties.get("fact")
        assert fact is not None
        assert "client" in fact.lower()

    def test_fact_stored_in_kg_relations(self, tmp_path):
        """When a relation has a fact, it should be stored in kg_relations.fact column."""
        from brainlayer.pipeline.kg_extraction import process_extraction_result
        from brainlayer.vector_store import VectorStore

        store = VectorStore(tmp_path / "test.db")

        # Pre-create entities so resolution finds them
        store.upsert_entity("person-yuval", "person", "Yuval Nir")
        store.upsert_entity("person-etan", "person", "Etan Heyman")

        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="Yuval Nir", entity_type="person", start=0, end=9, confidence=0.9, source="seed"),
                ExtractedEntity(
                    text="Etan Heyman", entity_type="person", start=15, end=26, confidence=0.9, source="seed"
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_text="Yuval Nir",
                    target_text="Etan Heyman",
                    relation_type="client_of",
                    confidence=0.8,
                    properties={"fact": "Yuval Nir is a client of Etan Heyman"},
                ),
            ],
            chunk_id="test-chunk",
        )
        process_extraction_result(store, result)

        cursor = store._read_cursor()
        rows = list(cursor.execute("SELECT fact FROM kg_relations WHERE relation_type = 'client_of'"))
        assert len(rows) >= 1
        assert rows[0][0] is not None
        assert "client" in rows[0][0].lower()
        store.close()


# ── Entity Importance Computation ─────────────────────────────────


class TestEntityImportanceComputation:
    """Entity importance should be computed from chunk links and relations."""

    def test_entity_importance_from_chunk_links(self, tmp_path):
        """Entity with more chunk links should have higher importance."""
        from brainlayer.pipeline.kg_extraction import compute_entity_importance
        from brainlayer.vector_store import VectorStore

        store = VectorStore(tmp_path / "test.db")
        cursor = store.conn.cursor()

        # Create two entities
        store.upsert_entity("person-popular", "person", "Popular Person")
        store.upsert_entity("person-rare", "person", "Rare Person")

        # Create dummy chunks and link them
        for i in range(10):
            cursor.execute(
                """INSERT INTO chunks (id, content, metadata, source_file, project,
                   content_type, char_count, source, importance)
                   VALUES (?, ?, '{}', 'test.jsonl', 'test', 'assistant_text', 100, 'claude_code', 7)""",
                (f"chunk-pop-{i}", f"Popular Person content {i}"),
            )
            store.link_entity_chunk("person-popular", f"chunk-pop-{i}", relevance=0.9)

        # Rare person: only 1 chunk
        cursor.execute(
            """INSERT INTO chunks (id, content, metadata, source_file, project,
               content_type, char_count, source, importance)
               VALUES (?, ?, '{}', 'test.jsonl', 'test', 'assistant_text', 100, 'claude_code', 5)""",
            ("chunk-rare-0", "Rare Person content"),
        )
        store.link_entity_chunk("person-rare", "chunk-rare-0", relevance=0.9)

        compute_entity_importance(store)

        # Check computed importance
        pop = store.get_entity("person-popular")
        rare = store.get_entity("person-rare")
        assert pop["importance"] > rare["importance"], (
            f"Popular ({pop['importance']}) should be > Rare ({rare['importance']})"
        )
        store.close()


# ── Prompt Quality ────────────────────────────────────────────────


class TestPromptQuality:
    """NER prompts should contain entity type guidance and direction constraints."""

    def test_groq_prompt_has_agent_type(self):
        """Groq NER prompt should list 'agent' as an entity type."""
        from brainlayer.pipeline.kg_extraction_groq import _MULTI_CHUNK_NER_PROMPT

        assert "agent" in _MULTI_CHUNK_NER_PROMPT

    def test_groq_prompt_has_direction_rules(self):
        """Groq NER prompt should contain direction guidance for relations."""
        from brainlayer.pipeline.kg_extraction_groq import _MULTI_CHUNK_NER_PROMPT

        assert "direction" in _MULTI_CHUNK_NER_PROMPT.lower() or "source" in _MULTI_CHUNK_NER_PROMPT.lower()

    def test_groq_prompt_has_fact_field(self):
        """Groq NER prompt should instruct LLM to provide fact for relations."""
        from brainlayer.pipeline.kg_extraction_groq import _MULTI_CHUNK_NER_PROMPT

        assert "fact" in _MULTI_CHUNK_NER_PROMPT.lower()

    def test_base_ner_prompt_has_agent_type(self):
        """Base NER prompt should list 'agent' as an entity type."""
        from brainlayer.pipeline.entity_extraction import _NER_PROMPT_TEMPLATE

        assert "agent" in _NER_PROMPT_TEMPLATE


# ── Seed Entity Coverage ──────────────────────────────────────────


class TestSeedEntityCoverage:
    """Seed entities should cover known agents and projects."""

    def test_seed_has_agent_type(self):
        """DEFAULT_SEED_ENTITIES should have 'agent' key (not 'golem')."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

        assert "agent" in DEFAULT_SEED_ENTITIES

    def test_seed_agents_include_ralph(self):
        """Ralph should be in agent seeds."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

        agents = DEFAULT_SEED_ENTITIES.get("agent", [])
        assert "Ralph" in agents

    def test_seed_projects_include_orchestrator(self):
        """orchestrator should be in project seeds."""
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

        projects = DEFAULT_SEED_ENTITIES.get("project", [])
        assert "orchestrator" in projects
