from __future__ import annotations

import hashlib
import re

from brainlayer.eval.abcde_variants import (
    ABCDE_VARIANTS,
    ABCDE_VARIANTS_BY_ID,
    VARIANT_IDS,
    load_abcde_variants,
)
from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT


def test_load_abcde_variants_freezes_all_five_ids_and_provenance() -> None:
    variants = load_abcde_variants()

    assert tuple(variant.id for variant in variants) == VARIANT_IDS == ("A", "B", "C", "D", "E")
    assert ABCDE_VARIANTS == variants
    assert set(ABCDE_VARIANTS_BY_ID) == set(VARIANT_IDS)
    assert [variant.provenance for variant in variants] == [
        "production",
        "shelf",
        "llm-proposed",
        "llm-proposed",
        "llm-proposed",
    ]


def test_prompt_hashes_are_sha256_of_frozen_templates() -> None:
    for variant in load_abcde_variants():
        expected = hashlib.sha256(variant.prompt_template.encode("utf-8")).hexdigest()

        assert variant.prompt_hash == expected
        assert len(variant.prompt_hash) == 64


def test_variant_a_is_current_production_enrichment_prompt() -> None:
    variant_a = ABCDE_VARIANTS_BY_ID["A"]

    assert variant_a.prompt_template == ENRICHMENT_PROMPT


def test_variant_b_is_located_2026_03_19_faceted_prompt() -> None:
    variant_b = ABCDE_VARIANTS_BY_ID["B"]
    unique_domains = set(re.findall(r"\bdom:[a-z0-9-]+\b", variant_b.prompt_template))
    unique_activities = set(re.findall(r"\bact:[a-z0-9-]+\b", variant_b.prompt_template))

    assert variant_b.source_path == "orchestrator/docs.local/plans/enrichment-prompt-v2.md"
    assert variant_b.source_section == "Enrichment Prompt (copy-paste into pipeline)"
    for field in ("a_reasoning", "b_topics", "c_activity", "d_domain", "e_confidence"):
        assert field in variant_b.prompt_template
    assert len(unique_domains) == 22
    assert len(unique_activities) == 10


def test_llm_proposed_variants_are_divergent_runtime_schema_prompts() -> None:
    axes = {variant.axis for variant in load_abcde_variants() if variant.provenance == "llm-proposed"}

    assert axes == {"density-max-recall", "entity-relation-first", "structure-hyde-faithfulness"}
    for variant_id in ("C", "D", "E"):
        prompt = ABCDE_VARIANTS_BY_ID[variant_id].prompt_template
        assert "{project}" in prompt
        assert "{content_type}" in prompt
        assert "{content}" in prompt
        assert "{context_section}" in prompt
        for field in (
            "summary",
            "key_facts",
            "tags",
            "importance",
            "intent",
            "primary_symbols",
            "resolved_query",
            "resolved_queries",
            "epistemic_level",
            "version_scope",
            "debt_impact",
            "external_deps",
            "entities",
            "sentiment_label",
            "sentiment_score",
            "sentiment_signals",
        ):
            assert field in prompt
