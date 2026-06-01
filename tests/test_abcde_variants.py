from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from brainlayer.eval.abcde_variants import (
    ABCDE_VARIANTS,
    ABCDE_VARIANTS_BY_ID,
    VARIANT_IDS,
    load_abcde_variants,
)
from brainlayer.pipeline.enrichment import ENRICHMENT_PROMPT
from scripts.generate_abcde_variants import (
    SHELF_SECTION,
    assert_freeze_integrity,
    compute_prompt_hash,
    extract_markdown_fenced_section,
    prepare_shelf_prompt_for_registry,
)

V2_PROMPT_PATH = Path("/Users/etanheyman/Gits/orchestrator/docs.local/plans/enrichment-prompt-v2.md")


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


def test_freeze_hashes_are_pairwise_distinct_and_match_anchors() -> None:
    variants = [
        {"id": variant.id, "prompt_hash": variant.prompt_hash, "prompt_template": variant.prompt_template}
        for variant in load_abcde_variants()
    ]

    # Verify all hashes are pairwise distinct
    hashes = {variant["id"]: variant["prompt_hash"] for variant in variants}
    assert len(set(hashes.values())) == len(hashes), "Prompt hashes must be pairwise distinct"

    # Verify variant A matches production
    assert hashes["A"] == compute_prompt_hash(ENRICHMENT_PROMPT), "Variant A must match production ENRICHMENT_PROMPT"

    # Verify variant B hash matches its own template (internal consistency)
    variant_b = next(v for v in variants if v["id"] == "B")
    assert variant_b["prompt_hash"] == compute_prompt_hash(variant_b["prompt_template"])

    # If external source exists, run full integrity check
    if V2_PROMPT_PATH.exists():
        v2_prompt = prepare_shelf_prompt_for_registry(extract_markdown_fenced_section(V2_PROMPT_PATH, SHELF_SECTION))
        assert_freeze_integrity(variants, ENRICHMENT_PROMPT, v2_prompt)


def test_freeze_integrity_fails_loud_on_prompt_hash_collision() -> None:
    variants = [
        {"id": variant.id, "prompt_hash": variant.prompt_hash, "prompt_template": variant.prompt_template}
        for variant in load_abcde_variants()
    ]
    variants[2]["prompt_hash"] = variants[0]["prompt_hash"]

    # Use dummy prompt for test since source file doesn't exist in CI
    dummy_prompt = "dummy shelf prompt"

    with pytest.raises(ValueError, match="collision"):
        assert_freeze_integrity(variants, ENRICHMENT_PROMPT, dummy_prompt)


def test_variant_a_is_current_production_enrichment_prompt() -> None:
    variant_a = ABCDE_VARIANTS_BY_ID["A"]

    assert variant_a.prompt_template == ENRICHMENT_PROMPT
    assert variant_a.prompt_hash == compute_prompt_hash(ENRICHMENT_PROMPT)


def test_variant_b_is_located_2026_03_19_faceted_prompt() -> None:
    variant_b = ABCDE_VARIANTS_BY_ID["B"]
    unique_domains = set(re.findall(r"\bdom:[a-z0-9-]+\b", variant_b.prompt_template))
    unique_activities = set(re.findall(r"\bact:[a-z0-9-]+\b", variant_b.prompt_template))

    assert variant_b.source_path == "orchestrator/docs.local/plans/enrichment-prompt-v2.md"
    assert variant_b.source_section == "Enrichment Prompt (copy-paste into pipeline)"

    # Verify hash matches prompt (internal consistency)
    assert variant_b.prompt_hash == compute_prompt_hash(variant_b.prompt_template)

    # Verify standard placeholders are present after fix
    for placeholder in ("{project}", "{content_type}", "{content}", "{context_section}"):
        assert placeholder in variant_b.prompt_template

    # Verify JSON examples are properly escaped (doubled braces)
    assert '{{"a_reasoning":' in variant_b.prompt_template or '{{"a_reasoning"' in variant_b.prompt_template

    for field in ("a_reasoning", "b_topics", "c_activity", "d_domain", "e_confidence"):
        assert field in variant_b.prompt_template
    assert len(unique_domains) == 22
    assert len(unique_activities) == 10

    # If external source exists, verify it matches
    if V2_PROMPT_PATH.exists():
        v2_prompt = prepare_shelf_prompt_for_registry(extract_markdown_fenced_section(V2_PROMPT_PATH, SHELF_SECTION))
        assert variant_b.prompt_template == v2_prompt


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
