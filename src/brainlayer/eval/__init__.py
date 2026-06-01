"""BrainLayer search quality evaluation framework using Ranx."""

from .abcde_variants import ABCDE_VARIANTS, ABCDE_VARIANTS_BY_ID, VARIANT_IDS, ABCDEVariant, load_abcde_variants
from .benchmark import (
    DEFAULT_COMPARE_METRICS,
    DEFAULT_QUERY_SUITE,
    DEFAULT_RUN_METRICS,
    PR3_RELEVANCE_QUERY_SUITE,
    ReadOnlyBenchmarkStore,
    SearchBenchmark,
    pipeline_fts5_only,
    pipeline_hybrid_entity,
    pipeline_hybrid_rrf,
    prewarm_benchmark_embedder,
)
from .enrichment_gold import sample_enrichment_gold
from .enrichment_graders import (
    check_meta_research_forced_importance,
    find_banned_summary_pattern,
    grade_candidate,
    score_entities,
    score_importance_calibration,
    score_key_facts_recall,
    validate_schema_gate,
)

__all__ = [
    "DEFAULT_COMPARE_METRICS",
    "DEFAULT_QUERY_SUITE",
    "DEFAULT_RUN_METRICS",
    "PR3_RELEVANCE_QUERY_SUITE",
    "ReadOnlyBenchmarkStore",
    "SearchBenchmark",
    "pipeline_fts5_only",
    "pipeline_hybrid_entity",
    "pipeline_hybrid_rrf",
    "prewarm_benchmark_embedder",
    "sample_enrichment_gold",
    "check_meta_research_forced_importance",
    "find_banned_summary_pattern",
    "grade_candidate",
    "score_entities",
    "score_importance_calibration",
    "score_key_facts_recall",
    "validate_schema_gate",
    "ABCDE_VARIANTS",
    "ABCDE_VARIANTS_BY_ID",
    "VARIANT_IDS",
    "ABCDEVariant",
    "load_abcde_variants",
]
