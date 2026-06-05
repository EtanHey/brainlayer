#!/usr/bin/env python3
"""Dry-run-first Braintrust ingestion for BrainLayer retrieval grading.

Doc-verified SDK/API notes:
- Braintrust SDK auth defaults to BRAINTRUST_API_KEY and prompts login without it:
  https://www.braintrust.dev/docs/reference/sdks/python/latest
- Dataset creation uses init_dataset(project=..., name=..., api_key=..., use_output=False).
- Dataset rows use Dataset.insert(input=..., expected=..., metadata=..., id=..., tags=...).
- Experiments use Eval(name, data=..., task=..., scores=..., experiment_name=..., metadata=...).
- Autoevals LLMClassifier supports prompt_template and choice_scores, but any semantic
  relevance judge needs raw query/evidence text, so this script never enables it for
  Braintrust cloud sends.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

DEFAULT_LABELED_PATH = Path("/tmp/labeled_eval_candidates.json")
DEFAULT_RETRIEVAL_PATH = Path("/tmp/retrieval_benchmark_results.json")
DEFAULT_RAW_ENRICHED_PATH = Path("/Users/etanheyman/Gits/brainlayer-abcde/eval_results/abcde_enrich_nebius.jsonl")
DEFAULT_PROJECT = "BrainLayer Retrieval Grading"
DEFAULT_DATASET = "retrieval-grading-candidates"
DEFAULT_EXPERIMENT = "abcde-retrieval-grading"
RELEVANT_LABEL = "relevant"
DOC_VERIFIED_SOURCES = [
    "https://www.braintrust.dev/docs/admin/self-hosting",
    "https://www.braintrust.dev/docs/admin/self-hosting/deploy",
    "https://www.braintrust.dev/docs/reference/sdks/python/latest",
    "https://www.braintrust.dev/docs/annotate/datasets/create",
    "https://github.com/braintrustdata/autoevals",
]


@dataclass(frozen=True)
class IngestionPayload:
    dataset_records: list[dict[str, Any]]
    retrieval_metrics: dict[str, Any]
    counts: dict[str, int]
    dataset_metadata: dict[str, Any]

    @property
    def eval_cases(self) -> list[dict[str, Any]]:
        return [
            {
                "input": record["input"],
                "expected": record["expected"],
                "metadata": record["metadata"],
                "tags": record["tags"],
            }
            for record in self.dataset_records
        ]

    def as_dry_run_dict(self) -> dict[str, Any]:
        sample_record = self.dataset_records[0] if self.dataset_records else None
        return {
            "mode": "dry-run",
            "counts": self.counts,
            "project_payload": {
                "dataset_records": "redacted records; raw query/chunk text omitted",
                "sample_record": sample_record,
                "retrieval_metrics": self.retrieval_metrics,
                "dataset_metadata": self.dataset_metadata,
            },
            "privacy": "No raw query text, chunk snippets, chunk rationale, or notes are included.",
        }


@dataclass(frozen=True)
class BraintrustSettings:
    project: str
    dataset: str
    experiment: str
    api_key: str | None


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _line in handle)


def _label_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        label = str(chunk.get("machine_label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _chunk_id_hash(chunk_id: Any) -> str:
    return stable_hash(str(chunk_id))


def _relevant_chunk_id_hashes(chunks: list[dict[str, Any]]) -> list[str]:
    return [
        _chunk_id_hash(chunk["id"])
        for chunk in chunks
        if chunk.get("id") and str(chunk.get("machine_label", "")).lower() == RELEVANT_LABEL
    ]


def _returned_chunk_id_hashes(chunks: list[dict[str, Any]]) -> list[str]:
    return [_chunk_id_hash(chunk["id"]) for chunk in chunks if chunk.get("id")]


def _redacted_source_meta(meta: dict[str, Any]) -> dict[str, Any]:
    safe_meta: dict[str, Any] = {}
    for key in ("n_chunks", "method", "generated"):
        if key in meta:
            safe_meta[key] = meta[key]
    if "source" in meta:
        safe_meta["source_redacted"] = True
    return safe_meta


def _sanitize_retrieval_metrics(retrieval_results: dict[str, Any]) -> dict[str, Any]:
    per_variant = retrieval_results.get("per_variant", {})
    sanitized_variants: dict[str, dict[str, int | float]] = {}
    for variant, metrics in sorted(per_variant.items()):
        sanitized_variants[str(variant)] = {
            str(key): value
            for key, value in sorted(metrics.items())
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    return {
        "meta": _redacted_source_meta(retrieval_results.get("meta", {})),
        "per_variant": sanitized_variants,
    }


def _build_dataset_record(item: dict[str, Any], index: int) -> dict[str, Any]:
    chunks = item.get("returned_chunks") or []
    query_id = f"query-{index:04d}"
    query_text = str(item.get("query") or "")
    label_source = str(item.get("label_source") or "unknown")
    metadata = {
        "source_index": index - 1,
        "label_source": label_source,
        "label_counts": _label_counts(chunks),
        "returned_count": len(chunks),
        "suspected_miss_present": item.get("suspected_miss") is not None,
        "suspected_misorder_present": item.get("suspected_misorder") is not None,
        "notes_redacted": bool(item.get("notes")),
        "raw_text_policy": "query_snippet_why_notes_redacted",
    }
    return {
        "id": query_id,
        "input": {
            "query_id": query_id,
            "query_sha256": stable_hash(query_text),
            "query_text_redacted": True,
            "returned_chunk_id_hashes": _returned_chunk_id_hashes(chunks),
        },
        "expected": {
            "relevant_chunk_id_hashes": _relevant_chunk_id_hashes(chunks),
        },
        "metadata": metadata,
        "tags": ["brainlayer", "retrieval-grading", label_source],
    }


def build_ingestion_payload(labeled_path: Path, retrieval_path: Path, raw_enriched_path: Path) -> IngestionPayload:
    labeled_candidates = _load_json(labeled_path)
    retrieval_results = _load_json(retrieval_path)
    if not isinstance(labeled_candidates, list):
        raise ValueError(f"Expected labeled candidates JSON array at {labeled_path}")
    if not isinstance(retrieval_results, dict):
        raise ValueError(f"Expected retrieval results JSON object at {retrieval_path}")

    dataset_records = [
        _build_dataset_record(item, index)
        for index, item in enumerate(labeled_candidates, start=1)
        if isinstance(item, dict)
    ]
    retrieval_metrics = _sanitize_retrieval_metrics(retrieval_results)
    raw_count = _count_jsonl_rows(raw_enriched_path)
    counts = {
        "labeled_queries": len(labeled_candidates),
        "dataset_records": len(dataset_records),
        "raw_enriched_rows_local_only": raw_count,
        "retrieval_variants": len(retrieval_metrics["per_variant"]),
    }
    dataset_metadata = {
        "privacy": "raw_query_and_chunk_text_redacted",
        "labeled_source_file": labeled_path.name,
        "retrieval_source_file": retrieval_path.name,
        "raw_enriched_rows_local_only": raw_count,
        "doc_verified_sources": DOC_VERIFIED_SOURCES,
    }
    return IngestionPayload(
        dataset_records=dataset_records,
        retrieval_metrics=retrieval_metrics,
        counts=counts,
        dataset_metadata=dataset_metadata,
    )


def make_recall_at_k_scorer(k: int) -> Callable[..., float | None]:
    def recall_at_k(input: dict[str, Any], output: dict[str, Any], expected: dict[str, Any] | None = None, **_kwargs):
        expected = expected or {}
        relevant = set(expected.get("relevant_chunk_id_hashes") or expected.get("relevant_chunk_ids") or [])
        if not relevant:
            return None
        retrieved = list(
            output.get("retrieved_chunk_id_hashes")
            or output.get("retrieved_chunk_ids")
            or input.get("returned_chunk_id_hashes")
            or input.get("returned_chunk_ids")
            or []
        )
        hits = relevant.intersection(retrieved[:k])
        return len(hits) / len(relevant)

    recall_at_k.__name__ = f"recall_at_{k}"
    return recall_at_k


def make_relevance_llm_judge(model: str | None = None):
    """Construct an Autoevals LLM judge for local/self-host-only semantic relevance.

    This scorer requires raw query/evidence text to be meaningful. It is intentionally
    not attached to Braintrust cloud sends in this script because the task forbids
    uploading Etan's personal query/chunk content.
    """

    from autoevals.llm import LLMClassifier

    return LLMClassifier(
        name="retrieval_relevance_llm_judge",
        prompt_template=(
            "Decide whether the retrieved evidence is relevant to the search query.\n"
            "Query: {{input}}\n"
            "Retrieved evidence: {{output}}\n"
            "Expected relevant evidence or label: {{expected}}\n"
            "Choose exactly one label."
        ),
        choice_scores={"relevant": 1, "not_relevant": 0},
        model=model,
        use_cot=True,
    )


def _task_return_retrieved_ids(input: dict[str, Any]) -> dict[str, Any]:
    return {"retrieved_chunk_id_hashes": input.get("returned_chunk_id_hashes") or []}


def _import_braintrust() -> ModuleType:
    try:
        import braintrust
    except ImportError as exc:
        raise RuntimeError("Install the Braintrust SDK first: python3 -m pip install braintrust autoevals") from exc
    return braintrust


def send_payload(
    payload: IngestionPayload,
    settings: BraintrustSettings,
    *,
    braintrust_module: Any | None = None,
) -> dict[str, Any]:
    api_key = settings.api_key
    if not api_key:
        raise RuntimeError("BRAINTRUST_API_KEY is required for --send")

    braintrust = braintrust_module or _import_braintrust()
    dataset = braintrust.init_dataset(
        project=settings.project,
        name=settings.dataset,
        api_key=api_key,
        use_output=False,
        metadata=payload.dataset_metadata,
    )
    for record in payload.dataset_records:
        dataset.insert(
            id=record["id"],
            input=record["input"],
            expected=record["expected"],
            metadata=record["metadata"],
            tags=record["tags"],
        )
    dataset.flush()

    braintrust.Eval(
        settings.project,
        data=payload.eval_cases,
        task=_task_return_retrieved_ids,
        scores=[make_recall_at_k_scorer(1), make_recall_at_k_scorer(5), make_recall_at_k_scorer(10)],
        experiment_name=settings.experiment,
        metadata={
            **payload.dataset_metadata,
            "retrieval_metrics": payload.retrieval_metrics,
            "privacy": "raw_query_and_chunk_text_redacted",
            "llm_judge_status": "not_enabled_for_cloud_personal_content_guardrail",
        },
        tags=["brainlayer", "retrieval-grading", "redacted"],
        no_send_logs=False,
    )
    return {"dataset_records_sent": len(payload.dataset_records), "eval_invoked": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-path", type=Path, default=DEFAULT_LABELED_PATH)
    parser.add_argument("--retrieval-path", type=Path, default=DEFAULT_RETRIEVAL_PATH)
    parser.add_argument("--raw-enriched-path", type=Path, default=DEFAULT_RAW_ENRICHED_PATH)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True, help="Print sanitized payload shape; send nothing")
    mode.add_argument("--send", action="store_false", dest="dry_run", help="Send redacted records to Braintrust")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_ingestion_payload(args.labeled_path, args.retrieval_path, args.raw_enriched_path)
    if args.dry_run:
        print(json.dumps(payload.as_dry_run_dict(), indent=2, sort_keys=True))
        return

    settings = BraintrustSettings(
        project=args.project,
        dataset=args.dataset,
        experiment=args.experiment,
        api_key=os.environ.get("BRAINTRUST_API_KEY"),
    )
    try:
        result = send_payload(payload, settings)
    except RuntimeError as exc:
        raise SystemExit(f"BLOCKED: {exc}") from None
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
