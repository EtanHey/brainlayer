#!/usr/bin/env python3
"""Build a deterministic per-variant report for ABCDE enrichment generations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from brainlayer.eval.enrichment_graders import (
    find_banned_summary_pattern,
    score_entities,
    validate_schema_gate,
)

DEFAULT_INPUT = Path("eval_results/abcde_enrich.jsonl")
DEFAULT_OUTPUT = Path("eval_results/abcde_results_summary.md")
TICK_USD = 1e-9
VARIANT_IDS = ("A", "B", "C", "D", "E")
LABELS = {
    "A": "production",
    "B": "faceted-v2",
    "C": "density-max",
    "D": "entity-first",
    "E": "hyde-structure",
}


@dataclass(frozen=True)
class VariantSummary:
    variant_id: str
    label: str
    n: int
    ok_n: int
    gen_error_n: int
    error_reasons: dict[str, int]
    schema_error_reasons: dict[str, int]
    schema_pass_pct: float
    banned_pattern_pct: float
    mean_importance: float
    mean_tags: float
    mean_key_facts: float
    mean_entities: float
    mean_unsupported_entities: float
    mean_resolved_queries: float
    mean_completion_tokens: float
    total_cost_usd: float


@dataclass(frozen=True)
class Report:
    variants: dict[str, VariantSummary]
    table: str
    totals_line: str
    read_paragraph: str
    markdown: str


@dataclass
class _VariantAccumulator:
    variant_id: str
    n: int = 0
    ok_n: int = 0
    gen_error_n: int = 0
    schema_pass_n: int = 0
    banned_pattern_n: int = 0
    importances: list[float] = field(default_factory=list)
    tags_counts: list[int] = field(default_factory=list)
    key_facts_counts: list[int] = field(default_factory=list)
    entities_counts: list[int] = field(default_factory=list)
    unsupported_entity_counts: list[int] = field(default_factory=list)
    resolved_queries_counts: list[int] = field(default_factory=list)
    completion_tokens: list[int] = field(default_factory=list)
    total_cost_usd: float = 0.0
    error_reasons: Counter[str] = field(default_factory=Counter)
    schema_error_reasons: Counter[str] = field(default_factory=Counter)

    def record(self, row: Mapping[str, Any], *, tick_usd: float) -> None:
        self.n += 1
        generation = _mapping(row.get("generation"))
        usage = _mapping(generation.get("usage"))
        self.completion_tokens.append(_non_negative_int(usage.get("completion_tokens")))
        self.total_cost_usd += _non_negative_int(usage.get("cost_in_usd_ticks")) * tick_usd

        status = generation.get("status")
        if status != "ok":
            self.gen_error_n += 1
            reason = str(generation.get("error") or status or "unknown_error")
            self.error_reasons[reason] += 1
            return

        self.ok_n += 1
        enrichment = _mapping(row.get("enrichment"))
        chunk_text = str(row.get("chunk_text") or "")

        schema = validate_schema_gate(enrichment)
        if schema.passed:
            self.schema_pass_n += 1
        else:
            self.schema_error_reasons.update(schema.errors)
        if find_banned_summary_pattern(str(enrichment.get("summary") or "")) is not None:
            self.banned_pattern_n += 1

        importance = enrichment.get("importance")
        if isinstance(importance, (int, float)) and not isinstance(importance, bool):
            self.importances.append(float(importance))

        self.tags_counts.append(_list_len(enrichment.get("tags")))
        self.key_facts_counts.append(_list_len(enrichment.get("key_facts")))
        self.entities_counts.append(_list_len(enrichment.get("entities")))
        self.resolved_queries_counts.append(_list_len(enrichment.get("resolved_queries")))
        entity_metrics = score_entities(enrichment, {"gold_entities": []}, chunk_text)
        self.unsupported_entity_counts.append(len(entity_metrics.hallucinated_entities))

    def summary(self) -> VariantSummary:
        return VariantSummary(
            variant_id=self.variant_id,
            label=LABELS.get(self.variant_id, "unknown"),
            n=self.n,
            ok_n=self.ok_n,
            gen_error_n=self.gen_error_n,
            error_reasons=dict(sorted(self.error_reasons.items())),
            schema_error_reasons=dict(sorted(self.schema_error_reasons.items())),
            schema_pass_pct=_pct(self.schema_pass_n, self.ok_n),
            banned_pattern_pct=_pct(self.banned_pattern_n, self.ok_n),
            mean_importance=_mean(self.importances),
            mean_tags=_mean(self.tags_counts),
            mean_key_facts=_mean(self.key_facts_counts),
            mean_entities=_mean(self.entities_counts),
            mean_unsupported_entities=_mean(self.unsupported_entity_counts),
            mean_resolved_queries=_mean(self.resolved_queries_counts),
            mean_completion_tokens=_mean(self.completion_tokens),
            total_cost_usd=self.total_cost_usd,
        )


def build_report(input_path: Path | str = DEFAULT_INPUT, *, tick_usd: float = TICK_USD) -> Report:
    rows = list(_read_jsonl(Path(input_path)))
    accumulators = {variant_id: _VariantAccumulator(variant_id) for variant_id in VARIANT_IDS}
    for row in rows:
        variant_id = str(row.get("variant_id") or "")
        if variant_id not in accumulators:
            accumulators[variant_id] = _VariantAccumulator(variant_id)
        accumulators[variant_id].record(row, tick_usd=tick_usd)

    variant_order = [*VARIANT_IDS, *sorted(set(accumulators) - set(VARIANT_IDS))]
    variants = {variant_id: accumulators[variant_id].summary() for variant_id in variant_order}
    table = _format_table(variants.values())
    totals_line = _format_totals(variants.values())
    read_paragraph = _format_read(variants.values())
    markdown = "\n\n".join(("# ABCDE Enrichment Results Summary", table, totals_line, read_paragraph)) + "\n"
    return Report(variants=variants, table=table, totals_line=totals_line, read_paragraph=read_paragraph, markdown=markdown)


def write_report(report: Report, output_path: Path | str = DEFAULT_OUTPUT) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.markdown, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tick-usd", type=float, default=TICK_USD)
    args = parser.parse_args(argv)

    report = build_report(args.input, tick_usd=args.tick_usd)
    write_report(report, args.output)
    print(report.table)
    return 0


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def _format_table(summaries: Iterable[VariantSummary]) -> str:
    lines = [
        "| Variant | label | n | ok | gen errors | schema pass % | banned % | mean importance | "
        "mean tags | mean key facts | mean entities | mean unsupported entities | mean resolved queries | "
        "mean completion tokens | total cost USD | error reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.variant_id} | {summary.label} | {summary.n} | {summary.ok_n} | {summary.gen_error_n} | "
            f"{summary.schema_pass_pct:.1f} | {summary.banned_pattern_pct:.1f} | "
            f"{summary.mean_importance:.2f} | {summary.mean_tags:.2f} | {summary.mean_key_facts:.2f} | "
            f"{summary.mean_entities:.2f} | {summary.mean_unsupported_entities:.2f} | "
            f"{summary.mean_resolved_queries:.2f} | {summary.mean_completion_tokens:.1f} | "
            f"{summary.total_cost_usd:.6f} | {_format_error_reasons(summary.error_reasons)} |"
        )
    return "\n".join(lines)


def _format_totals(summaries: Iterable[VariantSummary]) -> str:
    summary_list = list(summaries)
    total_calls = sum(summary.n for summary in summary_list)
    total_ok = sum(summary.ok_n for summary in summary_list)
    total_errors = sum(summary.gen_error_n for summary in summary_list)
    total_cost = sum(summary.total_cost_usd for summary in summary_list)
    return (
        f"TOTALS: {total_calls} calls, {total_ok} ok, {total_errors} generation errors, "
        f"${total_cost:.6f} metered in this JSONL. Smoke added ~$0.55 separately."
    )


def _format_read(summaries: Iterable[VariantSummary]) -> str:
    summary_list = list(summaries)
    candidates = [summary for summary in summary_list if summary.ok_n > 0 and summary.banned_pattern_pct == 0.0]
    if not candidates:
        return (
            "Read: no variant had ok generations without banned-pattern violations; deterministic signals do not "
            "identify a strongest variant. LLM judge pass is separate/pending."
        )

    strongest = max(
        candidates,
        key=lambda summary: (
            summary.schema_pass_pct,
            summary.mean_key_facts + summary.mean_entities,
            summary.mean_key_facts,
            summary.mean_entities,
            -summary.mean_unsupported_entities,
            summary.variant_id,
        ),
    )
    notes = _schema_notes(summary_list)
    return (
        f"Read: variant {strongest.variant_id} looks strongest on deterministic signals among variants with "
        f"0.0% banned-pattern hits: schema pass {strongest.schema_pass_pct:.1f}%, "
        f"mean key facts {strongest.mean_key_facts:.2f}, mean entities {strongest.mean_entities:.2f}, "
        f"mean unsupported entities {strongest.mean_unsupported_entities:.2f}. "
        f"{notes} LLM judge pass is separate/pending."
    )


def _schema_notes(summaries: Sequence[VariantSummary]) -> str:
    by_variant = {summary.variant_id: summary for summary in summaries}
    notes: list[str] = []

    variant_a = by_variant.get("A")
    if (
        variant_a is not None
        and variant_a.ok_n > 0
        and variant_a.schema_error_reasons == {"missing required key: resolved_query": variant_a.ok_n}
        and variant_a.mean_resolved_queries == 3.0
    ):
        notes.append(
            "Variant A fails schema_gate only on the missing legacy `resolved_query` key; "
            "it has `resolved_queries` x3."
        )

    variant_b = by_variant.get("B")
    if variant_b is not None and variant_b.ok_n > 0 and variant_b.schema_pass_pct == 0.0:
        notes.append("Variant B uses the faceted-tagger schema, not the enrichment schema.")

    return " ".join(notes)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _mean(values: Sequence[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator * 100.0) if denominator else 0.0


def _non_negative_int(value: Any) -> int:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0, int(value))
    return 0


def _format_error_reasons(reasons: Mapping[str, int]) -> str:
    if not reasons:
        return "-"
    return "; ".join(f"{reason} x{count}" for reason, count in sorted(reasons.items()))


if __name__ == "__main__":
    raise SystemExit(main())
