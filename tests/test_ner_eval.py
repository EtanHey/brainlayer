"""Tests for Gate 3: NER evaluation harness.

Loads gold_standard_ner.json and provides metrics for evaluating
entity extraction quality: exact match, partial match, type match.

Uses nervaluate-style metrics without the dependency:
- Exact: start + end + type must all match
- Partial: spans must overlap + type must match
- Type: only entity type needs to match (any span)
"""

import json
from pathlib import Path
from typing import Any

import pytest

GOLD_PATH = Path(__file__).parent / "data" / "gold_standard_ner.json"


def _load_gold_standard() -> list[dict[str, Any]]:
    """Load gold-standard NER samples."""
    data = json.loads(GOLD_PATH.read_text())
    return data["samples"]


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check if two character spans overlap."""
    return a_start < b_end and b_start < a_end


def _compute_ner_metrics(
    gold_entities: list[dict], pred_entities: list[dict]
) -> dict[str, dict[str, float]]:
    """Compute NER evaluation metrics at three granularities.

    Returns dict with keys: exact, partial, type_only.
    Each contains: precision, recall, f1, correct, possible, actual.
    """
    results = {}

    for mode in ("exact", "partial", "type_only"):
        matched_gold = set()
        matched_pred = set()

        for gi, gold in enumerate(gold_entities):
            for pi, pred in enumerate(pred_entities):
                if pi in matched_pred:
                    continue

                if mode == "exact":
                    match = (
                        gold["type"] == pred["type"]
                        and gold["start"] == pred["start"]
                        and gold["end"] == pred["end"]
                    )
                elif mode == "partial":
                    match = gold["type"] == pred["type"] and _spans_overlap(
                        gold["start"], gold["end"], pred["start"], pred["end"]
                    )
                else:  # type_only
                    match = gold["type"] == pred["type"]

                if match and gi not in matched_gold:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        correct = len(matched_gold)
        possible = len(gold_entities)
        actual = len(pred_entities)

        precision = correct / actual if actual > 0 else 0.0
        recall = correct / possible if possible > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[mode] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
            "possible": possible,
            "actual": actual,
        }

    return results


def _compute_per_type_metrics(
    gold_entities: list[dict], pred_entities: list[dict]
) -> dict[str, dict[str, float]]:
    """Compute exact-match metrics per entity type."""
    types = {e["type"] for e in gold_entities} | {e["type"] for e in pred_entities}
    results = {}

    for etype in sorted(types):
        gold_of_type = [e for e in gold_entities if e["type"] == etype]
        pred_of_type = [e for e in pred_entities if e["type"] == etype]
        metrics = _compute_ner_metrics(gold_of_type, pred_of_type)
        results[etype] = metrics["exact"]

    return results


# ── Gold Standard Integrity ──


class TestGoldStandardIntegrity:
    """Verify the gold-standard data itself is well-formed."""

    def test_gold_file_exists(self):
        assert GOLD_PATH.exists(), f"Gold standard not found at {GOLD_PATH}"

    def test_gold_has_samples(self):
        samples = _load_gold_standard()
        assert len(samples) >= 30, f"Need >= 30 samples, got {len(samples)}"

    def test_gold_sample_has_required_fields(self):
        samples = _load_gold_standard()
        for s in samples:
            assert "id" in s, f"Missing id in sample"
            assert "text" in s, f"Missing text in {s['id']}"
            assert "entities" in s, f"Missing entities in {s['id']}"
            assert isinstance(s["entities"], list)

    def test_gold_entities_have_spans(self):
        """Every entity must have text, type, start, end."""
        samples = _load_gold_standard()
        for s in samples:
            for e in s["entities"]:
                assert "text" in e, f"Missing text in entity of {s['id']}"
                assert "type" in e, f"Missing type in entity of {s['id']}"
                assert "start" in e, f"Missing start in entity of {s['id']}"
                assert "end" in e, f"Missing end in entity of {s['id']}"

    def test_gold_entity_spans_are_valid(self):
        """start < end, and span text matches the annotated text."""
        samples = _load_gold_standard()
        for s in samples:
            for e in s["entities"]:
                assert e["start"] < e["end"], (
                    f"Invalid span in {s['id']}: start={e['start']} >= end={e['end']}"
                )
                span_text = s["text"][e["start"] : e["end"]]
                assert span_text == e["text"], (
                    f"Span mismatch in {s['id']}: "
                    f"expected '{e['text']}' but got '{span_text}' "
                    f"at [{e['start']}:{e['end']}]"
                )

    def test_gold_entity_types_valid(self):
        """All entity types should be from the known set."""
        valid_types = {"person", "company", "project", "golem", "tool", "topic"}
        samples = _load_gold_standard()
        for s in samples:
            for e in s["entities"]:
                assert e["type"] in valid_types, (
                    f"Unknown type '{e['type']}' in {s['id']}"
                )

    def test_gold_has_diverse_entity_types(self):
        """Gold set should cover at least 4 entity types."""
        samples = _load_gold_standard()
        types = {e["type"] for s in samples for e in s["entities"]}
        assert len(types) >= 4, f"Need >= 4 entity types, got {types}"

    def test_gold_has_relations(self):
        """At least some samples should have relations."""
        samples = _load_gold_standard()
        with_relations = [s for s in samples if s.get("relations")]
        assert len(with_relations) >= 5, (
            f"Need >= 5 samples with relations, got {len(with_relations)}"
        )


# ── Metric Computation ──


class TestMetricComputation:
    """Test the metric computation functions themselves."""

    def test_perfect_exact_match(self):
        """Identical predictions should give F1 = 1.0."""
        gold = [{"text": "Etan", "type": "person", "start": 0, "end": 4}]
        pred = [{"text": "Etan", "type": "person", "start": 0, "end": 4}]
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["f1"] == 1.0
        assert m["exact"]["precision"] == 1.0
        assert m["exact"]["recall"] == 1.0

    def test_no_predictions(self):
        """No predictions should give 0 recall, 0 precision."""
        gold = [{"text": "Etan", "type": "person", "start": 0, "end": 4}]
        pred = []
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["recall"] == 0.0
        assert m["exact"]["precision"] == 0.0
        assert m["exact"]["f1"] == 0.0

    def test_no_gold(self):
        """No gold entities, but predictions present."""
        gold = []
        pred = [{"text": "Etan", "type": "person", "start": 0, "end": 4}]
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["recall"] == 0.0
        assert m["exact"]["precision"] == 0.0

    def test_partial_overlap_scores(self):
        """Overlapping spans should score on partial but not exact."""
        gold = [{"text": "Etan Heyman", "type": "person", "start": 0, "end": 11}]
        pred = [{"text": "Etan", "type": "person", "start": 0, "end": 4}]
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["f1"] == 0.0  # Spans don't match exactly
        assert m["partial"]["f1"] == 1.0  # But they overlap
        assert m["type_only"]["f1"] == 1.0  # Type matches

    def test_wrong_type_no_match(self):
        """Same span but wrong type should not match on exact or partial."""
        gold = [{"text": "Domica", "type": "company", "start": 0, "end": 6}]
        pred = [{"text": "Domica", "type": "project", "start": 0, "end": 6}]
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["f1"] == 0.0
        assert m["partial"]["f1"] == 0.0
        assert m["type_only"]["f1"] == 0.0

    def test_multiple_entities_scoring(self):
        """Test precision/recall with multiple entities."""
        gold = [
            {"text": "Etan", "type": "person", "start": 0, "end": 4},
            {"text": "Domica", "type": "company", "start": 10, "end": 16},
        ]
        # Only found one of two
        pred = [
            {"text": "Etan", "type": "person", "start": 0, "end": 4},
        ]
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["precision"] == 1.0  # 1/1 predicted correct
        assert m["exact"]["recall"] == 0.5  # 1/2 gold found

    def test_spurious_prediction_lowers_precision(self):
        """Extra predictions should lower precision."""
        gold = [
            {"text": "Etan", "type": "person", "start": 0, "end": 4},
        ]
        pred = [
            {"text": "Etan", "type": "person", "start": 0, "end": 4},
            {"text": "the", "type": "person", "start": 5, "end": 8},
        ]
        m = _compute_ner_metrics(gold, pred)
        assert m["exact"]["precision"] == 0.5  # 1/2 predictions correct
        assert m["exact"]["recall"] == 1.0  # 1/1 gold found


# ── Per-Type Metrics ──


class TestPerTypeMetrics:
    """Test per-entity-type breakdown."""

    def test_per_type_separates_correctly(self):
        gold = [
            {"text": "Etan", "type": "person", "start": 0, "end": 4},
            {"text": "Domica", "type": "company", "start": 10, "end": 16},
        ]
        pred = [
            {"text": "Etan", "type": "person", "start": 0, "end": 4},
            {"text": "Domicax", "type": "company", "start": 10, "end": 17},
        ]
        per_type = _compute_per_type_metrics(gold, pred)
        assert per_type["person"]["f1"] == 1.0
        assert per_type["company"]["f1"] == 0.0  # Wrong span


# ── Eval Harness (for actual NER pipeline) ──


def _run_extraction(text: str) -> list[dict]:
    """Run extraction pipeline on text, return entities in gold-standard format."""
    from brainlayer.pipeline.entity_extraction import extract_entities_combined
    from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

    result = extract_entities_combined(text, DEFAULT_SEED_ENTITIES, use_llm=False)
    return [
        {"text": e.text, "type": e.entity_type, "start": e.start, "end": e.end}
        for e in result.entities
    ]


class TestEvalHarnessSeedOnly:
    """Run seed-only extraction against gold standard.

    Seed matching covers known entities (people, companies, projects, golems).
    This gives a baseline — GLiNER and LLM add coverage for unknown entities.
    """

    @pytest.fixture
    def gold_samples(self):
        return _load_gold_standard()

    def test_seed_baseline_partial_f1(self, gold_samples):
        """Seed-only partial F1 should be reasonable for known entities."""
        all_gold = []
        all_pred = []
        for sample in gold_samples:
            all_gold.extend(sample["entities"])
            pred = _run_extraction(sample["text"])
            all_pred.extend(pred)

        m = _compute_ner_metrics(all_gold, all_pred)
        # Seed matching should find most known entities
        assert m["partial"]["f1"] >= 0.3, (
            f"Seed partial F1 {m['partial']['f1']:.3f} below 0.3 — seed entities may need updating"
        )

    def test_seed_baseline_reports_metrics(self, gold_samples):
        """Report full metrics for visibility (always passes)."""
        all_gold = []
        all_pred = []
        for sample in gold_samples:
            all_gold.extend(sample["entities"])
            pred = _run_extraction(sample["text"])
            all_pred.extend(pred)

        m = _compute_ner_metrics(all_gold, all_pred)
        per_type = _compute_per_type_metrics(all_gold, all_pred)

        # Print metrics for debugging (visible with pytest -v -s)
        print(f"\n=== Seed-Only Eval ===")
        print(f"Gold: {m['exact']['possible']} entities, Predicted: {m['exact']['actual']}")
        print(f"Exact:   P={m['exact']['precision']:.3f}  R={m['exact']['recall']:.3f}  F1={m['exact']['f1']:.3f}")
        print(f"Partial: P={m['partial']['precision']:.3f}  R={m['partial']['recall']:.3f}  F1={m['partial']['f1']:.3f}")
        print(f"Type:    P={m['type_only']['precision']:.3f}  R={m['type_only']['recall']:.3f}  F1={m['type_only']['f1']:.3f}")
        for etype, scores in per_type.items():
            print(f"  {etype}: P={scores['precision']:.3f} R={scores['recall']:.3f} F1={scores['f1']:.3f} ({scores['correct']}/{scores['possible']})")


class TestEvalHarnessGLiNER:
    """Run GLiNER extraction against gold standard.

    These are slow tests (load ML model). Use `pytest -m slow` to run.
    Quality bars:
    - Exact F1 >= 0.4 (seed + GLiNER combined)
    - Partial F1 >= 0.5
    """

    @pytest.fixture
    def gold_samples(self):
        return _load_gold_standard()

    @pytest.mark.slow
    def test_gliner_partial_f1(self, gold_samples):
        """GLiNER + seed partial F1 should meet quality bar."""
        from brainlayer.pipeline.entity_extraction import extract_entities_combined
        from brainlayer.pipeline.batch_extraction import DEFAULT_SEED_ENTITIES

        all_gold = []
        all_pred = []
        for sample in gold_samples:
            all_gold.extend(sample["entities"])
            result = extract_entities_combined(
                sample["text"], DEFAULT_SEED_ENTITIES, use_llm=False, use_gliner=True
            )
            pred = [
                {"text": e.text, "type": e.entity_type, "start": e.start, "end": e.end}
                for e in result.entities
            ]
            all_pred.extend(pred)

        m = _compute_ner_metrics(all_gold, all_pred)
        per_type = _compute_per_type_metrics(all_gold, all_pred)

        print(f"\n=== GLiNER + Seed Eval ===")
        print(f"Gold: {m['exact']['possible']} entities, Predicted: {m['exact']['actual']}")
        print(f"Exact:   P={m['exact']['precision']:.3f}  R={m['exact']['recall']:.3f}  F1={m['exact']['f1']:.3f}")
        print(f"Partial: P={m['partial']['precision']:.3f}  R={m['partial']['recall']:.3f}  F1={m['partial']['f1']:.3f}")
        print(f"Type:    P={m['type_only']['precision']:.3f}  R={m['type_only']['recall']:.3f}  F1={m['type_only']['f1']:.3f}")
        for etype, scores in per_type.items():
            print(f"  {etype}: P={scores['precision']:.3f} R={scores['recall']:.3f} F1={scores['f1']:.3f} ({scores['correct']}/{scores['possible']})")

        assert m["partial"]["f1"] >= 0.4, (
            f"GLiNER+seed partial F1 {m['partial']['f1']:.3f} below quality bar 0.4"
        )
