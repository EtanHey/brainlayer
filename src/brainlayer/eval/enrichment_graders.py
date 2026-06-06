"""Pure deterministic graders for BrainLayer enrichment eval labels."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence

VALID_INTENTS = frozenset(
    {"debugging", "designing", "configuring", "discussing", "deciding", "implementing", "reviewing"}
)
VALID_EPISTEMIC_LEVELS = frozenset({"hypothesis", "substantiated", "validated"})
VALID_DEBT_IMPACTS = frozenset({"introduction", "resolution", "none"})
VALID_ENTITY_TYPES = frozenset(
    {"person", "agent", "company", "project", "technology", "tool", "concept", "topic", "source"}
)
VALID_ENTITY_SUBTYPES = frozenset({"channel", "podcast", "brand", "newsletter"})
VALID_SENTIMENT_LABELS = frozenset({"frustration", "confusion", "positive", "satisfaction", "neutral"})

# Current runtime schema union: prompt fields plus the legacy singular resolved_query
# consumed by controller/drain paths.
REQUIRED_ENRICHMENT_KEYS = (
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
)

BANNED_SUMMARY_PATTERNS = (
    (
        "this chunk/message describes/details/outlines/provides/contains",
        re.compile(r"^\s*this\s+(?:chunk|message)\s+(?:describes|details|outlines|provides|contains)\b", re.I),
    ),
    (
        "the user/assistant is asking/instructing/discussing/explaining",
        re.compile(
            r"^\s*the\s+(?:user|assistant)\s+(?:is\s+)?(?:asking|instructing|discussing|explaining)\b",
            re.I,
        ),
    ),
    (
        "this is a conversation/discussion about",
        re.compile(r"^\s*this\s+is\s+a\s+(?:conversation|discussion)\s+about\b", re.I),
    ),
    (
        "the conversation covers/revolves around",
        re.compile(r"^\s*the\s+conversation\s+(?:covers|revolves\s+around)\b", re.I),
    ),
)


@dataclass(frozen=True)
class SchemaGateResult:
    passed: bool
    errors: list[str]
    value: dict[str, Any] | None = None


@dataclass(frozen=True)
class KeyFactsRecall:
    recall: float
    matched: tuple[str, ...]
    missed: tuple[str, ...]
    total: int


@dataclass(frozen=True)
class EntityMetrics:
    name_precision: float
    name_recall: float
    name_f1: float
    type_strict_precision: float
    type_strict_recall: float
    type_strict_f1: float
    hallucinated_entities: tuple[str, ...]


@dataclass(frozen=True)
class ImportanceCalibration:
    mae: float
    spearman_rho: float
    band_accuracy: float
    exact_accuracy: float


@dataclass(frozen=True)
class MetaResearchCheck:
    passed: bool
    expected_importance: int | None
    actual_importance: int | None


@dataclass(frozen=True)
class EnrichmentGrade:
    disqualified: bool
    overall_score: float
    schema: SchemaGateResult
    banned_pattern_hit: bool
    banned_pattern: str | None
    key_facts: KeyFactsRecall
    entities: EntityMetrics
    importance: ImportanceCalibration
    meta_research: MetaResearchCheck


def validate_schema_gate(candidate: str | Mapping[str, Any]) -> SchemaGateResult:
    """Validate candidate enrichment JSON without I/O or normalization side effects."""

    parsed: Any
    if isinstance(candidate, str):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            return SchemaGateResult(False, [f"invalid JSON: {exc.msg}"], None)
    elif isinstance(candidate, Mapping):
        parsed = dict(candidate)
    else:
        return SchemaGateResult(False, ["candidate must be a JSON object or mapping"], None)

    if not isinstance(parsed, dict):
        return SchemaGateResult(False, ["candidate must be a JSON object"], None)

    errors: list[str] = []
    for key in REQUIRED_ENRICHMENT_KEYS:
        if key not in parsed:
            errors.append(f"missing required key: {key}")

    if errors:
        return SchemaGateResult(False, errors, parsed)

    if not isinstance(parsed["summary"], str) or not parsed["summary"].strip():
        errors.append("summary must be a non-empty string")
    _validate_string_list(parsed, "key_facts", errors)
    _validate_string_list(parsed, "primary_symbols", errors)
    _validate_string_list(parsed, "external_deps", errors)
    _validate_string_list(parsed, "sentiment_signals", errors)

    tags = parsed["tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags) or not 3 <= len(tags) <= 7:
        errors.append("tags must contain 3 to 7 strings")

    importance = parsed["importance"]
    if not isinstance(importance, int) or isinstance(importance, bool) or not 1 <= importance <= 10:
        errors.append("importance must be an integer from 1 to 10")

    _validate_enum(parsed, "intent", VALID_INTENTS, errors)
    _validate_enum(parsed, "epistemic_level", VALID_EPISTEMIC_LEVELS, errors)
    _validate_enum(parsed, "debt_impact", VALID_DEBT_IMPACTS, errors)
    _validate_enum(parsed, "sentiment_label", VALID_SENTIMENT_LABELS, errors)

    resolved_query = parsed["resolved_query"]
    if resolved_query is not None and not isinstance(resolved_query, str):
        errors.append("resolved_query must be a string or null")

    resolved_queries = parsed["resolved_queries"]
    if (
        not isinstance(resolved_queries, list)
        or not all(isinstance(query, str) for query in resolved_queries)
        or len(resolved_queries) != 3
    ):
        errors.append("resolved_queries must contain exactly 3 strings")

    version_scope = parsed["version_scope"]
    if version_scope is not None and not isinstance(version_scope, str):
        errors.append("version_scope must be a string or null")

    sentiment_score = parsed["sentiment_score"]
    if (
        not isinstance(sentiment_score, (int, float))
        or isinstance(sentiment_score, bool)
        or not math.isfinite(float(sentiment_score))
        or not -1.0 <= float(sentiment_score) <= 1.0
    ):
        errors.append("sentiment_score must be a number in [-1, 1]")

    entities = parsed["entities"]
    if not isinstance(entities, list):
        errors.append("entities must be a list")
    else:
        for index, entity in enumerate(entities):
            if not isinstance(entity, Mapping):
                errors.append(f"entities[{index}] must be an object")
                continue
            name = entity.get("name")
            entity_type = entity.get("type")
            entity_subtype = entity.get("entity_subtype")
            relation = entity.get("relation")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"entities[{index}].name must be a non-empty string")
            if not isinstance(entity_type, str) or entity_type not in VALID_ENTITY_TYPES:
                errors.append(f"entities[{index}].type has invalid enum value: {entity_type}")
            if entity_subtype is not None:
                if entity_type != "source":
                    errors.append(f"entities[{index}].entity_subtype is only valid for source entities")
                elif not isinstance(entity_subtype, str) or entity_subtype not in VALID_ENTITY_SUBTYPES:
                    errors.append(f"entities[{index}].entity_subtype has invalid enum value: {entity_subtype}")
            if relation is not None and not isinstance(relation, str):
                errors.append(f"entities[{index}].relation must be a string or null")

    return SchemaGateResult(not errors, errors, parsed)


def find_banned_summary_pattern(summary: str) -> str | None:
    """Return the banned enrichment.py prompt pattern that starts this summary."""

    if not isinstance(summary, str):
        return None
    for label, pattern in BANNED_SUMMARY_PATTERNS:
        if pattern.search(summary):
            return label
    return None


def score_key_facts_recall(candidate: Mapping[str, Any], gold: Mapping[str, Any]) -> KeyFactsRecall:
    expected = tuple(str(fact) for fact in gold.get("must_capture_facts", ()) if str(fact).strip())
    if not expected:
        return KeyFactsRecall(1.0, (), (), 0)

    haystack = _candidate_fact_haystack(candidate)
    matched = tuple(fact for fact in expected if _contains_normalized(haystack, fact))
    missed = tuple(fact for fact in expected if fact not in matched)
    return KeyFactsRecall(len(matched) / len(expected), matched, missed, len(expected))


def score_entities(candidate: Mapping[str, Any], gold: Mapping[str, Any], chunk_text: str) -> EntityMetrics:
    candidate_entities = _entity_entries(candidate.get("entities", ()))
    gold_entities = _entity_entries(gold.get("gold_entities", ()))

    candidate_names = {_normalize_entity_name(entity["name"]) for entity in candidate_entities}
    gold_names = {_normalize_entity_name(entity["name"]) for entity in gold_entities}
    name_matches = candidate_names & gold_names

    candidate_typed = {(_normalize_entity_name(entity["name"]), entity["type"]) for entity in candidate_entities}
    gold_typed = {(_normalize_entity_name(entity["name"]), entity["type"]) for entity in gold_entities}
    typed_matches = candidate_typed & gold_typed

    hallucinated = tuple(
        entity["name"] for entity in candidate_entities if not _entity_supported_by_chunk(entity["name"], chunk_text)
    )

    return EntityMetrics(
        name_precision=_precision(len(name_matches), len(candidate_names)),
        name_recall=_recall(len(name_matches), len(gold_names)),
        name_f1=_f1(len(name_matches), len(candidate_names), len(gold_names)),
        type_strict_precision=_precision(len(typed_matches), len(candidate_typed)),
        type_strict_recall=_recall(len(typed_matches), len(gold_typed)),
        type_strict_f1=_f1(len(typed_matches), len(candidate_typed), len(gold_typed)),
        hallucinated_entities=hallucinated,
    )


def score_importance_calibration(
    predicted_importance: Sequence[int | float],
    gold_importance: Sequence[int | float],
) -> ImportanceCalibration:
    if len(predicted_importance) != len(gold_importance):
        raise ValueError("predicted_importance and gold_importance must have the same length")
    if not predicted_importance:
        return ImportanceCalibration(0.0, 0.0, 1.0, 1.0)

    predicted = [float(value) for value in predicted_importance]
    gold = [float(value) for value in gold_importance]
    mae = sum(abs(pred - expected) for pred, expected in zip(predicted, gold, strict=True)) / len(predicted)
    exact_accuracy = sum(pred == expected for pred, expected in zip(predicted, gold, strict=True)) / len(predicted)
    band_accuracy = sum(
        _importance_band(pred) == _importance_band(expected) for pred, expected in zip(predicted, gold, strict=True)
    ) / len(predicted)
    return ImportanceCalibration(
        mae=mae,
        spearman_rho=_spearman(predicted, gold),
        band_accuracy=band_accuracy,
        exact_accuracy=exact_accuracy,
    )


def check_meta_research_forced_importance(chunk_text: str, candidate: Mapping[str, Any]) -> MetaResearchCheck:
    if not _is_meta_research_chunk(chunk_text):
        return MetaResearchCheck(True, None, _safe_int(candidate.get("importance")))
    actual = _safe_int(candidate.get("importance"))
    return MetaResearchCheck(actual == 2, 2, actual)


def grade_candidate(candidate: str | Mapping[str, Any], gold: Mapping[str, Any], *, chunk_text: str) -> EnrichmentGrade:
    schema = validate_schema_gate(candidate)
    value = schema.value or {}
    empty_key_facts = KeyFactsRecall(0.0, (), (), 0)
    empty_entities = EntityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())
    empty_importance = ImportanceCalibration(0.0, 0.0, 0.0, 0.0)
    empty_meta = MetaResearchCheck(True, None, None)

    if not schema.passed:
        return EnrichmentGrade(
            True, 0.0, schema, False, None, empty_key_facts, empty_entities, empty_importance, empty_meta
        )

    banned_pattern = find_banned_summary_pattern(value.get("summary", ""))
    if banned_pattern is not None:
        key_facts = score_key_facts_recall(value, gold)
        entities = score_entities(value, gold, chunk_text)
        importance = score_importance_calibration([value["importance"]], [_gold_importance(gold)])
        meta = check_meta_research_forced_importance(chunk_text, value)
        return EnrichmentGrade(True, 0.0, schema, True, banned_pattern, key_facts, entities, importance, meta)

    key_facts = score_key_facts_recall(value, gold)
    entities = score_entities(value, gold, chunk_text)
    importance = score_importance_calibration([value["importance"]], [_gold_importance(gold)])
    meta = check_meta_research_forced_importance(chunk_text, value)
    disqualified = not meta.passed
    if disqualified:
        overall = 0.0
    else:
        # Deterministic PR-2 composite: weight band calibration higher than exact MAE.
        importance_component = (importance.band_accuracy * 0.8) + ((1.0 - min(importance.mae, 9.0) / 9.0) * 0.2)
        valid_entity_count = len(_entity_entries(value.get("entities", ())))
        hallucination_penalty = (
            1.0
            if not entities.hallucinated_entities
            else max(0.0, 1.0 - (len(entities.hallucinated_entities) / max(1, valid_entity_count)))
        )
        overall = (
            key_facts.recall * 0.35
            + entities.name_f1 * 0.25
            + entities.type_strict_f1 * 0.15
            + importance_component * 0.20
            + hallucination_penalty * 0.05
        )
    return EnrichmentGrade(disqualified, overall, schema, False, None, key_facts, entities, importance, meta)


def _validate_string_list(parsed: Mapping[str, Any], key: str, errors: list[str]) -> None:
    value = parsed[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"{key} must be a list of strings")


def _validate_enum(parsed: Mapping[str, Any], key: str, valid_values: frozenset[str], errors: list[str]) -> None:
    value = parsed[key]
    if not isinstance(value, str) or value not in valid_values:
        errors.append(f"{key} has invalid enum value: {value}")


def _candidate_fact_haystack(candidate: Mapping[str, Any]) -> str:
    values = [str(candidate.get("summary", ""))]
    key_facts = candidate.get("key_facts", ())
    if isinstance(key_facts, Sequence) and not isinstance(key_facts, (str, bytes)):
        values.extend(str(fact) for fact in key_facts)
    return "\n".join(values)


def _contains_normalized(haystack: str, needle: str) -> bool:
    return _normalize_text(needle) in _normalize_text(haystack)


def _entity_entries(raw_entities: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(raw_entities, Sequence) or isinstance(raw_entities, (str, bytes)):
        return ()
    entries: list[dict[str, str]] = []
    for entity in raw_entities:
        if not isinstance(entity, Mapping):
            continue
        name = entity.get("name")
        entity_type = entity.get("type")
        if (
            isinstance(name, str)
            and name.strip()
            and isinstance(entity_type, str)
            and entity_type in VALID_ENTITY_TYPES
        ):
            entries.append({"name": name.strip(), "type": entity_type})
    return tuple(entries)


def _entity_supported_by_chunk(name: str, chunk_text: str) -> bool:
    normalized_name = _normalize_entity_name(name)
    normalized_chunk = _normalize_text(chunk_text)
    if not normalized_name:
        return False
    if normalized_name in normalized_chunk:
        return True
    name_tokens = [token for token in normalized_name.split() if len(token) > 2]
    if name_tokens and all(token in normalized_chunk for token in name_tokens):
        return True
    return SequenceMatcher(None, normalized_name, normalized_chunk).ratio() >= 0.82


def _normalize_entity_name(value: str) -> str:
    normalized = _normalize_text(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value))
    if normalized.endswith("s") and len(normalized) > 3:
        normalized = normalized[:-1]
    return normalized


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", str(value)).lower()).strip()


def _precision(matches: int, candidate_total: int) -> float:
    if candidate_total == 0:
        return 1.0 if matches == 0 else 0.0
    return matches / candidate_total


def _recall(matches: int, gold_total: int) -> float:
    if gold_total == 0:
        return 1.0 if matches == 0 else 0.0
    return matches / gold_total


def _f1(matches: int, candidate_total: int, gold_total: int) -> float:
    precision = _precision(matches, candidate_total)
    recall = _recall(matches, gold_total)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _importance_band(value: float) -> str:
    if value <= 3:
        return "low"
    if value <= 6:
        return "moderate"
    if value <= 9:
        return "high"
    return "critical"


def _spearman(predicted: Sequence[float], gold: Sequence[float]) -> float:
    if len(predicted) < 2:
        return 1.0
    return _pearson(_rank(predicted), _rank(gold))


def _rank(values: Sequence[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(sorted_pairs):
        end = index + 1
        while end < len(sorted_pairs) and sorted_pairs[end][0] == sorted_pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for _, original_index in sorted_pairs[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_denominator = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    denominator = left_denominator * right_denominator
    if denominator == 0:
        return 0.0
    rho = numerator / denominator
    if math.isclose(rho, 1.0):
        return 1.0
    if math.isclose(rho, -1.0):
        return -1.0
    return rho


def _is_meta_research_chunk(chunk_text: str) -> bool:
    return "brain_search(" in chunk_text or "brain_entity(" in chunk_text


def _safe_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _gold_importance(gold: Mapping[str, Any]) -> int:
    value = gold.get("gold_importance")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError("gold must include integer gold_importance")
