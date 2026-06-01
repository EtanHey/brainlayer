"""Frozen ABCDE enrichment prompt registry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

VariantId = Literal["A", "B", "C", "D", "E"]
VariantProvenance = Literal["production", "shelf", "llm-proposed"]

VARIANT_IDS: tuple[VariantId, ...] = ("A", "B", "C", "D", "E")
REGISTRY_PATH = Path(__file__).with_name("abcde_variants.yaml")
HASH_ALGORITHM = "sha256"


@dataclass(frozen=True)
class ABCDEVariant:
    id: VariantId
    label: str
    prompt_template: str
    model: str
    params: dict[str, Any]
    provenance: VariantProvenance
    prompt_hash: str
    axis: str | None = None
    source_path: str | None = None
    source_section: str | None = None


def compute_prompt_hash(prompt_template: str) -> str:
    """Return the registry hash for an exact frozen prompt template."""

    return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()


def load_abcde_variants(registry_path: str | Path = REGISTRY_PATH) -> tuple[ABCDEVariant, ...]:
    """Load and validate the frozen ABCDE variant registry."""

    path = Path(registry_path)
    with path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)

    if not isinstance(registry, dict):
        raise ValueError("ABCDE registry must be a YAML mapping")
    if registry.get("hash_algorithm") != HASH_ALGORITHM:
        raise ValueError(f"ABCDE registry hash_algorithm must be {HASH_ALGORITHM!r}")

    raw_variants = registry.get("variants")
    if not isinstance(raw_variants, list):
        raise ValueError("ABCDE registry variants must be a list")

    variants: list[ABCDEVariant] = []
    for index, raw in enumerate(raw_variants):
        if not isinstance(raw, dict):
            raise ValueError(f"ABCDE variant at index {index} must be a mapping")
        variant = _variant_from_mapping(raw)
        expected_hash = compute_prompt_hash(variant.prompt_template)
        if variant.prompt_hash != expected_hash:
            raise ValueError(
                f"ABCDE variant {variant.id} prompt_hash mismatch: "
                f"registry={variant.prompt_hash} computed={expected_hash}"
            )
        variants.append(variant)

    ids = tuple(variant.id for variant in variants)
    if ids != VARIANT_IDS:
        raise ValueError(f"ABCDE registry must contain variants {VARIANT_IDS}, got {ids}")

    return tuple(variants)


def _variant_from_mapping(raw: dict[str, Any]) -> ABCDEVariant:
    missing = {
        key
        for key in ("id", "label", "prompt_template", "model", "params", "provenance", "prompt_hash")
        if key not in raw
    }
    if missing:
        raise ValueError(f"ABCDE variant is missing required keys: {sorted(missing)}")

    variant_id = raw["id"]
    if variant_id not in VARIANT_IDS:
        raise ValueError(f"Unknown ABCDE variant id: {variant_id!r}")

    provenance = raw["provenance"]
    if provenance not in ("production", "shelf", "llm-proposed"):
        raise ValueError(f"Unknown ABCDE variant provenance: {provenance!r}")

    label = raw["label"]
    prompt_template = raw["prompt_template"]
    model = raw["model"]
    params = raw["params"]
    prompt_hash = raw["prompt_hash"]
    if not isinstance(label, str) or not label:
        raise ValueError(f"ABCDE variant {variant_id} label must be a non-empty string")
    if not isinstance(prompt_template, str) or not prompt_template:
        raise ValueError(f"ABCDE variant {variant_id} prompt_template must be a non-empty string")
    if not isinstance(model, str) or not model:
        raise ValueError(f"ABCDE variant {variant_id} model must be a non-empty string")
    if not isinstance(params, dict):
        raise ValueError(f"ABCDE variant {variant_id} params must be a mapping")
    if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
        raise ValueError(f"ABCDE variant {variant_id} prompt_hash must be a 64-character hex string")

    return ABCDEVariant(
        id=variant_id,
        label=label,
        prompt_template=prompt_template,
        model=model,
        params=params,
        provenance=provenance,
        prompt_hash=prompt_hash,
        axis=_optional_string(raw, "axis"),
        source_path=_optional_string(raw, "source_path"),
        source_section=_optional_string(raw, "source_section"),
    )


def _optional_string(raw: dict[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"ABCDE variant {raw.get('id')} {key} must be a non-empty string or null")
    return value


ABCDE_VARIANTS = load_abcde_variants()
ABCDE_VARIANTS_BY_ID = {variant.id: variant for variant in ABCDE_VARIANTS}
