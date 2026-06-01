#!/usr/bin/env python3
"""Build the frozen ABCDE enrichment prompt registry without model/API calls."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PROMPT_PATH = REPO_ROOT / "src/brainlayer/pipeline/enrichment.py"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "src/brainlayer/eval/abcde_variants.yaml"
ORCHESTRATOR_ROOT = REPO_ROOT.parent / "orchestrator"
SHELF_SECTION = "Enrichment Prompt (copy-paste into pipeline)"
EXPERIMENT_MODEL = "gemini-2.5-flash-lite"
EXPERIMENT_PARAMS = {
    "temperature": 0.2,
    "max_output_tokens": 4096,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposals-json", type=Path, required=True, help="JSON file written by the subscription CLI agent"
    )
    parser.add_argument(
        "--shelf-prompt-path",
        type=Path,
        required=True,
        help="Markdown file containing the shelved faceted enrichment prompt section",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRY_PATH)
    args = parser.parse_args()

    production_prompt = extract_python_string_constant(PRODUCTION_PROMPT_PATH, "ENRICHMENT_PROMPT")
    shelf_prompt = extract_markdown_fenced_section(args.shelf_prompt_path, SHELF_SECTION)
    proposals = load_proposals(args.proposals_json)

    variants = [
        {
            "id": "A",
            "label": "production-enrichment-prompt",
            "prompt_template": production_prompt,
            "model": EXPERIMENT_MODEL,
            "params": dict(EXPERIMENT_PARAMS),
            "provenance": "production",
            "axis": "production-control",
            "source_path": format_source_path(PRODUCTION_PROMPT_PATH),
            "source_section": "ENRICHMENT_PROMPT",
        },
        {
            "id": "B",
            "label": "2026-03-19-v2-faceted-tags",
            "prompt_template": shelf_prompt,
            "model": EXPERIMENT_MODEL,
            "params": dict(EXPERIMENT_PARAMS),
            "provenance": "shelf",
            "axis": "faceted-tag-taxonomy",
            "source_path": format_source_path(args.shelf_prompt_path),
            "source_section": SHELF_SECTION,
        },
    ]

    for proposal in proposals:
        variants.append(
            {
                "id": proposal["id"],
                "label": proposal["label"],
                "prompt_template": proposal["prompt_template"],
                "model": EXPERIMENT_MODEL,
                "params": dict(EXPERIMENT_PARAMS),
                "provenance": "llm-proposed",
                "axis": proposal["axis"],
                "source_path": f"inline-llm-reasoning:{args.proposals_json.name}",
                "source_section": f"llm-proposed-{proposal['id']}",
            }
        )

    for variant in variants:
        variant["prompt_hash"] = compute_prompt_hash(variant["prompt_template"])
    assert_freeze_integrity(variants, production_prompt, shelf_prompt)

    registry = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "hash_algorithm": "sha256",
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(registry, handle, sort_keys=False, width=120, allow_unicode=True)


def extract_python_string_constant(path: Path, constant_name: str) -> str:
    module = ast.parse(path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == constant_name for target in node.targets):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError(f"{constant_name} in {path} is not a string")
            return value
    raise ValueError(f"Could not find {constant_name} in {path}")


def format_source_path(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    if resolved.is_relative_to(ORCHESTRATOR_ROOT):
        return f"orchestrator/{resolved.relative_to(ORCHESTRATOR_ROOT)}"
    return f"external:{resolved.name}"


def extract_markdown_fenced_section(path: Path, heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    pattern = rf"^## {re.escape(heading)}\s*\n\s*```(?:\w+)?\n(?P<prompt>.*?)\n```"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not find fenced section {heading!r} in {path}")
    return match.group("prompt")


def load_proposals(path: Path) -> list[dict[str, str]]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    variants = raw.get("variants") if isinstance(raw, dict) else None
    if not isinstance(variants, list):
        raise ValueError("Proposals JSON must contain a variants list")
    if [variant.get("id") for variant in variants if isinstance(variant, dict)] != ["C", "D", "E"]:
        raise ValueError("Proposals JSON must contain C, D, E variants in order")

    normalized = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("Each proposal variant must be a mapping")
        for key in ("id", "label", "axis", "prompt_template"):
            if not isinstance(variant.get(key), str) or not variant[key].strip():
                raise ValueError(f"Proposal {variant.get('id')} must include non-empty string {key}")
        normalized.append(
            {
                "id": variant["id"],
                "label": variant["label"],
                "axis": variant["axis"],
                "prompt_template": variant["prompt_template"],
            }
        )
    return normalized


def assert_freeze_integrity(variants: list[dict[str, Any]], production_prompt: str, shelf_prompt: str) -> None:
    hashes = {variant["id"]: variant["prompt_hash"] for variant in variants}
    if tuple(hashes) != ("A", "B", "C", "D", "E"):
        raise ValueError(f"freeze must contain A-E in order, got {tuple(hashes)}")
    if len(set(hashes.values())) != len(hashes):
        collisions: dict[str, list[str]] = {}
        for variant_id, prompt_hash in hashes.items():
            collisions.setdefault(prompt_hash, []).append(variant_id)
        duplicates = {prompt_hash: ids for prompt_hash, ids in collisions.items() if len(ids) > 1}
        raise ValueError(f"freeze prompt_hash collision detected: {duplicates}")

    expected_a = compute_prompt_hash(production_prompt)
    if hashes["A"] != expected_a:
        raise ValueError(f"variant A hash does not match live ENRICHMENT_PROMPT hash: {hashes['A']} != {expected_a}")
    expected_b = compute_prompt_hash(shelf_prompt)
    if hashes["B"] != expected_b:
        raise ValueError(f"variant B hash does not match shelved v2 prompt hash: {hashes['B']} != {expected_b}")


def compute_prompt_hash(prompt_template: str) -> str:
    return hashlib.sha256(prompt_template.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
