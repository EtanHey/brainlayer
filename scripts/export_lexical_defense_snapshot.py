#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from brainlayer.lexical_defense import load_lexical_defense_dictionary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export lexical-defense artifacts for downstream consumers.")
    parser.add_argument(
        "--format",
        choices=("json", "voicelayer", "gbnf"),
        default="json",
        help="Output artifact format.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination file path.")
    args = parser.parse_args()

    dictionary = load_lexical_defense_dictionary()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        payload = {
            "version": dictionary.version,
            "generated_at": dictionary.generated_at,
            "entries": [
                {
                    "canonical": entry.canonical,
                    "category": entry.category,
                    "script": entry.script,
                    "protect_from_split": entry.protect_from_split,
                    "swift_override_priority": entry.swift_override_priority,
                    "aliases": list(entry.aliases),
                    "split_forms": list(entry.split_forms),
                    "sources": list(entry.sources),
                }
                for entry in dictionary.entries
            ],
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.format == "voicelayer":
        args.output.write_text(
            json.dumps(dictionary.voicelayer_snapshot(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    args.output.write_text(dictionary.whisper_entity_gbnf() + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
