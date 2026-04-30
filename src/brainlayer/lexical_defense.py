from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().with_name("lexical_defense_dictionary.json")


def _normalize_surface(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return "".join(ch for ch in normalized if ch.isalnum())


@dataclass(frozen=True)
class LexicalDefenseEntry:
    canonical: str
    category: str
    script: str
    protect_from_split: bool
    swift_override_priority: int
    aliases: tuple[str, ...]
    split_forms: tuple[str, ...]
    sources: tuple[str, ...]

    @property
    def all_surfaces(self) -> tuple[str, ...]:
        surfaces = [self.canonical, *self.aliases, *self.split_forms]
        deduped: list[str] = []
        seen: set[str] = set()
        for surface in surfaces:
            if surface not in seen:
                deduped.append(surface)
                seen.add(surface)
        return tuple(deduped)


class LexicalDefenseDictionary:
    def __init__(self, version: str, generated_at: str, entries: list[LexicalDefenseEntry]):
        self.version = version
        self.generated_at = generated_at
        self.entries = tuple(entries)
        self.by_canonical = {entry.canonical: entry for entry in self.entries}
        self.by_surface = self._build_surface_index()
        self.replacement_map = self._build_replacement_map()

    def _build_surface_index(self) -> dict[str, LexicalDefenseEntry]:
        index: dict[str, LexicalDefenseEntry] = {}
        for entry in self.entries:
            for surface in entry.all_surfaces:
                index[_normalize_surface(surface)] = entry
        return index

    def _build_replacement_map(self) -> dict[str, str]:
        pairs: dict[str, str] = {}
        for entry in self.entries:
            for split_form in entry.split_forms:
                pairs[split_form] = entry.canonical
        return dict(sorted(pairs.items(), key=lambda item: (-len(item[0]), item[0])))

    def lookup(self, surface: str) -> LexicalDefenseEntry | None:
        return self.by_surface.get(_normalize_surface(surface))

    def protected_entities(self) -> list[str]:
        return [entry.canonical for entry in self.entries if entry.protect_from_split]

    def grammar_literals(self) -> list[str]:
        literals: set[str] = set()
        for entry in self.entries:
            literals.add(entry.canonical)
            literals.update(entry.aliases)
        return sorted(literals, key=lambda value: (-len(value), value.casefold()))

    def slm_entity_lines(self) -> list[str]:
        return [f"- {entry.canonical} [{entry.category}]" for entry in self.entries if entry.protect_from_split]

    def swift_override_patterns(self) -> list[dict[str, str | int]]:
        patterns: list[dict[str, str | int]] = []
        for entry in sorted(self.entries, key=lambda item: (-item.swift_override_priority, item.canonical.casefold())):
            for split_form in entry.split_forms:
                patterns.append(
                    {
                        "match": split_form,
                        "replacement": entry.canonical,
                        "priority": entry.swift_override_priority,
                    }
                )
        return patterns

    def voicelayer_snapshot(self) -> dict[str, object]:
        return {
            "updated_at": self.generated_at,
            "prompt_terms": self.protected_entities(),
            "aliases": [{"from": match, "to": replacement} for match, replacement in self.replacement_map.items()],
        }

    def whisper_entity_gbnf(self) -> str:
        protected = [entry for entry in self.entries if entry.protect_from_split]
        lines = ["root ::= protected_entity", ""]
        lines.append("protected_entity ::= " + " | ".join(f"entity_{index}" for index, _entry in enumerate(protected)))
        lines.append("")
        for index, entry in enumerate(protected):
            literal = entry.canonical.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'entity_{index} ::= "{literal}"')
        return "\n".join(lines)


def _load_entries(payload: dict) -> list[LexicalDefenseEntry]:
    return [
        LexicalDefenseEntry(
            canonical=item["canonical"],
            category=item["category"],
            script=item["script"],
            protect_from_split=item["protect_from_split"],
            swift_override_priority=item["swift_override_priority"],
            aliases=tuple(item.get("aliases", [])),
            split_forms=tuple(item.get("split_forms", [])),
            sources=tuple(item.get("sources", [])),
        )
        for item in payload["entries"]
    ]


@lru_cache(maxsize=1)
def load_lexical_defense_dictionary(path: Path | None = None) -> LexicalDefenseDictionary:
    dictionary_path = path or DATA_PATH
    payload = json.loads(dictionary_path.read_text(encoding="utf-8"))
    return LexicalDefenseDictionary(
        version=payload["version"],
        generated_at=payload["generated_at"],
        entries=_load_entries(payload),
    )
