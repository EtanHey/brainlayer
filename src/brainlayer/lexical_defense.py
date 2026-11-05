from __future__ import annotations

import json
import logging
import os
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().with_name("lexical_defense_dictionary.json")

# Optional per-user alias source (#995). It lives OUTSIDE the package so personal
# terms are never shipped; the BrainBar native fallback reads the same file with the
# same bounds (brain-bar/Sources/BrainBar/LexicalDefenseUserAliases.swift).
USER_ALIASES_ENV = "BRAINLAYER_LEXICAL_DEFENSE_LOCAL"
USER_ALIASES_SCHEMA_VERSION = 1
USER_ALIASES_MAX_BYTES = 64 * 1024
USER_ALIASES_MAX_ENTRIES = 256
USER_ALIASES_MAX_SURFACES_PER_ENTRY = 16
USER_ALIASES_MAX_SURFACE_CHARS = 64


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


class UserAliasConfigError(ValueError):
    """The user alias file is present but violates the bounded schema; none of it is used."""


def default_user_aliases_path(environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    override = env.get(USER_ALIASES_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "brainlayer" / "lexical_defense.local.json"


def _user_surface(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise UserAliasConfigError(f"{where}: surface must be a string")
    surface = value.strip()
    if not surface or len(surface) > USER_ALIASES_MAX_SURFACE_CHARS:
        raise UserAliasConfigError(f"{where}: surface must be 1..{USER_ALIASES_MAX_SURFACE_CHARS} chars")
    if not _normalize_surface(surface):
        raise UserAliasConfigError(f"{where}: surface has no letters or digits")
    return surface


def parse_user_alias_entries(raw: bytes) -> list[LexicalDefenseEntry]:
    """Validate the user file strictly. Any violation rejects the WHOLE file."""
    if len(raw) > USER_ALIASES_MAX_BYTES:
        raise UserAliasConfigError(f"file exceeds {USER_ALIASES_MAX_BYTES} bytes")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UserAliasConfigError(f"not UTF-8 JSON: {exc}") from None
    if not isinstance(payload, dict) or set(payload) != {"version", "entries"}:
        raise UserAliasConfigError('top level must be exactly {"version", "entries"}')
    if payload["version"] != USER_ALIASES_SCHEMA_VERSION or isinstance(payload["version"], bool):
        raise UserAliasConfigError(f"version must be {USER_ALIASES_SCHEMA_VERSION}")
    items = payload["entries"]
    if not isinstance(items, list) or len(items) > USER_ALIASES_MAX_ENTRIES:
        raise UserAliasConfigError(f"entries must be a list of at most {USER_ALIASES_MAX_ENTRIES}")
    entries: list[LexicalDefenseEntry] = []
    for index, item in enumerate(items):
        where = f"entries[{index}]"
        if (
            not isinstance(item, dict)
            or "canonical" not in item
            or not set(item) <= {"canonical", "aliases", "split_forms"}
        ):
            raise UserAliasConfigError(f"{where}: keys must be canonical plus optional aliases/split_forms")
        lists = {}
        for key in ("aliases", "split_forms"):
            values = item.get(key, [])
            if not isinstance(values, list):
                raise UserAliasConfigError(f"{where}.{key}: must be a list")
            lists[key] = tuple(_user_surface(value, f"{where}.{key}") for value in values)
        if len(lists["aliases"]) + len(lists["split_forms"]) > USER_ALIASES_MAX_SURFACES_PER_ENTRY:
            raise UserAliasConfigError(f"{where}: more than {USER_ALIASES_MAX_SURFACES_PER_ENTRY} surfaces")
        entries.append(
            LexicalDefenseEntry(
                canonical=_user_surface(item["canonical"], f"{where}.canonical"),
                category="user_alias",
                script="user",
                protect_from_split=False,
                swift_override_priority=0,
                aliases=lists["aliases"],
                split_forms=lists["split_forms"],
                sources=("user_config",),
            )
        )
    return entries


def merge_user_alias_entries(
    base: LexicalDefenseDictionary, user_entries: list[LexicalDefenseEntry]
) -> tuple[LexicalDefenseDictionary, list[str]]:
    """Add user entries without ever changing a packaged surface.

    Conflict policy: a user entry any of whose normalized surfaces is already claimed
    (by the package, or by an EARLIER user entry) is skipped whole. That also breaks
    every cross-entry cycle (A->B plus B->A), and expansion is one hop, never transitive.
    """
    claimed = set(base.by_surface)
    accepted: list[LexicalDefenseEntry] = []
    skipped: list[str] = []
    for entry in user_entries:
        surfaces = {_normalize_surface(surface) for surface in entry.all_surfaces}
        if surfaces & claimed:
            skipped.append(entry.canonical)
            continue
        claimed |= surfaces
        accepted.append(entry)
    if not accepted:
        return base, skipped
    return LexicalDefenseDictionary(base.version, base.generated_at, [*base.entries, *accepted]), skipped


_effective_lock = threading.Lock()
_effective_cache: tuple[tuple[str, int, int] | None, LexicalDefenseDictionary] | None = None


def _user_file_stamp(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        return (str(path), -1, -1)  # present but not stat()able: read below fails and warns once
    return (str(path), stat.st_mtime_ns, stat.st_size)


def load_effective_lexical_defense_dictionary(user_path: Path | None = None) -> LexicalDefenseDictionary:
    """Packaged dictionary plus the optional user alias file (#995).

    Freshness: the user file is stat()ed on every call and re-read only when its
    (path, mtime_ns, size) changes, so an edit applies to the next search without a
    restart. A missing file yields exactly the packaged dictionary. A malformed,
    oversized or unreadable file is ignored whole with one warning per file version.
    """
    global _effective_cache
    base = load_lexical_defense_dictionary()
    path = user_path or default_user_aliases_path()
    stamp = _user_file_stamp(path)
    with _effective_lock:
        if _effective_cache is not None and _effective_cache[0] == stamp:
            return _effective_cache[1]
        effective = base
        if stamp is not None:
            try:
                with path.open("rb") as handle:
                    raw = handle.read(USER_ALIASES_MAX_BYTES + 1)
                effective, skipped = merge_user_alias_entries(base, parse_user_alias_entries(raw))
                if skipped:
                    logger.warning("lexical defense: %d user alias entries skipped on conflict: %s", len(skipped), path)
            except (OSError, UserAliasConfigError) as exc:
                logger.warning("lexical defense: ignoring user alias file %s: %s", path, exc)
        _effective_cache = (stamp, effective)
        return effective
