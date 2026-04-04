"""Cross-script entity resolution via Beider-Morse Phonetic Matching."""

from __future__ import annotations

import re
import unicodedata

from abydos.phonetic import BeiderMorse

_BM = BeiderMorse(name_mode="gen", match_mode="approx")
_HEBREW_RE = re.compile(r"[\u0590-\u05FF]{2,}")
_HEBREW_CHAR_MAP = {
    "א": "a",
    "ב": "b",
    "ג": "g",
    "ד": "d",
    "ה": "h",
    "ו": "o",
    "ז": "z",
    "ח": "h",
    "ט": "t",
    "י": "i",
    "כ": "k",
    "ך": "k",
    "ל": "l",
    "מ": "m",
    "ם": "m",
    "נ": "n",
    "ן": "n",
    "ס": "s",
    "ע": "a",
    "פ": "p",
    "ף": "p",
    "צ": "ts",
    "ץ": "ts",
    "ק": "k",
    "ר": "r",
    "ש": "sh",
    "ת": "t",
}
_HEBREW_SEQ_MAP = (
    ("היי", "hei"),
    ("אי", "ei"),
    ("תן", "tan"),
    ("מן", "man"),
    ("יי", "i"),
)


def looks_hebrew(text: str) -> bool:
    """Return True when the text contains a substantial Hebrew token."""
    return bool(_HEBREW_RE.search(text or ""))


def _normalize_text(name: str) -> str:
    """Normalize text and transliterate Hebrew into a BMPM-friendly Latin form."""
    text = "".join(c for c in unicodedata.normalize("NFD", name.strip()) if unicodedata.category(c) != "Mn")
    if not looks_hebrew(text):
        return text
    normalized = text
    for source, target in _HEBREW_SEQ_MAP:
        normalized = normalized.replace(source, target)
    return "".join(_HEBREW_CHAR_MAP.get(char, char) for char in normalized)


def phonetic_key(name: str) -> str:
    """Generate a normalized BMPM phonetic key for a name."""
    if not name or not name.strip():
        return ""
    encoded = _BM.encode(_normalize_text(name))
    if not encoded:
        return ""
    if isinstance(encoded, (list, tuple, set)):
        parts = [str(part).strip() for part in encoded if str(part).strip()]
        return " ".join(sorted(set(parts)))
    return str(encoded).strip()


def phonetic_tokens(name: str) -> set[str]:
    """Split a BMPM key into comparable tokens."""
    return {token for token in phonetic_key(name).split() if token}


def phonetic_match(name1: str, name2: str, threshold: float = 0.7) -> bool:
    """Check if two names match phonetically."""
    set1 = phonetic_tokens(name1)
    set2 = phonetic_tokens(name2)
    if not set1 or not set2:
        return False
    if set1 & set2:
        return True
    similarity = len(set1 & set2) / len(set1 | set2)
    return similarity >= threshold
