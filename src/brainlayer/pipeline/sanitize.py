"""PII sanitization pipeline for BrainLayer chunks.

Strips personally identifiable information from chunk content before sending
to external LLM APIs (Gemini, Groq). Three detection layers:

1. Regex — owner name, emails, file paths, IPs, JWTs, phone numbers, op:// refs
2. Known names dictionary — WhatsApp contacts + manual list (Hebrew + English)
3. spaCy NER — unknown English person names (en_core_web_sm)

Usage:
    from brainlayer.pipeline.sanitize import Sanitizer

    sanitizer = Sanitizer.from_env()
    result = sanitizer.sanitize("Etan said hello to David")
    print(result.sanitized)   # "[OWNER] said hello to [PERSON_a1b2c3d4]"
    print(result.pii_detected)  # True

    # Batch mode
    results = sanitizer.sanitize_batch(chunks, parallel=4)

    # Build name dictionary from WhatsApp contacts in DB
    names = sanitizer.build_name_dictionary(store)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# AIDEV-NOTE: spaCy is lazy-loaded to avoid slow import on every pipeline import.
# Only loaded when use_spacy_ner=True and sanitize() is first called.


# ── Types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Replacement:
    """Single PII replacement record."""

    category: str  # "owner", "person_name", "email", "file_path", "ip", "jwt", "op_ref", "phone", "github"
    original: str  # The matched text
    placeholder: str  # What it was replaced with
    start: int  # Position in original text
    end: int  # End position in original text
    source: str  # "regex", "spacy", "name_dict"


@dataclass
class SanitizeResult:
    """Output of sanitization — the cleaned text + audit metadata."""

    sanitized: str
    original_length: int
    replacements: list[Replacement] = field(default_factory=list)
    pii_detected: bool = False


@dataclass(frozen=True)
class SanitizeConfig:
    """What to sanitize and how."""

    owner_names: tuple[str, ...] = ()
    owner_emails: tuple[str, ...] = ()
    owner_paths: tuple[str, ...] = ()
    known_names: frozenset[str] = frozenset()
    strip_emails: bool = True
    strip_ips: bool = True
    strip_jwts: bool = True
    strip_op_refs: bool = True
    strip_phone_numbers: bool = True
    use_spacy_ner: bool = True


# ── Helpers ─────────────────────────────────────────────────────────────

# Hebrew nikud (diacritics) range: U+0591 to U+05C7
_NIKUD_RE = re.compile(r"[\u0591-\u05c7]")


def _strip_nikud(text: str) -> str:
    """Remove Hebrew nikud (diacritical marks) for fuzzy name matching."""
    return _NIKUD_RE.sub("", text)


def _nikud_offset_map(original: str) -> list[int]:
    """Build mapping from nikud-stripped positions to original positions.

    Returns a list where map[stripped_idx] = original_idx, plus one extra
    entry at the end for end-of-string positions.
    """
    offset_map: list[int] = []
    for orig_idx, ch in enumerate(original):
        if not _NIKUD_RE.match(ch):
            offset_map.append(orig_idx)
    # Sentinel for end positions
    offset_map.append(len(original))
    return offset_map


# ── Regex patterns ─────────────────────────────────────────────────────

# Email: simplified RFC 5322
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")

# IPv4
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

# JWT tokens (3 base64 segments separated by dots)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")

# 1Password references
_OP_REF_RE = re.compile(r"op://[^\s\"']+")

# Phone numbers: international format (+972..., +1..., etc.)
_PHONE_RE = re.compile(r"\+\d{1,3}[\s.-]?\d{1,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b")

# Code blocks (to exclude from NER)
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")

# GitHub URLs and @mentions
_GITHUB_URL_RE = re.compile(r"github\.com/([a-zA-Z0-9_-]+)")
_GITHUB_MENTION_RE = re.compile(r"@([a-zA-Z0-9_-]+)")


# ── Sanitizer ──────────────────────────────────────────────────────────


class Sanitizer:
    """Reusable PII sanitizer for BrainLayer chunks.

    Thread-safe for batch processing — spaCy model is loaded once and shared.
    """

    def __init__(self, config: SanitizeConfig) -> None:
        self.config = config
        self._nlp = None  # Lazy-loaded spaCy model
        self._name_to_pseudo: dict[str, str] = {}  # name.lower() → placeholder
        self._pseudo_lock = threading.Lock()  # Thread-safe pseudonym access
        self._owner_re: Optional[re.Pattern[str]] = None
        self._known_names_re: Optional[re.Pattern[str]] = None

        self._build_owner_regex()
        self._build_known_names_regex()

    def _build_owner_regex(self) -> None:
        """Build compiled regex for owner name variants."""
        if not self.config.owner_names:
            return
        # Sort by length descending so longer matches take priority
        sorted_names = sorted(self.config.owner_names, key=len, reverse=True)
        escaped = [re.escape(name) for name in sorted_names]
        self._owner_re = re.compile(
            r"\b(?:" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )

    def _build_known_names_regex(self) -> None:
        """Build compiled regex for known names dictionary.

        Uses word boundaries for Latin names. For Hebrew names (containing
        Hebrew Unicode chars), uses lookahead/lookbehind on whitespace since
        \\b doesn't work reliably with Hebrew script.
        """
        if not self.config.known_names:
            return

        latin_names: list[str] = []
        hebrew_names: list[str] = []

        for name in sorted(self.config.known_names, key=len, reverse=True):
            name = name.strip()
            if not name or len(name) < 2:
                continue
            # Check if name contains Hebrew characters (Unicode block 0x0590-0x05FF)
            if any("\u0590" <= ch <= "\u05ff" for ch in name):
                # Normalize: strip nikud (diacritics U+0591-U+05C7) for matching
                normalized = _strip_nikud(name)
                hebrew_names.append(re.escape(normalized))
            else:
                latin_names.append(re.escape(name))

        parts: list[str] = []
        if latin_names:
            parts.append(r"\b(?:" + "|".join(latin_names) + r")\b")
        if hebrew_names:
            # Hebrew word boundary: preceded/followed by whitespace, start, or end
            parts.append(r"(?:^|(?<=\s))(?:" + "|".join(hebrew_names) + r")(?=\s|$)")

        if parts:
            self._known_names_re = re.compile("|".join(parts), re.IGNORECASE | re.MULTILINE)

    def _get_nlp(self):
        """Lazy-load spaCy model on first use."""
        if self._nlp is None and self.config.use_spacy_ner:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except (ImportError, OSError) as e:
                import sys

                print(f"  spaCy unavailable ({e}), skipping NER layer", file=sys.stderr)
                self._nlp = False  # Sentinel: tried and failed
        return self._nlp if self._nlp is not False else None

    def _pseudonym(self, name: str) -> str:
        """Get or create a stable pseudonym for a name. Thread-safe."""
        key = name.lower().strip()
        with self._pseudo_lock:
            if key not in self._name_to_pseudo:
                h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
                self._name_to_pseudo[key] = f"[PERSON_{h}]"
            return self._name_to_pseudo[key]

    def sanitize(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SanitizeResult:
        """Sanitize a single chunk's content. Returns result with cleaned text.

        Args:
            content: The raw chunk text to sanitize.
            metadata: Optional chunk metadata (source, sender, etc.) for context.
        """
        if not content:
            return SanitizeResult(sanitized="", original_length=0, pii_detected=False)

        original_length = len(content)
        replacements: list[Replacement] = []
        text = content

        # Track already-replaced spans to avoid double-replacement
        replaced_spans: list[tuple[int, int]] = []

        def _apply_replacements(
            text: str,
            matches: list[tuple[int, int, str, str, str]],
        ) -> str:
            """Apply a list of (start, end, placeholder, category, source) replacements.

            Works backwards to preserve positions.
            """
            # Sort by start position descending
            sorted_matches = sorted(matches, key=lambda m: m[0], reverse=True)
            for start, end, placeholder, category, source in sorted_matches:
                # Skip if overlaps with already-replaced span
                if any(s <= start < e or s < end <= e for s, e in replaced_spans):
                    continue
                original = text[start:end]
                replacements.append(
                    Replacement(
                        category=category,
                        original=original,
                        placeholder=placeholder,
                        start=start,
                        end=end,
                        source=source,
                    )
                )
                text = text[:start] + placeholder + text[end:]
                replaced_spans.append((start, start + len(placeholder)))
            return text

        # ── Layer 1: Regex (owner + known patterns) ──
        # Order matters: match longer/more-specific patterns first to avoid
        # partial matches (e.g., "jane" inside "jane@example.com").

        # Owner emails FIRST (before owner names, to avoid partial match)
        for email in self.config.owner_emails:
            if email.lower() in text.lower():
                email_matches = [
                    (m.start(), m.end(), "[OWNER_EMAIL]", "email", "regex")
                    for m in re.finditer(re.escape(email), text, re.IGNORECASE)
                ]
                text = _apply_replacements(text, email_matches)

        # Owner file paths SECOND (before owner names, same reason)
        for path_prefix in self.config.owner_paths:
            if path_prefix in text:
                path_matches = [
                    (
                        m.start(),
                        m.end(),
                        m.group(0).replace(path_prefix, "/Users/[OWNER]"),
                        "file_path",
                        "regex",
                    )
                    for m in re.finditer(re.escape(path_prefix) + r"[^\s\"']*", text)
                ]
                text = _apply_replacements(text, path_matches)

        # GitHub username THIRD
        for owner_name in self.config.owner_names:
            for pattern in [_GITHUB_URL_RE, _GITHUB_MENTION_RE]:
                for m in pattern.finditer(text):
                    if m.group(1).lower() == owner_name.lower():
                        matches = [(m.start(), m.end(), "[OWNER_GITHUB]", "github", "regex")]
                        text = _apply_replacements(text, matches)

        # Owner names LAST (after emails/paths/github are already replaced)
        if self._owner_re:
            matches = [(m.start(), m.end(), "[OWNER]", "owner", "regex") for m in self._owner_re.finditer(text)]
            text = _apply_replacements(text, matches)

        # General emails
        if self.config.strip_emails:
            counter = 0
            general_emails = []
            for m in _EMAIL_RE.finditer(text):
                counter += 1
                general_emails.append((m.start(), m.end(), f"[EMAIL_{counter}]", "email", "regex"))
            text = _apply_replacements(text, general_emails)

        # IPs
        if self.config.strip_ips:
            ip_matches = [
                (m.start(), m.end(), "[IP_ADDR]", "ip", "regex")
                for m in _IPV4_RE.finditer(text)
                # Skip common non-PII IPs
                if not m.group(0).startswith(("127.", "0.", "255.")) and m.group(0) != "0.0.0.0"
            ]
            text = _apply_replacements(text, ip_matches)

        # JWTs
        if self.config.strip_jwts:
            jwt_matches = [(m.start(), m.end(), "[JWT_TOKEN]", "jwt", "regex") for m in _JWT_RE.finditer(text)]
            text = _apply_replacements(text, jwt_matches)

        # 1Password refs
        if self.config.strip_op_refs:
            op_matches = [(m.start(), m.end(), "[OP_REF]", "op_ref", "regex") for m in _OP_REF_RE.finditer(text)]
            text = _apply_replacements(text, op_matches)

        # Phone numbers
        if self.config.strip_phone_numbers:
            phone_matches = [(m.start(), m.end(), "[PHONE]", "phone", "regex") for m in _PHONE_RE.finditer(text)]
            text = _apply_replacements(text, phone_matches)

        # ── Layer 2: Known names dictionary ──

        if self._known_names_re:
            # Match against nikud-stripped text but replace in original
            text_no_nikud = _strip_nikud(text)
            if text_no_nikud != text:
                # Hebrew text with nikud — match on stripped version, map positions back
                omap = _nikud_offset_map(text)
                name_matches = [
                    (
                        omap[m.start()],
                        omap[m.end()],
                        self._pseudonym(m.group(0)),
                        "person_name",
                        "name_dict",
                    )
                    for m in self._known_names_re.finditer(text_no_nikud)
                ]
            else:
                name_matches = [
                    (m.start(), m.end(), self._pseudonym(m.group(0)), "person_name", "name_dict")
                    for m in self._known_names_re.finditer(text)
                ]
            text = _apply_replacements(text, name_matches)

        # ── Layer 3: spaCy NER (English names only) ──

        nlp = self._get_nlp()
        if nlp is not None:
            # Find code blocks to exclude
            code_spans: set[tuple[int, int]] = set()
            for m in _CODE_BLOCK_RE.finditer(text):
                code_spans.add((m.start(), m.end()))

            doc = nlp(text)
            ner_matches = []
            for ent in doc.ents:
                if ent.label_ != "PERSON":
                    continue
                # Skip if inside a code block
                if any(cs <= ent.start_char < ce for cs, ce in code_spans):
                    continue
                # Skip very short entities (likely false positives)
                if len(ent.text.strip()) < 3:
                    continue
                # Skip if already replaced
                if any(s <= ent.start_char < e for s, e in replaced_spans):
                    continue
                ner_matches.append(
                    (
                        ent.start_char,
                        ent.end_char,
                        self._pseudonym(ent.text),
                        "person_name",
                        "spacy",
                    )
                )
            text = _apply_replacements(text, ner_matches)

        pii_detected = len(replacements) > 0
        return SanitizeResult(
            sanitized=text,
            original_length=original_length,
            replacements=replacements,
            pii_detected=pii_detected,
        )

    def sanitize_batch(
        self,
        chunks: list[dict[str, Any]],
        content_key: str = "content",
        metadata_key: str = "metadata",
        parallel: int = 1,
    ) -> list[SanitizeResult]:
        """Sanitize a batch of chunks.

        Args:
            chunks: List of chunk dicts with at least a content field.
            content_key: Key for content in each chunk dict.
            metadata_key: Key for metadata in each chunk dict.
            parallel: Number of parallel workers (1=sequential).

        Returns:
            List of SanitizeResult in same order as input chunks.
        """
        if not chunks:
            return []

        # Pre-load spaCy model before parallel execution
        if self.config.use_spacy_ner:
            self._get_nlp()

        if parallel <= 1:
            return [
                self.sanitize(
                    chunk.get(content_key, ""),
                    chunk.get(metadata_key),
                )
                for chunk in chunks
            ]

        # Parallel execution — spaCy model is thread-safe for inference
        results: list[Optional[SanitizeResult]] = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {
                pool.submit(
                    self.sanitize,
                    chunk.get(content_key, ""),
                    chunk.get(metadata_key),
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]

    def build_name_dictionary(self, store: "VectorStore") -> set[str]:
        """Extract unique sender names from WhatsApp chunks in the DB.

        Queries the chunks table for distinct sender values where source='whatsapp'.
        Returns the set of names found (can be passed to SanitizeConfig.known_names).
        """
        cursor = store.conn.cursor()
        rows = list(
            cursor.execute(
                "SELECT DISTINCT sender FROM chunks WHERE source = 'whatsapp' AND sender IS NOT NULL AND sender != ''"
            )
        )
        names = {row[0].strip() for row in rows if row[0] and row[0].strip()}
        return names

    def save_mapping(self, path: Path) -> None:
        """Save the name→pseudonym mapping to a JSON file for reversibility.

        This file should NEVER be uploaded to external services.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        mapping = {
            "name_to_pseudonym": self._name_to_pseudo,
            "pseudonym_to_name": {v: k for k, v in self._name_to_pseudo.items()},
        }
        path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))

    def load_mapping(self, path: Path) -> None:
        """Load a previously saved mapping to maintain pseudonym consistency."""
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            existing = data.get("name_to_pseudonym", {})
            self._name_to_pseudo.update(existing)
        except (json.JSONDecodeError, KeyError, OSError) as e:
            import sys

            print(f"  Warning: could not load PII mapping from {path}: {e}", file=sys.stderr)

    @classmethod
    def from_env(cls) -> Sanitizer:
        """Create a Sanitizer from environment variables with sensible defaults."""
        owner_names = tuple(
            n.strip()
            for n in os.environ.get(
                "BRAINLAYER_SANITIZE_OWNER_NAMES",
                "",
            ).split(",")
            if n.strip()
        )
        owner_emails = tuple(
            e.strip()
            for e in os.environ.get(
                "BRAINLAYER_SANITIZE_OWNER_EMAILS",
                "",
            ).split(",")
            if e.strip()
        )
        owner_paths = tuple(
            p.strip()
            for p in os.environ.get(
                "BRAINLAYER_SANITIZE_OWNER_PATHS",
                "",
            ).split(",")
            if p.strip()
        )
        extra_names = frozenset(
            n.strip() for n in os.environ.get("BRAINLAYER_SANITIZE_EXTRA_NAMES", "").split(",") if n.strip()
        )
        use_spacy = os.environ.get("BRAINLAYER_SANITIZE_USE_SPACY", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        config = SanitizeConfig(
            owner_names=owner_names,
            owner_emails=owner_emails,
            owner_paths=owner_paths,
            known_names=extra_names,
            use_spacy_ner=use_spacy,
        )
        return cls(config)
