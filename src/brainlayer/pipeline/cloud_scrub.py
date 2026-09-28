"""The secret-scrub chokepoint between BrainLayer text and any remote LLM.

Two directions, both fail closed:

- ``scrub_for_cloud`` runs on every text sent to a cloud model. If scrubbing
  raises, it raises ``CloudScrubError`` and the caller must not send.
- ``scrub_llm_output`` runs on every LLM output value before it is persisted.
  A cloud model copies tokens from its prompt into summaries and key facts,
  so output is scrubbed even when the input already was. If scrubbing raises,
  nothing is persisted.

The PII ``Sanitizer`` is a separate layer (names, emails, paths); it does not
look for credentials, so it is not a substitute for this module.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TypeVar

from .secret_scrub import scrub_secrets

T = TypeVar("T")


class CloudScrubError(RuntimeError):
    """Secret scrubbing failed; the text must not be sent or persisted."""


def _scrub_text(text: str) -> str:
    try:
        scrubbed = scrub_secrets(text).text
    except Exception as exc:
        raise CloudScrubError(f"secret scrub failed ({type(exc).__name__}); refusing to pass text on") from None
    if not isinstance(scrubbed, str):
        raise CloudScrubError("secret scrub returned non-text; refusing to pass text on")
    return scrubbed


def scrub_for_cloud(text: str) -> str:
    """Return ``text`` with secrets redacted, ready to send to a remote LLM."""
    if not isinstance(text, str):
        raise CloudScrubError(f"remote LLM payload must be text, got {type(text).__name__}")
    return _scrub_text(text)


def scrub_llm_output(value: T) -> T:
    """Redact secrets from every string inside an LLM output value.

    Walks dicts, lists and tuples, and scrubs string dict keys as well as values:
    a model writes free-form mappings (relation properties, tool stats), so a key
    can be model-authored content. Schema field names never match a secret shape,
    so scrubbing them is a no-op. Two keys that redact to the same placeholder
    collapse into one entry and the later value wins — no secret survives, and
    that loss is deliberate. Non-text leaves pass through.
    """
    if isinstance(value, str):
        return _scrub_text(value)  # type: ignore[return-value]
    if isinstance(value, dict):
        return {  # type: ignore[return-value]
            (_scrub_text(key) if isinstance(key, str) else key): scrub_llm_output(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [scrub_llm_output(item) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(scrub_llm_output(item) for item in value)  # type: ignore[return-value]
    return value


MAX_JSON_STRING_DEPTH = 3


def normalize_json_strings(value: T, *, depth_used: int = 0) -> T:
    """Rewrite every string that is itself JSON into an escape-free form.

    JSON may spell any character as a ``\\uXXXX`` escape, and a string can hold a
    whole JSON document, so a token can sit behind any number of encodings that a
    text scrub never sees. Every string (value or key) that decodes to a JSON
    container or JSON string is decoded, normalized recursively, and re-encoded
    with ``ensure_ascii=False``, so the token is literal text for
    ``scrub_llm_output``. The string keeps its type, so the stored shape does not
    change.

    ``depth_used`` counts decodes the caller already did. A string still holding
    JSON after ``MAX_JSON_STRING_DEPTH`` decodes on one path raises
    ``CloudScrubError``: nothing that deep is written.
    """
    if isinstance(value, str):
        return _normalize_json_text(value, depth_used)  # type: ignore[return-value]
    if isinstance(value, dict):
        return {  # type: ignore[return-value]
            normalize_json_strings(key, depth_used=depth_used): normalize_json_strings(item, depth_used=depth_used)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalize_json_strings(item, depth_used=depth_used) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(normalize_json_strings(item, depth_used=depth_used) for item in value)  # type: ignore[return-value]
    return value


def _normalize_json_text(text: str, depth_used: int) -> str:
    try:
        decoded = json.loads(text)
    except ValueError:
        return text
    except RecursionError:
        # Too deep to decode means too deep to inspect; never pass it through.
        raise CloudScrubError("JSON nested in a string too deep to decode; refusing to persist it") from None
    if not isinstance(decoded, (dict, list, str)):
        return text  # numbers, booleans and null cannot hide text
    if depth_used >= MAX_JSON_STRING_DEPTH:
        raise CloudScrubError(
            f"JSON nested in strings deeper than {MAX_JSON_STRING_DEPTH} levels; refusing to persist it"
        )
    return json.dumps(normalize_json_strings(decoded, depth_used=depth_used + 1), ensure_ascii=False)


def scrub_gemini_batch_jsonl(path: Path | str) -> None:
    """Rewrite a Gemini batch request file in place with every prompt part scrubbed.

    Export files can outlive the code that wrote them, so the upload step
    re-scrubs whatever is on disk rather than trusting the exporter.
    """
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            for content in row.get("request", {}).get("contents", []):
                for part in content.get("parts", []):
                    if "text" in part:
                        part["text"] = scrub_for_cloud(part["text"])
            out.append(json.dumps(row, ensure_ascii=False))
        fd, tmp_name = tempfile.mkstemp(prefix=f".{source.name}.", suffix=".scrub", dir=source.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("\n".join(out) + ("\n" if out else ""))
            os.replace(tmp_name, source)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except CloudScrubError:
        raise
    except Exception as exc:
        raise CloudScrubError(
            f"could not scrub batch file {source.name} ({type(exc).__name__}); refusing to upload"
        ) from None
