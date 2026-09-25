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

    Walks dicts, lists and tuples; dict keys are left alone because they are
    schema field names, not model-authored content. Non-text leaves pass through.
    """
    if isinstance(value, str):
        return _scrub_text(value)  # type: ignore[return-value]
    if isinstance(value, dict):
        return {key: scrub_llm_output(item) for key, item in value.items()}  # type: ignore[return-value]
    if isinstance(value, list):
        return [scrub_llm_output(item) for item in value]  # type: ignore[return-value]
    if isinstance(value, tuple):
        return tuple(scrub_llm_output(item) for item in value)  # type: ignore[return-value]
    return value


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
