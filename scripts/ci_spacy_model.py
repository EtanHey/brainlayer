#!/usr/bin/env python3
"""Probe real keg spaCy; optional candidate wheel is extracted, never installed.

The resource pin is shared with the exact tap diff in the PR body. A candidate
PASS proves real model load/NER under keg Python, not published packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

MODEL_URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
MODEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
BUG_SHA = "38521b5296eddaaa7d2cdc722119f2622f3456b2"
PROBE = """
import spacy
from brainlayer.pipeline.sanitize import Sanitizer, SanitizeConfig
nlp = spacy.load('en_core_web_sm')
if 'ner' not in nlp.pipe_names:
    raise SystemExit('NER component missing')
result = Sanitizer(SanitizeConfig()).sanitize('John Smith lives in London.')
if 'John Smith' in result.sanitized:
    raise SystemExit('Person was not redacted')
if not any(r.source == 'spacy' for r in result.replacements):
    raise SystemExit('No NER replacement recorded')
"""


def probe(python: Path, overlay: Path | None = None) -> bool:
    if not python.is_file():
        return False
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    if overlay is not None:
        env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(Path(__file__).resolve().parents[1] / "src")))
    try:
        result = subprocess.run(
            [str(python), "-s", "-c", PROBE],
            env=env,
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Probe text is synthetic; loader diagnostics stay out of the report.
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--fix-sha", required=True)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    published = probe(args.python)
    payload = dict(
        bug_sha=BUG_SHA,
        fix_sha=args.fix_sha,
        python=str(args.python.absolute()),
        python_prefix=str(args.python.parent.parent.resolve()),
        published_loaded=published,
        loaded=published,
        scope="published",
    )
    if args.candidate_wheel is not None:
        payload.update(scope="candidate", loaded=False)
        try:
            if hashlib.sha256(args.candidate_wheel.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError("model wheel hash mismatch")
            with tempfile.TemporaryDirectory(prefix="brainlayer-spacy-") as scratch:
                overlay = Path(scratch)
                with zipfile.ZipFile(args.candidate_wheel) as wheel:
                    wheel.extractall(overlay)
                payload["loaded"] = probe(args.python, overlay)
        except (OSError, ValueError, zipfile.BadZipFile):
            pass  # Failure is published, never skipped.
    args.out.write_text(json.dumps(payload))
    print(json.dumps(payload))
    return 0 if payload["loaded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
