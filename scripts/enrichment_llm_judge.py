#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""CLI wrapper for local-vs-Flex enrichment LLM judge."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.eval.enrichment_llm_judge import main


if __name__ == "__main__":
    raise SystemExit(main())
