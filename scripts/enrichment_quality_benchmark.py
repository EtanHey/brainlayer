#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""CLI wrapper for Track A enrichment quality benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.eval.enrichment_quality_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
