#!/usr/bin/env python3
"""Backfill KG identities from chunks.raw_entities_json."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.kg_promotion import main  # noqa: I001


if __name__ == "__main__":
    main()
