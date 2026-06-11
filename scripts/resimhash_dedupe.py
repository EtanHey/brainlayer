#!/usr/bin/env python3
"""Recompute stored dedupe simhashes after the content-only SimHash migration."""

import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from brainlayer.dedupe import resimhash_main


if __name__ == "__main__":
    if hasattr(os, "nice"):
        try:
            os.nice(5)
        except OSError:
            pass
    raise SystemExit(resimhash_main())
