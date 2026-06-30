#!/usr/bin/env python3
"""Compatibility wrapper for the packaged P0 longitudinal counter module."""

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if (_REPO_SRC / "brainlayer").is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from brainlayer.p0_longitudinal_count import main

if __name__ == "__main__":
    raise SystemExit(main())
