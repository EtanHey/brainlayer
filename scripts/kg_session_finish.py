#!/usr/bin/env python3
"""CLI wrapper for brainlayer.kg_session_finish."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brainlayer.kg_session_finish import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
