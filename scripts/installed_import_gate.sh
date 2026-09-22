#!/usr/bin/env bash
set -euo pipefail
wheel="${1:?wheel path required}" hooks="${2:?hooks directory required}" venv="${3:?fresh venv path required}"
python3 -m venv "$venv"
"$venv/bin/python" -m pip install "${wheel}[cloud]"
"$venv/bin/python" -m pip check
(cd "$(dirname "$venv")" && "$venv/bin/python" -m brainlayer.import_sweep \
  --wheel "$wheel" --hooks-dir "$hooks" --timeout 60 --jobs 4)
