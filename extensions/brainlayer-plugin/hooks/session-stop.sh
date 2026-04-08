#!/usr/bin/env bash
set -euo pipefail
PLUGIN_ROOT="${CLAUDE_PLUGIN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python3 "${PLUGIN_ROOT}/scripts/brainbar-hook.py" session-stop
