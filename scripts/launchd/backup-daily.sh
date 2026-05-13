#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
export PYTHONUNBUFFERED=1
BRAINLAYER_DIR="${BRAINLAYER_DIR:-__BRAINLAYER_DIR__}"
if [ "$BRAINLAYER_DIR" = "__BRAINLAYER_DIR__" ]; then
    BRAINLAYER_DIR="$HOME/Gits/brainlayer"
fi
export PYTHONPATH="$BRAINLAYER_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "${BRAINLAYER_PYTHON:-python3}" -m brainlayer.backup_daily
