#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
export PYTHONUNBUFFERED=1
: "${BRAINLAYER_BACKUP_TIMEOUT_SECONDS:=300}"
export BRAINLAYER_BACKUP_TIMEOUT_SECONDS
BRAINLAYER_DIR="${BRAINLAYER_DIR:-__BRAINLAYER_DIR_VALUE__}"
case "$BRAINLAYER_DIR" in
    __BRAINLAYER_DIR_*) BRAINLAYER_DIR="$HOME/Gits/brainlayer" ;;
esac
export PYTHONPATH="$BRAINLAYER_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

exec "${BRAINLAYER_PYTHON:-python3}" -m brainlayer.backup_daily
