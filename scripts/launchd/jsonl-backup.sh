#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
export PYTHONUNBUFFERED=1
: "${BRAINLAYER_BACKUP_TIMEOUT_SECONDS:=1800}"
export BRAINLAYER_BACKUP_TIMEOUT_SECONDS
: "${BRAINLAYER_PYTHON:?installer must render the prefix-aware keg interpreter}"
unset PYTHONPATH

exec "$BRAINLAYER_PYTHON" -m brainlayer.jsonl_backup
