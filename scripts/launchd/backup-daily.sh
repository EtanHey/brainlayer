#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin"
export PYTHONUNBUFFERED=1

exec "${BRAINLAYER_PYTHON:-python3}" -m brainlayer.backup_daily
