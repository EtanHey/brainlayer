#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/newsyslog.d/brainlayer.conf"
DST="/etc/newsyslog.d/brainlayer.conf"
OWNER="${BRAINLAYER_LOG_OWNER:-${SUDO_USER:-$(id -un)}}"
GROUP="${BRAINLAYER_LOG_GROUP:-staff}"
RENDERED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/brainlayer-newsyslog.XXXXXX")"
trap 'rm -f "$RENDERED_CONFIG"' EXIT

escape_sed_replacement() {
    printf '%s' "$1" | sed 's/[\/#&\\]/\\&/g'
}

if ! id -u "$OWNER" >/dev/null 2>&1; then
    echo "ERROR: log owner does not exist: $OWNER" >&2
    exit 1
fi

if ! OWNER_HOME="$(dscl . -read "/Users/$OWNER" NFSHomeDirectory 2>/dev/null | sed 's/^NFSHomeDirectory:[[:space:]]*//')"; then
    echo "ERROR: could not resolve home directory for $OWNER" >&2
    exit 1
fi
if [ -z "$OWNER_HOME" ]; then
    echo "ERROR: could not resolve home directory for $OWNER" >&2
    exit 1
fi
LOG_DIR="${BRAINLAYER_LOG_DIR:-$OWNER_HOME/Library/Logs/brainlayer}"
if [[ "$LOG_DIR" =~ [[:space:]] ]]; then
    echo "ERROR: newsyslog log paths cannot contain whitespace: $LOG_DIR" >&2
    exit 1
fi

if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
fi

LOG_DIR_ESCAPED="$(escape_sed_replacement "$LOG_DIR")"
OWNER_GROUP_ESCAPED="$(escape_sed_replacement "$OWNER:$GROUP")"

sed \
    -e "s#/Users/etanheyman/Library/Logs/brainlayer#$LOG_DIR_ESCAPED#g" \
    -e "s#etanheyman:staff#$OWNER_GROUP_ESCAPED#g" \
    "$SRC" >"$RENDERED_CONFIG"

sudo mkdir -p "$LOG_DIR"
# Ensure already-created logs are writable by user LaunchAgents before the first rotation.
sudo chown "$OWNER:$GROUP" "$LOG_DIR"
for log in "$LOG_DIR"/*.log; do
    [ -e "$log" ] || continue
    if [ -L "$log" ] || [ ! -f "$log" ]; then
        echo "Skipping non-regular log path: $log" >&2
        continue
    fi
    sudo chown "$OWNER:$GROUP" "$log"
    sudo chmod 0644 "$log"
done

sudo newsyslog -nv -f "$RENDERED_CONFIG"
sudo mkdir -p /etc/newsyslog.d
sudo install -o root -g wheel -m 0644 "$RENDERED_CONFIG" "$DST"
sudo newsyslog -nv -f "$DST"
echo "Installed $DST"
