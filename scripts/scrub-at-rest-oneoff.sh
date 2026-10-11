#!/bin/sh
# Schedule this wrapper with an explicit merged-code interpreter/import path.
# The CLI already returns nonzero on refusal; save that status before running date.
set -u
if [ "$#" -lt 2 ]; then
    echo 'usage: scrub-at-rest-oneoff.sh LOG_PATH ABSOLUTE_PYTHON [scrub-at-rest options]' >&2
    exit 64
fi
scrub_log=$1
scrub_python=$2
shift 2
case "$scrub_python" in
    /*) ;;
    *) echo 'scrub-at-rest requires an absolute interpreter path' >&2; exit 64 ;;
esac
"$scrub_python" -m brainlayer scrub-at-rest "$@" >> "$scrub_log" 2>&1
scrub_rc=$?
echo "$(date -u +%FT%TZ) exit=$scrub_rc" >> "$scrub_log"
exit "$scrub_rc"
