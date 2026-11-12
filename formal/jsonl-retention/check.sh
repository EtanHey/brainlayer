#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
out="$here/../../.build.noindex/tlc/jsonl-retention"
jar=${TLC_JAR:-$HOME/.local/share/tlaplus/tla2tools-1.7.4.jar}
mkdir -p "$out"
cp "$here/Retention.tla" "$out/Retention.tla"
cd "$out"
java -Xmx512m -cp "$jar" tlc2.TLC -workers 1 -config "$here/current.cfg" -metadir current-states Retention > current.log 2>&1
for bug in verified age; do
    if java -Xmx512m -cp "$jar" tlc2.TLC -workers 1 -config "$here/$bug.cfg" -metadir "$bug-states" Retention > "$bug.log" 2>&1; then
        echo "FAIL: $bug mutation survived"; exit 1
    fi
    case "$bug" in
        verified) grep -F 'Invariant NeverDeleteWithoutVerifiedCopy is violated' "$bug.log" ;;
        age) grep -F 'Invariant NeverDeleteYoungerThan14Days is violated' "$bug.log" ;;
    esac
done
echo 'PASS: current invariants; both mutations rejected'
