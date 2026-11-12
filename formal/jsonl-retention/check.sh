#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
out="$here/../../.build.noindex/tlc/jsonl-retention"
jar=${TLC_JAR:-$HOME/.local/share/tlaplus/tla2tools-1.7.4.jar}
mkdir -p "$out/tmp"
cp "$here/Retention.tla" "$out/Retention.tla"
cd "$out"
java -Djava.io.tmpdir="$out/tmp" -Xmx512m -cp "$jar" tlc2.TLC -workers 1 -config "$here/current.cfg" -metadir current-states Retention > current.log 2>&1
for bug in live md5 newest age restore external; do
    if java -Djava.io.tmpdir="$out/tmp" -Xmx512m -cp "$jar" tlc2.TLC -workers 1 -config "$here/$bug.cfg" -metadir "$bug-states" Retention > "$bug.log" 2>&1; then
        echo "FAIL: $bug mutation survived"; exit 1
    fi
    case "$bug" in
        live|md5) expected=NeverDeleteWithoutVerifiedCopy ;;
        newest) expected=NeverDeleteNewest ;;
        age) expected=NeverDeleteYoungerThan14Days ;;
        restore) expected=RestoreOnlySurvivors ;;
        external) expected=AlwaysAtLeastOneCopy ;;
    esac
    grep -F "Invariant $expected is violated" "$bug.log"
done
echo 'PASS: current invariants; five rule mutations and external-loss limit rejected'
