#!/bin/sh
set -eu
here=$(CDPATH= cd "$(dirname "$0")" && pwd)
out="$here/../../.build.noindex/tlc/policy-reconcile"
jar=${TLC_JAR:-$HOME/.local/share/tlaplus/tla2tools-1.7.4.jar}
mkdir -p "$out"
cp "$here/PolicyReconcile.tla" "$out/PolicyReconcile.tla"
cd "$out"
java -Xmx512m -cp "$jar" tlc2.TLC -workers 1 -config "$here/current.cfg" -metadir current-states PolicyReconcile > current.log 2>&1
cat current.log
if java -Xmx512m -cp "$jar" tlc2.TLC -workers 1 -config "$here/bug-rename-first.cfg" -metadir bug-states PolicyReconcile > bug.log 2>&1; then
    echo 'FAIL: rename-before-journal bug did not violate an invariant' >&2
    exit 1
fi
grep -F 'Invariant JournalOwnsManifest is violated' bug.log
echo 'PASS: current transition and crash recovery; rename-first counterexample reproduced'
