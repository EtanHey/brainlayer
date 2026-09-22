#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    printf 'Usage: %s [brainlayer-keg]\n' "$0" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    keg_path="$1"
elif command -v brew >/dev/null 2>&1; then
    keg_path="$(brew --prefix brainlayer)"
else
    printf 'ERROR: pass a BrainLayer keg path or install Homebrew\n' >&2
    exit 2
fi

if [[ ! -d "$keg_path" ]]; then
    printf 'ERROR: BrainLayer keg not found: %s\n' "$keg_path" >&2
    exit 2
fi
# The public opt/ prefix may be a symlink to the Cellar; resolve that one trusted entrypoint.
keg_path="$(cd "$keg_path" && pwd -P)"
native_root="$keg_path/libexec/venv"
codesign_bin="${BRAINLAYER_CODESIGN_BIN:-codesign}"
if [[ ! -d "$native_root" ]]; then
    printf 'ERROR: native extension root not found: %s\n' "$native_root" >&2
    exit 2
fi
if [[ "$(cd "$native_root" && pwd -P)" != "$native_root" ]]; then
    printf 'ERROR: native extension root escapes keg: %s\n' "$native_root" >&2
    exit 2
fi
if ! command -v "$codesign_bin" >/dev/null 2>&1; then
    printf 'ERROR: codesign executable not found: %s\n' "$codesign_bin" >&2
    exit 2
fi

# find -type f does not follow links. Refuse linked libraries or package directories instead of
# silently skipping a native file outside the keg. Venv python launchers under bin/ are expected.
native_symlink="$(find "$native_root" -type l \( ! -path "$native_root/bin/*" -o -name '*.so' -o -name '*.dylib' \) -print -quit)"
if [[ -n "$native_symlink" ]]; then
    printf 'ERROR: symlink in native library tree: %s\n' "$native_symlink" >&2
    exit 2
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
# -type f skips symlinked extensions (none in current wheels); descends into dot-dirs like PIL/.dylibs.
find "$native_root" -type f \( -name '*.so' -o -name '*.dylib' \) -print0 >"$tmp_dir/native-files"

valid=0
invalid=0
while IFS= read -r -d '' native_file; do
    error_file="$tmp_dir/codesign-error"
    if "$codesign_bin" --verify --verbose=4 "$native_file" 2>"$error_file"; then
        valid=$((valid + 1))
    else
        invalid=$((invalid + 1))
        failure_class="$(awk 'NF { print; exit }' "$error_file")"
        failure_class="${failure_class#"$native_file: "}"
        printf 'INVALID %s: %s\n' "${native_file#"$native_root/"}" "${failure_class:-unknown codesign failure}"
    fi
done <"$tmp_dir/native-files"

printf 'valid: %d\ninvalid: %d\n' "$valid" "$invalid"
printf 'valid-signature: %d\n' "$valid"
if [[ $((valid + invalid)) -eq 0 ]]; then
    printf 'ERROR: no native extensions found under %s\n' "$native_root" >&2
    exit 1
fi

keg_python="$native_root/bin/python"
if [[ ! -x "$keg_python" ]]; then
    printf 'ERROR: keg python not executable: %s\n' "$keg_python" >&2
    exit 2
fi

# A valid signature does not prove dyld can load the Mach-O (e.g. a misaligned LINKEDIT string pool).
# Each extension runs in its own bounded process, so a native crash cannot abort this inventory.
"$keg_python" -I - "$native_root" "$tmp_dir/native-files" <<'PY'
import fnmatch
import os
from pathlib import Path
import subprocess
import sys

native_root = Path(sys.argv[1])
inventory = Path(sys.argv[2]).read_bytes().split(b"\0")
files = [Path(os.fsdecode(path)) for path in inventory if path]
loadable = 0
allowed_optional = 0
failed = 0
child = """import ctypes, os, sys
try:
    ctypes.CDLL(sys.argv[1])
except OSError as exc:
    print(f"dlopen: {exc}", file=sys.stderr)
    os._exit(3)
os._exit(0)
"""

for native_file in files:
    relative = native_file.relative_to(native_root)
    parts = relative.parts
    # numba's optional OpenMP threading layer needs libomp at an unavailable @rpath on this keg.
    # Numba itself imports and its default-threading trivial njit works without this extension.
    optional_omppool = (
        len(parts) == 7
        and parts[0] == "lib"
        and parts[1].startswith("python")
        and parts[2:6] == ("site-packages", "numba", "np", "ufunc")
        and fnmatch.fnmatchcase(parts[6], "omppool*.so")
    )
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", child, str(native_file)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"LOAD_FAILED {relative}: timeout after 10s")
        failed += 1
        continue
    except OSError as exc:
        print(f"LOAD_FAILED {relative}: subprocess error: {exc}")
        failed += 1
        continue
    if result.returncode == 0:
        loadable += 1
        continue
    reason = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "no stderr"
    if result.returncode == 3 and optional_omppool:
        print(f"ALLOWED_OPTIONAL {relative}: {reason}")
        allowed_optional += 1
    else:
        kind = "signal" if result.returncode < 0 else "exit"
        print(f"LOAD_FAILED {relative}: {kind} {result.returncode}: {reason}")
        failed += 1

print(f"loadable: {loadable}")
print(f"allowed-optional: {allowed_optional}")
print(f"load-failed: {failed}")
sys.exit(1 if failed else 0)
PY
[[ "$invalid" -eq 0 ]]
