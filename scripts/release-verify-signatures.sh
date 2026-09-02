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

native_root="$keg_path/libexec/venv"
codesign_bin="${BRAINLAYER_CODESIGN_BIN:-codesign}"
if [[ ! -d "$native_root" ]]; then
    printf 'ERROR: native extension root not found: %s\n' "$native_root" >&2
    exit 2
fi
if ! command -v "$codesign_bin" >/dev/null 2>&1; then
    printf 'ERROR: codesign executable not found: %s\n' "$codesign_bin" >&2
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
if [[ $((valid + invalid)) -eq 0 ]]; then
    printf 'ERROR: no native extensions found under %s\n' "$native_root" >&2
    exit 1
fi
[[ "$invalid" -eq 0 ]]
