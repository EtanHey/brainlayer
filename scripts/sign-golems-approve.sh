#!/bin/bash
# Hosted ephemeral runner ONLY. Mirrors BrainBar's temporary-keychain/notary flow.
set -euo pipefail
set +x
umask 077
python3 scripts/golems_approve_release.py secrets
python3 scripts/golems_approve_release.py profile .github/signing/golems-approve/Golems_Approve_Developer_ID.provisionprofile
python3 scripts/golems_approve_release.py source payload
keychain_path="$RUNNER_TEMP/golems-approve-signing.keychain-db"
cert_path="$RUNNER_TEMP/golems-approve-developer-id.p12"
cleanup() {
  if [[ -e "$keychain_path" ]]; then security delete-keychain "$keychain_path" >/dev/null 2>&1; fi
  if [[ -e "$cert_path" ]]; then /bin/rm -f "$cert_path"; fi
}
trap cleanup EXIT
keychain_password="$(uuidgen)"
# Mask the generated password too. No credentials go into GITHUB_ENV or outputs.
printf '::add-mask::%s\n' "$keychain_password"
printf '%s' "$BRAINBAR_DEVELOPER_ID_CERTIFICATE_BASE64" | base64 -D -o "$cert_path"
# Suppress security/notary output, including errors: those can echo credential-bearing argv.
security create-keychain -p "$keychain_password" "$keychain_path" >/dev/null 2>&1
security set-keychain-settings -lut 21600 "$keychain_path" >/dev/null 2>&1
security unlock-keychain -p "$keychain_password" "$keychain_path" >/dev/null 2>&1
security import "$cert_path" -P "$BRAINBAR_DEVELOPER_ID_CERTIFICATE_PASSWORD" -A -t cert -f pkcs12 -k "$keychain_path" >/dev/null 2>&1
security set-key-partition-list -S apple-tool:,apple: -s -k "$keychain_password" "$keychain_path" >/dev/null 2>&1
security list-keychains -d user -s "$keychain_path" >/dev/null 2>&1
app="$PWD/dist/GolemsApprove.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp payload/golems-approve "$app/Contents/MacOS/golems-approve"
chmod 755 "$app/Contents/MacOS/golems-approve"
cp payload/Info.plist "$app/Contents/Info.plist"
cp payload/Info.plist payload/GolemsApprove.entitlements "$app/Contents/Resources/"
cp .github/signing/golems-approve/Golems_Approve_Developer_ID.provisionprofile "$app/Contents/embedded.provisionprofile"
codesign --force --options runtime --timestamp --identifier com.golems.approve --keychain "$keychain_path" --entitlements payload/GolemsApprove.entitlements --sign "$BRAINBAR_DEVELOPER_ID_APPLICATION" "$app" >/dev/null 2>&1
ditto -c -k --keepParent "$app" "$RUNNER_TEMP/GolemsApprove-notary.zip"
xcrun notarytool submit "$RUNNER_TEMP/GolemsApprove-notary.zip" --apple-id "$BRAINBAR_NOTARY_APPLE_ID" --password "$BRAINBAR_NOTARY_PASSWORD" --team-id "$BRAINBAR_NOTARY_TEAM_ID" --wait >/dev/null 2>&1
# Same ephemeral-runner ticket-delivery transport workaround as brainbar-release.yml.
echo "block return out quick proto udp from any to any port 443" | sudo pfctl -f -
sudo pfctl -E
xcrun stapler staple "$app"
xcrun stapler validate "$app"
