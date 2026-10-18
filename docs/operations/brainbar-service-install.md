# BrainBar per-user service rendering
For an existing signed released `BrainBar.app`, use the installed package's Python and CLI explicitly. The released app bundles `Contents/Resources/install-services.py`.
It verifies the signature and requires current-account-owned app/config paths; canonical `/Applications/BrainBar.app` is accepted only when owned by this account. An explicit own venv Python symlink may resolve the pre-existing shared interpreter.
Render first with explicit paths:

```sh
"$venv/bin/python" -I "$app/Contents/Resources/install-services.py" \
  --app "$app" --python "$venv/bin/python" --cli "$venv/bin/brainlayer" \
  --socket "$HOME/Library/Application Support/BrainBar/brainbar.sock" --render-only
```

`--install` installs the reviewed pair into own `~/Library/LaunchAgents` and creates owned socket/DB/log parent directories. It does not activate, stop or restart jobs, remove sockets, install packages, build, sign or invoke Homebrew.
Activation stays in the separate current-UID LaunchAgent procedure after release/ownership disposition. Labels and global default socket remain compatible; a foreign default socket/lock is refused. A dedicated account must explicitly select its private socket.
Only derived runtime paths enter the environment; inherited source paths, provider/key references and secrets are not copied. Optional `--db` must remain inside the account home. Templates stay authoritative.
Checks are metadata observations, not a same-UID mutation lease. Logs/plists must remain owned and safe at activation. Cask activation uses the separate explicit command below; rendering alone does not establish native proof.
The DB must identify a different resource from the socket, derived `.lock`, and both installed LaunchAgent plist destinations: existing same-file identities are refused.
Case-folded, canonically Unicode-equivalent full paths are conservatively refused even when missing or on case-sensitive volumes; choose names that differ beyond case/normalization.
This covers prospective ambiguous names without creating probe files or relaxing ownership/type checks; it is a point-in-time check, not an atomic namespace lease.

## Homebrew cask upgrades: defer activation

The cask upgrade stops BrainBar; the supervisor re-activates it automatically after
Homebrew returns. Manual: `brainlayer brainbar activate`.
This requires the released formula's always-on supervisor to be running.
Upgrade the `brainlayer` formula to a release with `brainbar activate` first. Run these
steps in order, stopping on any non-zero exit:

```sh
brew upgrade --cask brainbar
brainlayer brainbar activate --verify --timeout 90
brainlayer setup --no-migrate-mcp --verify-mcp
```

For a first install/reinstall, replace the first command with `brew install --cask brainbar`.
Postflight writes the two cask LaunchAgent plists and migrates MCP configuration without
launching or verifying BrainBar. Homebrew can then finish its own approval inheritance.
Activation leaves both current-bundle loaded labels untouched when MCP verification passes;
a probe failure on that current running bundle also preserves the service and reports failure.
Otherwise it replaces only the current user's cask daemon/UI labels, waits for `/tmp/brainbar.sock`,
and requires initialize, tools/list and a safe tool call through the installed bridge.
The startup/MCP budget defaults to 90 seconds; `--timeout` or
`BRAINLAYER_BRAINBAR_ACTIVATION_TIMEOUT_S` sets it independently of setup's 10-second probe.
`--no-verify` skips MCP probing and reports the result as unverified; the default is `--verify`.
Disabled labels are reported explicitly, never silently enabled.
This command covers the cask's default socket, not custom service-renderer socket paths.

A non-zero activation result includes JSON `stage`, `reason` and `cleanup_errors` fields.
It unloads attempted labels to stop KeepAlive retries; it does not restore old app files,
undo MCP migration, or retry launches. Cleanup failure is an explicit unresolved service state.
Stop deployment and investigate the receipt; do not loop activation. This change alone does
not guarantee uninterrupted service or no consent prompt on macOS 26/27: approved-predecessor
upgrade, failure recovery and real native MCP evidence remain release acceptance gates.
Other package jobs retain their own deployment restart/verification procedure.


### Supervisor recovery and deploy-local-prod step

The formula supervisor checks every five seconds for the installed bundle and both cask
plists with neither label loaded. Any active brew entrypoint/Homebrew child defers activation;
the automatic command checks again before launch mutations. The supervised worker has a
180-second runtime limit and its activation subprocess a 110-second limit. It records the
Info.plist GitCommit/version in `supervisor/brainbar-activation.json` **before** attempting
activation, so failure or supervisor restart does not repeat the attempt for that build.
Failure raises the `brainbar-activation` job alert (BrainBar attention/observability); an operator
can run `brainlayer brainbar activate` to retry. A successful manual verification clears the alert.

In deploy-local-prod, immediately after the cask installation/upgrade returns, insert:

```sh
brainlayer brainbar activate --verify
brainlayer brainbar verify-running --app /Applications/BrainBar.app
brainlayer setup --no-migrate-mcp --verify-mcp
```

Stop on a non-zero result. `verify-running` binds both launchd PIDs' mapped executable
inode/device to the installed bundle before reporting its GitCommit/version; a process name,
path or leftover socket alone cannot pass. `scripts/brainlayer-update-brainbar.sh` runs these
activation/identity steps itself, including when optional summary checks are skipped.
The tap change must merge atomically with the formula+cask bump to the release proved to
contain this implementation. The lead coordinates that bump after release; no versions are
changed in these implementation PRs. Native no-consent/real-MCP acceptance remains outstanding.
