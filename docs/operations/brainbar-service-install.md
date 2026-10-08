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
Checks are metadata observations, not a same-UID mutation lease. Logs/plists must remain owned and safe at activation. Existing cask postflight is separate; this does not change it or establish native proof.
The DB must identify a different resource from the socket and derived `.lock`: existing same-file identities are refused.
Case-folded, canonically Unicode-equivalent full paths are conservatively refused even when missing or on case-sensitive volumes; choose names that differ beyond case/normalization.
This covers prospective ambiguous names without creating probe files or relaxing ownership/type checks; it is a point-in-time check, not an atomic namespace lease.
