# BrainLayer newsyslog

BrainLayer LaunchAgents write logs as the user, not as root. macOS `newsyslog`
creates rotated replacement files as `root:admin` unless the config line
specifies an owner and group. A root-owned replacement silently breaks later
appends from user-level daemons.

This drop-in only rotates finite scheduled LaunchAgent jobs. Long-running jobs
such as BrainBar, watch, and enrichment keep their `StandardOutPath` and
`StandardErrorPath` descriptors open; macOS `newsyslog` has no post-rotate hook
or copy-truncate mode. The separate `com.brainlayer.log-cap` job checks installed
BrainLayer plists every five minutes and trims oversized logs in place, keeping
the newest 2 MiB on the inode held by launchd's append descriptor. It refuses
symlinks, hard links, non-regular files, and files owned by another user. Drain
also owns `drain.err.log` through a rotating Python handler; launchd captures
stderr for the daemon's whole life in `drain.bootstrap.err.log`.

Install `brainlayer.conf` into `/etc/newsyslog.d/` with:

```sh
launchd/install-newsyslog.sh
sudo newsyslog -nv -f /etc/newsyslog.d/brainlayer.conf
```

The checked-in config targets Etan's LaunchAgent account as `etanheyman:staff`.
The installer renders that config for `BRAINLAYER_LOG_OWNER`,
`BRAINLAYER_LOG_GROUP`, and `BRAINLAYER_LOG_DIR` before installing it into
`/etc/newsyslog.d/`. The rendered config is validated with `newsyslog -nv`
before replacing the live drop-in. Because `newsyslog.conf` is
whitespace-delimited, the installer rejects log directories containing
whitespace.

Every installed entry uses mode `644`, `J` compression, and `N` so rotation does
not signal user LaunchAgents. Launchd owns job lifecycle.
