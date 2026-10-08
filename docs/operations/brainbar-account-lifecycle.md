# BrainBar account lifecycle

BrainBar UI and daemon heartbeats belong to the logged-in process account:
`~/Library/Application Support/BrainBar/{ui,daemon}.heartbeat`.
Both peers resolve their own account home. The writer creates an owned directory,
requires a non-writable-by-others parent, and writes an owned, single-link regular
file with mode 0600. Symlinks, foreign ownership and writable-by-others files are
refused. Read failures remain unmeasured; write failures reach unified logging.

The watchdog accepts a peer only with the current real/effective UID and expected
installed executable path. UID, executable and process start time are checked
again before TERM and KILL. A replacement process prevents `kickstart -k`.
These are point-in-time checks, not an atomic kernel PID lease.
The awake clock, responding-daemon veto and missing/legacy heartbeat grace remain.

Upgrade UI and daemon together through the existing quiesced release procedure.
Old `/tmp/brainbar-{ui,daemon}.heartbeat` files are neither adopted nor removed.
A mixed old/new pair does not share heartbeat paths; it must not be left resident.
The default socket remains `/tmp/brainbar.sock`; a separate account still needs
the supported server/bridge socket configuration and its own installed services.
This lifecycle change does not relocate legacy service-template log paths or the
global toggle flag, install services, grant TCC access or establish native proof.

The CGEventTap hotkey fallback defaults OFF. The UI returns before starting a tap
when OFF; the permission-request path belongs to an explicitly enabled fallback.
