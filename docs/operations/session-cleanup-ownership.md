# SessionStart cleanup ownership

Status: source fix; installation is a separate post-merge action owned by orc.

`hooks/session-cleanup.py` no longer discovers kill targets by process-command
substrings or elapsed age. A socket path in argv cannot establish ownership.
With no ownership directory/records, cleanup returns zero without inspecting the
process table. WAL checkpoint behavior is unchanged.

The optional `start_owned_mcp([absolute_executable, ...])` launch helper records
only its direct child, with PID, UID, kernel birth identity and executable path.
Records live in `~/.local/state/brainlayer/session-cleanup-owned/` (0700 directory,
0600 regular files). Registration cannot adopt an unrelated process. A failed
registration leaves the caller its process handle but grants no cleanup rights.
SessionStart currently launches no MCPs, so existing Claude, Codex, T3, Cursor,
VoiceLayer fixture and socat processes are deliberately never adopted.

Cleanup requires PPID 1 plus the recorded executable, UID and birth identity,
revalidated immediately before TERM. It ignores attached processes regardless of
age, unsafe/symlinked/malformed records, reused PIDs, changed executables,
zombies and unavailable identities. Darwin uses libproc microsecond birth time
and executable identity; Linux uses procfs birth ticks plus boot ID. Unsupported
platforms cannot authorize a signal. The same-user private record is the trust
boundary; there is no atomic birth-bound signal primitive around `os.kill`.

## Read-only impact receipt

On 2026-10-17, the unchanged installed selector was replayed with its `os.kill`
namespace replaced by a recorder. It matched 28 command strings and selected
**0** processes for TERM: Codex app-server had 12 BrainLayer matches and 11
VoiceLayer socats; T3 had 2 BrainLayer matches and 1 VoiceLayer socat; 2 other
BrainLayer matches remained. No signals were sent. This does not establish
historical or future safety of the old six-hour rule. VoiceLayer's native sender
evidence independently establishes the fixture impact.

## Update the installed copy after merge

The active M4 settings command invokes the independent file
`~/.claude/hooks/session-cleanup.py` using
`/opt/homebrew/opt/brainlayer/libexec/venv/bin/python`. At diagnosis both source
and installed copy had SHA256
`77c9cf5e5308ef2f2dbda2770c97904fced36d202e5237759d19680134a09ed8`.

BrainLayer does not ship this `hooks/` directory in its wheel/sdist and has no
installer for this copy. Golems' `scripts/hooks/manifest.json` lists
`session-cleanup` as `kind: external`; its installer preserves that registration
and does not copy BrainLayer's script. A Homebrew upgrade or golems hook install
therefore does not deploy this fix.

After merge, orc should use a checkout verified to contain the merged commit:

```sh
install -m 0755 /absolute/merged-brainlayer-checkout/hooks/session-cleanup.py \
  "$HOME/.claude/hooks/session-cleanup.py.new"
mv "$HOME/.claude/hooks/session-cleanup.py.new" \
  "$HOME/.claude/hooks/session-cleanup.py"
shasum -a 256 /absolute/merged-brainlayer-checkout/hooks/session-cleanup.py \
  "$HOME/.claude/hooks/session-cleanup.py"
```

This atomically replaces the script without running SessionStart or checkpointing
the live DB. Verify the two hashes match and the settings command still uses the
pinned interpreter. No installed file is changed by this PR.

Read-only `ssh m1` checks found neither `~/.claude/hooks/session-cleanup.py`
nor a `session-cleanup` registration in `~/.claude/settings.json`. M1 therefore
has no copy to update at that path in this snapshot; do not add a new hook there.
