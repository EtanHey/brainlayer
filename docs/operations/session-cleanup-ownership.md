# SessionStart cleanup ownership

Status: source fix; installation is a separate post-merge action owned by orc.

`hooks/session-cleanup.py` never sends signals. It only reconciles its private
ownership records and logs still-live orphan candidates. A socket path in argv
cannot establish ownership; there is no command-substring or age sweep.
With no ownership directory/records, cleanup returns zero without inspecting the
process table. WAL checkpoint behavior is unchanged.

The optional `start_owned_mcp([absolute_executable, ...])` launch helper records
only its direct child, with PID, UID, kernel birth identity and executable path.
Records live in `~/.local/state/brainlayer/session-cleanup-owned/` (0700 directory,
0600 regular files). Registration cannot adopt an unrelated process. A failed
registration removes the partial record (including fsync/write failures) and
leaves the caller its process handle. An existing record is never removed by a
failed exclusive-create attempt.
SessionStart currently launches no MCPs, so existing Claude, Codex, T3, Cursor,
VoiceLayer fixture and socat processes are deliberately never adopted.

Reconciliation removes a record only when kernel inspection proves the process
absent (`ESRCH` on Darwin or a missing process stat file on Linux), or its kernel
birth identity differs from the record. A changed executable alone is not proof
of a changed birth. Permission failures, unreadable identities and zombies leave
records intact. Unsafe/symlinked/malformed records are ignored.

A matching PID, UID, executable and birth with PPID 1 is logged as an owned orphan
candidate and its record is retained. No process is stopped, even a verified
orphan. Darwin cannot bind a numeric signal atomically to birth identity, so an
extra identity check cannot close the PID-reuse race. Etan's MCP connection rule
outranks orphan cleanup; a leaked orphan is preferable to a disconnected seat.

Follow-up row **SC-OWN-PARENT-EXIT**: make bridges exit when their parent dies.
This orphan-prevention work needs a separate PR and is not implemented here.

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
