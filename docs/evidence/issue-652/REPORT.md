# Issue #652: copy-to-continue namespace trace (read-only report)

Marker: `cloud-credit-20261105-brainlayer-last-proof-652` · Tree: `origin/main` `d6b82f04f6baa00081c9dc38afca2ada9d114c78`.
Proof: `python3 -I docs/evidence/issue-652/proof.py` → **7/7 claims hold** (source text plus a synthetic model; no DB, no Swift run).

## Verdict

**Not fixed, and not reproducible from current source either.** Current `main` has **no "copy to continue" action**:
the command builder exists, but no production code calls it. So the report Etan gave cannot happen in a
build cut from this tree. The ID wiring the builder would consume is still wrong in two ways (W1, W2 below).
Reconnecting the button without fixing them would bring the #652 bug back.

## Trace (file:line on d6b82f04)

| Step | Location | What it does |
|---|---|---|
| Builder | `brain-bar/Sources/BrainBar/InjectionContinuation.swift:9-24` | `resumeCommand(conversationID:fallbackSessionID:projectPath:)` → `[cd <path> &&] claude --resume <id>` or `claude --continue` |
| Callers | none (proof C1) | `BrainBarDaemon/InjectionContinuation.swift` is a symlink to the same file, not a caller |
| Clipboard | `brain-bar/Sources/BrainBar/BrainBarWorkspace.swift:46-49` | the only `NSPasteboard` writer. Its callers are Settings/backup only (`BrainBarSettingsView.swift:638-643`, `BrainBarBackupChecks.swift:292`, `BrainBarBackupSources.swift:17`) |
| ID feed | `BrainDatabase.swift:6058` → `:6142-6155` | `resumableClaudeConversationID(event:chunks:)` sets `InjectionEvent.claudeConversationID` (`:6099`) |
| Event row | `BrainDatabase.swift:5999-6005` | `SELECT … FROM injection_events` does **not** project `claude_conversation_id`, so `event.claudeConversationID` is always `""` (`InjectionEvent.swift:481`) |
| Chunk row | `BrainDatabase.swift:6286-6291` | chunk `claude_conversation_id` = `metadata.$.claude_conversation_id` of the **stored memory chunk** |
| Project dir | `InjectionEvent.swift:96-98`, `InjectionContinuation.swift:26-35` | project path decoded from the chunk's `source_file` (`-Users-x-Gits-repo` → `/Users/x/Gits/repo`). This decode is lossy for names containing `-` |

## The four ID namespaces

| Namespace | Example shape | Where it appears in BrainBar |
|---|---|---|
| BrainLayer row/session ID | `injection_events.session_id`, chunk `conversation_id`, `chunks.id` | stored by hooks/BrainBar. **Not** a Claude CLI ID by contract |
| Native Claude CLI session UUID | `<uuid>.jsonl` stem under `~/.claude/projects/<enc>/` | the only thing `claude --resume` accepts |
| Cloud session locator (`--teleport`, `session_…`/`cse_…`) | — | **absent**: no emitter in BrainBar (proof C3) |
| T3 thread link | — | **absent**: no emitter in BrainBar (proof C3) |

## Wrong-ID mappings that would come back if the button were reconnected

- **W1 (wrong conversation, proof C5+C6+C7):** the recipient session's UUID is never read. Because the event
  column is not selected, the resumable ID falls through to the **injected memory chunk's** own conversation
  (metadata, then `source_file` stem). The synthetic fixture yields `2222…` (the memory's source conversation)
  instead of `1111…` (the session that received the injection). The project `cd` path comes from the same chunk,
  so it points at the memory's repo, not the recipient's. The UUID is a real Claude ID, but it resolves to the
  *wrong* transcript.
- **W2 (wrong namespace, proof C4):** for a non-UUID conversation ID, `resumeCommand` emits
  `claude --resume <fallbackSessionID>` with whatever safe string it is given. `InjectionContinuationTests.swift:26-31,40-45`
  pin `claude --resume sess-42`, which is exactly the BrainLayer-session-ID-as-CLI-ID shape #652 describes.
  The doc comment at `InjectionContinuation.swift:5-7` says the opposite.
- **W3 (cloud/T3):** a cloud-run or T3-run session has no local JSONL UUID. Today it would get
  `claude --continue`, which resumes whatever was most recent in the cwd, never `--teleport` and never a T3 link.

## Suggested direction (not implemented; the owning lead decides)

Resolve the resume target from the artifact it names (#652's "stronger form"): read the **recipient's**
transcript path/UUID from the injection event (store it at injection time, then select it). Emit `claude --resume`
only when that `<uuid>.jsonl` is confirmed to exist. Emit `--teleport <locator>` only from a recorded cloud
locator. Remove the BrainLayer-session-ID fallback, and flip the two tests above.

## Scope notes

- No product or source changes. No cmuxlayer changes. The global `~/.claude/CLAUDE.md` wording stays Etan's, and was not authored or installed here.
- **UNRUN:** Swift build/tests (no toolchain in the cloud container); native `claude --resume`/`--teleport`; any live BrainBar UI click; any real DB, user data, or Mac access.
- The history of when the caller was removed is unclear: the file was re-added in large merges (`870dabc`, `666b2dc`, `7c69eb9`, all 2026-10-18), and `-S` search could not pin the removal. Not investigated further within the time box.
