# Release receipt

Status: BLOCKED until every required gate has evidence.

- Release tag / commit:
- `scripts/release_tag_contains.py <tag> <PRs/SHAs>` table:
- Installed keg path/version and `release-verify-signatures.sh` result:
- Installed module import assertions:
- Supervisor health (`scan_error: null`, watch `source: supervisor`):
- MBP harness probe exit code, start time, worktree commit, private artifact paths:
- M1 harness probe exit code, start time, worktree commit, private artifact paths:
- Paste each host's complete `table.md` here:
- Cask result (only after gates pass):
- Limits / unresolved failures:

## Harness acceptance

The live gate is `scripts/harness_probe.py --output <new-private-directory>`, run under the
installed keg Python. It preserves user MCP/auth/model settings; Cursor receives no model flag.
Every row requires structured native search/store results, a separately connected installed
bridge finding the tagged unique nonce, observed child bridge/socket transport, and an independent
transcript marker found in that fresh session's JSONL. Config inventory is diagnostic, not proof
of effective precedence. Stores use project `brainlayer`, tag `harness-probe`, importance 1.

Claude, Codex and Cursor use the watcher JSONL roots. Gemini CLI is out of convergence scope
per Etan's 2026-10-16 harness ruling 1 (`harness-probe-rulings-1.md`); agy is the Google harness.
Antigravity/agy has no BrainLayer watcher adapter; its ingest is explicitly
NOT_APPLICABLE. Codex app and T3 remain separate required rows: the current script fails those
rows until a native session-launch/evidence adapter exists. CLI success cannot substitute.
The narrow T3 adapter is an authenticated command to the existing running server's
`orchestration.dispatchCommand` / `thread.turn.start`, retaining its real provider config and
observing the resulting tool events and projected transcript. Do not start another T3 server or
mint auth to get a green row. Codex app needs a fresh app thread and native tool-event evidence.

Offline contracts run in normal Python CI and the changed-file pre-push gate. The live portion
requires each operator's authenticated harnesses and BrainBar, unavailable on GitHub-hosted CI;
the required operator deploy gate above is its closest gate. A unit-suite PASS is not live proof.
Any FAIL exits nonzero; partial selections also exit nonzero and cannot serve as release receipts.
