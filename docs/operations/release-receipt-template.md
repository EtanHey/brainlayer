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
NOT_APPLICABLE. Codex app and T3 remain separate required rows. Codex app connects by WebSocket to the existing
app daemon Unix control socket, without starting/restarting a daemon or changing MCP settings.
T3 uses the existing installed `~/.t3/bin/t3 acp-mcp-bridge` only when the invoking
agent already has `T3_ACP_MCP_ENDPOINT` and `T3_ACP_MCP_AUTHORIZATION`. The endpoint must match
the local installed runtime port. It launches a fresh scratch thread, reads native dynamic-tool
events and binds the exact T3 thread to its strong Codex provider reference before checking the
Codex JSONL marker. Other T3 providers' transcript bindings are not supported by this adapter.
No server starts and no token is minted or extracted. Missing context yields `FAIL(auth)`.
A connected T3 MCP tool in Codex does not imply these environment variables reach its shell:
the measured Codex-backed T3 shell lacked both. Such a run remains FAIL; native connector
observations may be attached as diagnostics but cannot replace a complete script receipt.
Native adapters reserve 15 seconds within the outer timeout for their own-turn cleanup.
CLI success cannot substitute for app or T3 proof.

Offline contracts run in normal Python CI and the changed-file pre-push gate. The live portion
requires each operator's authenticated harnesses and BrainBar, unavailable on GitHub-hosted CI;
the required operator deploy gate above is its closest gate. A unit-suite PASS is not live proof.
Any FAIL exits nonzero; partial selections also exit nonzero and cannot serve as release receipts.

An agy row is admitted only with an existing unexpired CLI bearer file (at least five minutes
remaining). Missing/expired/unverifiable credentials produce `FAIL(auth)` before spawning agy;
the probe never refreshes, logs in, unlocks a keychain, or starts OAuth. A failed SSH credential
path does not establish that the GUI user is signed out.


For a macOS run launched over SSH, use `--aqua`. A unique temporary one-shot LaunchAgent runs
the probe in the existing `gui/<uid>` session. The child verifies `launchctl managername` is Aqua
before any harness starts; the parent waits for its terminal receipt and removes only that job.
No unlock, sign-in, persistent LaunchAgent installation or existing service restart occurs.
Receipts include the GUI job identity and private parent/child logs.

Claude's session grants only the already authorized `mcp__brainlayer__brain_search` and
`mcp__brainlayer__brain_store` tools. Cursor receives workspace trust for the chosen probe
directory; use the probe's own scratch directory. These do not replace real user MCP configs
or select a Cursor model, and do not bypass general tool permissions.
