## A1 live implementation

Two implementations exist at base `cc1d21f66534747ee4a50367c290858972629a70`: Python `src/brainlayer/mcp/store_handler.py:88` (dispatched by `src/brainlayer/mcp/__init__.py:1482`) and Swift `brain-bar/Sources/BrainBar/BrainDatabase.swift:6457` (called by `brain-bar/Sources/BrainBar/MCPRouter.swift:993`). The live path is Swift: the running `/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon` owned `/tmp/brainbar.sock`, which is the socket default at `brain-bar/Sources/BrainBar/BrainBarServer.swift:181`; the stdio bridges connected to that socket. `brain-bar/Sources/BrainBarDaemon/BrainDatabase.swift` is a symlink to the same Swift source, not a third implementation.

## A2 measured cutoff

At base `cc1d21f66534747ee4a50367c290858972629a70`, a `tools/call` to `brain_digest` with exactly 701 characters returned success but the temporary database contained exactly 503 characters: the first 500 input characters plus `...`. Measured cutoff: 500. The RED assertion reported `brain_digest stored 503 characters from a 701-character input`; the defect is the prefix construction at `brain-bar/Sources/BrainBar/BrainDatabase.swift:6507` plus the normal success formatting at `brain-bar/Sources/BrainBar/MCPRouter.swift:1001`.

## A3 stub count

298 truncated stubs, six fewer than the unverified estimate of about 304.

```sql
SELECT COUNT(*) AS truncated_stubs FROM chunks WHERE created_at >= '2026-04-01' AND source = 'digest' AND length(content) >= 503 AND substr(content, -3) = '...';
```

Detection rule: rows created on/after 2026-04-01 with `source = 'digest'`, at least 503 characters, and a terminal `...`. This matches the base Swift behavior at `cc1d21f66534747ee4a50367c290858972629a70` (`brain-bar/Sources/BrainBar/BrainDatabase.swift:6507`): 500 content characters plus `...`, optionally preceded by a title. The query was run read-only with `sqlite3 "file:$HOME/.local/share/brainlayer/brainlayer.db?mode=ro"`.

## A4 recoverable

recoverable. The chunk rows contain only the stub (`source_file = 'brainbar-store'`, no original-content column), but matching stored prefixes and post-title body prefixes against pre-2026-08-01 local Claude/Codex/Cursor/Gemini transcripts located full tool-call inputs for 198 of 298 rows. The configured Google Drive `claude-jsonl` backup folder contains daily transcript archives through 2026-07-31, covering the source class for the remainder; the remaining 100 were not individually extracted from those archives during this task.

## A5 fix chosen

full digest — storage now receives the full input (with the optional title prepended), because the schema already accepts up to 200,000 characters and silent data loss is unnecessary.

## B false ack

Reproduction at base `cc1d21f66534747ee4a50367c290858972629a70`: `brain_ack({"agent_id":"cmuxlayer-owned-agent","seq":42})` returned `{"status":"acked"}` and created a BrainBar agent row with both cursors at 42. The cause is `brain-bar/Sources/BrainBar/BrainBarServer.swift:866`, which called an acknowledge path that auto-created unknown agents. The fix requires an existing BrainBar subscription; otherwise the MCP result has `isError: true`, text `No BrainBar subscription for agent: ...`, and no row is created. Dropped/missed messages: undetermined. The production DB has three acknowledged agent rows with zero subscription tags, but cursor state cannot show whether a caller discarded or overlooked an external cmux message; settling that requires the cmux inbox/consumer event history at each ack time.

## tests

RED on `cc1d21f66534747ee4a50367c290858972629a70`: `swift test --package-path brain-bar --filter 'MCPRouterTests.testBrainDigestStoresTheFullInputContent|SocketIntegrationTests.testMCPBrainAckRejectsAgentWithoutBrainBarSubscription'` executed 2 tests with 5 assertion failures: digest 503 versus 701, and false ack success plus created cursor row. GREEN after the fix: the same command executed 2 tests with 0 failures. Full suite command `swift test --package-path brain-bar` executed 842 tests with 10 skipped and 1 unrelated failure in `DashboardTests.testHeartbeatMarksStatsSnapshotPendingDuringCoalescedRefresh`; that test passed when rerun alone with `swift test --package-path brain-bar --filter 'DashboardTests.testHeartbeatMarksStatsSnapshotPendingDuringCoalescedRefresh'` (1 test, 0 failures). The changed suites passed in the full run: `MCPRouterTests` 74/74 and `SocketIntegrationTests` 21/21.

## uncertainties

The remaining 100 truncated inputs were not individually extracted from the Drive transcript archives. The full-suite dashboard timing failure is reproducible only in the full run so far and is outside Items A and B. Whether any caller missed an external message after a false ack is undetermined because no per-message cmux consumption history was available in the BrainLayer DB.
