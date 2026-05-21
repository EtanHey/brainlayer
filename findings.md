# Phase 1 Findings

## Formatting Deltas

- Search result cards now have a typed `SearchResult` model with a separate compact metadata row (`project • date • imp • score`) for SwiftUI rendering. The terminal formatter still preserves Python `_format.py` structure for MCP text output.
- `brain_entity` in `MCPRouter` now follows Python's simple entity lookup structure rather than the richer card formatter. This matches the Python handler path that formats lookup results with relations, associated memories, and metadata.
- `brain_recall` stats required a small DB payload expansion. `BrainDatabase.recallStats()` now returns `projects` and `content_types` arrays so the Swift formatter can match Python's stats lines instead of showing count-only output.
- Quick Capture search results now render through the new shared `SearchResultsList`/`SearchResultCard` components by bridging existing `QuickCaptureSearchRow` data into `SearchResult`.

## Branch Repairs

- The worktree had existing compile blockers unrelated to formatting: `HotkeyRouteStatus` and `BrainBarURLAction` were referenced but missing. Minimal compatibility implementations were restored so phase-1 formatter tests could run.

## Residual Notes

- `brain-bar/build-app.sh` succeeds, but the release build still emits two pre-existing warnings in `BrainDatabase.swift` about unused bindings in `listTags(query:)` and `digest(content:)`.

## Dashboard PID + Lane Decouple Dispatch

- 2026-05-21: Read the Fix A dispatch, source handoff Fix A section, and benchmark RESULT Step 2 before code.
- 2026-05-21: Memory search completed for BrainBar dashboard `targetPID:0`, `DashboardFlowSummary`, and PR #304 writable-store routing context. No more specific prior PID fix surfaced beyond dashboard-state memories.
- 2026-05-21: Source inspection confirmed `BrainBarAppSupport.makeUIStatsCollector` is currently in `BrainBarApp.swift` and passes `targetPID: 0`; no separate `BrainBarAppSupport.swift` exists yet in this worktree.
- 2026-05-21: TDD RED authored in `DashboardTests`: monitor returns non-nil for a real PID, UI stats collector accepts a discovered daemon PID provider, and DB-derived flow lanes render with `daemon: nil`. Targeted `swift test --filter ...` failed at compile time on missing `daemonPIDProvider`, confirming RED before production edits.
- 2026-05-21: Targeted GREEN passed: 5 DashboardTests, 0 failures, covering PID sampling, injected UI PID provider, nil-daemon lanes, and updated pipeline state expectations.
- 2026-05-21: Full BrainBar Swift suite passed: `swift test` in `brain-bar` ran 377 tests with 0 failures.
- 2026-05-21: Python verification notes: ambient `pytest` failed at collection due missing/incompatible eval deps (`deepchecks`, `numba` vs NumPy 2.4). `uv run --extra dev pytest` created `.venv` and completed with 2087 passed, 6 failed, 62 errors; failures/errors were live production DB lock setup (`apsw.BusyError: database is locked`) and eval/numba stack failures, not touched Swift dashboard files.
- 2026-05-21: Operator probes: `pgrep -fl BrainBarDaemon` returned PID 56862 from `/Users/etanheyman/Applications/BrainBar.app/Contents/MacOS/BrainBarDaemon`; `/tmp/brainbar.sock` exists; `/tmp/brainbar-daemon.pid` does not exist; `launchctl print gui/501/com.brainlayer.brainbar-daemon` reports `pid = 56862`.
- 2026-05-21: PR #304 verification: commit `12fecb4e` only changed `src/brainlayer/mcp/_shared.py` and `src/brainlayer/mcp/search_handler.py`; Swift UI still opens dashboard stats read-only. Existing Swift tests verify read-only dashboard stats reads, so no UI writable-store change is included in this PR.
- 2026-05-21: Runtime build evidence: `./brain-bar/build-app.sh --force-worktree-build --force-dirty` succeeded and produced `/Users/etanheyman/Applications/BrainBar-DEV-fix-dashboard-pid-and-lane-decouple.app` signed at 21:15:43. The script preserved the canonical daemon/socket.
- 2026-05-21: Runtime UI evidence: temporarily booted out canonical UI LaunchAgent, ran the dev app directly with canonical daemon PID 56862 alive, triggered `/tmp/.brainbar-toggle`, and captured `/tmp/brainbar-dashboard-pid-after.png`. Visual read: dashboard shows DB-derived lane labels `Writes` = `recent`, `Enrichments` = `live`, headline `Enrichment is draining backlog`; no `Queue offline` or `Writes unavailable` labels are visible in the captured dashboard. Canonical UI LaunchAgent restored afterward; `pgrep -fl BrainBar` shows canonical BrainBar and BrainBarDaemon running again.
- 2026-05-21: Opened PR #307 (`fix(brainbar): wire daemon PID into dashboard`) at https://github.com/EtanHey/brainlayer/pull/307 with reproducible RED/GREEN, push-hook, pgrep/launchctl, and screenshot evidence. Requested CodeRabbit, Codex, Cursor, and Bugbot reviews via PR comments.
