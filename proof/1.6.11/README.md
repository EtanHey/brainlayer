# BrainLayer 1.6.11 offscreen visual comparison

Status: all four scenarios captured and all 18 PNGs visually inspected. Requested fixes are visibly demonstrated for states 3 and 4 only. States 1 and 2 are retained as failure evidence, not claimed fixed.

Before: `a6b65831615af8043bc83f3a8ffd60017c20f7bf` (v1.6.10).
After: `8a6f1c4a4f42f4880279905d888c381536b93671` (release head).

Method: a temporary DEBUG-only helper in each detached local worktree calls the existing `NDRenderHarness.window` and `NDRenderHarness.menu`. The process sets `NSApplication` activation policy to prohibited and exits before `BrainBarApp.main()`. Window views are rendered using `NSHostingView.cacheDisplay` to a 2x bitmap; menu content uses the real `NDMenuBuilder` hosted rows on the harness's stand-in menu material. No window, menu or status item is shown. These are offscreen fixture renders, not installed-app or physical-wake captures.

Inputs are invented and identical at both commits. Fixture time starts at `2026-11-15T08:00:00+02:00`; there are no reads of the production database, BrainBar socket, personal files or screen.

1. Wake: an injected wall/uptime clock, an in-memory config document, static launchd and embedding probes, and no health verdict. Call `systemWillSleep()`, advance wall time 600 seconds without advancing uptime, then call `systemDidWake()`. Render Overview and menu content, dark and light. No fresh verdict is supplied. Both commits have these handlers. The after helper also uses its new `observeSystemLifecycle: false` injection; before has no such option. Both use explicit injected-clock handler calls.
2. Wake timeout: from the same model, advance both clocks by 361 seconds, fire the last scheduled deadline callback, and render Overview and menu content in dark mode. No fresh verdict is supplied.
3. Stalled upload: the existing `upload-stalled` fixture at both commits, rendered on Backups and Overview in dark mode. Its confirmed byte checkpoint is normalized to 21 minutes old at both commits; its prior verified Drive copy is preserved. The before tag's stock fixture was only 15 minutes old, below the actual 20-minute stall threshold, whereas the release head changed it to 21. The temporary before-worktree fixture uses the after fixture's 21-minute input, without changing the before progress reader or views. With identical raw inputs the before reader drops the stale timestamp, so the true baseline says `Uploading` without a time, rather than the task's anticipated clock-only baseline.
4. Retention error: parse these invented sessions-log rows with each commit's `NDBackupHistoryReader.parseConversations`: a verified upload six hours ago, `{"event":"local_retention_error","error":"invented inventory failure"}` with no timestamp, and a verified upload three hours ago. Put the parser's actual result into the calm fixture's Sessions lane, retaining its other lanes; render Backups in dark mode.

Only PNGs and this README are published on this orphan proof branch. No harness or product source is published here, and this branch is never merged into main.

## Observed results

| State | Actual before | Actual after | Requested visual proof |
| --- | --- | --- | --- |
| 1. Immediately after wake, Overview and menu, dark/light | `Not confirmed`, dashed pipes | Still `Not confirmed`, dashed pipes | **Not demonstrated.** Generic presenter output is `Checking after sleep…`, but these actual New-design surfaces do not show it. |
| 2. Wake timeout after 361 awake seconds, Overview and menu, dark | `Not confirmed`, no wake reason | Still `Not confirmed`, no wake reason | **Not demonstrated.** The underlying published snapshot has an amber dot and `No health reading from after wake.`, but the requested New-design pictures omit the reason. These pictures do not prove a status-item dot. |
| 3. Stalled Drive upload, Backups and Overview, dark | `Uploading` / `Uploading to Google Drive`, no stale checkpoint time | `not uploading since 07:39`, last-good line retained | **Demonstrated**, with the normalized identical-input baseline caveat above. |
| 4. Undated retention-error row, Backups, dark | Sessions lane dashed, `Record couldn't be read` | Solid Sessions lane with a red failure at the following run's date | **Demonstrated.** The actual parser changes from `skipped=1, attempts=nil` to `skipped=0`, preserving both uploads and the dated error. |

The wake gap is source-consistent: at this release head `NewDesign/NDR1.swift`'s `pipeline.allUnconfirmed` headline branch chooses `Not confirmed` without using the capture's wake flag. The generic presenter and status snapshot improvements therefore cannot be reported as New-design visual acceptance. The wake fixture passes `model.menuCapture(...)` into the otherwise invented calm ND inputs; it does not instantiate a live runtime or claim to exercise its additional warnings.

Both commit builds and all offscreen render runs exited 0. Swift builds were gated by disk, memory pressure, two swap-out samples 30 seconds apart, load, and an owned heavy slot. Every acquired slot was logged and released. No test-suite, installed-app or real-wake acceptance is claimed.
