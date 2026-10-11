# Coverage honesty render evidence

These are DEBUG fixture captures from offscreen `NSWindow`s at 1200-point height. The window was ordered front for bitmap capture but never made key. No installed BrainBar or production database was used.

`before/` was rendered from exact `origin/main` commit `96d2f19ed0d1c4ad1c7a398ec004ee8ba77e59c4` in a temporary detached worktree. It used a temporary copy of `BrainBarCoverageVisualTests`, with the new coverage-loading case removed; that worktree was deleted. `after/` was rendered by the test at rebased honesty commit `c5820a368a040a4e60f942fb409c65e91a819eab`. The subsequent artifact commit changes no app code.

| Pair | Width | Evidence |
| --- | ---: | --- |
| [Before compact](before/compact.png) / [After compact](after/compact.png) | 760 pt | Exact indexed / eligible and missing counts fit in all three cards. |
| [Before default](before/default.png) / [After default](after/default.png) | 960 pt | FTS5 changes from a misleading `100%` to `99%`, `296,980 / 297,412`, `432 not indexed`. |
| [Before wide](before/wide.png) / [After wide](after/wide.png) | 1280 pt | The same denominator is visible across Vector, FTS5, and Trigram. |
| [Before stale](before/stale.png) / [After stale](after/stale.png) | 960 pt | Stale dashboard framing remains visible with the honest coverage labels. |
| [Before Details collapsed](before/details-collapsed.png) / [After Details collapsed](after/details-collapsed.png) | 960 pt | The two frames are byte-identical (SHA-256 `779bc3d14cc4e01a538818c75178a7d2db5fa047fa228dbef6f566bea3e9bd62`). |

[After coverage loading](after/loading.png) shows `Computing…` and `Counting eligible chunks…` with empty tracks during first load. The fixture includes no corresponding baseline loading frame. The frames do not verify VoiceOver, Vector hover, keyboard Tab reachability, or installed-app behavior.

At the rebased source head, full `swift test` ran 949 BrainBar tests with six skipped and two failures: the known panel geometry assertion at `BrainBarDashboardPanelControllerTests.swift:188`, plus the previously observed heartbeat timestamp comparison at `DashboardTests.swift:2788`. The heartbeat test passed when rerun alone. BrainBarDaemon passed 7/7, and both `BrainBar` and `BrainBarDaemon` target builds passed.
