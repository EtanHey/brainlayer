# BrainBar: New is the only app design

Source commit: `8909073145318fc6755147c4c253fc2c500062fe`. Baseline resolver/Settings/wording: `07720cbbacba43abfb25ca0414eb5dafaf5e0245`.

All inputs are invented. These are offscreen `NDRenderHarness.host` renders, not a running installed app or native window capture.
Settings uses the view's fitting height at its fixed native 480-point width. The initial-window views use their native widths/heights (900 points wide). There is no mobile/390-point app layout.

| State | Before | After |
|---|---|---|
| Settings, dark | ![](before-settings-dark.png) | ![](after-settings-dark.png) |
| Settings, light | ![](before-settings-light.png) | ![](after-settings-light.png) |
| Stored `BrainBarDesign=classic`, initial window, dark | ![](before-stored-classic-window-dark.png) | ![](after-stored-classic-window-dark.png) |
| Stored `BrainBarDesign=classic`, initial window, light | ![](before-stored-classic-window-light.png) | ![](after-stored-classic-window-light.png) |

## Method and evidence boundary

The original `BrainBarDesign.swift`, `NDSettings.swift` and `BrainBarMenuSentences.swift` were restored from the baseline for the before run. Other files remained on the implementation branch. The retained Classic initial root is unchanged; its nil-collector state renders a loading view and starts no collector, database or socket. The after run renders New's equivalent unconfirmed initial state from an invented `BrainBarMenuCapture`.

`selected-root-render-test.swift.txt` preserves the temporary baseline-compatible selection renderer. Its Classic render arm is removed from the committed regression test. The app's actual controller has separate New-only root/page-sizing integration coverage for CI; those window tests were not run on the owner's desk.

The baseline ran 3 tests with 12 expected assertion failures (stored preference resolution and OCR of Settings/selected root). The implementation ran the same checks with 0 failures. The final focused set passed 73 tests; the truth test enumerated 47,257 captures. No full Swift suite, installed-app or native acceptance claim is made.

All eight PNGs were opened and visually inspected. The production Settings loses its Design selector and switch-back caption. New's other shipping views retain their rendering; a DEBUG-only backup fixture's Classic-card substitution was also removed.


## State-table extension

Source: `0dbd657c84455d0988a2952b31c6de7e9c95f099`. Missing previously: setup/pre-open and refreshed menus, long/error/signed-out Settings, and new/garbage/absent stored-value initial-window renders.

| State | Dark | Light |
|---|---|---|
| Menu at setup, before any open; unconfirmed | ![](menu-setup-before-open-dark.png) | ![](menu-setup-before-open-light.png) |
| Menu after refresh; fresh | ![](menu-after-refresh-fresh-dark.png) | ![](menu-after-refresh-fresh-light.png) |
| Menu after refresh; signed out/error | ![](menu-after-refresh-signed-out-dark.png) | ![](menu-after-refresh-signed-out-light.png) |
| Menu after refresh; waiting for browser | ![](menu-after-refresh-waiting-dark.png) | ![](menu-after-refresh-waiting-light.png) |
| Settings; long saved-file summary | ![](settings-long-text-dark.png) | ![](settings-long-text-light.png) |
| Settings; signed out/error | ![](settings-signed-out-error-dark.png) | ![](settings-signed-out-error-light.png) |
| Settings; waiting for browser | ![](settings-waiting-dark.png) | ![](settings-waiting-light.png) |
| Settings; unconfirmed/unknown | ![](settings-unconfirmed-dark.png) | ![](settings-unconfirmed-light.png) |
| Initial window; stored classic | ![](window-stored-classic-dark.png) | ![](window-stored-classic-light.png) |
| Initial window; stored new | ![](window-stored-new-dark.png) | ![](window-stored-new-light.png) |
| Initial window; stored garbage | ![](window-stored-garbage-dark.png) | ![](window-stored-garbage-light.png) |
| Initial window; stored absent | ![](window-stored-absent-dark.png) | ![](window-stored-absent-light.png) |


Widths: menu 320 pt, Settings 480 pt with natural fitting height, initial window 900 pt with natural Overview height. All images are 2x. The four stored-value window pairs are deliberately visually identical: New is the only installed root, regardless of the stored preference; the key is left unchanged. They are offscreen views of the initial root, not opened NSWindows or installed-app screenshots.

Menu proof captures the actual menu attached by the production status controller to `TestStatusItemHost`, before calling any menu delegate. After-rows inject invented fixture data and call the real pre-open refresh delegate. No native status item is allocated and no menu tracking begins. The production hosted rows are composed offscreen using NDRenderHarness's row-composition implementation; standard system rows/material use the existing flat stand-in, so this is not a claim about AppKit's tracking/material rendering. The new replay includes assertions that setup already contains actionable Settings and Quit.

Settings long-text uses an invented 12,345-backup summary with a long nested fixture path. Signed-out/error shows Drive's real Reconnect row; waiting shows its disabled replacement after a reconnect request; unconfirmed shows unknown evidence. No new UI control was introduced by this PR. Pressed/confirmation/saved selector states are not applicable because the Design selector is deleted. Existing actions' native click/confirmation behavior remains in installed acceptance, outside this offscreen proof.

Replay: copy `state-proof-render-test.swift.txt` to `brain-bar/Tests/BrainBarTests/BrainBarW243StateProofTests.swift`, then run `BRAINBAR_W243_STATE_PROOF=<absolute output directory> swift test --package-path brain-bar --filter BrainBarW243StateProofTests` with the lane's required scratch/gate settings. This proof-only fixture uses the shipping sources; it is not shipped in the app. Its drawn-state assertions and PNGs are retained here for visual regression review. All 24 new images were opened and inspected at the proof boundary.


## Update-control merge integration

Source `079fc3e7d9a472b44dbb7bb9c0572f4bace675c9` merges main `8393656df7ac204e4c4c6c039723b796c96344f1` while retaining the single New design. Only NDSettings.swift conflicted: keep showsUpdateControl, omit the obsolete Design binding. App, Runtime, NDStore and NDWindow are identical to main's updater wiring.

These additional invented-data captures come from the full run's `BrainBarUpdatesTests.testDrawnUpdateStates`, at the real 480pt Settings width, with long recovery text. Both were opened and inspected. The update row is below Diagnostics and there is no Design row. No extra render invocation or production app was launched.

| Integrated Settings state | Dark | Light |
|---|---|---|
| Update recovery, long text, Classic removed | ![](merged-update-settings-dark.png) | ![](merged-update-settings-light.png) |

The version numbers and recovery filename are fixed test fixtures. The independent updater states are exercised unchanged by the merged full test suite. These are offscreen captures, not installed-app proof. There are now 34 PNGs on this proof branch.
