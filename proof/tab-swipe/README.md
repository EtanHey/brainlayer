# Proof: two-finger swipe between Overview, Backups and Jobs (w241)

Offscreen `NDWindowRoot` renders (the window root with its drawn title bar and tabs) at 2x, dark and light, written by
`BrainBarTabSwipeTests.testASwipeMovesTheSelectedTabWithThePage` with `BRAINBAR_SWIPE_RENDER_DIR` set. Invented fixture
data (`calm`). One hosting view per appearance, updated in place: each step sends a simulated trackpad gesture
(began, ten changed steps of 12 pt, ended) through `NDPageSwipe`, applies its step to the store, then draws.

| Step | Gesture | Page and selected tab after it |
|---|---|---|
| 0 | none (start) | Overview |
| 1 | left, from Overview | Backups |
| 2 | left, from Backups | Jobs |
| 3 | left, at Jobs | Jobs (the end stops; same pixels as step 2) |
| 4 | right, from Jobs | Backups |
| 5 | right, from Backups | Overview |
| 6 | right, at Overview | Overview (the end stops) |

Before this change no page handles the gesture: the window stays on the page it was on, which is the previous step's
picture. No motion recording: nothing may be shown on screen on the M4.

## State table (added 24 Nov, owner rule "no state table, no merge")

The control this PR touches is the window's tab control (`NDSegments`, a native `NSSegmentedControl` with drawn labels and
count pills). Pictures from `BrainBarTabSwipeStateRenderTests.swift` (in this folder; run with
`BRAINBAR_SWIPE_STATE_DIR=<dir>`), offscreen, at the window's real width (900 pt) and each page's real height (its
`NDPageHeight.window` on a screen tall enough not to cap it), 2x, dark and light.

| State | Files | What the test asserts |
|---|---|---|
| Each page selected after a click vs after a swipe | `state-<page>-from-<page>-by-click/-by-swipe-*.png` | same page, same selected tab, same height, under 0.1% of pixels differ (they are byte-identical) |
| Keyboard focus on the tab control | `state-focus-<page>-*.png` | the ring is drawn; drawn as AppKit draws it (`NSFocusRingPlacement.only` + the control's own `drawFocusRingMask`), since an offscreen view cannot hold first responder |
| Count pills on Backups and Jobs (two faults), each page selected | `state-two-faults-<page>-*.png` | the selected tab names the page; the fixture carries counts |
| Calm, each page selected, after swipes left and right, ends stopping | `swipe-*.png` (above) | the drawn selected tab and the laid-out page follow each swipe |

States the control does not have: disabled (no code path disables it), busy, waiting, saved, confirm, error and empty
(it always shows the same three tabs). Pressed (mouse held on a segment) is the native control's own tracking highlight
and cannot be drawn offscreen.
