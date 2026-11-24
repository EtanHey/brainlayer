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
