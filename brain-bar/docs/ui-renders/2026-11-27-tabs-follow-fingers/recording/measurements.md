# Tabs follow the fingers — 120 Hz recording and per-frame measurements

PR #1896. BEFORE = main `7ee776fb` (the 1.6.13 tab behaviour), AFTER = `388b358b` (the PR head when built; the head is now `f6e8e6a7`, which only adds 6 lines to `BrainBarTabsFollowFingersTests.swift`, so the app is the same). Capture builds (`com.brainlayer.brainbar.capture`, fixture `calm`, `BRAINBAR_DESIGN_NOW=2026-11-27T21:00:00Z`), one app at a time, on the M1 Pro (14" XDR, ProMotion 120 Hz, macOS 26.5.1), on mains power (charger attached; battery behaviour not measured). System appearance: **Dark** (the capture builds have no appearance switch and changing System Settings was out of scope), so every frame here is dark mode.

Clips (`clips/`, real time, 120 fps, 1240×1488 px = 620×744 pt): `<gesture>.before.mp4` / `<gesture>.after.mp4`. Per-frame data: `data/{before,after}.csv` (host time, pill centre/width, content offset, match error); per-gesture results: `data/{before,after}.json`; inputs and window-server bounds: `data/{before,after}.events`.

## Results per gesture

Limits: one frame may move the pill at most 1/3 of the slot pitch (BEFORE 86.75 pt → 28.9 pt; AFTER 77.0 pt → 25.7 pt) and the content at most 1/3 of a page (300 pt; one page = the 900 pt window); rest exactly on the destination slot within 500 ms of the release (mouse-up for clicks; for BEFORE this includes the window's height animation).

Columns: **first motion** = frames (120 Hz) from the input to the first frame that moved (swipes: from the scroll event that crossed the 6 pt axis slop; clicks: from mouse-down; the drag: from its first drag event). **Stall** = for swipes and the drag, time from the fingers lifting to the first frame that moved after it, and that frame's pill / content step; for clicks, mouse-up (selection) to the first moved frame and its pill / content step. **Max step** = largest move between two consecutive frames, pt (in tabs). **Repeats / jumps** = frames where neither pill nor content moved between moving neighbours (a frame the app did not draw), and steps over twice both neighbours (a catch-up), both over one-frame intervals only. **Rest** = ms from release to the first frame after which pill centre and width are within 0.5 pt of the slot's own resting values and the content within 1 pt. **Labels** = largest change of any tab label's ink extent against the gesture's first frame, and frames over 0.5 pt.

| gesture | build | tabs | first motion (fr) | stall | max pill step pt (tabs) | max content step pt (tabs) | repeats / jumps | rest ms | window height changes | labels | limits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g1a slow swipe Overview→Backups (1.5 s, 600 pt) | before | Overview→Backups | 179.0 | 78.6 ms, then 93.5 / 900.0 pt | 93.5 (1.08) | 900.0 (1.00) | 0 / 0 | 219.4 | yes | 2.0 pt, 136 fr | FAIL |
| g1a | after | Overview→Backups | 4.9 | 94.6 ms, then 14.5 / 155.0 pt | 14.5 (0.19) | 155.0 (0.17) | 0 / 2 | 377.9 | no | 0.0 pt, 0 fr | pass |
| g1b slow swipe Backups→Jobs (1.5 s, 600 pt) | before | Backups→Jobs | 182.4 | 103.4 ms, then 80.0 / 900.0 pt | 80.0 (0.92) | 900.0 (1.00) | 0 / 0 | 128.6 | yes | 1.0 pt, 132 fr | FAIL |
| g1b | after | Backups→Jobs | 11.9 | 44.6 ms, then 4.0 / 49.5 pt | 4.0 (0.05) | 49.5 (0.06) | 1 / 5 | 378.0 | no | 0.5 pt, 0 fr | pass |
| g2 fast flick Jobs→Backups (216 pt in 83 ms + momentum) | before | Jobs→Backups | 17.0 | 61.8 ms, then 80.0 / 900.0 pt | 80.0 (0.92) | 900.0 (1.00) | 0 / 0 | 210.6 | yes | 1.0 pt, 174 fr | FAIL |
| g2 | after | Jobs→Backups | 4.4 | 2.8 ms, then 2.5 / 30.0 pt | 21.2 (0.28) | 274.5 (0.30) | 0 / 3 | 411.1 | no | 0.0 pt, 0 fr | pass |
| g3 click a far tab Overview→Jobs | before | Overview→Jobs | 19.6 | 101.8 ms, then 173.5 / 1800.0 pt | 173.5 (2.00) | 1800.0 (2.00) | 0 / 0 | 219.6 | yes | 1.5 pt, 124 fr | FAIL |
| g3 | after | Overview→Jobs | 20.3 | 110.9 ms, then 100.0 / 1105.4 pt | 100.0 (1.30) | 1105.4 (1.23) | 1 / 2 | 477.5 | no | 0.5 pt, 0 fr | FAIL |
| g4 click Backups, then Jobs 120 ms later | before | Overview→Jobs | 1.7 | moves at the press (13.9 ms after mouse-down), 45.5 / 0.0 pt | 48.0 (0.55) | 900.0 (1.00) | 1 / 3 | 223.5 | yes | 2.5 pt, 154 fr | FAIL |
| g4 | after | Overview→Jobs | 21.3 | 121.3 ms, then 55.2 / 594.0 pt | 55.2 (0.72) | 594.0 (0.66) | 1 / 4 | 447.8 | no | 0.5 pt, 0 fr | FAIL |
| g5 press the pill, drag one tab (Overview→Backups), release | before | Overview→Backups | 38.4 | 68.7 ms, then 48.0 / 900.0 pt | 48.0 (0.55) | 900.0 (1.00) | 0 / 0 | 210.7 | yes | 2.5 pt, 166 fr | FAIL |
| g5 | after | Overview→Backups | 13.3 | on the slot at release; the press swell (starts within 45 ms, 72→75 pt) settles back to 68 pt by 258 ms | 6.2 (0.08) | 66.5 (0.07) | 3 / 7 | 257.8 | no | 0.5 pt, 0 fr | pass |
| g6 held swipe 0.45 tab (405 pt) toward Jobs, still 1 s, release | before | Backups→Jobs | 201.0 | 110.4 ms, then 80.0 / 900.0 pt | 80.0 (0.92) | 900.0 (1.00) | 0 / 0 | 126.6 | yes | 1.0 pt, 135 fr | FAIL |
| g6 | after | Backups→Backups | 11.9 | 19.4 ms, then 0.8 / 9.0 pt | 3.5 (0.04) | 45.0 (0.05) | 0 / 2 | 394.4 | no | 0.5 pt, 0 fr | pass |
| g7 flick past the last tab (Jobs) | before | Jobs→Jobs | — | — | 0.0 (0.00) | 0.0 (0.00) | 0 / 0 | no motion | no | 0.5 pt, 0 fr | pass |
| g7 | after | Jobs→Jobs | 4.9 | 2.9 ms, then 0.0 / 13.5 pt | 0.0 (0.00) | 32.5 (0.04) | 1 / 1 | 361.3 | no | 0.0 pt, 0 fr | pass |
| g8 double flick Overview→Backups→Jobs, 0.5 s apart | before | Overview→Jobs | 17.1 | 95.0 ms, then 80.0 / 900.0 pt | 93.5 (1.08) | 900.0 (1.00) | 0 / 0 | 128.7 | yes | 2.0 pt, 265 fr | FAIL |
| g8 | after | Overview→Jobs | 5.1 | 19.4 ms, then 0.0 / 1.0 pt | 36.5 (0.47) | 435.5 (0.48) | 0 / 6 | 411.1 | no | 0.0 pt, 0 fr | FAIL |

### What fails the limits

- g1a before: pill step 93.5 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g1b before: pill step 80.0 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g2 before: pill step 80.0 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g3 before: pill step 173.5 pt > 28.9; content step 1800.0 pt > 300; window height changes; labels change
- g3 after: pill step 100.0 pt > 25.7; content step 1105.4 pt > 300
- g4 before: pill step 48.0 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g4 after: pill step 55.2 pt > 25.7; content step 594.0 pt > 300
- g5 before: pill step 48.0 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g6 before: pill step 80.0 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g8 before: pill step 93.5 pt > 28.9; content step 900.0 pt > 300; window height changes; labels change
- g8 after: pill step 36.5 pt > 25.7; content step 435.5 pt > 300

BEFORE fails by design (no slide: the page swaps in one frame, the window height animates, the selected label turns bold). The AFTER failures are all one pattern: **when a selection commits, the window draws no new frame for ~45–120 ms, then lands where the spring already is.** Measured: far click Overview→Jobs, 13 consecutive recorded frames (reference strip moving every frame) with no change after mouse-up, then the first drawn frame moves the pill 100 pt and the content 1105 pt — the pill never draws over Backups; slow swipe release, 11 frames without a change, then 14.5 / 155 pt. A swipe that starts on Backups (g1b, g6) draws its first frame ~12 frames after the slop is crossed; from Overview (g1a) ~5.

## Tracking (AFTER)

- slow swipe Overview→Backups: 165 tracking frames; pill width vs the straight line between the two slots' resting widths (as a function of its centre): max deviation 0.46 pt (measurement step 0.5 pt); largest width change in one frame 0.5 pt; pill progress vs content progress max 0.003 tab.
- slow swipe Backups→Jobs: 158 tracking frames; pill width vs the straight line between the two slots' resting widths (as a function of its centre): max deviation 0.47 pt (measurement step 0.5 pt); largest width change in one frame 1.0 pt; pill progress vs content progress max 0.003 tab.
- held half swipe (incl. the 1 s hold): 176 tracking frames; pill width vs the straight line between the two slots' resting widths (as a function of its centre): max deviation 0.41 pt (measurement step 0.5 pt); largest width change in one frame 1.0 pt; pill progress vs content progress max 0.003 tab.
- Held half swipe, 0.5 s into the hold: pill centre 497.0 pt, width 57.0 pt, content 1305.0 pt (0.45 of the way to Jobs, as the fingers). Released with no velocity it returns to Backups (BEFORE: the same 405 pt commits to Jobs on release, its 88 pt threshold).
- Pill drag (g5): the press starts swelling the pill within 45 ms (72→75 pt). The pill starts following ~110 ms after the pointer starts moving (pointer 7.2 pt along at +100 ms, pill still on its slot; first move 6.2 pt), then tracks it within 2 pt (+200 ms: pointer +23.4, pill +20.3; +600 ms: both +84). 3 frames not drawn mid-drag (repeats at 303, 353, 419 ms), each followed by a ~30–37 pt content catch-up; all within the limits.
- Pill drag (g5): lands at centre 465.5 / width 68.0 pt; Backups' resting pill is 465.5 / 68.0 pt.

## The taste reviewer's checks

- **Glass pill visible against the track, dark: measured, light: NOT CHECKED.** Dark: the pill fill is grey level 34 on a track of 49 (8-bit, contrast ≈1.2:1) with a lighter rim (52–65), see `stills/after-tabbar-rest-overview-x2.png`. Visible, low contrast. Light mode was not recorded: the capture builds have no appearance switch and System Settings was out of bounds. On macOS 26.5.1 the code path is `glassEffect(.regular, in: .capsule)` (GlassTabPager.swift `glassPill`); the pixels cannot prove which API drew them.
- **No double capsule from the system toolbar: pass (measured).** In an 8 pt band around the AFTER track the toolbar takes only grey levels 33 and 34 (a one-level soft shade, no edge).
- **Dropped frame at the first frame of a swipe: FAIL from Backups, pass from Overview.** See first motion / repeats above: g1b and g6 start ~12 frames after the slop with a 19 / 45 pt catch-up on the next frame; g1a starts 5 frames after it (first steps 6, 6, 1, 2 pt: the 6 pt slop handed over, then the fingers).
- **Far click passing Backups: FAIL.** The pill's first drawn frame after mouse-up is already past Backups (centre 481.5 pt vs Backups 465.5 pt); no frame shows it on Backups.
- **Held half swipe: label colour seam: clean (visual, not a number); pill width interpolates: pass (measured above).** `stills/after-tabbar-held-half-swipe-x2.png`: the white/grey split runs exactly along the pill edge through "Backups" and "Jobs", one weight, no doubled glyphs. Labels never moved or resized in any AFTER frame (largest ink-extent change 0.5 pt = one pixel, at the threshold).
- **Pill drag lands at exactly the slot's resting size: pass** (numbers above).
- **Reduce Motion: NOT CHECKED** (a system setting; not changed).

## Method

- **Recorder**: ScreenCaptureKit (`tools/sck.swift`), region = the window's x 280–900 pt, full height, 2 px/pt, cursor drawn, H.264. `ffmpeg -f avfoundation -framerate 120` measured 77–80 fps on this Mac in every variant tried (full screen, cropped, nv12, no cursor, VideoToolbox), so it was not used — see `cadence.txt`. SCK at `minimumFrameInterval` 1/120 s skipped ~1 frame in 10 against the display; at 1/240 s it delivers every 120 Hz frame.
- **Reference strip** (`tools/ref.swift`): a 600×14 pt window below BrainBar's, inside the recorded region, whose bar moves 4 pt per frame as a render-server Core Animation animation. It changes every display frame, so a recorded interval over one frame is a recorder/compositor miss, never the app's; and a one-frame interval where BrainBar did not change is the app's.
- **Why only x ≥ 280 pt**: a system dialog ("“BrainBar” is damaged and can't be opened", CoreServicesUIAgent, plus a second UserNotificationCenter dialog behind it) sits at screen x 626–886, y 189–503 pt. It was not touched. No placement of the 900 pt window avoids it: the app keeps the window inside the screen's visible frame (measured: a window moved to x −280 pt was pulled back to x 0 on the first page change). At (612, 33) the dialog covers window x 14–274 pt only, so the recording starts at x 280. The tab bar (x 322–620) and the right two-thirds of each page are in every frame.
- **Input** (`tools/drv.swift`): CGEvents at the HID tap through the real NSEvent path, 120 per second: scroll phases began/changed/ended plus momentum (began/continue/end) for flicks; mouse down/drag/up. macOS truncates a synthesized point delta to whole points (checked with an event monitor), so deltas are sent as whole points with the remainder carried; total travel is exact (logged). AFTER moves one tab per 900 pt of finger travel. Slow swipe = 600 pt over 1.5 s, ease-in-out (no flick on release); flick = 216 pt in 83 ms (last step 30 pt) + 32 momentum events; held swipe = 405 pt over 0.6 s, still 1 s, release; clicks = 60 ms press; pill drag = press, 150 ms, 0.6 s drag to the Backups slot centre, 100 ms, release. The pointer rests on the page header (window 760, 130 pt) for swipes. One click on that empty header first makes the window key.
- **Clock**: SCK frame times and every input / window-bounds log line are host time (`CACurrentMediaTime`).
- **Measure** (`measure.py`): pill = longest run of columns on the pill's side of the midpoint between the 10th/90th percentile grey levels, in the tab bar's rows above the labels (15–20 pt), inside the track's rim (resolution 0.5 pt); content = masked template match of the page band (y 55–400 pt) against the three rested pages laid 900 pt apart (blurred coarse search + local search, 1 pt fine search, parabolic sub-point; rested frames match at error 0); window height = the window server's bounds polled every 2 ms. Every per-frame value is in the CSVs.
- **Not measured**: light appearance; Reduce Motion; battery.
