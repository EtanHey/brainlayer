# Tabs follow the fingers — AFTER-2 (13bb20c8) re-recorded, compared 1:1

PR #1896 after its round-2 fix (`13bb20c8`, "no frame is lost when a tab selection commits"). Recorder w308, M1 Pro (14" XDR, ProMotion 120 Hz, macOS 26.5.1), mains power, appearance **Dark** (not changed). App: `.build.noindex/bbui-phase18/app-after2-13bb20c8/BrainBarCapture.app` (Info.plist GitCommit 13bb20c8), fixture `calm`, launched fresh before each run, window at (612, 33) 900×742 pt as before (window server, constant for the whole of both runs).

**Same tools and gestures as w300** (`../tools/`): ScreenCaptureKit recorder, minimumFrameInterval 1/240 s, region window x 280–900 pt, reference strip inside the region; HID-tap point-delta driver with the same deltas, timings and pointer positions (`TABS=overview:380,backups:464,jobs:533.5`, read back from w300's after.events); the same `measure.py`. Changes, all in `tools/` here: `drv.swift` gains step 9; `record.sh` records 38 s (was 33) for it; `measure.py` gains build `after2` (the round-2 pill is *lighter* than the track, so its polarity flips; label ink threshold 100 with ±8 px around the pill's edges masked, because the new rim reaches grey ~160, as bright as an unselected label) and the g9 rows. `compare.py` computes the frozen-frames column with one rule for all three builds. BEFORE (`7ee776fb`) and AFTER (`388b358b`) numbers are w300's own recordings (`data/w300-*`), not re-recorded.

Clips: `clips/<gesture>.after2.mp4` (real time, 120 fps, 1240×1488 px). Per-frame: `data/after2.csv` (evidence run), `data/after2-repeat.csv` (a second full run, also fresh-launched); per-gesture: `data/after2*.json`; inputs and window bounds: `data/after2*.events`; comparison: `data/compare.json`.

## Comparison (cells: before / after 388b358b / **after2 13bb20c8**, evidence run)

Limits: pill step ≤ 1/3 slot pitch (after2 76.75 pt → 25.6 pt; after 25.7; before 28.9), content step ≤ 300 pt, at rest ≤ 500 ms after the release (mouse-up for clicks), window height constant, a swipe's first motion ≤ 2 display frames after its 6 pt slop is crossed. **Frozen frames at commit** = display frames after a commit (mouse-up / fingers lifting) until the first recorded frame where the pill centre, pill width or content changed, counted only while the gesture is not yet at rest; the largest over its commits.

| gesture | max pill step pt | max content step pt | frozen frames at commit (ms) | time to rest ms | window height change | first motion after slop, frames (after2) |
|---|---|---|---|---|---|---|
| g1a slow swipe Overview→Backups (1.5 s, 600 pt) | 93.5 / 14.5 / 2.0 | 900.0 / 155.0 / 19.0 | 9 (78.6) / 11 (94.6) / 3 (27.9) | 219.4 / 377.9 / 377.9 | yes / no / no | 3.4 |
| g1b slow swipe Backups→Jobs (1.5 s, 600 pt) | 80.0 / 4.0 / 1.8 | 900.0 / 49.5 / 19.6 | 12 (103.4) / 5 (44.6) / 3 (27.9) | 128.6 / 378.0 / 377.9 | yes / no / no | 2.2 |
| g2 fast flick Jobs→Backups | 80.0 / 21.2 / 6.5 | 900.0 / 274.5 / 81.0 | 7 (61.8) / 0 (0.0) / 0 (0.0) | 210.6 / 411.1 / 411.3 | yes / no / no | 2.1 |
| g3 click a far tab Overview→Jobs | 173.5 / 100.0 / 10.8 | 1800.0 / 1105.4 / 115.5 | 12 (101.8) / 13 (110.8) / 3 (28.5) | 219.6 / 477.5 / 470.2 | yes / no / no | — |
| g4 click Backups, then Jobs 120 ms later | 48.0 / 55.2 / 8.0 | 900.0 / 594.0 / 104.5 | 9 (83.3) / 14 (121.3) / 2 (20.0) | 223.5 / 447.8 / 447.9 | yes / no / no | — |
| g5 press the pill, drag one tab, release | 48.0 / 6.2 / 2.2 | 900.0 / 66.5 / 38.6 | 8 (68.7) / 12 (107.8) / 12 (103.1) | 210.7 / 257.8 / 253.1 | yes / no / no | — |
| g6 held swipe 0.45 tab toward Jobs, still 1 s, release | 80.0 / 3.5 / 2.2 | 900.0 / 45.0 / 26.0 | 13 (110.4) / 2 (19.4) / 3 (26.7) | 126.6 / 394.4 / 393.3 | yes / no / no | 2.4 |
| g7 flick past the last tab (Jobs) | 0.0 / 0.0 / 0.0 | 0.0 / 32.5 / 30.5 | 0 (0.0) / 0 (0.0) / 0 (0.0) | no motion / 361.3 / 361.2 | no / no / no | 2.1 |
| g8 double flick Overview→Backups→Jobs, 0.5 s apart | 93.5 / 36.5 / 12.0 | 900.0 / 435.5 / 156.0 | 11 (95.0) / 2 (19.4) / 0 (0.0) | 128.7 / 411.1 / 411.2 | yes / no / no | 2.9 |
| g9c g9's far click Overview→Jobs (as g3) | — / — / 10.8 | — / — / 115.5 | — / — / 3 (26.2) | — / — / 476.2 | — / — / no | — |
| g9 **new:** flick Jobs→Backups (as g2) started 100 ms after a far click rests | — / — / 6.5 | — / — / 81.5 | — / — / 0 (0.0) | — / — / 410.8 | — / — / no | 3.1 |

### Repeat run (after2, second fresh launch) — the same columns, to show the spread

| gesture | max pill step | max content step | frozen frames (ms) | rest ms | height change | first motion after slop |
|---|---|---|---|---|---|---|
| g1a | 2.0 | 19.5 | 3 (25.7) | 375.7 | no | 3.1 |
| g1b | 1.8 | 19.0 | 3 (25.5) | 375.5 | no | 1.8 |
| g2 | 6.2 | 82.5 | 0 (0.0) | 408.8 | no | 2.7 |
| g3 | 21.5 | 229.5 | 3 (27.1) | 477.1 | no | — |
| g4 | 10.2 | 111.0 | 2 (25.0) | 447.8 | no | — |
| g5 | 3.5 | 38.6 | 17 (141.9) | 300.2 | no | — |
| g6 | 3.5 | 39.5 | 2 (22.8) | 397.8 | no | 1.7 |
| g7 | 0.0 | 29.0 | 0 (0.0) | 358.8 | no | 2.7 |
| g8 | 9.8 | 127.0 | 0 (0.0) | 409.0 | no | 2.7 |
| g9c | 10.8 | 115.0 | 3 (27.3) | 477.3 | no | — |
| g9 | 7.5 | 89.5 | 0 (0.0) | 414.0 | no | 2.3 |

## What still fails (after2)

- after2 g1a: first motion 3.4 frames after the slop > 2
- after2 g1b: first motion 2.2 frames after the slop > 2
- after2 g2: first motion 2.1 frames after the slop > 2
- after2 g6: first motion 2.4 frames after the slop > 2
- after2 g7: first motion 2.1 frames after the slop > 2
- after2 g8: first motion 2.9 frames after the slop > 2
- after2 g9: first motion 3.1 frames after the slop > 2
- after2_repeat g1a: first motion 3.1 frames after the slop > 2
- after2_repeat g2: first motion 2.7 frames after the slop > 2
- after2_repeat g7: first motion 2.7 frames after the slop > 2
- after2_repeat g8: first motion 2.7 frames after the slop > 2
- after2_repeat g9: first motion 2.3 frames after the slop > 2

Everything else passes in both runs: max pill step, max content step, rest ≤ 500 ms, window height constant.

**The only limit that fails is the ≤ 2-frame first motion, and 2 frames is about the platform's floor.** The same recorder and the same HID-tap scroll event, sent to a bare borderless window whose CALayer changes colour on the event (`tools/probe.swift`, `tools/pdrv.swift`, no SwiftUI): post->first changed recorded frame, display frames (120 Hz): n 40 min 1.35 p50 1.80 mean 1.81 max 2.56. So a swipe's first motion at 2.1–3.4 frames (after: 4.4–11.9) is 0.3–1.6 frames above the probe's median, and a 2.0 limit fails 11 of the probe's own 40 samples too (any value up to 2.56 is the platform).

## g9: a swipe 100 ms after a far click rests

The far click before it rests 476.2 ms after mouse-up (repeat 477.3 ms; measured g3: 470.2), and the flick's first event went out 99.1 ms after that rest (repeat 106.2 ms; `G9_DELAY=0.577` s after mouse-up from the first run's g3/g9c rest). The flick is g2's (216 pt in 83 ms + momentum), so g2 is its control:

| | first motion after slop (frames) | first steps pill / content pt | frozen at release | rest ms | max pill / content step |
|---|---|---|---|---|---|
| g9 evidence | 3.1 | [1.0, 0.8, 2.5, 1.8] / [9.0, 12.0, 33.0, 21.0] | 0 | 410.8 | 6.5 / 81.5 |
| g2 evidence | 2.1 | [1.0, 0.8, 1.2, 1.2] / [9.0, 12.0, 15.0, 18.0] | 0 | 411.3 | 6.5 / 81.0 |
| g9 repeat | 2.3 | [1.0, 0.8, 1.2, 1.2] / [9.0, 12.0, 15.0, 18.0] | 0 | 414.0 | 7.5 / 89.5 |
| g2 repeat | 2.7 | [1.8, 0.0, 2.5, 1.8] / [21.0, 0.0, 33.0, 21.0] | 0 | 408.8 | 6.2 / 82.5 |

Per frame (content offset from `data/after2*.csv`): in the evidence run the recorder missed the display frame at +2.1 frames after g9's first event (a 2-frame recorded interval), and g9's first drawn motion, the flick's own first 9 pt, is on the next frame at +3.1; its steps then run exactly one frame behind g2's (g2: 1791 at +2.1, 1779 at +3.1; g9: 1791 at +3.1, 1779 at +4.1). In the repeat run g9 moves at +2.3 with g2-evidence's exact step sequence, while g2-repeat itself skips one frame at its start (1779 twice). So one of two g9 runs is one frame late, on the same frame the recorder missed; the other is not. Not a consistent delay at the measured resolution: the one late start cannot be told apart from the recorder's own miss on that frame.

## Commit freeze (the round-1 failure)

AFTER's 11–14 frozen frames (95–121 ms) at a slow swipe's release (g1a) and a click's mouse-up (g3, g4) are gone: after2 shows 0–3 frames for every gesture but g5: the first change lands 20–29 ms after the commit (the probe's bare input-to-screen latency is 11–21 ms). The far click now draws the pill passing Backups: one frame within 3 pt of its slot in each far click of both runs (after: the nearest drawn frame was 16 pt past it, after a 100 pt jump); max pill step 10.8 pt.

**Still there: the pill-drag release (g5).** After mouse-up the pill is already on the Backups slot and the content at 900 pt, but the pressed pill keeps its swollen width (76.5 pt) for 12 frames / 103.1 ms (repeat 17 / 141.9 ms; after: 12 / 107.8 ms) before shrinking to its resting 73.5 pt, at rest 253.1 ms after release. Within every limit; nothing else moves in that time, so pixels cannot tell a delayed un-press from frames the app did not draw.

## Selected pill visibility (dark, the current appearance)

At rest on Overview, in the pill band (window y 15–20 pt): pill fill grey **90** on a track of **49** (8-bit), luminance contrast **1.9:1**, plus a rim up to grey ~156–161 (4.9:1 on the track). AFTER was 34 on 49 (1.2:1, a darker pill). **Clearly visible**: the selected pill reads at a glance as a lighter capsule with a bright edge (`stills/after2-tabbar-rest-overview-x2.png`). Held half swipe (`stills/after2-tabbar-held-half-swipe-x2.png`): the white/grey label split runs along the pill edge through "Backups" and "Jobs", no doubled glyphs. Light appearance: NOT CHECKED (not switched).

## Labels

Largest label ink-extent change, after2: g1a 1.5 pt / 11 fr, g1b 1.5 pt / 7 fr, g2 1.5 pt / 1 fr, g3 1.5 pt / 3 fr, g4 1.5 pt / 2 fr, g5 1.5 pt / 4 fr, g6 1.5 pt / 3 fr, g7 0.0 pt / 0 fr, g8 1.5 pt / 3 fr, g9c 0.5 pt / 0 fr, g9 1.5 pt / 2 fr. These are the 0.5–1.5 pt edges of a label where the pill's edge crosses it (the mask is ±8 px); not a limit.

## Screen

The two system dialogs (CoreServicesUIAgent "“BrainBar” is damaged…", 260×280 at x 626 y 201; UserNotificationCenter, 260×314 at x 626 y 189) were still on screen at the start and the end of the session, untouched; the window was placed at (612, 33) as w300 did, so they cover window x 14–274 pt, left of the recorded region.
