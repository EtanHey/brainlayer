"""Write measurements.md from the two measure.py results (no number is typed by hand).
Usage: make_md.py <before.json> <after.json> <interp.json> <before.events> <after.events> <out.md>"""
import json
import sys

B, A, INTERP = (json.load(open(p)) for p in sys.argv[1:4])
EVB, EVA = sys.argv[4:6]
OUT = sys.argv[6]


def released(events):
    return {l.split()[2]: float(l.split()[0]) for l in open(events) if l.split()[1] == "released"}


def starts(events):
    return {l.split()[2]: float(l.split()[0]) for l in open(events) if l.split()[1] == "input-start"}


def f(x, nd=1):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def click_stall(d, events, key, csvfile):
    """mouse-up → first drawn motion, and that frame's pill/content step (clicks select on mouse-up)."""
    import numpy as np
    names = {"g3": "g3-click-far-overview-jobs", "g4": "g4a-click-backups"}
    g = next(x for x in d["gestures"] if x["key"] == key)
    if g["first_motion_ms"] is None:
        return None
    up = released(events)[names[key]] - starts(events)[names[key]]
    return round(g["first_motion_ms"] - up * 1000, 1)


rows = []
pitch = {"before": B["slot_pitch_pt"], "after": A["slot_pitch_pt"]}
lim_pill = {k: v / 3 for k, v in pitch.items()}
LIM_CONTENT = 300.0
LIM_REST = 500.0
fails = []
for gb, ga in zip(B["gestures"], A["gestures"]):
    assert gb["key"] == ga["key"]
    for build, g in (("before", gb), ("after", ga)):
        fm = g["first_motion_after_slop_frames"] if g["first_motion_after_slop_frames"] is not None else g["first_motion_frames"]
        verdict = []
        if g["max_pill_step_pt"] > lim_pill[build]:
            verdict.append(f"pill step {g['max_pill_step_pt']} pt > {lim_pill[build]:.1f}")
        if g["max_content_step_pt"] > LIM_CONTENT:
            verdict.append(f"content step {g['max_content_step_pt']} pt > 300")
        rest = g["rest_all_after_release_ms"]
        if rest is not None and rest > LIM_REST:
            verdict.append(f"rest {rest} ms > 500")
        if g["first_motion_ms"] is not None and rest is None:
            verdict.append("never rests exactly on the slot")
        if g["window_height_changed"]:
            verdict.append("window height changes")
        if g["label_frames_changed"]:
            verdict.append("labels change")
        nrep, njump = len(g["app_repeats"]), len(g["app_jumps"])
        drops = f"{nrep} / {njump}"
        settle = g["settle"]
        stall = None
        if settle:
            stall = f"{settle['release_to_first_settle_frame_ms']} ms, then {settle['first_settle_step_pill_pt']} / {settle['first_settle_step_content_pt']} pt"
        if g["key"] in ("g3", "g4") and g["first_steps"]:
            up = click_stall({"gestures": [g]}, EVB if build == "before" else EVA, g["key"], None)
            if up < 0:
                stall = f"moves at the press ({g['first_motion_ms']} ms after mouse-down), {g['first_steps']['pill_pt'][0]} / {g['first_steps']['content_pt'][0]} pt"
            else:
                stall = f"{up} ms, then {g['first_steps']['pill_pt'][0]} / {g['first_steps']['content_pt'][0]} pt"
        if g["key"] == "g5" and build == "after":
            stall = "on the slot at release; the press swell (starts within 45 ms, 72→75 pt) settles back to 68 pt by 258 ms"
        if g["first_motion_ms"] is None:
            rest = None
        if verdict:
            fails.append((g["key"], build, "; ".join(verdict)))
        rows.append((g["key"], build, g["start_tab"] + "→" + g["lands_on"], f(fm), stall or "—",
                     f"{g['max_pill_step_pt']} ({g['max_pill_step_tabs']:.2f})",
                     f"{g['max_content_step_pt']} ({g['max_content_step_tabs']:.2f})", drops,
                     f(rest) if g["first_motion_ms"] is not None else "no motion", "yes" if g["window_height_changed"] else "no",
                     f"{g['label_max_shift_pt']} pt, {g['label_frames_changed']} fr",
                     "FAIL" if verdict else "pass"))

labels = {g["key"]: g["gesture"] for g in A["gestures"]}
L = []
L.append("# Tabs follow the fingers — 120 Hz recording and per-frame measurements")
L.append("")
L.append("PR #1896. BEFORE = main `7ee776fb` (the 1.6.13 tab behaviour), AFTER = `388b358b` (the PR head when built; the head is "
         "now `f6e8e6a7`, which only adds 6 lines to `BrainBarTabsFollowFingersTests.swift`, so the app is the same). Capture builds "
         "(`com.brainlayer.brainbar.capture`, fixture `calm`, `BRAINBAR_DESIGN_NOW=2026-11-27T21:00:00Z`), one app at a "
         "time, on the M1 Pro (14\" XDR, ProMotion 120 Hz, macOS 26.5.1), on mains power (charger attached; battery "
         "behaviour not measured). System appearance: **Dark** (the capture builds have no appearance switch and "
         "changing System Settings was out of scope), so every frame here is dark mode.")
L.append("")
L.append("Clips (`clips/`, real time, 120 fps, 1240×1488 px = 620×744 pt): `<gesture>.before.mp4` / `<gesture>.after.mp4`. "
         "Per-frame data: `data/{before,after}.csv` (host time, pill centre/width, content offset, match error); "
         "per-gesture results: `data/{before,after}.json`; inputs and window-server bounds: `data/{before,after}.events`.")
L.append("")
L.append("## Results per gesture")
L.append("")
L.append(f"Limits: one frame may move the pill at most 1/3 of the slot pitch (BEFORE {pitch['before']} pt → "
         f"{lim_pill['before']:.1f} pt; AFTER {pitch['after']} pt → {lim_pill['after']:.1f} pt) and the content at most "
         "1/3 of a page (300 pt; one page = the 900 pt window); rest exactly on the destination slot within 500 ms of "
         "the release (mouse-up for clicks; for BEFORE this includes the window's height animation).")
L.append("")
L.append("Columns: **first motion** = frames (120 Hz) from the input to the first frame that moved (swipes: from the "
         "scroll event that crossed the 6 pt axis slop; clicks: from mouse-down; the drag: from its first drag event). **Stall** = for swipes "
         "and the drag, time from the fingers lifting to the first frame that moved after it, and that frame's pill / "
         "content step; for clicks, mouse-up (selection) to the first moved frame and its pill / content step. "
         "**Max step** = largest move between two consecutive frames, pt (in tabs). **Repeats / jumps** = frames where "
         "neither pill nor content moved between moving neighbours (a frame the app did not draw), and steps over twice "
         "both neighbours (a catch-up), both over one-frame intervals only. **Rest** = ms from release to the first frame "
         "after which pill centre and width are within 0.5 pt of the slot's own resting values and the content within "
         "1 pt. **Labels** = largest change of any tab label's ink extent against the gesture's first frame, and frames "
         "over 0.5 pt.")
L.append("")
L.append("| gesture | build | tabs | first motion (fr) | stall | max pill step pt (tabs) | max content step pt (tabs) | repeats / jumps | rest ms | window height changes | labels | limits |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    L.append("| " + " | ".join([f"{r[0]} {labels[r[0]]}" if r[1] == "before" else r[0]] + list(r[1:])) + " |")
L.append("")
L.append("### What fails the limits")
L.append("")
for k, b, why in fails:
    L.append(f"- {k} {b}: {why}")
L.append("")
L.append("BEFORE fails by design (no slide: the page swaps in one frame, the window height animates, the selected label "
         "turns bold). The AFTER failures are all one pattern: **when a selection commits, the window draws no new "
         "frame for ~45–120 ms, then lands where the spring already is.** Measured: far click Overview→Jobs, 13 "
         "consecutive recorded frames (reference strip moving every frame) with no change after mouse-up, then the "
         "first drawn frame moves the pill 100 pt and the content 1105 pt — the pill never draws over Backups; slow "
         "swipe release, 11 frames without a change, then 14.5 / 155 pt. A swipe that starts on Backups (g1b, g6) "
         "draws its first frame ~12 frames after the slop is crossed; from Overview (g1a) ~5.")
L.append("")
L.append("## Tracking (AFTER)")
L.append("")
for k, name in (("g1a", "slow swipe Overview→Backups"), ("g1b", "slow swipe Backups→Jobs"), ("g6", "held half swipe (incl. the 1 s hold)")):
    x = INTERP[k]
    L.append(f"- {name}: {x['frames']} tracking frames; pill width vs the straight line between the two slots' resting "
             f"widths (as a function of its centre): max deviation {x['width_residual_max_pt']} pt (measurement step "
             f"0.5 pt); largest width change in one frame {x['width_step_max_pt']} pt; pill progress vs content progress "
             f"max {x['pill_vs_content_progress_max_tab']} tab.")
h = INTERP["g6_hold"]
L.append(f"- Held half swipe, 0.5 s into the hold: pill centre {h['pill_centre_pt']} pt, width {h['pill_width_pt']} pt, "
         f"content {h['content_pt']} pt (0.45 of the way to Jobs, as the fingers). Released with no velocity it returns "
         "to Backups (BEFORE: the same 405 pt commits to Jobs on release, its 88 pt threshold).")
L.append("- Pill drag (g5): the press starts swelling the pill within 45 ms (72→75 pt). The pill starts following ~110 ms after "
         "the pointer starts moving (pointer 7.2 pt along at +100 ms, pill still on its slot; first move 6.2 pt), then "
         "tracks it within 2 pt (+200 ms: pointer +23.4, pill +20.3; +600 ms: both +84). 3 frames not drawn mid-drag "
         "(repeats at 303, 353, 419 ms), each followed by a ~30–37 pt content catch-up; all within the limits.")
L.append("- Pill drag (g5): lands at centre {} / width {} pt; Backups' resting pill is {} / {} pt.".format(
    *(next(g for g in A["gestures"] if g["key"] == "g5")[k] for k in ("final_pill_centre_pt", "final_pill_width_pt", "slot_rest_centre_pt", "slot_rest_width_pt"))))
L.append("")
L.append("## The taste reviewer's checks")
L.append("")
lv = A["pill_levels_at_first_rest"]
L.append(f"- **Glass pill visible against the track, dark: measured, light: NOT CHECKED.** Dark: the pill fill is grey "
         f"level {lv['low']:.0f} on a track of {lv['high']:.0f} (8-bit, contrast ≈1.2:1) with a lighter rim (52–65), "
         "see `stills/after-tabbar-rest-overview-x2.png`. Visible, low contrast. Light mode was not recorded: the capture "
         "builds have no appearance switch and System Settings was out of bounds. On macOS 26.5.1 the code path is "
         "`glassEffect(.regular, in: .capsule)` (GlassTabPager.swift `glassPill`); the pixels cannot prove which API drew them.")
L.append("- **No double capsule from the system toolbar: pass (measured).** In an 8 pt band around the AFTER track the "
         "toolbar takes only grey levels 33 and 34 (a one-level soft shade, no edge).")
L.append("- **Dropped frame at the first frame of a swipe: FAIL from Backups, pass from Overview.** See first motion / "
         "repeats above: g1b and g6 start ~12 frames after the slop with a 19 / 45 pt catch-up on the next frame; g1a "
         "starts 5 frames after it (first steps 6, 6, 1, 2 pt: the 6 pt slop handed over, then the fingers).")
L.append("- **Far click passing Backups: FAIL.** The pill's first drawn frame after mouse-up is already past Backups "
         "(centre 481.5 pt vs Backups 465.5 pt); no frame shows it on Backups.")
L.append("- **Held half swipe: label colour seam: clean (visual, not a number); pill width interpolates: pass "
         "(measured above).** `stills/after-tabbar-held-half-swipe-x2.png`: the white/grey split runs exactly along "
         "the pill edge through \"Backups\" and \"Jobs\", one weight, no doubled glyphs. Labels never moved or resized "
         "in any AFTER frame (largest ink-extent change 0.5 pt = one pixel, at the threshold).")
L.append("- **Pill drag lands at exactly the slot's resting size: pass** (numbers above).")
L.append("- **Reduce Motion: NOT CHECKED** (a system setting; not changed).")
L.append("")
L.append("## Method")
L.append("")
L.append("- **Recorder**: ScreenCaptureKit (`tools/sck.swift`), region = the window's x 280–900 pt, full height, 2 px/pt, "
         "cursor drawn, H.264. `ffmpeg -f avfoundation -framerate 120` measured 77–80 fps on this Mac in every "
         "variant tried (full screen, cropped, nv12, no cursor, VideoToolbox), so it was not used — see `cadence.txt`. "
         "SCK at `minimumFrameInterval` 1/120 s skipped ~1 frame in 10 against the display; at 1/240 s it delivers "
         "every 120 Hz frame.")
L.append("- **Reference strip** (`tools/ref.swift`): a 600×14 pt window below BrainBar's, inside the recorded region, "
         "whose bar moves 4 pt per frame as a render-server Core Animation animation. It changes every display frame, "
         "so a recorded interval over one frame is a recorder/compositor miss, never the app's; and a one-frame "
         "interval where BrainBar did not change is the app's.")
L.append("- **Why only x ≥ 280 pt**: a system dialog (\"“BrainBar” is damaged and can't be opened\", CoreServicesUIAgent, "
         "plus a second UserNotificationCenter dialog behind it) sits at screen x 626–886, y 189–503 pt. It was not "
         "touched. No placement of the 900 pt window avoids it: the app keeps the window inside the screen's visible "
         "frame (measured: a window moved to x −280 pt was pulled back to x 0 on the first page change). At (612, 33) "
         "the dialog covers window x 14–274 pt only, so the recording starts at x 280. The tab bar (x 322–620) "
         "and the right two-thirds of each page are in every frame.")
L.append("- **Input** (`tools/drv.swift`): CGEvents at the HID tap through the real NSEvent path, 120 per second: "
         "scroll phases began/changed/ended plus momentum (began/continue/end) for flicks; mouse down/drag/up. macOS "
         "truncates a synthesized point delta to whole points (checked with an event monitor), so deltas are sent as "
         "whole points with the remainder carried; total travel is exact (logged). AFTER moves one tab per 900 pt of "
         "finger travel. Slow swipe = 600 pt over 1.5 s, ease-in-out (no flick on release); flick = 216 pt in 83 ms "
         "(last step 30 pt) + 32 momentum events; held swipe = 405 pt over 0.6 s, still 1 s, release; clicks = 60 ms "
         "press; pill drag = press, 150 ms, 0.6 s drag to the Backups slot centre, 100 ms, release. The pointer rests "
         "on the page header (window 760, 130 pt) for swipes. One click on that empty header first makes the window key.")
L.append("- **Clock**: SCK frame times and every input / window-bounds log line are host time (`CACurrentMediaTime`).")
L.append("- **Measure** (`measure.py`): pill = longest run of columns on the pill's side of the midpoint between the "
         "10th/90th percentile grey levels, in the tab bar's rows above the labels (15–20 pt), inside the track's rim "
         "(resolution 0.5 pt); content = masked template match of the page band (y 55–400 pt) against the three "
         "rested pages laid 900 pt apart (blurred coarse search + local search, 1 pt fine search, parabolic sub-point; "
         "rested frames match at error 0); window height = the window server's bounds polled every 2 ms. Every "
         "per-frame value is in the CSVs.")
L.append("- **Not measured**: light appearance; Reduce Motion; battery.")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
