"""w308: before / after(388b358b) / after2(13bb20c8) per gesture, from measure.py's JSON + per-frame CSV + the driver's
events, the same rules for all three builds.
Usage: compare.py <before prefix> <after prefix> <after2 prefix> [<after2 repeat prefix>] --out compare.json

Frozen frames at commit: for every commit of a gesture (a click's mouse-up, the drag's mouse-up, the fingers lifting
off a swipe: the driver's `released` lines inside the gesture), the display frames (120 Hz, from the recorder's host
times) after the commit until the first recorded frame whose pill centre (>0.25 pt), pill width (>=0.5 pt) or content
(>0.5 pt) differs from the last frame at or before the commit; only while the gesture has not yet come to rest (if
nothing changes before the rest frame, 0). The largest over the gesture's commits is reported, with its ms."""
import json
import sys

import numpy as np

GEST = {  # key -> (first input name, last input name)
    "g1a": ("g1a-slow-overview-backups",) * 2, "g1b": ("g1b-slow-backups-jobs",) * 2,
    "g2": ("g2-flick-jobs-backups",) * 2, "g3": ("g3-click-far-overview-jobs",) * 2,
    "g4": ("g4a-click-backups", "g4b-click-jobs-120ms"), "g5": ("g5-pill-drag-overview-backups",) * 2,
    "g6": ("g6-held-half-backups-jobs",) * 2, "g7": ("g7-flick-past-last-jobs",) * 2,
    "g8": ("g8a-double-flick-1", "g8b-double-flick-2"), "g9c": ("g9a-click-far-overview-jobs",) * 2,
    "g9": ("g9b-flick-after-rest",) * 2,
}
SWIPES = ("g1a", "g1b", "g2", "g6", "g7", "g8", "g9")


def load(prefix):
    d = json.load(open(prefix + ".json"))
    c = np.loadtxt(prefix + ".csv", delimiter=",", skiprows=1)
    ev = [l.split() for l in open(prefix + ".events") if len(l.split()) > 1]
    return d, c, ev


def frozen(c, ev, key, rest_ms):
    first, last = GEST[key]
    names = {first, last}
    rel = sorted(float(e[0]) for e in ev if e[1] == "released" and e[2] in names)
    t = c[:, 0]
    best = (0, 0.0)
    t_last_rel = rel[-1]
    t_rest = t_last_rel + rest_ms / 1000 if rest_ms is not None else t[-1]
    for tc in rel:
        k0 = int(np.searchsorted(t, tc) - 1)
        j = k0 + 1
        while j < len(t) and t[j] < t_rest:
            if (abs(c[j, 1] - c[k0, 1]) > 0.25 or abs(c[j, 2] - c[k0, 2]) >= 0.5 or abs(c[j, 3] - c[k0, 3]) > 0.5):
                break
            j += 1
        if j >= len(t) or t[j] >= t_rest:
            continue  # nothing moves after this commit before rest
        n = int(round((t[j] - tc) * 120 - 0.5))  # display frames strictly between the commit and the changed frame
        if n > best[0]:
            best = (n, round((t[j] - tc) * 1000, 1))
    return best


def row(prefix):
    d, c, ev = load(prefix)
    out = {}
    for g in d["gestures"]:
        k = g["key"]
        rest = g["rest_all_after_release_ms"]
        fz = frozen(c, ev, k, rest) if g["first_motion_ms"] is not None else (0, 0.0)
        out[k] = {"max_pill_step_pt": g["max_pill_step_pt"], "max_content_step_pt": g["max_content_step_pt"],
                  "frozen_frames_at_commit": fz[0], "frozen_ms": fz[1],
                  "rest_ms": rest if g["first_motion_ms"] is not None else None, "moved": g["first_motion_ms"] is not None,
                  "height_change": g["window_height_changed"],
                  "first_motion_after_slop_frames": g["first_motion_after_slop_frames"] if k in SWIPES else None,
                  "start_tab": g["start_tab"], "lands_on": g["lands_on"], "repeats": len(g["app_repeats"]),
                  "jumps": len(g["app_jumps"]), "labels": [g["label_max_shift_pt"], g["label_frames_changed"]],
                  "first_steps": g["first_steps"]}
    return {"pitch": d["slot_pitch_pt"], "cadence": d["cadence"], "levels": d["pill_levels_at_first_rest"],
            "rest_pill": d["rest_pill"], "g": out}


args = sys.argv[1:]
outp = args[args.index("--out") + 1]
pre = [a for a in args[:args.index("--out")]]
res = {name: row(p) for name, p in zip(("before", "after", "after2", "after2_repeat"), pre)}
json.dump(res, open(outp, "w"), indent=1, ensure_ascii=False)
for name, r in res.items():
    print(name, "pitch", r["pitch"])
    for k, v in r["g"].items():
        print(" ", k, v["start_tab"], "->", v["lands_on"], "pill", v["max_pill_step_pt"], "cont", v["max_content_step_pt"],
              "frozen", v["frozen_frames_at_commit"], v["frozen_ms"], "rest", v["rest_ms"], "H", v["height_change"],
              "fm/slop", v["first_motion_after_slop_frames"], "rep/jmp", v["repeats"], v["jumps"])
