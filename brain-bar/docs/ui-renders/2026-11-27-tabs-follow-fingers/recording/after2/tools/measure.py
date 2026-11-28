#!/usr/bin/env python3
"""w300 per-frame measurement of BrainBar's window tabs (PR #1896 evidence).

Input: one recording made by sck (ScreenCaptureKit, 120 Hz, region = the window's x 280..900 pt, full height, 2 px/pt),
its per-frame host-clock timestamps (<prefix>.pts) and the gesture driver's log (<prefix>.events: every input and the
window server's window bounds, polled every 2 ms, on the same CACurrentMediaTime clock).

Per frame it measures, from pixels only:
  * the selection pill's horizontal centre and width (pt, window coordinates): the tab bar's band just above the
    labels (rows 30..40 px, 15..20 pt), where only the pill and the track are drawn; the pill is the longest run of
    columns on the pill's side of the midpoint between the two levels (after: the glass pill is darker than the track;
    before: the selected segment is lighter than the bar);
  * the page content's horizontal offset (pt): the frame's content band (window y 55..400 pt) template-matched against
    a panorama of the three pages at rest laid side by side 900 pt apart (the window width), cursor box masked,
    columns never seen at rest masked. 0 = Overview, 900 = Backups, 1800 = Jobs. In the BEFORE build the content does
    not slide, so the measure jumps by a whole page in one frame (reported as a swap);
  * the tab labels' ink extents (rows 44..62 px), compared with the gesture's starting frame, cursor columns masked.
The window height comes from the window server (the poller lines), not from pixels.

Usage: measure.py <prefix> <before|after|after2> [--json out.json] [--csv out.csv]
"""
import json
import os
import subprocess
import sys

import numpy as np

W_PX, H_PX = 1240, 1528          # recorded region in pixels (620 x 764 pt; the reference strip at rows 1494..1521)
REGION_X_PT = 280.0              # region's left edge in window points
PX = 2.0                         # pixels per point
PAGE_W = 900.0                   # window width: one page of content travel
BAR_ROWS = (30, 41)              # pill band (px)
LABEL_ROWS = (44, 63)            # label band (px)
BAND_Y = (110, 800, 4)           # content band rows (px): window y 55..400 pt, every 2 pt
BAND_XSTEP = 2                   # content band columns: every 1 pt
CURSOR_HDR = (755, 805, 118, 165)  # header cursor box, window pt (x0, x1, y0, y1)
EXTENT = {"after": (124, 557), "after2": (124, 557), "before": (84, 680)}   # tab control's interior in region px (inside the track's rim)
# w308: AFTER-2 (13bb20c8) tints the glass pill white 26 % on the dark track, so its pill is LIGHTER than the track
POLARITY = {"after": -1, "after2": +1, "before": +1}  # pill darker (-1) or lighter (+1) than its surroundings
FRAME = 1 / 120


def decode(prefix):
    cache = prefix + ".cache.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return z["t"], z["bar"], z["band"], z["ref"]
    t = np.array([float(x) for x in open(prefix + ".pts") if x.strip()])
    assert (t > 0).all(), "the recorder's writer dropped frames"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", prefix + ".mov", "-fps_mode", "passthrough",
                             "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    bars, bands, refs = [], [], []
    size = W_PX * H_PX
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H_PX, W_PX)
        bars.append(f[0:100].copy())
        bands.append(f[BAND_Y[0]:BAND_Y[1]:BAND_Y[2], ::BAND_XSTEP].copy())
        refs.append(f[1506].copy())
    proc.wait()
    assert len(bars) == len(t), f"decoded {len(bars)} frames, {len(t)} timestamps"
    bar, band, ref = np.stack(bars), np.stack(bands), np.stack(refs)
    np.savez(cache, t=t, bar=bar, band=band, ref=ref)
    return t, bar, band, ref


def ref_slots(ref):
    """The reference bar's slot per frame (ref window at x 900 pt, region x 892 pt, 4 pt slots, 150 slots)."""
    out = np.full(len(ref), np.nan)
    for i, r in enumerate(ref):
        on = np.flatnonzero(r > 128)
        if len(on):
            out[i] = (on.mean() / PX - 8 - 2) / 4
    return out


def parse_events(prefix):
    ev, win, cursor = [], [], []
    for line in open(prefix + ".events"):
        p = line.split()
        if len(p) < 2:
            continue
        t, kind = float(p[0]), p[1]
        if kind == "window":
            win.append((t, *map(float, p[2:6])))
        elif kind == "cursor":
            cursor.append((t, float(p[2]), float(p[3])))
        else:
            ev.append((t, kind, p[2] if len(p) > 2 else ""))
    return ev, np.array(win), np.array(cursor)


def runs(mask, close=6):
    """Runs of True as (start, end) inclusive, after closing gaps up to `close` px."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    out, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > close + 1:
            out.append((s, prev)); s = i
        prev = i
    out.append((s, prev))
    return out


def pill(bar, build):
    """Per frame: threshold midway between the 10th and 90th percentile of the band inside the control (adaptive, so
    the window's key/non-key colours both work)."""
    lo, hi = EXTENT[build]
    prof = bar[:, BAR_ROWS[0]:BAR_ROWS[1], :].mean(axis=1)
    centre = np.full(len(bar), np.nan); width = np.full(len(bar), np.nan)
    for i, p in enumerate(prof):
        seg = p[lo:hi]
        a, b = np.percentile(seg, 10), np.percentile(seg, 90)
        if b - a < 6:
            continue
        thr = (a + b) / 2
        m = seg < thr if POLARITY[build] < 0 else seg > thr
        rs = runs(m, close=4)
        if not rs:
            continue
        a, b = max(rs, key=lambda r: r[1] - r[0])
        a += lo; b += lo
        centre[i] = ((a + b + 1) / 2) / PX + REGION_X_PT
        width[i] = (b - a + 1) / PX
    return centre, width


def pill_levels(bar0, build):
    lo, hi = EXTENT[build]
    seg = bar0[BAR_ROWS[0]:BAR_ROWS[1], lo:hi].mean(axis=0)
    a, b = np.percentile(seg, 10), np.percentile(seg, 90)
    return (a + b) / 2, a, b


def build_panorama(band, rest_idx):
    """Pages at rest side by side, 900 pt apart; columns not seen at rest are NaN."""
    R, C = band.shape[1], band.shape[2]
    off = 300
    P = np.full((R, off + 3 * int(PAGE_W) + 600), np.nan, np.float32)
    mask = cursor_mask(R, C)
    for page, i in enumerate(rest_idx):
        img = band[i].astype(np.float32)
        img[mask] = np.nan
        c0 = off + page * int(PAGE_W) + int(REGION_X_PT)
        P[:, c0:c0 + C] = img
    return P, off


def cursor_mask(R, C):
    m = np.zeros((R, C), bool)
    x0, x1, y0, y1 = CURSOR_HDR
    rows = np.arange(R) * BAND_Y[2] / PX + BAND_Y[0] / PX
    cols = np.arange(C) * BAND_XSTEP / PX + REGION_X_PT
    m[np.ix_((rows >= y0) & (rows <= y1), (cols >= x0) & (cols <= x1))] = True
    return m


def _box(img, kx, ky):
    """Box blur (kx columns, ky rows); NaN wherever the box touches NaN."""
    def one(x, k, axis):
        nan = np.isnan(x)
        z = np.where(nan, 0, x)
        pad = [(0, 0), (0, 0)]; pad[axis] = (1, 0)
        cs = np.cumsum(np.pad(z, pad), axis=axis)
        cn = np.cumsum(np.pad(nan.astype(np.int32), pad), axis=axis)
        sl = lambda a, b: (slice(None), slice(a, b)) if axis == 1 else (slice(a, b), slice(None))
        n = x.shape[axis]
        sm = cs[sl(k, n + 1)] - cs[sl(0, n + 1 - k)]
        bad = (cn[sl(k, n + 1)] - cn[sl(0, n + 1 - k)]) > 0
        out = np.where(bad, np.nan, sm / k)
        padout = [(0, 0), (0, 0)]; padout[axis] = (k // 2, k - 1 - k // 2)
        return np.pad(out, padout, constant_values=np.nan)
    out = img
    if kx > 1:
        out = one(out, kx, 1)
    if ky > 1:
        out = one(out, ky, 0)
    return out


class Matcher:
    """Content offset s (pt): the frame's band against the panorama, masked mean squared difference. Candidates: a
    global coarse search on blurred images (6 pt steps) and a local search around the previous frame's offset; each
    refined at full resolution (1 pt), the better one kept, then a parabola through the best three."""

    def __init__(self, P, off, C):
        self.P, self.off, self.C = P, off, C
        self.Pb = _box(P, 7, 3)
        self.prev = None

    def _scores(self, P, f, shifts, rstep, cstep):
        C = self.C
        base = self.off + int(REGION_X_PT)
        lo, hi = int(shifts.min()), int(shifts.max())
        sub = P[::rstep, base + lo: base + hi + C]
        win = np.lib.stride_tricks.sliding_window_view(sub, C, axis=1)[:, shifts - lo, ::cstep]
        d = win - f[::rstep, None, ::cstep]
        v = ~np.isnan(d)
        n = v.sum(axis=(0, 2))
        sc = np.where(n >= 0.15 * d.shape[0] * d.shape[2], np.nansum(d * d, axis=(0, 2)) / np.maximum(n, 1), np.inf)
        return sc, n / (d.shape[0] * d.shape[2])

    def _refine(self, f, s0, lo, hi):
        fine = np.arange(max(lo, s0 - 6), min(hi, s0 + 6) + 1)
        sf, fr = self._scores(self.P, f, fine, 1, 1)
        k = int(np.argmin(sf))
        return fine, sf, fr, k

    def __call__(self, img, lo=-280, hi=2080):
        f = img.astype(np.float32)
        f[cursor_mask(*f.shape)] = np.nan
        fb = _box(f, 7, 3)
        coarse = np.arange(lo, hi + 1, 6)
        sc, _ = self._scores(self.Pb, fb, coarse, 4, 6)
        cands = [int(coarse[int(np.argmin(sc))])]
        if self.prev is not None:
            local = np.arange(max(lo, int(self.prev) - 40), min(hi, int(self.prev) + 40) + 1)
            sl, _ = self._scores(self.P, f, local, 2, 2)
            cands.append(int(local[int(np.argmin(sl))]))
        best = None
        for s0 in cands:
            fine, sf, fr, k = self._refine(f, s0, lo, hi)
            if best is None or sf[k] < best[1][best[3]]:
                best = (fine, sf, fr, k)
        fine, sf, fr, k = best
        s = float(fine[k])
        if 0 < k < len(fine) - 1 and np.isfinite(sf[k - 1:k + 2]).all():
            a, b, c = sf[k - 1], sf[k], sf[k + 1]
            den = a - 2 * b + c
            if den > 0:
                s += 0.5 * (a - c) / den
        self.prev = s
        return s, float(sf[k]), float(fr[k])


# w308: AFTER-2's pill is filled grey 90 with a rim up to grey ~160, as bright as an unselected label (164), so for it the
# label ink threshold is 100 (above the fill) and the columns within 8 px of either pill edge (this frame's and the
# baseline frame's, the same mask on both) are not compared.
LABEL_THR = {"before": 85, "after": 85, "after2": 100}


def label_runs(bar_frame, cursor_x=None, thr=85, mask=None):
    prof = bar_frame[LABEL_ROWS[0]:LABEL_ROWS[1]].max(axis=0).astype(int)
    ink = prof > thr
    if cursor_x is not None:
        cx = (cursor_x - REGION_X_PT) * PX
        ink[max(0, int(cx - 30)):int(cx + 60)] = False
    if mask is not None:
        ink[mask] = False
    return runs(ink, close=14)


def edge_mask(n, pills):
    m = np.zeros(n, bool)
    for c, w in pills:
        if not (np.isfinite(c) and np.isfinite(w)):
            continue
        for x in (c - w / 2, c + w / 2):
            px = int(round((x - REGION_X_PT) * PX))
            m[max(0, px - 8):max(0, px + 9)] = True
    return m


def slop_event_index(key):
    """The driver's deltas, rebuilt exactly (drv.swift: smooth()/flick() with whole-point error diffusion): the index of
    the first scroll event at which the sideways travel reaches the 6 pt slop, the earliest a swipe may move a page."""
    def smooth(total, seconds):
        n = int(seconds * 120); out = []; prev = 0.0
        for i in range(1, n + 1):
            u = i / n; s = u * u * (3 - 2 * u); out.append(total * (s - prev)); prev = s
        return out
    profile = {"g1a": smooth(-600, 1.5), "g1b": smooth(-600, 1.5), "g6": smooth(-405, 0.6)}.get(key)
    if profile is None:
        return 0  # flicks: the first event is already 9 pt
    acc = sent = 0.0
    for i, d in enumerate(profile):
        acc += d; k = round(acc) - sent; sent += k
        if abs(sent) >= 6:
            return i
    return None


def main():
    prefix, build = sys.argv[1], sys.argv[2]
    t, bar, band, ref = decode(prefix)
    slots = ref_slots(ref)
    dtf = np.round(np.diff(t) * 120).astype(int)
    rstep = np.mod(np.round(np.diff(slots)), 150)
    ev, win, cur = parse_events(prefix)
    starts = {name: tt for tt, kind, name in ev if kind == "input-start"}
    released = {name: tt for tt, kind, name in ev if kind == "released"}
    momentum = {name: tt for tt, kind, name in ev if kind == "momentum-end"}
    firstmove = {name: tt for tt, kind, name in ev if kind == "first-move"}
    marks = sorted([tt for tt, kind, name in ev if kind in ("input-start", "script-end")])

    def last_before(tt):
        return int(np.searchsorted(t, tt) - 1)

    levels, lev_lo, lev_hi = pill_levels(bar[last_before(starts["g1a-slow-overview-backups"])], build)
    pc, pw = pill(bar, build)

    rest_idx = [last_before(starts[n]) for n in ("g1a-slow-overview-backups", "g1b-slow-backups-jobs",
                                                 "g2-flick-jobs-backups")]
    P, off = build_panorama(band, rest_idx)
    content = np.zeros(len(t)); score = np.zeros(len(t)); valid = np.zeros(len(t))
    matcher = Matcher(P, off, band.shape[2])
    for i in range(len(t)):
        content[i], score[i], valid[i] = matcher(band[i])
    rest_pill = [(pc[i], pw[i]) for i in rest_idx]

    def cursor_at(tt):
        if len(cur) == 0:
            return None
        k = np.searchsorted(cur[:, 0], tt) - 1
        return cur[k, 1:] if k >= 0 else None

    gestures = [
        ("g1a", "g1a-slow-overview-backups", "slow swipe Overview→Backups (1.5 s, 600 pt)"),
        ("g1b", "g1b-slow-backups-jobs", "slow swipe Backups→Jobs (1.5 s, 600 pt)"),
        ("g2", "g2-flick-jobs-backups", "fast flick Jobs→Backups (216 pt in 83 ms + momentum)"),
        ("g3", "g3-click-far-overview-jobs", "click a far tab Overview→Jobs"),
        ("g4", "g4a-click-backups", "click Backups, then Jobs 120 ms later"),
        ("g5", "g5-pill-drag-overview-backups", "press the pill, drag one tab (Overview→Backups), release"),
        ("g6", "g6-held-half-backups-jobs", "held swipe 0.45 tab (405 pt) toward Jobs, still 1 s, release"),
        ("g7", "g7-flick-past-last-jobs", "flick past the last tab (Jobs)"),
        ("g8", "g8a-double-flick-1", "double flick Overview→Backups→Jobs, 0.5 s apart"),
        # w308: g9 = a far click (g9c, the same click as g3), then a new flick Jobs→Backups (g9, the same flick as g2)
        # started 100 ms after that click rests
        ("g9c", "g9a-click-far-overview-jobs", "far click Overview→Jobs that g9 waits on (as g3)"),
        ("g9", "g9b-flick-after-rest", "flick Jobs→Backups started 100 ms after a far click rests"),
    ]
    gestures = [g for g in gestures if g[1] in starts]
    last_input = {"g4": "g4b-click-jobs-120ms", "g8": "g8b-double-flick-2"}
    tab_names = ["Overview", "Backups", "Jobs"]
    results = []
    for key, name, label in gestures:
        t0 = starts[name]
        tm = firstmove.get(name, t0)          # first input that moves anything (a click's press is its input)
        end_name = last_input.get(key, name)
        t_rel_fingers = released[end_name]
        t_rel = momentum.get(end_name, t_rel_fingers)
        t_end = min([m for m in marks if m > t_rel + 0.01] + [t[-1]])
        b = last_before(t0)
        idx = np.flatnonzero((t > t0) & (t <= t_end))
        idx = idx[np.isfinite(pc[idx])]
        c0, s0 = pc[b], content[b]
        moved = idx[(np.abs(pc[idx] - c0) > 0.75) | (np.abs(content[idx] - s0) > 1.5)]
        first = int(moved[0]) if len(moved) else None
        fin = int(idx[-1])
        dest = int(np.argmin([abs(pc[fin] - rc) for rc, _ in rest_pill]))
        rc, rw = rest_pill[dest]
        at_rest = (np.abs(pc[idx] - rc) <= 0.5) & (np.abs(pw[idx] - rw) <= 0.5) & (np.abs(content[idx] - dest * PAGE_W) <= 1.0)
        not_rest = np.flatnonzero(~at_rest)
        if len(not_rest) == 0:
            rest_i = int(idx[0]) if len(idx) else None
        elif not_rest[-1] + 1 < len(idx):
            rest_i = int(idx[not_rest[-1] + 1])
        else:
            rest_i = None  # never rests exactly at the slot inside the window
        stop = rest_i if rest_i is not None else fin
        span = np.concatenate([[b], idx[idx <= stop]])
        dpill = np.abs(np.diff(pc[span])); dcont = np.abs(np.diff(content[span]))
        # Frame accounting while moving (first motion .. rest). With the reference strip changing every display
        # frame, a recorded interval over one frame is a frame the RECORDER (or compositor) missed. Over one-frame
        # intervals, a frame where neither pill nor content moved between moving neighbours is a frame the APP did not
        # draw (repeat); a step over twice both neighbours is a catch-up jump.
        rec_missed, repeats, jumps = [], [], []
        if first is not None:
            mv = span[span >= first - 1]
            st_p = np.abs(np.diff(pc[mv])); st_c = np.abs(np.diff(content[mv])); dt_ = np.round(np.diff(t[mv]) * 120)
            for k in range(len(mv) - 1):
                at = round((t[mv[k]] - tm) * 1000, 1)
                if dt_[k] >= 2:
                    rec_missed.append({"at_ms": at, "frames": int(dt_[k])})
                    continue
                if 0 < k < len(mv) - 2:
                    prev_mv = st_p[k - 1] > 0.5 or st_c[k - 1] > 3
                    next_mv = st_p[k + 1] > 0.5 or st_c[k + 1] > 3
                    if st_p[k] < 0.2 and st_c[k] < 0.5 and prev_mv and next_mv:
                        repeats.append({"at_ms": at})
                    for st in (st_p, st_c):
                        nb = max(st[k - 1], st[k + 1])
                        if st[k] > 2 * nb and st[k] > (1.0 if st is st_p else 6.0) and dt_[k - 1] == 1 and dt_[k + 1] == 1:
                            jumps.append({"at_ms": at, "step_pt": round(float(st[k]), 1), "neighbours_pt": round(float(nb), 1),
                                          "of": "pill" if st is st_p else "content"})
        # the settle after the fingers lift (swipes, drag): first changed frame after the release
        settle = None
        if key in ("g1a", "g1b", "g2", "g5", "g6", "g7", "g8", "g9"):
            after_rel = idx[t[idx] > t_rel_fingers]
            if len(after_rel):
                kb = int(np.searchsorted(t, t_rel_fingers) - 1)
                ch = [i for i in after_rel if abs(pc[i] - pc[kb]) > 0.25 or abs(content[i] - content[kb]) > 0.5]
                if ch:
                    i1 = ch[0]
                    settle = {"release_to_first_settle_frame_ms": round((t[i1] - t_rel_fingers) * 1000, 1),
                              "first_settle_step_pill_pt": round(float(abs(pc[i1] - pc[i1 - 1])), 1),
                              "first_settle_step_content_pt": round(float(abs(content[i1] - content[i1 - 1])), 1),
                              "next_steps_pill_pt": [round(float(abs(pc[j] - pc[j - 1])), 1) for j in range(i1 + 1, min(i1 + 4, len(t)))],
                              "next_steps_content_pt": [round(float(abs(content[j] - content[j - 1])), 1) for j in range(i1 + 1, min(i1 + 4, len(t)))]}
        # first steps of the motion
        first_steps = None
        if first is not None:
            first_steps = {"pill_pt": [round(float(abs(pc[j] - pc[j - 1])), 1) for j in range(first, min(first + 4, len(t)))],
                           "content_pt": [round(float(abs(content[j] - content[j - 1])), 1) for j in range(first, min(first + 4, len(t)))],
                           "interval_frames": [int(round((t[j] - t[j - 1]) * 120)) for j in range(first, min(first + 4, len(t)))]}
        # window height (window server)
        wsel = win[(win[:, 0] >= t0 - 0.05) & (win[:, 0] <= t_end)] if len(win) else np.zeros((0, 5))
        wbefore = win[win[:, 0] < t0 - 0.05]
        h0 = wbefore[-1, 4] if len(wbefore) else (win[0, 4] if len(win) else np.nan)
        heights = [h0] + list(wsel[:, 4])
        hchange = bool(len(wsel) and np.any(np.abs(np.array(heights) - h0) > 0.5))
        hdesc = None
        if hchange:
            hdesc = {"from": h0, "to": float(wsel[-1, 4]), "min": float(min(heights)), "max": float(max(heights)),
                     "first_change_ms": round((wsel[0, 0] - tm) * 1000, 1),
                     "last_change_ms": round((wsel[-1, 0] - tm) * 1000, 1), "steps": int(len(wsel))}
        # labels: ink runs against the gesture's starting frame
        thr = LABEL_THR[build]
        base_runs = label_runs(bar[b], None, thr)
        lab_max, lab_frames, lab_count_change = 0.0, 0, 0
        for i in idx:
            cxy = cursor_at(t[i])
            cx = cxy[0] if (cxy is not None and cxy[1] < 45) else None
            mask = edge_mask(bar.shape[2], [(pc[i], pw[i]), (pc[b], pw[b])]) if build == "after2" else None
            r = label_runs(bar[i], cx, thr, mask)
            rb = label_runs(bar[b], cx, thr, mask) if (cx is not None or mask is not None) else base_runs
            if len(r) != len(rb):
                lab_count_change += 1; lab_frames += 1
                continue
            d = max([max(abs(x[0] - y[0]), abs(x[1] - y[1])) for x, y in zip(r, rb)] + [0]) / PX
            if d > 0.5:
                lab_frames += 1
            lab_max = max(lab_max, d)
        pitch = np.mean(np.diff([rp[0] for rp in rest_pill]))
        si = slop_event_index(key) if key in ("g1a", "g1b", "g2", "g6", "g7", "g8", "g9") else None
        t_slop = t0 + si / 120 if si is not None else None
        w_rest = None
        if hchange:
            w_rest = round((wsel[-1, 0] - t_rel_fingers) * 1000, 1)
        rest_ms = round((t[rest_i] - t_rel_fingers) * 1000, 1) if rest_i is not None else None
        results.append({
            "key": key, "gesture": label,
            "first_motion_after_slop_ms": round((t[first] - t_slop) * 1000, 1) if (first is not None and t_slop) else None,
            "first_motion_after_slop_frames": round((t[first] - t_slop) * 120, 1) if (first is not None and t_slop) else None,
            "window_height_rest_after_release_ms": w_rest,
            "rest_all_after_release_ms": (max(x for x in (rest_ms, w_rest) if x is not None)
                                          if rest_ms is not None or w_rest is not None else None),
            "frames_in_window": int(len(idx)),
            "start_tab": tab_names[int(np.argmin([abs(c0 - rc_) for rc_, _ in rest_pill]))],
            "lands_on": tab_names[dest],
            "first_motion_ms": round((t[first] - tm) * 1000, 1) if first is not None else None,
            "first_motion_frames": round((t[first] - tm) * 120, 1) if first is not None else None,
            "first_steps": first_steps,
            "max_pill_step_pt": round(float(dpill.max()), 1) if len(dpill) else 0.0,
            "max_pill_step_tabs": round(float(dpill.max()) / pitch, 3) if len(dpill) else 0.0,
            "max_content_step_pt": round(float(dcont.max()), 1) if len(dcont) else 0.0,
            "max_content_step_tabs": round(float(dcont.max()) / PAGE_W, 3) if len(dcont) else 0.0,
            "settle": settle,
            "rest_after_release_ms": round((t[rest_i] - t_rel_fingers) * 1000, 1) if rest_i is not None else None,
            "rest_after_input_ms": round((t[rest_i] - tm) * 1000, 1) if rest_i is not None else None,
            "final_pill_centre_pt": round(float(pc[fin]), 2), "slot_rest_centre_pt": round(float(rc), 2),
            "final_pill_width_pt": round(float(pw[fin]), 2), "slot_rest_width_pt": round(float(rw), 2),
            "final_content_pt": round(float(content[fin]), 1),
            "recorder_missed": rec_missed, "app_repeats": repeats, "app_jumps": jumps,
            "window_height_changed": hchange, "window_height": hdesc,
            "label_max_shift_pt": round(lab_max, 1), "label_frames_changed": lab_frames,
            "label_run_count_changes": lab_count_change,
        })
    ok = np.isfinite(slots[1:]) & np.isfinite(slots[:-1])
    cad = {"frames": int(len(t)), "seconds": round(float(t[-1] - t[0]), 3),
           "avg_fps": round(float((len(t) - 1) / (t[-1] - t[0])), 2),
           "interval_frames_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(dtf, return_counts=True))},
           "ref_step_equals_interval": int(((rstep == dtf) & ok).sum()),
           "ref_total_advance": int(np.nansum(rstep[ok])), "recorded_total_intervals": int(dtf[ok].sum())}
    out = {"build": build, "prefix": os.path.basename(prefix), "frames": int(len(t)), "cadence": cad,
           "pill_levels_at_first_rest": {"threshold": round(levels, 1), "low": round(lev_lo, 1), "high": round(lev_hi, 1)},
           "rest_pill": [{"tab": n, "centre_pt": round(float(c), 2), "width_pt": round(float(w), 2)}
                         for n, (c, w) in zip(tab_names, rest_pill)],
           "slot_pitch_pt": round(float(np.mean(np.diff([rp[0] for rp in rest_pill]))), 2),
           "content_match": {"median_score": round(float(np.median(score)), 1), "p99_score": round(float(np.percentile(score, 99)), 1),
                             "min_valid_fraction": round(float(valid.min()), 3)},
           "gestures": results}
    args = sys.argv[3:]
    if "--json" in args:
        json.dump(out, open(args[args.index("--json") + 1], "w"), indent=1, ensure_ascii=False)
    if "--csv" in args:
        with open(args[args.index("--csv") + 1], "w") as fh:
            fh.write("t_host_s,pill_centre_pt,pill_width_pt,content_offset_pt,match_mse,valid_frac\n")
            for i in range(len(t)):
                fh.write(f"{t[i]:.6f},{pc[i]:.2f},{pw[i]:.2f},{content[i]:.2f},{score[i]:.1f},{valid[i]:.3f}\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
