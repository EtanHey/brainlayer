"""Cut one real-time 120 fps clip per gesture group from a recording (window crop: the recorded region without the
reference strip). Usage: clips.py <prefix> <build> <outdir>"""
import subprocess, sys
prefix, build, outdir = sys.argv[1:4]
t0 = float(open(prefix + ".pts").readline())
ev = [(float(l.split()[0]), l.split()[1], (l.split() + [""])[2]) for l in open(prefix + ".events")]
starts = [(t, n) for t, k, n in ev if k == "input-start"]
end_mark = [t for t, k, n in ev if k == "script-end"][0]
rel = {n: t for t, k, n in ev if k in ("released", "momentum-end")}
groups = [("g1-slow-swipe", "g1a-slow-overview-backups", "g1b-slow-backups-jobs"),
          ("g2-fast-flick", "g2-flick-jobs-backups", "g2-flick-jobs-backups"),
          ("g3-far-click", "g3-click-far-overview-jobs", "g3-click-far-overview-jobs"),
          ("g4-click-during-settle", "g4a-click-backups", "g4b-click-jobs-120ms"),
          ("g5-pill-drag", "g5-pill-drag-overview-backups", "g5-pill-drag-overview-backups"),
          ("g6-held-half-swipe", "g6-held-half-backups-jobs", "g6-held-half-backups-jobs"),
          ("g7-flick-past-last", "g7-flick-past-last-jobs", "g7-flick-past-last-jobs"),
          ("g8-double-flick", "g8a-double-flick-1", "g8b-double-flick-2")]
for slug, first, last in groups:
    a = dict((n, t) for t, n in starts)[first] - 0.4
    after = [t for t, n in starts if t > rel[last]]
    b = (after[0] - 0.1) if after else end_mark
    out = f"{outdir}/{slug}.{build}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", prefix + ".mov", "-ss", f"{a - t0:.4f}", "-to", f"{b - t0:.4f}",
                    "-vf", "crop=1240:1488:0:0,fps=120", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out], check=True)
    print(out, f"{b - a:.2f} s")
