"""Reference-strip check: per recorded frame, the reference bar's slot (rows 1500..1514 px of the region); its step
between consecutive frames against the frame interval. step == round(dt*120) means the reference drew every display
frame in between; a recorded interval over 1 frame with step 1 means the reference itself missed a frame."""
import subprocess, sys
import numpy as np
mov, pts = sys.argv[1], sys.argv[2]
W, H = 1240, int(sys.argv[3]) if len(sys.argv) > 3 else 1528
t = np.array([float(x) for x in open(pts) if x.strip()])
p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", mov, "-fps_mode", "passthrough", "-vf", f"crop={W}:12:0:1500",
                      "-f", "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
xs = []
while True:
    b = p.stdout.read(W * 12)
    if len(b) < W * 12: break
    r = np.frombuffer(b, np.uint8).reshape(12, W)[6]
    on = np.flatnonzero(r > 128)
    xs.append((on.mean() / 2 - 8) / 4 if len(on) else np.nan)   # region x 892 pt, ref x 900 pt: 8 pt offset
xs = np.array(xs); assert len(xs) == len(t), (len(xs), len(t))
slots = 150
step = np.mod(np.round(np.diff(xs)), slots)
dtf = np.round(np.diff(t) * 120)
np.savez(sys.argv[1] + ".ref.npz", t=t, slot=xs)
print("frames", len(t), "intervals", len(dtf))
print("interval(frames) histogram:", dict(zip(*np.unique(dtf, return_counts=True))))
print("ref step == interval:", int((step == dtf).sum()), " step < interval (ref missed):", int((step < dtf).sum()),
      " step > interval:", int((step > dtf).sum()))
