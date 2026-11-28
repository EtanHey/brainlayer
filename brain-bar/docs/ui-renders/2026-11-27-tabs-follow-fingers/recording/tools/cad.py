import sys, numpy as np
t = np.array([float(l.strip().strip(',')) for l in open(sys.argv[1]) if l.strip().strip(',')])
d = np.diff(t) * 1000
print('frames', len(t), 'span %.3f s' % (t[-1] - t[0]), 'avg fps %.2f' % ((len(t) - 1) / (t[-1] - t[0])))
print('dt ms p5/25/50/75/95/99', np.percentile(d, [5, 25, 50, 75, 95, 99]).round(2), 'max %.1f' % d.max())
for w in sys.argv[2:]:
    a, b = map(float, w.split(':'))
    s = t[(t > a) & (t < b)]; dd = np.diff(s) * 1000
    print('window %.2f-%.2f' % (a, b), len(s), 'fps %.1f' % ((len(s) - 1) / (s[-1] - s[0])), 'gaps>12ms', int((dd > 12).sum()), 'max %.1f' % dd.max())
