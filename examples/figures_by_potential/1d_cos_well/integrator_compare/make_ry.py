"""simple_ry.png: x + history [0, 1] with history types ['y(t-d),y(t)', 'y(t-d),r(t-d)'], seed 2611, 26000 iterations,
EM / BBK / ASE: F(q) plot | q(x, v) | fit - exact | exact. Reuses load_md from make_simple.py's definitions."""
import json, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.image as mpimg
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'make_simple.py')).read()
ns = {'__file__': os.path.abspath(__file__)}
exec(src[:src.index('# 1. Z_q overview')], ns)          # HERE, WELL, ASE, N, LA, LB, load_md, CASES
HERE, ASE, N, LA, LB, load_md = (ns[k] for k in ('HERE', 'ASE', 'N', 'LA', 'LB', 'load_md'))
CASE = {'EM': (f'{HERE}/EM/history_ry/seed2611', 'q_traj.npy'), 'BBK': (f'{HERE}/BBK/history_ry/seed2611', 'q_traj.npy'),
        'ASE': (f'{ASE}/history_x/seed2611', 'q_traj2611.npy')}
fig = plt.figure(figsize=(26, 4.6 * 3)); gs = fig.add_gridspec(3, 4, width_ratios=[2.6, 1, 1, 1]); out = {}
for i, (integ, (f, qn)) in enumerate(CASE.items()):
    x, v, qe = load_md(integ)
    free = (x > LA) & (x < LB)
    el, ev = np.linspace(LA, LB, 71), np.linspace(-2.6, 2.6, 53)
    n = np.histogram2d(x[free], v[free], [el, ev])[0]
    mean = lambda r: np.where(n > 20, np.histogram2d(x[free], v[free], [el, ev], weights=r[free])[0] / np.maximum(n, 1), np.nan).T
    r = np.load(f'{f}/{qn}').astype(float)[:len(x)]
    assert (r[x <= LA] == 0).all() and (r[x >= LB] == 1).all(), integ
    rms, rms_b = float(np.sqrt(np.mean((r[free] - qe[free]) ** 2))), float(np.sqrt(np.nanmean((mean(r) - mean(qe)) ** 2)))
    out[integ] = dict(rms_frame=rms, rms_bin=rms_b)
    a = fig.add_subplot(gs[i, 0]); a.imshow(mpimg.imread(f'{f}/it26000_feps.png')); a.axis('off')
    a.set_title(f'{integ}, x + history [0, 1], new history type (seed 2611): $F(q)$, $F(q)$ natural, $Z_q$', fontsize=11)
    for k, (m, cm, lo, hi, t) in enumerate([(mean(r), 'coolwarm', 0, 1, 'fit $q(x, v)$'),
                                            (mean(r - qe), 'RdBu_r', -0.1, 0.1, f'fit $-$ exact (RMS {rms:.4f} / frame, {rms_b:.4f} / bin)'),
                                            (mean(qe), 'coolwarm', 0, 1, f'exact ({integ})')]):
        b = fig.add_subplot(gs[i, k + 1]); im = b.pcolormesh(el, ev, m, cmap=cm, vmin=lo, vmax=hi)
        fig.colorbar(im, ax=b); b.set(title=t, xlabel='$x$', ylabel='$v$' if k == 0 else None)
fig.tight_layout(); fig.savefig(f'{HERE}/simple_ry.png', dpi=80)
json.dump(out, open(f'{HERE}/simple_ry.json', 'w'), indent=1); print(json.dumps(out, indent=1))
