"""Plateau offset and RMS against iteration for the five data-selection fits (parsed from the logs)."""
import re, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
runs = {'base (1500 paths/ens., all frames)': 'fit_base.log', 'flat_qn (frame weights flat in q_n)': 'fit_flat_qn.log',
        'select_paths (paths kept by coverage)': 'fit_select_paths.log',
        'blind flat_y (3000-path pool, 10M starts flat in y)': 'blind_flat_y.log',
        'blind uniform (3000-path pool, 10M starts)': 'blind_uniform.log'}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8))
for lab, f in runs.items():
    rows = re.findall(r'^seed 0, (\d+) iterations.*RMS\(fit - exact\) per bin ([\d.]+) \| fit/shooting: (.*)$', open(f).read(), re.M)
    it = np.array([int(r[0]) for r in rows]); rms = np.array([float(r[1]) for r in rows])
    off = np.array([np.mean([float(a) - float(b) for a, b in (p.split('/') for p in r[2].split())]) for r in rows])
    a1.plot(it, off, 'o-', label=lab); a2.plot(it, rms, 'o-', label=lab)
    print(f'{lab:52s} ' + ' '.join(f'{i}:{o:+.3f}/{r:.3f}' for i, o, r in zip(it, off, rms)))
a1.axhline(0, color='k', lw=0.8)
a1.set(xlabel='iteration', ylabel='mean (fit - shooting) at 4 plateau points', title='plateau level')
a2.set(xlabel='iteration', ylabel='RMS(fit - exact) per (y, x) bin', title='whole map', yscale='log')
a1.legend(fontsize=8); a1.grid(alpha=0.3); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('summary.png', dpi=110)
