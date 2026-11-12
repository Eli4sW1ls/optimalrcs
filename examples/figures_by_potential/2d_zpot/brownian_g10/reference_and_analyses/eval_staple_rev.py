"""Staple refit with mirrored pieces vs the exact committor, as for the original fit."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
F = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/'
ref = np.load(F + '2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
f = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
for name, path in (('original fit (forward only)', 'istarz_br_g10/committor.npz'),
                   ('refit, forward + mirrored', 'istarz_br_g10_reversed/committor.npz')):
    d = np.load(F + path); y, x, r = d['X_all'][:, 0], d['X_all'][:, 1], d['r']
    free = ~(d['boundary0'] | d['boundary1']); qx = np.clip(f(np.c_[y, x]), 0, 1)
    ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
    n = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
    bm = lambda v: np.where(n >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(n, 1), np.nan)
    Fq, X = bm(r), bm(qx); tr = (X > 0.05) & (X < 0.95)
    plateau = free & (qx > 0.55) & (qx < 0.85)
    pts = []
    for (a, b) in [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]:
        m = free & ((y - a) ** 2 + (x - b) ** 2 < 0.01 ** 2); pts.append(f'{r[m].mean():.3f}')
    print(f'{name:30s}: RMS per bin {np.sqrt(np.nanmean((Fq - X)[tr] ** 2)):.4f} | mean(fit - exact) on plateau '
          f'{np.mean(r[plateau] - qx[plateau]):+.4f} | at the shooting points {" ".join(pts)} '
          f'(shooting 0.639 0.664 0.698 0.747)', flush=True)
