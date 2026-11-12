"""Compare a saved fit of the 1D Langevin well MD (first 6M frames of md_istarwell2_100M) with the
brute-force reference and the MD-outcome committor, as fit_vs_exact_vs_md.png, plus the same numbers
for the earlier lag-1 and lag-16 fits.  Usage: python compare_seed.py <q_traj.npy> <tag>"""
import sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
W = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures/well_md_fit/'
QFILE, TAG = sys.argv[1], sys.argv[2]
N = 6_000_000
d = np.load('/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy', mmap_mode='r')[:N]
lam, v = np.array(d[:, 0], float), np.array(d[:, 1], float)
A, B = lam <= -0.35, lam >= 0.35; free = ~A & ~B
ref = np.load(W + '../well_check/reference.npz')
f = RegularGridInterpolator((ref['xs'], ref['vs']), ref['q'], bounds_error=False, fill_value=None)
qx = np.clip(f(np.c_[np.clip(lam, ref['xs'][0], ref['xs'][-1]), np.clip(v, ref['vs'][0], ref['vs'][-1])]), 0, 1)
idx = np.where(A | B, np.arange(N), N); nxt = np.minimum.accumulate(idx[::-1])[::-1]
out = np.where(nxt < N, B[np.minimum(nxt, N - 1)].astype(float), np.nan)
ok = free & np.isfinite(out)

def V(x, wall_dx=0.1, k=100., bleft=-0.25, bright=0.25, deltaf=1.2, height=0.15, width=0.1):
    L = bright - bleft; xl = bleft - wall_dx - width; xr = bright + wall_dx + width
    return np.select([x < xl, x > xr, x < bleft - width, x > bright + width, x < bleft, x > bright],
                     [k * (x - xl) ** 2 / 2, k * (x - xr) ** 2 / 2, 0 * x, 0 * x,
                      height / 2 * (1 - np.cos(np.pi * (x - bleft + width) / width)),
                      height / 2 * (1 - np.cos(np.pi * (x - bright - width) / width))],
                     height + deltaf / 2 * (-1 + np.cos(2 * np.pi * (x - bleft) / L)))
band = ok & (np.abs(0.5 * v ** 2 + V(lam) - 0.15) < 0.05)

el, ev = np.linspace(-0.35, 0.35, 36), np.linspace(-3, 3, 31)
n_bin = np.histogram2d(lam[ok], v[ok], [el, ev])[0]
def bmap(val):
    s = np.histogram2d(lam[ok], v[ok], [el, ev], weights=val[ok])[0]
    return np.where(n_bin > 200, s / np.maximum(n_bin, 1), np.nan)
rms = lambda a: np.sqrt(np.nanmean(a ** 2))
X, Emd = bmap(qx), bmap(out)

def numbers(r):
    F = bmap(r)
    return (rms(F - X), np.sqrt(np.mean((r[ok] - qx[ok]) ** 2)), np.sqrt(np.mean((r[band] - qx[band]) ** 2)),
            np.nanmax(np.abs(F - X)), rms(F - Emd)), F

print(f'noise yardstick RMS(exact - MD outcome) per bin {rms(X - Emd):.4f}')
print(f'{"fit":34s} RMS bin   frame   band   max bin   RMS(fit - MD outcome)')
fits = [(TAG, QFILE), ('lag 1, 95000 it. (long_xv)', W + 'long_xv/q_latest.npy'),
        ('lag 16, 12000 it. (lag_xv)', W + 'lag_xv/q_latest.npy')]
for name, path in fits:
    r = np.load(path).astype(float)
    (rb, rf, rbd, mx, rmd), F = numbers(r)
    print(f'{name:34s} {rb:.4f}  {rf:.4f}  {rbd:.4f}  {mx:.3f}     {rmd:.4f}')
    if name == TAG:
        Fmain, rmain = F, (rb, rf, rbd, mx)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
panels = [(Fmain, f'fit ({TAG})', 'coolwarm', 1), (X, 'exact (brute-force reference)', 'coolwarm', 1),
          (Emd, 'MD outcome (fraction reaching B first)', 'coolwarm', 1),
          (Fmain - Emd, 'fit - MD outcome', 'PuOr', 0.15), (X - Emd, f'exact - MD outcome (noise, RMS {rms(X - Emd):.3f})', 'PuOr', 0.15),
          (Fmain - X, f'fit - exact (RMS {rmain[0]:.3f} per bin, {rmain[2]:.3f} on the E ≈ 0.15 band)', 'PuOr', 0.15)]
for ax, (M, t, cm, lim) in zip(axes.ravel(), panels):
    im = ax.pcolormesh(el, ev, M.T, cmap=cm, vmin=0 if lim == 1 else -lim, vmax=lim)
    ax.set(title=t, xlabel='lambda', ylabel='v'); fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(W + f'compare_{TAG}.png', dpi=90)
print('saved', W + f'compare_{TAG}.png')
