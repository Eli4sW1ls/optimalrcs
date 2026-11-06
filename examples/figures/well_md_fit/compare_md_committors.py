"""MD-outcome committor q(x, v) (fraction of frames reaching B before A, no fit) of the Euler-Maruyama well MD
against the toytis BBK runs (dt = 0.005, and dt = 0.0005 saved every 10 steps) and the brute-force reference.
All on the same number of rows (= same physical time, 0.005 per row). Noise of each map: RMS(half1 - half2)/2."""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
W = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures/well_md_fit/'
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/'
N = 3_000_000
LA, LB = -0.35, 0.35
runs = {'Euler-Maruyama, dt 0.005': W + 'euler_md_wel_lgv_3M.npy',
        'BBK, dt 0.005': SIM + 'md_istarwell2_100M/md_path.npy',
        'BBK, dt 0.0005 (every 10th)': SIM + 'md_istarwell2_dt5e-4_10M/md_path.npy'}
ref = np.load(W + '../well_check/reference.npz')
f = RegularGridInterpolator((ref['xs'], ref['vs']), ref['q'], bounds_error=False, fill_value=None)
el, ev = np.linspace(LA, LB, 36), np.linspace(-3, 3, 31)

def bmap(x, v, val, m, min_n):
    n = np.histogram2d(x[m], v[m], [el, ev])[0]
    s = np.histogram2d(x[m], v[m], [el, ev], weights=val[m])[0]
    return np.where(n > min_n, s / np.maximum(n, 1), np.nan)

maps = {}
for name, path in runs.items():
    a = np.load(path, mmap_mode='r')[:N]; x, v = np.array(a[:, 0]), np.array(a[:, 1])
    A, B = x <= LA, x >= LB
    idx = np.where(A | B, np.arange(N), N); nxt = np.minimum.accumulate(idx[::-1])[::-1]
    out = np.where(nxt < N, B[np.minimum(nxt, N - 1)].astype(float), np.nan)
    ok = ~A & ~B & np.isfinite(out)
    s = np.where(x <= LA, 0, np.where(x >= LB, 1, -1)); s = s[s >= 0]
    h = np.arange(N) < N // 2
    M = bmap(x, v, out, ok, 200)
    R = bmap(x, v, np.clip(f(np.c_[np.clip(x, ref['xs'][0], ref['xs'][-1]), np.clip(v, ref['vs'][0], ref['vs'][-1])]), 0, 1), ok, 200)
    noise = np.sqrt(np.nanmean((bmap(x, v, out, ok & h, 100) - bmap(x, v, out, ok & ~h, 100)) ** 2)) / 2
    maps[name] = (M, R, noise)
    print(f'{name:30s}: A->B transitions {int(np.sum(np.diff(s) == 1)):5d} | <v^2> {np.mean(v ** 2):.4f} | noise per bin {noise:.4f}')

rms = lambda a: np.sqrt(np.nanmean(a ** 2))
names = list(runs)
print('\npairwise, per (x, v) bin:                     RMS     mean   | combined noise')
pairs = [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
for a_, b_ in pairs:
    D = maps[a_][0] - maps[b_][0]
    print(f'  {a_:28s} - {b_:28s}: {rms(D):.4f} {np.nanmean(D):+.4f} | {np.hypot(maps[a_][2], maps[b_][2]):.4f}')
for n_ in names:
    D = maps[n_][0] - maps[n_][1]
    print(f'  {n_:28s} - reference (BBK shooting)   : {rms(D):.4f} {np.nanmean(D):+.4f} | {maps[n_][2]:.4f}')

E, Bo = maps[names[0]][0], maps[names[1]][0]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
panels = [(E, 'MD outcome, Euler-Maruyama', 'coolwarm', 1), (Bo, 'MD outcome, BBK dt 0.005', 'coolwarm', 1),
          (maps[names[0]][1], 'reference (BBK shooting, dt 0.005)', 'coolwarm', 1),
          (E - Bo, f'EM - BBK (RMS {rms(E - Bo):.3f}; noise {np.hypot(maps[names[0]][2], maps[names[1]][2]):.3f})', 'PuOr', 0.15),
          (E - maps[names[0]][1], f'EM - reference (RMS {rms(E - maps[names[0]][1]):.3f})', 'PuOr', 0.15),
          (Bo - maps[names[1]][1], f'BBK - reference (RMS {rms(Bo - maps[names[1]][1]):.3f})', 'PuOr', 0.15)]
for ax, (Mp, t, cm, lim) in zip(axes.ravel(), panels):
    im = ax.pcolormesh(el, ev, Mp.T, cmap=cm, vmin=0 if lim == 1 else -lim, vmax=lim)
    ax.set(title=t, xlabel='x', ylabel='v'); fig.colorbar(im, ax=ax)
fig.suptitle(f'1D Langevin well, MD-outcome committor (no fit), {N // 10**6}M rows per run')
fig.tight_layout(); fig.savefig(W + 'md_committor_euler_vs_bbk.png', dpi=90)
print('saved', W + 'md_committor_euler_vs_bbk.png')
