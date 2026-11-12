"""1D Langevin well MD (md_istarwell2_100M, first 6M frames): CommittorNE on (lambda, v, E) with
E = v^2/2 + V(lambda) (CosWellWalls, kT = m = 1), same settings as the notebook's saved fit
(envelope 0.001, gamma 0.2, ny 6). Compared with the exact committor and the MD outcomes."""
import sys, time, numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/')
import optimalrcs
OUT = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures/well_md_fit/'
EX = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/'
MAX_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
GAMMA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
SEED_R = sys.argv[3] if len(sys.argv) > 3 else None     # .npy of a previous fit to continue from
TAG = sys.argv[4] if len(sys.argv) > 4 else str(MAX_ITER)
N = 6_000_000
d = np.load('/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy', mmap_mode='r')
lam, v = np.array(d[:N, 0], float), np.array(d[:N, 1], float)
A, B = lam <= -0.35, lam >= 0.35

def V_coswell(x, wall_dx=0.1, k=100., bleft=-0.25, bright=0.25, deltaf=1.2, height=0.15, width=0.1):
    """CosWellWalls.potential (toytis/potentials/cos_well.py, default parameters, db = 0), vectorized."""
    L = bright - bleft; xl = bleft - wall_dx - width; xr = bright + wall_dx + width
    return np.select(
        [x < xl, x > xr, x < bleft - width, x > bright + width, x < bleft, x > bright],
        [k * (x - xl) ** 2 / 2, k * (x - xr) ** 2 / 2, 0 * x, 0 * x,
         height / 2 * (1 - np.cos(np.pi * (x - bleft + width) / width)),
         height / 2 * (1 - np.cos(np.pi * (x - bright - width) / width))],
        height + deltaf / 2 * (-1 + np.cos(2 * np.pi * (x - bleft) / L)))
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis')
from potentials.cos_well import CosWellWalls
pot = CosWellWalls()
for x0 in (-0.5, -0.4, -0.3, -0.27, -0.1, 0.0, 0.2, 0.3, 0.42):
    assert abs(pot.potential((np.array([x0]),)) - V_coswell(np.array(x0))) < 1e-12, x0
E = 0.5 * v ** 2 + V_coswell(lam)

Y = cp.asarray(np.c_[lam, v, E])
optimalrcs.envelope_scale = 0.001
np.random.seed(0); cp.random.seed(0)
q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=np.ones(N), t_traj=np.arange(N),
                           seed_r=None if SEED_R is None else np.load(OUT + SEED_R))
t0 = time.time()
q.fit_transform(lambda: Y[:, np.random.randint(3)], gamma=GAMMA, ny=6, max_iter=MAX_ITER, print_step=250,
                save_min_delta_zq=False, lazy_basis=True)
r = cp.asnumpy(q.r_traj)
np.save(OUT + f'q_energy_cv_{TAG}.npy', r)
print(f'fit done in {time.time() - t0:.0f} s', flush=True)

# comparison with the exact committor and the MD outcomes, as for the saved (lambda, v) fit
nxt = np.full(N, -1.0); last = -1.0
for i in range(N - 1, -1, -1):
    if A[i]: last = 0.0
    elif B[i]: last = 1.0
    nxt[i] = last
free = ~A & ~B; ok = free & (nxt >= 0)
ref = np.load(EX + 'figures/well_check/reference.npz')
f = RegularGridInterpolator((ref['xs'], ref['vs']), ref['q'], bounds_error=False, fill_value=None)
qx = np.clip(f(np.c_[np.clip(lam, ref['xs'][0], ref['xs'][-1]), np.clip(v, ref['vs'][0], ref['vs'][-1])]), 0, 1)
r_lv = np.load(EX + 'q_traj_langevin_env0.001g0.2.npy')
el, ev = np.linspace(-0.35, 0.35, 36), np.linspace(-3, 3, 31)
def bmap(val):
    n = np.histogram2d(lam[ok], v[ok], [el, ev])[0]; s = np.histogram2d(lam[ok], v[ok], [el, ev], weights=val[ok])[0]
    return np.where(n > 200, s / np.maximum(n, 1), np.nan)
Emd, X, F3, F2 = bmap(nxt), bmap(qx), bmap(r), bmap(r_lv)
rms = lambda a: np.sqrt(np.nanmean(a ** 2))
band = free & (np.abs(E - 0.15) < 0.05)
print(f'per bin: RMS(exact - MD) {rms(X - Emd):.4f} | (lambda, v) fit 10000 it: RMS(fit - exact) {rms(F2 - X):.4f} '
      f'| (lambda, v, E) fit {TAG}: RMS(fit - exact) {rms(F3 - X):.4f}', flush=True)
print(f'per frame RMS(fit - exact): (lambda, v) {np.sqrt(np.mean((r_lv[free] - qx[free]) ** 2)):.4f}, '
      f'(lambda, v, E) {np.sqrt(np.mean((r[free] - qx[free]) ** 2)):.4f}; on the band |E - 0.15| < 0.05: '
      f'{np.sqrt(np.mean((r_lv[band] - qx[band]) ** 2)):.4f} vs {np.sqrt(np.mean((r[band] - qx[band]) ** 2)):.4f}', flush=True)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
for ax, (M, t, cm, lim) in zip(axes, [(F3, f'(lambda, v, E) fit, {TAG}', 'coolwarm', 1), (F2 - X, '(lambda, v) fit 10000 it - exact', 'PuOr', 0.15),
                                       (F3 - X, f'(lambda, v, E) fit {TAG} - exact', 'PuOr', 0.15)]):
    im = ax.pcolormesh(el, ev, M.T, cmap=cm, vmin=0 if lim == 1 else -lim, vmax=lim); ax.set(title=t, xlabel='lambda', ylabel='v')
    fig.colorbar(im, ax=ax)
fig.tight_layout(); fig.savefig(OUT + f'energy_cv_{TAG}.png', dpi=90)
print('ALL DONE', flush=True)
