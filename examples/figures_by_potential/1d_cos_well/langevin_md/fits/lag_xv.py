"""Lag-LAG NPNEq fit (windows stopped at the first A/B frame: sum_t f_k(t)[q(min(t+LAG, tau_t)) - q(t)] = 0),
otherwise as long_xv.py. Original docstring of long_xv.py:
Long CommittorNE fit on the 1D Langevin well MD (md_istarwell2_100M) with the full state (lambda, v)
as CVs, one setting throughout: envelope 0.001, gamma 0.1, ny 6, lazy basis, N_FRAMES frames.

The run is split into chunks of CHUNK iterations, each continuing from the previous q (the
envelope and update do not depend on the iteration counter, so this equals one continuous run).
After every chunk q is saved and compared with the exact committor (brute-force reference) and
the MD-outcome committor; the convergence curve is written to long_xv_progress.csv/.png.
Stops when converged by its own metrics: the cross-entropy improved by less than XENT_TOL over the
last 3 chunks (the exact reference is only used for reporting). Otherwise stops starting new chunks
after HOURS hours.
"""
import os, sys, time, io, contextlib
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/optimalrcs-cupy/')
import optimalrcs

OUT = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/figures_by_potential/1d_cos_well/langevin_md/fits/lag_xv/'
EX = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs/examples/'
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
HOURS = float(sys.argv[3]) if len(sys.argv) > 3 else 7.0
GAMMA, ENVELOPE, NY = 0.1, 0.001, 6
LAG = int(sys.argv[4]) if len(sys.argv) > 4 else 16
XENT_TOL = 3e-5
os.makedirs(OUT, exist_ok=True)

d = np.load('/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/md_istarwell2_100M/md_path.npy',
            mmap_mode='r')
lam, v = np.array(d[:N_FRAMES, 0], float), np.array(d[:N_FRAMES, 1], float)
A, B = lam <= -0.35, lam >= 0.35
free = ~A & ~B

# MD outcome of every frame (which boundary comes next), vectorized
idx = np.where(A | B, np.arange(N_FRAMES), N_FRAMES)
nxt_idx = np.minimum.accumulate(idx[::-1])[::-1]
has = nxt_idx < N_FRAMES
outcome = np.where(has, B[np.minimum(nxt_idx, N_FRAMES - 1)].astype(float), np.nan)
ok = free & has

ref = np.load(EX + 'figures_by_potential/1d_cos_well/langevin_staple/checks/reference.npz')
f = RegularGridInterpolator((ref['xs'], ref['vs']), ref['q'], bounds_error=False, fill_value=None)
qx = np.clip(f(np.c_[np.clip(lam, ref['xs'][0], ref['xs'][-1]), np.clip(v, ref['vs'][0], ref['vs'][-1])]), 0, 1)
E = 0.5 * v ** 2   # + V(lambda): only used to locate the separatrix band below
def V_coswell(x, wall_dx=0.1, k=100., bleft=-0.25, bright=0.25, deltaf=1.2, height=0.15, width=0.1):
    L = bright - bleft; xl = bleft - wall_dx - width; xr = bright + wall_dx + width
    return np.select([x < xl, x > xr, x < bleft - width, x > bright + width, x < bleft, x > bright],
                     [k * (x - xl) ** 2 / 2, k * (x - xr) ** 2 / 2, 0 * x, 0 * x,
                      height / 2 * (1 - np.cos(np.pi * (x - bleft + width) / width)),
                      height / 2 * (1 - np.cos(np.pi * (x - bright - width) / width))],
                     height + deltaf / 2 * (-1 + np.cos(2 * np.pi * (x - bleft) / L)))
E = E + V_coswell(lam)
band = free & (np.abs(E - 0.15) < 0.05)

el, ev = np.linspace(-0.35, 0.35, 36), np.linspace(-3, 3, 31)
n_bin = np.histogram2d(lam[ok], v[ok], [el, ev])[0]
def bmap(val):
    s = np.histogram2d(lam[ok], v[ok], [el, ev], weights=val[ok])[0]
    return np.where(n_bin > 200, s / np.maximum(n_bin, 1), np.nan)
Emd, X = bmap(outcome), bmap(qx)
rms = lambda a: float(np.sqrt(np.nanmean(a ** 2)))
print(f'{N_FRAMES} frames, {int((A[:-1] & ~A[1:]).sum())} departures from A; RMS(exact - MD outcome) per bin '
      f'{rms(X - Emd):.4f} (noise yardstick)', flush=True)


# ---- lag-LAG NPNEq update (as examples/figures_by_potential/2d_zpot/langevin_g2/fd_velocity/fd_test.py) ----
import nonparametrics
tau_np = np.where(A | B, np.arange(N_FRAMES), nxt_idx)          # first A/B frame at or after t
tau_np = np.minimum(tau_np, N_FRAMES - 1)
u_np = np.minimum(np.arange(N_FRAMES) + LAG, tau_np)
U = cp.asarray(u_np)
STARTS = cp.asarray(np.flatnonzero(free & (u_np > np.arange(N_FRAMES))))
class LazyRY:
    """basis_poly_ry evaluated on chunks of frames, normalized over all frames as basis_poly_ry does."""
    def __init__(self, r, y, n, fenv):
        self.r, self.y, self.n = r / cp.max(cp.abs(r)), y / cp.max(cp.abs(y)), n
        self.fenv = cp.ones_like(r) if fenv is None else fenv
    def cols(self, idx):
        return optimalrcs._poly_ry(self.r[idx], self.y[idx], self.n, self.fenv[idx])
def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
    akj, bb = 0, 0
    for s0 in range(0, STARTS.size, 400_000):
        st = STARTS[s0:s0 + 400_000]
        fa, fb = fk.cols(st), fk.cols(U[st])
        akj = akj + fa @ (fb - fa * (1 + gamma)).T
        bb = bb + fa @ (-(r_traj[U[st]] - r_traj[st]))
    al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
    new = r_traj.copy()
    for s0 in range(0, r_traj.size, 1_000_000):
        idx = cp.arange(s0, min(s0 + 1_000_000, r_traj.size))
        new[idx] = r_traj[idx] + al @ fk.cols(idx)
    return cp.clip(new, 0, 1)
optimalrcs.nonparametrics.npneq = npneq_lag
print(f'lag {LAG}: {int(STARTS.size)} windows', flush=True)

Y = cp.asarray(np.c_[lam, v])
optimalrcs.envelope_scale = ENVELOPE
rows, r_prev, t_start, it_total, k = [], None, time.time(), 0, 0
while time.time() - t_start < HOURS * 3600:
    np.random.seed(k); cp.random.seed(k)
    q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=np.ones(N_FRAMES), t_traj=np.arange(N_FRAMES),
                               seed_r=r_prev)
    t0 = time.time()
    q.fit_transform(lambda: Y[:, np.random.randint(2)], gamma=GAMMA, ny=NY, max_iter=CHUNK, print_step=1000,
                    metrics_print=('iter', 'cross_entropy', 'mse', 'max_sd_zq', 'auc', 'delta_x'),
                    save_min_delta_zq=False,
                    basis_functions=lambda r_, y_, n_, f_: LazyRY(r_, y_, n_, f_))
    it_total += CHUNK
    r = cp.asnumpy(q.r_traj); r_prev = r
    h = q.metrics_history
    F = bmap(r)
    row = dict(iteration=it_total, hours=(time.time() - t_start) / 3600, xent=h['cross_entropy'][-1],
               sdzq=h['max_sd_zq'][-1], auc=h['auc'][-1], dx=h['delta_x'][-1],
               rms_bin=rms(F - X), rms_frame=float(np.sqrt(np.mean((r[free] - qx[free]) ** 2))),
               rms_band=float(np.sqrt(np.mean((r[band] - qx[band]) ** 2))), max_bin=float(np.nanmax(np.abs(F - X))))
    rows.append(row)
    print(' | '.join(f'{k_}={v_:.5g}' for k_, v_ in row.items()), flush=True)
    np.save(OUT + 'q_latest.npy', r)
    if it_total % 5000 == 0:
        np.save(OUT + f'q_{it_total}.npy', r)
    keys = list(rows[0])
    np.savetxt(OUT + 'long_xv_progress.csv', np.array([[rw[c] for c in keys] for rw in rows]), delimiter=',',
               header=','.join(keys), comments='')
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    it = [rw['iteration'] for rw in rows]
    axes[0].plot(it, [rw['rms_bin'] for rw in rows], 'o-', label='per bin')
    axes[0].plot(it, [rw['rms_frame'] for rw in rows], 'o-', label='per frame')
    axes[0].plot(it, [rw['rms_band'] for rw in rows], 'o-', label='band |E - 0.15| < 0.05')
    axes[0].axhline(rms(X - Emd), color='k', ls='--', lw=1, label='exact - MD (noise)')
    axes[0].set(xlabel='iteration', ylabel='RMS(fit - exact)', yscale='log'); axes[0].legend(frameon=False)
    axes[1].plot(it, [rw['xent'] for rw in rows], 'o-'); axes[1].set(xlabel='iteration', ylabel='cross-entropy')
    im = axes[2].pcolormesh(el, ev, (F - X).T, cmap='PuOr', vmin=-0.15, vmax=0.15)
    axes[2].set(title=f'fit - exact after {it_total} iterations', xlabel='lambda', ylabel='v'); fig.colorbar(im, ax=axes[2])
    fig.tight_layout(); fig.savefig(OUT + 'long_xv_progress.png', dpi=90); plt.close(fig)
    del q; cp.get_default_memory_pool().free_all_blocks()
    k += 1
    if len(rows) >= 4 and rows[-4]['xent'] - rows[-1]['xent'] < XENT_TOL:
        print(f'converged: cross-entropy improved by {rows[-4]["xent"] - rows[-1]["xent"]:.2e} over the last '
              f'{3 * CHUNK} iterations', flush=True)
        break
print('ALL DONE', flush=True)
