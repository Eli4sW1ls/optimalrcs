"""Blind data balancing by choosing window starts (2D Brownian Z, sim_istarz_br_g10).

No committor information: only the order parameter y. From a pool of TIS pieces (forward + mirrored,
rejection weights), every free frame t is flagged as a window start with a probability that depends on its
y only. Each flagged start contributes the transition t -> t+1 (lag-1 NPNEq), with its path weight. The flags
depend on the current frame only, so E[q(x_{t+1}) - q(x_t) | x_t, flagged] = 0 still holds: unbiased.
To keep memory bounded, each maximal run of flagged frames is kept as its own segment (frames s .. e+1).

  flat_y    p(y) = min(1, c / rho_w(y bin)): the weighted density of starts is flat over 40 y bins
            between lambda_A and lambda_B (bins with too few frames keep all of them)
  uniform   p = const (control: same pool and budget, no balancing)
c (or the constant) is set so the expected number of starts is BUDGET.
    python blind_starts.py <flat_y|uniform> [PATHS_PER_ENSEMBLE] [BUDGET] [MAX_ITER]   (MAX_ITER 0: data only)
"""
import argparse, contextlib, io, sys, time
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import tis_data
plt.show = lambda *a, **k: None
OUT = REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/staple/selection_test/'
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10'
MODE = sys.argv[1]
PPE = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
BUDGET = int(float(sys.argv[3])) if len(sys.argv) > 3 else 10_000_000
MAX_ITER = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
N_BINS = 40
TAG = f'blind_{MODE}_p{PPE}_b{BUDGET:.0e}'.replace('+0', '')
t0 = time.time()

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([SIM, '--reversed', '--paths-per-ensemble', str(PPE), '--seed', '1']))
y, x = data.X_all[:, 0].astype(float), data.X_all[:, 1].astype(float)
A, B, it, w, grp, tt = data.boundary0, data.boundary1, data.i_traj, data.path_weights, data.group_traj, data.t_traj
la, lb = data.lam[A].max() if A.any() else 0.05, 0.85
del data
n = len(y)
print(f'pool: {PPE} paths per ensemble, {n:,} frames ({time.time() - t0:.0f} s)', flush=True)

# possible starts: free frames with a successor in the same piece
can = ~(A | B)
can[:-1] &= it[1:] == it[:-1]
can[-1] = False
edges = np.linspace(0.05, 0.85, N_BINS + 1)
k = np.clip(np.digitize(y, edges) - 1, 0, N_BINS - 1)
n_b = np.bincount(k[can], minlength=N_BINS).astype(float)
rho = np.bincount(k[can], weights=w[can], minlength=N_BINS)
if MODE == 'flat_y':
    def expected(c):
        return np.sum(n_b * np.minimum(1, c / np.maximum(rho, 1e-12)))
    lo, hi = 0.0, rho.max()
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if expected(mid) < BUDGET else (lo, mid)
    p_b = np.minimum(1, hi / np.maximum(rho, 1e-12))
elif MODE == 'uniform':
    p_b = np.full(N_BINS, min(1.0, BUDGET / can.sum()))
else:
    raise ValueError(MODE)
flag = can & (np.random.default_rng(0).random(n) < p_b[k])

# segments: each run of flagged frames s..e plus its successor e+1
keep = flag.copy(); keep[1:] |= flag[:-1]
new_seg = keep.copy(); new_seg[1:] &= ~flag[:-1]
seg = np.cumsum(new_seg) - 1
ix = np.flatnonzero(keep)
y, x, A, B, w, grp, tt, seg = y[ix], x[ix], A[ix], B[ix], w[ix], grp[ix], tt[ix], seg[ix]
flag = flag[ix]
print(f'{flag.sum():,} starts ({flag.sum() / can.sum():.1%} of {can.sum():,} possible), {len(ix):,} frames in {seg.max() + 1:,} segments; '
      f'probability per y bin: min {p_b.min():.3f}, max {p_b.max():.3f}', flush=True)

ref = np.load(REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
f_ref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
q_exact = np.clip(f_ref(np.c_[y, x]), 0, 1); q_exact[A] = 0; q_exact[B] = 1
band = np.abs(y + x - 0.6) < 0.05 * np.sqrt(2)
qe_bins = np.bincount(np.clip((q_exact[flag] * 10).astype(int), 0, 9), weights=w[flag], minlength=10)
print(f'starts in the ridge band: {band[flag].mean():.2%} (weighted {w[flag & band].sum() / w[flag].sum():.2%}); '
      f'weighted starts per bin of the exact q: ' + ' '.join(f'{v / qe_bins.sum():.3f}' for v in qe_bins), flush=True)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
a1.plot((edges[1:] + edges[:-1]) / 2, rho / rho.sum(), 'k--', label='all possible starts')
a1.plot((edges[1:] + edges[:-1]) / 2, np.bincount(np.clip(np.digitize(y[flag], edges) - 1, 0, N_BINS - 1), weights=w[flag], minlength=N_BINS) / w[flag].sum(), 'o-', ms=3, label='flagged')
a1.set(xlabel='y', ylabel='weighted fraction', yscale='log', title=f'{MODE}: starts along y'); a1.legend()
a2.bar(np.arange(10) / 10 + 0.05, qe_bins / qe_bins.sum(), width=0.09)
a2.set(xlabel='exact q', ylabel='weighted fraction of starts', yscale='log', title='starts along the exact committor (not used)')
fig.tight_layout(); fig.savefig(OUT + f'{TAG}_coverage.png', dpi=110)
if MAX_ITER == 0:
    sys.exit()

import cupy as cp
import optimalrcs, seed_runs
pts = [((0.50, 0.25), 0.639), ((0.55, 0.10), 0.664), ((0.62, 0.15), 0.698), ((0.70, 0.15), 0.747)]
free = ~(A | B)
NEAR = [np.flatnonzero(free & ((y - y0) ** 2 + (x - x0) ** 2 < 0.01 ** 2)) for (y0, x0), _ in pts]
ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
bm = seed_runs.BinnedMean((y, x), (ey, ex), free, 20)
Qe = bm(q_exact); trans = (Qe > 0.05) & (Qe < 0.95)
def report(q):
    r = cp.asnumpy(q.r_traj); d = bm(r) - Qe
    return (f'| RMS(fit - exact) per bin {np.sqrt(np.nanmean(d[trans] ** 2)):.4f} | fit/shooting: '
            + ' '.join(f'{r[m].mean():.3f}/{qs:.3f}' for m, (_, qs) in zip(NEAR, pts)))
X_cv = cp.asarray(np.column_stack([y, x]))
def make_q():
    return optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=seg, t_traj=tt.astype(float), path_weights=w, group_traj=grp)
results, q = seed_runs.run_seeds(
    optimalrcs, make_q, comp_y=lambda: X_cv[:, np.random.randint(2)],
    fit_kw=dict(ny=6, print_step=500, min_delta_x=1e-6, lazy_basis=True, save_min_metric='delta_r2',
                metrics_print=('iter', 'delta_r2', 'delta_x', 'time_elapsed')),
    seeds=[0], max_iter=MAX_ITER, chunk=2500, out_dir=OUT + TAG, envelope=0.01, gamma=0.1,
    warmup_iter=1000, warmup_envelope=1, warmup_gamma=0.1,
    plots=('metrics',), plot_options=dict(metrics=['delta_r2', 'delta_x']),
    extra_plot={'q_map': seed_runs.committor_map([seed_runs.plane(y, ey, 'y', x, ex, 'x')], mask=free, q_ref=q_exact)},
    report=report)
np.savez(OUT + f'{TAG}/fit.npz', r=cp.asnumpy(q.r_traj).astype(np.float32), y=y.astype(np.float32), x=x.astype(np.float32),
         flag=flag, w=w.astype(np.float32))
print('done', TAG, f'({time.time() - t0:.0f} s)', flush=True)
