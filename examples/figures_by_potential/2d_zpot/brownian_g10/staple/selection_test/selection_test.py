"""Can the TIS data be balanced over regions without extra MD? (2D Brownian Z, sim_istarz_br_g10)

Three versions of the same data (1500 paths per ensemble, seed 0, forward + mirrored pieces, rejection weights):
  base          as they are
  flat_qn       frame weights: path weight x g(q_n(x_t)), g flattening the weighted frame density over
                30 bins of the natural coordinate q_n of a first fit (r_continued.npy, fitted on these data).
                g depends on the frame's position only, so E[q(x_{t+1}) - q(x_t) | x_t] = 0 still holds.
  select_paths  whole paths (both pieces) kept with probability min(1, K / N_rarest), N_rarest the number of
                paths visiting the rarest q_n bin the path visits, K the 25% quantile of the bin counts;
                weights not corrected. The selection depends on where the path goes (its future).
    python selection_test.py diag              bias tests with the exact committor (no fit)
    python selection_test.py fit <version>     NPNEq fit (warm-up 1000 at envelope 1, 10000 at 0.01, gamma 0.1)
"""
import argparse, contextlib, io, os, sys, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import optimalrcs, plots, seed_runs, tis_data, toy2d
plt.show = lambda *a, **k: None
OUT = REPO + '/examples/figures/selection_test/'
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10'
N_BINS = 30

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([SIM, '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']))
y, x = data.X_all[:, 0], data.X_all[:, 1]
A, B = data.boundary0, data.boundary1
b = A | B
free = ~b
it, grp, w0, t_traj = data.i_traj, data.group_traj, data.path_weights, data.t_traj
ref = np.load(REPO + '/examples/figures/z_brownian/committor_reference_h0.001.npz')
f_ref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
q_exact = np.clip(f_ref(np.c_[y, x]), 0, 1); q_exact[A] = 0; q_exact[B] = 1

# natural coordinate of a first fit on these data
r_first = np.load(REPO + '/examples/figures/istarz_br_g10_reversed/r_continued.npy')
assert len(r_first) == len(y)
qn = plots.transform_q2qn(cp.asarray(r_first), i_traj=cp.asarray(it), t_traj=cp.asarray(t_traj),
                          w_traj=cp.asarray(w0), dt_sim=0.002)
qn_bin = np.clip((qn / qn.max() * N_BINS).astype(int), 0, N_BINS - 1)

def version(name):
    """(frame weights, kept frames mask) of a data version."""
    if name == 'base':
        return w0.copy(), np.ones(len(y), bool)
    if name == 'flat_qn':
        rho = np.bincount(qn_bin, weights=w0, minlength=N_BINS)
        g = np.clip(rho.mean() / np.maximum(rho, 1e-12), 0, 20)          # cap: no frame weight above 20x
        w = w0 * g[qn_bin]
        return w / w.mean(), np.ones(len(y), bool)
    if name == 'select_paths':
        _, g_idx = np.unique(grp, return_inverse=True)
        n_g = g_idx.max() + 1
        visits = np.zeros((n_g, N_BINS), bool)
        visits[g_idx, qn_bin] = True
        n_bin = visits.sum(0)                                             # paths visiting each bin
        rarest = np.where(visits, n_bin[None, :], np.iinfo(int).max).min(1)
        K = np.quantile(n_bin, 0.25)
        keep_g = np.random.default_rng(0).random(n_g) < np.minimum(1, K / rarest)
        return w0.copy(), keep_g[g_idx]
    raise ValueError(name)

def coverage(w, keep):
    return np.bincount(qn_bin[keep], weights=w[keep], minlength=N_BINS)

def martingale(w, keep, band):
    """Per piece: D = q_exact(stop) - q_exact(first frame in band); weighted mean +- s.e. over paths."""
    yk, itk, bk, qk, wk, gk = y[keep], it[keep], b[keep], q_exact[keep], w[keep], grp[keep]
    n = len(qk)
    ends = np.r_[np.flatnonzero(itk[1:] != itk[:-1]), n - 1]; starts = np.r_[0, ends[:-1] + 1]
    marker = np.where(bk, np.arange(n), n); nxt = np.minimum.accumulate(marker[::-1])[::-1]
    stop = np.minimum(nxt, np.repeat(ends, ends - starts + 1))
    idx = np.where((~bk) & (qk > band[0]) & (qk < band[1]), np.arange(n), n)
    first = np.minimum.reduceat(idx, starts); ok = first < n; f = first[ok]
    D, W, G = qk[stop[f]] - qk[f], wk[f], gk[f]          # weight of the frame where the piece enters the band
    m = np.sum(W * D) / np.sum(W)
    gs = np.bincount(np.unique(G, return_inverse=True)[1], weights=W * (D - m))
    return m, np.sqrt(np.sum(gs ** 2)) / np.sum(W), len(D)

if sys.argv[1] == 'diag':
    LAGS = (1, 16, 256, 4096)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ('base', 'flat_qn', 'select_paths'):
        w, keep = version(name)
        z = toy2d.z_per_lag(q_exact[keep], b[keep], it[keep], w[keep], LAGS, grp[keep])
        ess = w[keep].sum() ** 2 / (w[keep] ** 2).sum() / keep.sum()
        print(f'=== {name}: {keep.sum():,} frames ({keep.mean():.0%}), {len(np.unique(grp[keep]))} paths, frame-weight ESS/N {ess:.2f}')
        print(f'  exact committor, worst |stopped z| at lags {LAGS}: ' + ' '.join(f'{v:.1f}' for v in z))
        for band in ((0.05, 0.3), (0.3, 0.55), (0.55, 0.7), (0.7, 0.85)):
            m, se, nD = martingale(w, keep, band)
            print(f'  martingale, band {band}: E[D] = {m:+.4f} +- {se:.4f} ({m / se:+.1f} sigma, {nD} pieces)')
        cov = coverage(w, keep)
        ax.plot(np.arange(N_BINS), cov / cov.sum(), 'o-', ms=3, label=name)
        sys.stdout.flush()
    ax.set(xlabel=f'q_n bin (of {N_BINS})', ylabel='weighted fraction of frames', yscale='log', title='coverage along q_n')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT + 'coverage.png', dpi=110)
    sys.exit()

name = sys.argv[2]
w, keep = version(name)
ix = np.flatnonzero(keep)
pts = [((0.50, 0.25), 0.639), ((0.55, 0.10), 0.664), ((0.62, 0.15), 0.698), ((0.70, 0.15), 0.747)]
yk, xk, fk = y[ix], x[ix], free[ix]
NEAR = [np.flatnonzero(fk & ((yk - y0) ** 2 + (xk - x0) ** 2 < 0.01 ** 2)) for (y0, x0), _ in pts]
ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
bm = seed_runs.BinnedMean((yk, xk), (ey, ex), fk, 20)
Qe = bm(q_exact[ix]); trans = (Qe > 0.05) & (Qe < 0.95)
def report(q):
    r = cp.asnumpy(q.r_traj)
    d = bm(r) - Qe
    return (f'| RMS(fit - exact) per bin {np.sqrt(np.nanmean(d[trans] ** 2)):.4f} | fit/shooting: '
            + ' '.join(f'{r[n].mean():.3f}/{qs:.3f}' for n, (_, qs) in zip(NEAR, pts)))
X_cv = cp.asarray(np.column_stack([yk, xk]))
def make_q():
    q = optimalrcs.CommittorNE(boundary0=A[ix], boundary1=B[ix], i_traj=it[ix], t_traj=t_traj[ix],
                               path_weights=w[ix], group_traj=grp[ix])
    q.set_fixed_traj_length_trap(b[ix], 2 ** 16)
    return q
del data
results, q = seed_runs.run_seeds(
    optimalrcs, make_q, comp_y=lambda: X_cv[:, np.random.randint(2)],
    fit_kw=dict(ny=6, print_step=500, min_delta_x=1e-6, lazy_basis=True, save_min_metric='max_z_zq_stopped',
                metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x', 'time_elapsed')),
    seeds=[0], max_iter=10000, chunk=2500, out_dir=OUT + name, envelope=0.01, gamma=0.1,
    warmup_iter=1000, warmup_envelope=1, warmup_gamma=0.1,
    plots=('metrics', 'tis'), plot_options=dict(metrics=['delta_r2', 'max_z_zq_stopped', 'delta_x']),
    extra_plot={'q_map': seed_runs.committor_map([seed_runs.plane(yk, ey, 'y', xk, ex, 'x')], mask=fk, q_ref=q_exact[ix])},
    report=report)
np.save(OUT + f'{name}/r.npy', cp.asnumpy(q.r_traj).astype(np.float32))
print('done', name, flush=True)
