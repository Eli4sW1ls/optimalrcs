"""Lag-Delta NPNEq on the two disjoint halves (even / odd path numbers) of the mirrored-pieces staple data
(sim_istarz_br_g10, rejection weights, --reversed, 1500 paths per ensemble, seed 0). Windows
[t, min(t + LAG, tau_t)], tau_t = first A/B frame or the last frame of the piece; each window weighted by its
path weight. Otherwise the staple settings (gamma 0.1, envelope 0.01, ny 6, A/B traps for the metrics).
    python lag_halves.py LAG N_ITER [halves, e.g. even,odd or all]
"""
import argparse, contextlib, io, sys, time
import numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import optimalrcs, tis_data
OUT = REPO + '/examples/figures/istarz_br_g10_reversed/'
LAG = int(sys.argv[1]); N_ITER = int(sys.argv[2])
HALVES = sys.argv[3].split(',') if len(sys.argv) > 3 else ['even', 'odd']
import os
PPE = os.environ.get('PPE', '1500')   # paths per ensemble ('all' = every accepted path)
CHUNK = 1000

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([
        '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
        '--reversed'] + ([] if PPE == 'all' else ['--paths-per-ensemble', PPE, '--seed', '0'])))
ref = np.load(REPO + '/examples/figures/z_brownian/committor_reference_h0.001.npz')
f_ref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]


class LazyRY:
    def __init__(self, r, y, k, fenv):
        self.r, self.y, self.n = r / cp.max(cp.abs(r)), y / cp.max(cp.abs(y)), k
        self.fenv = cp.ones_like(r) if fenv is None else fenv
    def cols(self, idx):
        return optimalrcs._poly_ry(self.r[idx], self.y[idx], self.n, self.fenv[idx])


for half in HALVES:
    if half == 'all':          # use the arrays in place: copies of 42M-frame arrays do not fit in memory
        keep = slice(None)
        y, x = data.X_all[:, 0], data.X_all[:, 1]
        b0, b1, it, tt, w, g = data.boundary0, data.boundary1, data.i_traj, data.t_traj, data.path_weights, data.group_traj
    else:
        keep = (data.group_traj % 2) == (0 if half == 'even' else 1)
        y, x = data.X_all[keep, 0], data.X_all[keep, 1]
        b0, b1, it, tt = data.boundary0[keep], data.boundary1[keep], data.i_traj[keep], data.t_traj[keep]
        w, g = data.path_weights[keep], data.group_traj[keep]; w = w / w.mean()
    n = len(y); b = b0 | b1; free = ~b
    ends = np.r_[np.flatnonzero(it[1:] != it[:-1]), n - 1]; starts_ = np.r_[0, ends[:-1] + 1]
    end_of = np.repeat(ends, ends - starts_ + 1)
    nxt = np.minimum.accumulate(np.where(b, np.arange(n), n)[::-1])[::-1]
    tau = np.minimum(nxt, end_of)
    u_np = np.minimum(np.arange(n) + LAG, tau)
    U = cp.asarray(u_np.astype(np.int32)); W = cp.asarray(w)
    STARTS = cp.asarray(np.flatnonzero(free & (u_np > np.arange(n))).astype(np.int32))
    del nxt, end_of, tau, u_np

    def npneq_lag(r_traj, fk, i_traj=None, gamma=0, stable=False, train_mask=None, chunk=None, w_traj=None, **kw):
        akj, bb = 0, 0
        for s0 in range(0, STARTS.size, 400_000):
            st = STARTS[s0:s0 + 400_000]
            fa, fb, wt = fk.cols(st), fk.cols(U[st]), W[st]
            akj = akj + fa @ (fb * wt - fa * (wt + gamma)).T
            bb = bb + fa @ (-(r_traj[U[st]] - r_traj[st]) * wt)
        al = cp.linalg.lstsq(akj, bb, rcond=None)[0]
        new = r_traj.copy()
        for s0 in range(0, r_traj.size, 1_000_000):
            idx = cp.arange(s0, min(s0 + 1_000_000, r_traj.size))
            new[idx] = r_traj[idx] + al @ fk.cols(idx)
        return cp.clip(new, 0, 1)
    optimalrcs.nonparametrics.npneq = npneq_lag

    q_ex = np.clip(f_ref(np.c_[y, x]), 0, 1)
    ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
    nb = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
    bm = lambda v: np.where(nb >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(nb, 1), np.nan)
    QX = bm(q_ex); trans = (QX > 0.05) & (QX < 0.95); plateau = free & (q_ex > 0.55) & (q_ex < 0.85)
    near = [free & ((y - a) ** 2 + (x - c) ** 2 < 0.015 ** 2) for a, c in PTS]
    print(f'lag {LAG}, {half} paths: {n} frames, {int(STARTS.size)} windows', flush=True)
    X = cp.asarray(data.X[keep], dtype=cp.float64); r_prev = None
    optimalrcs.envelope_scale = 0.01; t0 = time.time()
    for k in range(N_ITER // CHUNK):
        np.random.seed(k); cp.random.seed(k)
        q = optimalrcs.CommittorNE(boundary0=b0, boundary1=b1, i_traj=it, t_traj=tt, path_weights=w, group_traj=g, seed_r=r_prev)
        q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], basis_functions=lambda r_, y_, k_, f_: LazyRY(r_, y_, k_, f_),
                        gamma=0.1, ny=6, max_iter=CHUNK, print_step=CHUNK, metrics_print=('iter', 'delta_x'), save_min_delta_zq=False)
        r_prev = cp.asnumpy(q.r_traj)
        print(f'  lag {LAG} {half} {CHUNK * (k + 1)} it. ({time.time() - t0:.0f} s, |dx| {q.metrics_history["delta_x"][-1]:.1e}): '
              f'RMS per bin {np.sqrt(np.nanmean((bm(r_prev) - QX)[trans] ** 2)):.4f} | plateau (fit - exact) '
              f'{np.mean(r_prev[plateau] - q_ex[plateau]):+.4f} | at the points ' + ' '.join(f'{r_prev[m].mean():.3f}' for m in near)
              + ' (shooting 0.639 0.664 0.698 0.747)', flush=True)
        del q; cp.get_default_memory_pool().free_all_blocks()
    np.save(OUT + f'r_lag{LAG}_{half}_ppe{PPE}.npy', r_prev)
    del X, U, W, STARTS; cp.get_default_memory_pool().free_all_blocks()
print('ALL DONE', flush=True)
