"""Independent NPNEq fits on disjoint halves (odd / even path numbers) of the mirrored-pieces staple data
(sim_istarz_br_g10, rejection weights, --reversed, 1500 paths per ensemble, seed 0), with the settings of
load_tis_data_cupy.py (gamma 0.1, envelope 0.01, ny 6, lazy basis, A/B traps), from scratch.
Reports the error against the exact committor every 2500 iterations.
    python half_fits.py [N_ITER]
"""
import argparse, contextlib, io, sys, time
import numpy as np, cupy as cp
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import optimalrcs, tis_data
OUT = REPO + '/examples/figures/istarz_br_g10_reversed/'
N_ITER = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
CONTINUE = len(sys.argv) > 2 and sys.argv[2] == 'continue'
START_IT = 10000   # iterations already done by the saved halves

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([
        '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
        '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']))
ref = np.load(REPO + '/examples/figures/z_brownian/committor_reference_h0.001.npz')
f_ref = RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)
PTS = [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]

for half in ('even', 'odd'):
    keep = (data.group_traj % 2) == (0 if half == 'even' else 1)        # whole paths: both pieces stay together
    y, x = data.X_all[keep, 0], data.X_all[keep, 1]
    b0, b1, it, tt = data.boundary0[keep], data.boundary1[keep], data.i_traj[keep], data.t_traj[keep]
    w, g = data.path_weights[keep], data.group_traj[keep]
    w = w / w.mean()
    free = ~(b0 | b1); q_ex = np.clip(f_ref(np.c_[y, x]), 0, 1)
    ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
    nb = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
    bm = lambda v: np.where(nb >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(nb, 1), np.nan)
    QX = bm(q_ex); trans = (QX > 0.05) & (QX < 0.95); plateau = free & (q_ex > 0.55) & (q_ex < 0.85)
    near = [free & ((y - a) ** 2 + (x - b) ** 2 < 0.015 ** 2) for a, b in PTS]
    print(f'{half} paths: {keep.sum()} frames, {np.unique(g).size} paths', flush=True)
    X = cp.asarray(data.X[keep])
    import os
    r_file = OUT + f'r_half_{half}.npy'
    r_prev = np.load(r_file) if (CONTINUE and os.path.exists(r_file)) else None
    it0 = START_IT if r_prev is not None else 0
    last_plateau = None
    optimalrcs.envelope_scale = 0.01; t0 = time.time()
    for k in range(N_ITER // 2500):
        np.random.seed(k); cp.random.seed(k)
        q = optimalrcs.CommittorNE(boundary0=b0, boundary1=b1, i_traj=it, t_traj=tt, path_weights=w, group_traj=g, seed_r=r_prev)
        q.set_fixed_traj_length_trap(b0 | b1, 2 ** 16)
        q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], gamma=0.1, ny=6, max_iter=2500, print_step=2500,
                        metrics_print=('iter', 'max_z_zq_stopped', 'delta_x'), save_min_metric='max_z_zq_stopped', lazy_basis=True)
        r_prev = cp.asnumpy(q.r_traj); h = q.metrics_history
        plat = np.mean(r_prev[plateau] - q_ex[plateau])
        np.save(r_file, r_prev)
        print(f'  {half} {it0 + 2500 * (k + 1)} it. ({time.time() - t0:.0f} s, zqs_z {h["max_z_zq_stopped"][-1]:.2f}, |dx| {h["delta_x"][-1]:.1e}): '
              f'RMS per bin {np.sqrt(np.nanmean((bm(r_prev) - QX)[trans] ** 2)):.4f} | plateau (fit - exact) '
              f'{np.mean(r_prev[plateau] - q_ex[plateau]):+.4f} | at the points ' + ' '.join(f'{r_prev[m].mean():.3f}' for m in near)
              + ' (shooting 0.639 0.664 0.698 0.747)', flush=True)
        del q; cp.get_default_memory_pool().free_all_blocks()
        if last_plateau is not None and abs(plat - last_plateau) < 0.001 and h['delta_x'][-1] < 2e-4:
            print(f'  {half}: converged (plateau changed {plat - last_plateau:+.4f} over the last 2500 it.)', flush=True)
            break
        last_plateau = plat
    del X; cp.get_default_memory_pool().free_all_blocks()
print('ALL DONE', flush=True)
