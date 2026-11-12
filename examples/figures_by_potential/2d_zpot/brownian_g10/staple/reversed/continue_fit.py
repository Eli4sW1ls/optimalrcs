"""Continue the staple refit with mirrored pieces (sim_istarz_br_g10, rejection weights, --reversed,
1500 paths per ensemble, seed 0) from its saved q, with the same settings as load_tis_data_cupy.py
(gamma 0.1, envelope 0.01, ny 6, lazy basis, A/B traps). Chunks of CHUNK iterations; after each chunk
the RMS against the exact committor and q at the shooting points are logged and q is saved.
    python continue_fit.py [N_CHUNKS] [CHUNK]
"""
import argparse, contextlib, io, os, sys, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import optimalrcs, tis_data
plt.show = lambda *a, **k: None
OUT = REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/staple/reversed/'
N_CHUNKS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CHUNK = int(sys.argv[2]) if len(sys.argv) > 2 else 5000

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args([
        '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
        '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']))
saved = np.load(OUT + 'committor.npz')
assert np.array_equal(saved['lam'], data.lam) and np.array_equal(saved['i_traj'], data.i_traj), 'frames differ'
r_prev = saved['r']
print(f'{len(data.lam)} frames, same as the saved fit; continuing for {N_CHUNKS} x {CHUNK} iterations', flush=True)

y, x = data.X_all[:, 0], data.X_all[:, 1]
free = ~(data.boundary0 | data.boundary1)
ref = np.load(REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/reference_and_analyses/committor_reference_h0.001.npz')
q_ex = np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)
ey, ex = np.linspace(0.05, 0.85, 81), np.linspace(-0.01, 0.31, 33)
nb = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
bm = lambda v: np.where(nb >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(nb, 1), np.nan)
QX = bm(q_ex); trans = (QX > 0.05) & (QX < 0.95); plateau = free & (q_ex > 0.55) & (q_ex < 0.85)
near = [free & ((y - a) ** 2 + (x - b) ** 2 < 0.01 ** 2) for a, b in [(0.50, 0.25), (0.55, 0.10), (0.62, 0.15), (0.70, 0.15)]]
def report(r, label):
    print(f'{label}: RMS per bin {np.sqrt(np.nanmean((bm(r) - QX)[trans] ** 2)):.4f} | mean(fit - exact) on plateau '
          f'{np.mean(r[plateau] - q_ex[plateau]):+.4f} | at the shooting points '
          + ' '.join(f'{r[m].mean():.3f}' for m in near) + ' (shooting 0.639 0.664 0.698 0.747)', flush=True)
report(r_prev, 'start (10000 it.)')

X = cp.asarray(data.X)
optimalrcs.envelope_scale = 0.01
t0 = time.time()
for k in range(N_CHUNKS):
    np.random.seed(100 + k); cp.random.seed(100 + k)
    q = optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj, t_traj=data.t_traj,
                               path_weights=data.path_weights, group_traj=data.group_traj, seed_r=r_prev)
    q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)
    q.fit_transform(lambda: X[:, np.random.randint(X.shape[1])], gamma=0.1, ny=6, max_iter=CHUNK, print_step=500,
                    min_delta_x=1e-6, metrics_print=('iter', 'delta_r2', 'max_z_zq_stopped', 'max_sd_zq_stopped', 'delta_x'),
                    save_min_metric='max_z_zq_stopped', lazy_basis=True)
    r_prev = cp.asnumpy(q.r_traj)
    h = q.metrics_history
    report(r_prev, f'{10000 + CHUNK * (k + 1)} it. ({time.time() - t0:.0f} s, zqs_z {h["max_z_zq_stopped"][-1]:.2f}, '
                   f'|dx| {h["delta_x"][-1]:.1e})')
    np.save(OUT + 'r_continued.npy', r_prev)
    if k == N_CHUNKS - 1:
        tis_data.save_fit_figures(q, OUT + 'continued', 'iter')
        q.plots_obs_pred(); plt.gcf().savefig(OUT + 'continued/obs_pred.png', dpi=100); plt.close('all')
    del q; cp.get_default_memory_pool().free_all_blocks()
print('ALL DONE', flush=True)
