"""Complete plot sets for docs/results_overview.md. One example per call (memory):
    python regen_all.py flat | wellpath | hist01 | zlgv_extra | zstaple_extra

flat, wellpath, hist01: refit with the original settings, recording every metric, and save
    metrics (original + convergence), feps, obs_pred (+ log), tis_diagnostics (path data) and
    example-specific comparison plots.
zlgv_extra, zstaple_extra: extra plots from the saved fits (no refit).
"""
import argparse, contextlib, io, os, sys, time
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, f'{REPO}/optimalrcs-cupy/'); sys.path.insert(0, f'{REPO}/examples/')
import optimalrcs, tis_data
plt.show = lambda *a, **k: None
FIG = f'{REPO}/examples/figures_by_potential/'
OVERVIEW_DIRS = [('flat', '1d_flat/brownian_staple'), ('md_well', '1d_cos_well/langevin_md'),
                 ('wellpath', '1d_cos_well/langevin_staple'), ('zmd', '2d_zpot/brownian_g10'),
                 ('zstaple', '2d_zpot/brownian_g10'), ('zlgv', '2d_zpot/langevin_g2')]


def overview_out(name):
    """Path of an overview plot: overview_plots/ of its potential, chosen by the start of the file name."""
    sub = next((d for p, d in OVERVIEW_DIRS if name.startswith(p)), '_shared')
    os.makedirs(f'{FIG}{sub}/overview_plots', exist_ok=True)
    return f'{FIG}{sub}/overview_plots/{name}'


SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/'
PATH_METRICS = ('iter', 'cross_entropy', 'mse', 'auc', 'delta_r2', 'max_sd_zq', 'max_z_zq_stopped', 'delta_x')
MD_METRICS = ('iter', 'cross_entropy', 'mse', 'auc', 'delta_r2', 'max_sd_zq', 'delta_x')


def save(name):
    plt.gcf().savefig(overview_out(name), dpi=100); plt.close('all')


def all_plots(q, tag, path_data, dt_sim=1):
    q.plots_metrics(['cross_entropy', 'mse', 'auc']); save(f'{tag}_metrics_original.png')
    q.plots_metrics(['delta_r2', 'max_z_zq_stopped' if path_data else 'max_sd_zq', 'delta_x'])
    save(f'{tag}_metrics_convergence.png')
    q.plots_feps(delta_t_sim=dt_sim); save(f'{tag}_feps.png')
    q.plots_obs_pred(); save(f'{tag}_obs_pred.png')
    q.plots_obs_pred(log_scale=True); save(f'{tag}_obs_pred_log.png')
    if path_data:
        q.plots_tis(); save(f'{tag}_tis_diagnostics.png')


def load_paths(argv):
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    with contextlib.redirect_stdout(io.StringIO()):
        return tis_data.prepare_data(p.parse_args(argv))


def fit_paths(data, X, max_iter, gamma=0.1, seed=0):
    q = optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj,
                               t_traj=data.t_traj, path_weights=data.path_weights, group_traj=data.group_traj)
    q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)
    Xg = cp.asarray(X); np.random.seed(seed); cp.random.seed(seed)
    q.fit_transform(lambda: Xg[:, np.random.randint(Xg.shape[1])], gamma=gamma, ny=6, max_iter=max_iter,
                    print_step=250, metrics_print=PATH_METRICS, save_min_metric='max_z_zq_stopped', lazy_basis=True)
    return q


def q_map(ax, a, b, ea, eb, r, mask, labels, vmin=0, vmax=1, cmap='coolwarm'):
    n = np.histogram2d(a[mask], b[mask], [ea, eb])[0]
    s = np.histogram2d(a[mask], b[mask], [ea, eb], weights=r[mask])[0]
    m = np.where(n > 20, s / np.maximum(n, 1), np.nan)
    im = ax.pcolormesh(ea, eb, m.T, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set(xlabel=labels[0], ylabel=labels[1]); plt.colorbar(im, ax=ax)
    return m


which = sys.argv[1]
t0 = time.time()
if which == 'flat':
    # 1D flat overdamped staple run, as load_tis_data_cupy.py with its defaults (3000 iterations)
    # all 33M frames do not fit the GPU for the stopped-Z_q metric: 12000 paths per ensemble (~8M frames)
    data = load_paths([SIM + 'sim_istar0br10intf', '--paths-per-ensemble', '12000', '--seed', '0'])
    print(f'{len(data.lam)} frames', flush=True)
    q = fit_paths(data, data.X, 3000)
    all_plots(q, 'flat_staple', True)
    r = cp.asnumpy(q.r_traj); lam = data.lam
    la, lb, dt, gam, kT = -0.1, 0.8, 2e-4, 1.0, 1.0
    s = 0.5826 * np.sqrt(2 * kT * dt / gam)                       # absorption checked every step
    edges = np.linspace(la, lb, 91); mid = (edges[1:] + edges[:-1]) / 2
    k = np.digitize(lam, edges) - 1; ok = (k >= 0) & (k < len(mid))
    qb = np.bincount(k[ok], weights=r[ok], minlength=len(mid)) / np.maximum(np.bincount(k[ok], minlength=len(mid)), 1)
    exact = (mid - la + s) / (lb - la + 2 * s)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(mid, qb, label='NPNEq (staple paths)'); a1.plot(mid, exact, 'k--', label='exact'); a1.legend(); a1.grid()
    a1.set(xlabel='$\\lambda$', ylabel='q')
    a2.plot(mid, qb - exact); a2.axhline(0, color='k', lw=0.8); a2.grid()
    a2.set(xlabel='$\\lambda$', ylabel='fit - exact', title=f'RMS {np.sqrt(np.mean((qb - exact) ** 2)):.4f}')
    fig.tight_layout(); save('flat_staple_vs_exact.png')
elif which == 'wellpath':
    # 1D underdamped well staple run with recorded velocities, lag-1 fit on (x, v)
    data = load_paths([SIM + 'sim_istarwell2rv', '--paths-per-ensemble', '2000', '--seed', '0'])
    print(f'{len(data.lam)} frames, CVs {data.X.shape[1]}', flush=True)
    q = fit_paths(data, data.X[:, :2], 4000)
    all_plots(q, 'wellpath', True, dt_sim=0.005)
    r = cp.asnumpy(q.r_traj); x, v = data.X[:, 0], data.X[:, 1]
    ref = np.load(FIG + '1d_cos_well/langevin_staple/checks/reference.npz')
    f = RegularGridInterpolator((ref['xs'], ref['vs']), ref['q'], bounds_error=False, fill_value=None)
    qx = np.clip(f(np.c_[np.clip(x, ref['xs'][0], ref['xs'][-1]), np.clip(v, ref['vs'][0], ref['vs'][-1])]), 0, 1)
    free = ~(data.boundary0 | data.boundary1)
    ex, ev = np.linspace(-0.35, 0.35, 36), np.linspace(-3, 3, 31)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    q_map(axes[0], x, v, ex, ev, r, free, ('x', 'v')); axes[0].set_title('fit')
    q_map(axes[1], x, v, ex, ev, qx, free, ('x', 'v')); axes[1].set_title('exact (brute-force reference)')
    q_map(axes[2], x, v, ex, ev, r - qx, free, ('x', 'v'), -0.15, 0.15, 'PuOr')
    axes[2].set_title(f'fit - exact, RMS per frame {np.sqrt(np.mean((r[free] - qx[free]) ** 2)):.3f}')
    fig.tight_layout(); save('wellpath_q_x_v.png')
elif which == 'hist01':
    # 1D underdamped well MD, CV x only with history [0, 1] (base_documented_1d_langevin.ipynb)
    d = np.load(SIM + 'md_istarwell2_100M/md_path.npy', mmap_mode='r')[:6_000_000]
    lam, v = np.array(d[:, 0], float), np.array(d[:, 1], float)
    optimalrcs.envelope_scale = 0.001
    q = optimalrcs.CommittorNE(boundary0=lam <= -0.35, boundary1=lam >= 0.35, i_traj=np.ones(len(lam)),
                               t_traj=np.arange(len(lam)))
    y = cp.asarray(lam); np.random.seed(0); cp.random.seed(0)
    q.fit_transform(lambda: y, history_delta_t=[0, 1], history_type=['y(t-d),y(t)'], gamma=0.2, ny=6,
                    max_iter=1500, print_step=100, metrics_print=MD_METRICS, save_min_delta_zq=False, lazy_basis=True)
    all_plots(q, 'md_well_hist01', False, dt_sim=0.005)
    r = cp.asnumpy(q.r_traj)
    fig, ax = plt.subplots(figsize=(7, 5))
    q_map(ax, lam, v, np.linspace(-0.35, 0.35, 71), np.linspace(-3, 3, 61), r, np.ones(len(r), bool), ('$\\lambda$', '$v$'))
    ax.set_title('history [0, 1] fit (v not used in training)'); save('md_well_hist01_q_x_v.png')
elif which == 'zlgv_extra':
    # 2D underdamped Z staple, saved lag-4 fit with ridge-normal CVs
    f = np.load(FIG + '2d_zpot/langevin_g2/fd_velocity/frames.npz'); r = np.load(FIG + '2d_zpot/langevin_g2/fd_velocity/q_diag_lag4.npy')
    lam, x, vy, vx = f['lam'], f['x'], f['vy'] / 0.002, f['vx'] / 0.002
    free = ~(f['b0'] | f['b1'])
    el = np.linspace(0.05, 0.85, 81); vmax = 4 * np.sqrt(0.05)
    sys.path.insert(0, '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis')
    from potentials.sjoelbak_angle import RectangularGridWithAngledBarrierPotential
    pot = RectangularGridWithAngledBarrierPotential(Lx=0.3, angle=45)
    gy, gx = np.linspace(-0.01, 0.91, 185), np.linspace(-0.01, 0.31, 65)
    V = np.array([[pot.effective_potential((np.array([a, b]), np.zeros(2))) for b in gx] for a in gy])
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    q_map(axes[0], lam, x, el, np.linspace(-0.01, 0.31, 33), r, free, ('y (order parameter)', 'x'))
    axes[0].contour(gy, gx, V.T, levels=np.linspace(np.percentile(V, 1), np.percentile(V, 92), 14), colors='0.4', linewidths=0.5)
    axes[0].set_xlim(0.05, 0.85)
    q_map(axes[1], lam, vy, el, np.linspace(-vmax, vmax, 41), r, free, ('y', '$v_y$ (finite difference)'))
    q_map(axes[2], lam, vx, el, np.linspace(-vmax, vmax, 41), r, free, ('y', '$v_x$ (finite difference)'))
    fig.tight_layout(); save('zlgv_planes.png')
    fig, ax = plt.subplots(figsize=(7, 5)); show = np.random.default_rng(0).permutation(len(r))[:300_000]
    ax.scatter(lam[show], r[show], s=1, alpha=0.2, edgecolors='none'); ax.set(xlabel='y (order parameter)', ylabel='q')
    save('zlgv_committor_vs_order.png')
elif which == 'zstaple_extra':
    d = np.load(FIG + '2d_zpot/brownian_g10/staple/original/committor.npz')
    keep = d['i_traj'] % 2 == 0
    q = optimalrcs.CommittorNE(boundary0=d['boundary0'][keep], boundary1=d['boundary1'][keep], i_traj=d['i_traj'][keep],
                               t_traj=d['t_traj'][keep], path_weights=d['path_weights'][keep])
    q.set_fixed_traj_length_trap(q.b_traj > 0, 2 ** 16)
    q.r_traj = cp.asarray(d['r'][keep]); del d
    q.plots_obs_pred(log_scale=True); save('zstaple_obs_pred_log.png')
print(f'done {which} in {time.time() - t0:.0f} s', flush=True)
