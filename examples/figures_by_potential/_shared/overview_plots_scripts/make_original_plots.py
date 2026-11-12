"""Original optimalrcs plots (plots_feps, plots_obs_pred, metrics) for the examples in
docs/results_overview.md that did not save them. One example per call, to keep memory low:
    python make_original_plots.py zmd | zstaple | zlgv | wellpath
"""
import argparse, contextlib, io, os, re, sys
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
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


def save_feps_obs(q, tag, dt_sim=1):
    q.plots_feps(delta_t_sim=dt_sim); plt.gcf().savefig(overview_out(f'{tag}_feps.png'), dpi=100); plt.close('all')
    q.plots_obs_pred(); plt.gcf().savefig(overview_out(f'{tag}_obs_pred.png'), dpi=100); plt.close('all')


def metrics_plot(series, tag, title):
    fig, axes = plt.subplots(1, len(series), figsize=(5 * len(series), 3.8))
    for ax, (lab, x, y) in zip(np.atleast_1d(axes), series):
        ax.plot(x, y, 'o-', ms=3); ax.set(xlabel='iteration', ylabel=lab); ax.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(overview_out(f'{tag}_metrics.png'), dpi=100); plt.close(fig)


def load_paths(argv):
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    with contextlib.redirect_stdout(io.StringIO()):
        return tis_data.prepare_data(p.parse_args(argv))


def committor_ne(b0, b1, i_traj, t_traj, w=None, traps=True):
    q = optimalrcs.CommittorNE(boundary0=b0, boundary1=b1, i_traj=i_traj, t_traj=t_traj, path_weights=w)
    if traps:
        q.set_fixed_traj_length_trap(b0 | b1, 2 ** 16)
    return q


which = sys.argv[1]
if which == 'zmd':
    d = np.load(FIG + '2d_zpot/brownian_g10/reference_and_analyses/npq_md/npq_md.npz')
    it = d['metric_iter']
    metrics_plot([('delta_r2', it, d['metric_delta_r2']), ('max_sd_zq', it, d['metric_max_sd_zq']),
                  ('auc', it, d['metric_auc']), ('|dx|', it, d['metric_delta_x'])],
                 'zmd', '2D Brownian Z, long MD, NPq (all 16 seeds)')
elif which == 'zstaple':
    d = np.load(FIG + '2d_zpot/brownian_g10/staple/original/committor.npz')
    keep = d['i_traj'] % 2 == 0                            # every second path, to limit memory
    q = committor_ne(d['boundary0'][keep], d['boundary1'][keep], d['i_traj'][keep], d['t_traj'][keep],
                     d['path_weights'][keep])
    q.r_traj = cp.asarray(d['r'][keep])
    del d
    save_feps_obs(q, 'zstaple')
elif which == 'zlgv':
    data = load_paths([SIM + 'sim_istarz_2603', '--paths-per-ensemble', '1000', '--seed', '0', '--fd-velocity'])
    q = committor_ne(data.boundary0, data.boundary1, data.i_traj, data.t_traj, data.path_weights)
    q.r_traj = cp.asarray(np.load(FIG + '2d_zpot/langevin_g2/fd_velocity/q_diag_lag4.npy'))
    save_feps_obs(q, 'zlgv')
    line = [l for l in open(FIG + '2d_zpot/langevin_g2/fd_velocity/run_diag.log') if l.startswith('lag4:') and 'zqs_z' in l][0]
    z = [float(v) for v in re.search(r'\[(.*?)\]', line).group(1).split(',')]
    metrics_plot([('max_z_zq_stopped (zqs_z)', 250 * np.arange(1, len(z) + 1), z)], 'zlgv',
                 '2D Langevin Z staple, CVs y, x, dy, dx, y+x, dy+dx, lag-4 fit')
elif which == 'wellpath':
    data = load_paths([SIM + 'sim_istarwell2rv', '--paths-per-ensemble', '2000', '--seed', '0', '--fd-velocity'])
    q = committor_ne(data.boundary0, data.boundary1, data.i_traj, data.t_traj, data.path_weights)
    q.r_traj = cp.asarray(np.load(FIG + '1d_cos_well/langevin_staple/checks/fit_recorded_lag1.npz')['r'])
    save_feps_obs(q, 'wellpath', dt_sim=0.005)
print('done', which, flush=True)
