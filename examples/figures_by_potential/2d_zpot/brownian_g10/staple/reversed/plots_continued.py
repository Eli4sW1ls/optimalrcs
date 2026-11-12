"""Plots of the continued mirrored-pieces staple fit (r_continued.npy, 25000 iterations): feps, TIS
diagnostics, obs vs pred, committor vs order and on the potential, and the convergence over both runs."""
import argparse, contextlib, io, re, sys, types
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import optimalrcs, tis_data
plt.show = lambda *a, **k: None
D = REPO + '/examples/figures_by_potential/2d_zpot/brownian_g10/staple/reversed/'
OUT = D + 'continued/'

p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
argv = ['/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10',
        '--reversed', '--paths-per-ensemble', '1500', '--seed', '0']
with contextlib.redirect_stdout(io.StringIO()):
    data = tis_data.prepare_data(p.parse_args(argv))
r = np.load(D + 'r_continued.npy')
assert len(r) == len(data.lam)

q = optimalrcs.CommittorNE(boundary0=data.boundary0, boundary1=data.boundary1, i_traj=data.i_traj, t_traj=data.t_traj,
                           path_weights=data.path_weights, group_traj=data.group_traj)
q.set_fixed_traj_length_trap(data.boundary0 | data.boundary1, 2 ** 16)
q.r_traj = cp.asarray(r)
for name, call in (('feps', q.plots_feps), ('tis_diagnostics', q.plots_tis), ('obs_pred', q.plots_obs_pred),
                   ('obs_pred_log', lambda: q.plots_obs_pred(log_scale=True))):
    call(); plt.gcf().savefig(OUT + f'{name}.png', dpi=100); plt.close('all')
args = types.SimpleNamespace(figure_dir=OUT, tis_dir=argv[0], lambda_a=0.05, lambda_b=0.85, no_potential=False,
                             potential='sjoelbak_angle:RectangularGridWithAngledBarrierPotential',
                             pot_kwargs=['Lx=0.3', 'angle=45'], engine_class=None)
tis_data.plot_committor(data, r, args)

# convergence over both runs: zqs_z from the 10000-iteration run's log, per-chunk numbers from continue.log
z1 = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r'#=(\d+), dr2=[-\d.e+]+, zqs_z=([-\d.e+]+)', open(D + 'run.log').read())]
chunks = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for m in
          re.finditer(r'^(\d+) it\..*?RMS per bin ([\d.]+) \| mean\(fit - exact\) on plateau ([+-][\d.]+)', open(D + 'continue.log').read(), re.M)]
start = re.search(r'start \(10000 it\.\): RMS per bin ([\d.]+) \| mean\(fit - exact\) on plateau ([+-][\d.]+)', open(D + 'continue.log').read())
chunks = [(10000, float(start.group(1)), float(start.group(2)))] + chunks
zc = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r'^(\d+) it\. \(\d+ s, zqs_z ([\d.]+)', open(D + 'continue.log').read(), re.M)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
a1.plot(*zip(*z1), 'o-', ms=3, label='first run (from scratch)'); a1.plot(*zip(*zc), 's-', label='continued (per 5000)')
a1.axhline(2, color='k', ls='--', lw=1); a1.set(xlabel='iteration', ylabel='zqs_z', yscale='log', title='convergence criterion (≲ 2)'); a1.legend(frameon=False)
it, rms, plat = zip(*chunks)
a2.plot(it, rms, 'o-', label='RMS fit − exact per bin'); a2.plot(it, plat, 's-', label='mean (fit − exact) on the plateau')
a2.axhline(0, color='k', lw=0.8); a2.set(xlabel='iteration', title='error against the exact committor'); a2.legend(frameon=False); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT + 'convergence.png', dpi=100)
print('saved', OUT, flush=True)
