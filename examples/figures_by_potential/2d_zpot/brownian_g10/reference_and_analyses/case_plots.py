"""Per-case plots for the 2D Brownian Z investigation (docs/results_overview.md, notes section 9.2): for every
fit, the original optimalrcs plots (feps, obs vs pred), the path-data diagnostics (plots_tis), the fit vs the
exact committor on the potential, and the convergence from the run's log. Output: cases/<case>/.
    python case_plots.py synth | staple1500 | staple2250 | forward | md
"""
import argparse, contextlib, io, os, re, sys
import numpy as np, cupy as cp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
REPO = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/OP_improvement/optimalrcs'
sys.path.insert(0, REPO + '/optimalrcs-cupy/'); sys.path.insert(0, REPO + '/examples/')
import optimalrcs, tis_data
plt.show = lambda *a, **k: None
F = REPO + '/examples/figures_by_potential/'
Z = F + '2d_zpot/brownian_g10/reference_and_analyses/'; R = F + '2d_zpot/brownian_g10/staple/reversed/'
OUT = Z + 'cases/'
SIM = '/mnt/0bf0c339-34bb-4500-a5fb-f3c2a863de29/DATA/PyRETIS3/toytis/simulations/sim_istarz_br_g10'
ref_st = np.load(Z + 'committor_reference_h0.001.npz')                     # A 0.05, B 0.85, every step
ref_md = np.load(Z + 'committor_reference_h0.001_A0.1_B0.85_s20.npz')      # A 0.1, B 0.85, every 20 steps
V = ref_st['V_eff']; vy, vx = ref_st['y'], ref_st['x']; inside = (vy >= -0.01) & (vy <= 0.91)
LEV = np.linspace(np.percentile(V[inside], 1), np.percentile(V[inside], 92), 14)


def interp(ref, y, x):
    return np.clip(RegularGridInterpolator((ref['y'], ref['x']), ref['q'], bounds_error=False, fill_value=None)(np.c_[y, x]), 0, 1)


def save(path):
    plt.gcf().savefig(path, dpi=95); plt.close('all')


def map_plot(case, title, y, x, r, free, ref, lam_a):
    q_ex = interp(ref, y, x)
    ey, ex = np.linspace(lam_a, 0.85, 81), np.linspace(-0.01, 0.31, 33)
    n = np.histogram2d(y[free], x[free], bins=(ey, ex))[0]
    bm = lambda v: np.where(n >= 20, np.histogram2d(y[free], x[free], bins=(ey, ex), weights=v[free])[0] / np.maximum(n, 1), np.nan)
    Fq, E = bm(r), bm(q_ex); tr = (E > 0.05) & (E < 0.95)
    rms = np.sqrt(np.nanmean((Fq - E)[tr] ** 2)); plateau = free & (q_ex > 0.55) & (q_ex < 0.85)
    off = np.mean(r[plateau] - q_ex[plateau])
    cy, cx = (ey[1:] + ey[:-1]) / 2, (ex[1:] + ex[:-1]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.3))
    for ax, (M, t, cm, lim) in zip(axes, [(Fq, f'{title}: fit', 'coolwarm', 1), (E, 'exact committor', 'coolwarm', 1),
                                          (Fq - E, f'fit − exact (RMS {rms:.3f}, plateau {off:+.3f})', 'PuOr', 0.15)]):
        im = ax.imshow(M.T, origin='lower', extent=[ey[0], ey[-1], ex[0], ex[-1]], aspect='auto', cmap=cm,
                       vmin=0 if lim == 1 else -lim, vmax=lim)
        ax.contour(vy[inside], vx, V[inside].T, levels=LEV, colors='0.4', linewidths=0.5)
        if lim == 1:
            ax.contour(cy, cx, np.nan_to_num(M, nan=0.5).T, levels=[0.1, 0.5, 0.9], colors='k', linewidths=1, linestyles='--')
        ax.set(title=t, xlabel='order parameter y', ylabel='x', xlim=(lam_a - 0.01, 0.86), ylim=(-0.01, 0.31)); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT + case + '/map.png', dpi=95); plt.close(fig)
    np.savez(OUT + case + '/map.npz', fit=Fq, exact=E, ey=ey, ex=ex, rms=rms, plateau=off, title=title)   # for summary_plots.py
    return rms, off


def convergence_plot(case, log, pattern):
    """Plateau offset, RMS and zqs_z per logged chunk: lines of `log` matching `pattern` (regex with groups it, rest)."""
    if not os.path.exists(log):
        return
    cum = pattern.startswith('CUM:'); pattern = pattern.removeprefix('CUM:')   # CUM: iteration counts restart per stage
    rows = []; base = 0; last = 0
    for line in open(log):
        m = re.search(pattern, line)
        if not m:
            continue
        it = int(m.group(1)); rest = line
        if cum:
            base += last if it <= last else 0; last = it; it += base
        rms = re.search(r'RMS per bin ([\d.]+)', rest); off = re.search(r'plateau(?: \(fit - exact\))? ([+-][\d.]+)', rest)
        z = re.search(r'zqs_z ([\d.]+)', rest)
        rows = [r_ for r_ in rows if r_[0] != it]          # a rerun in the same log replaces earlier lines
        rows.append((it, float(rms.group(1)) if rms else np.nan, float(off.group(1)) if off else np.nan,
                     float(z.group(1)) if z else np.nan))
    if not rows:
        return
    it, rms, off, z = map(np.array, zip(*rows))
    has_z = np.isfinite(z).any()
    fig, axes = plt.subplots(1, 2 if has_z else 1, figsize=(12 if has_z else 6.5, 4)); a1 = axes[0] if has_z else axes
    a1.plot(it, off, 'o-', label='plateau (fit − exact)'); a1.plot(it, rms, 's-', label='RMS fit − exact per bin')
    a1.axhline(0, color='k', lw=0.8); a1.set(xlabel='iteration', title=f'{case}: error against the exact committor'); a1.legend(frameon=False); a1.grid(alpha=0.3)
    if has_z:
        a2 = axes[1]; a2.plot(it, z, 'o-'); a2.axhline(2, color='k', ls='--', lw=1)
        a2.set(xlabel='iteration', ylabel='zqs_z', title='convergence criterion (≲ 2)'); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT + case + '/convergence.png', dpi=95); plt.close(fig)


def all_plots(case, title, q, y, x, free, ref, lam_a, path_data=True, log=None, pattern=None):
    os.makedirs(OUT + case, exist_ok=True)
    q.plots_feps(); save(OUT + case + '/feps.png')
    q.plots_obs_pred(); save(OUT + case + '/obs_pred.png')
    if path_data:
        try:
            q.plots_tis(); save(OUT + case + '/tis_diagnostics.png')
        except cp.cuda.memory.OutOfMemoryError:
            plt.close('all'); print(f'{case}: tis_diagnostics skipped (GPU memory)', flush=True)
            cp.get_default_memory_pool().free_all_blocks()
    rms, off = map_plot(case, title, y, x, cp.asnumpy(q.r_traj), free, ref, lam_a)
    if log:
        convergence_plot(case, log, pattern)
    print(f'{case:24s} RMS {rms:.4f}  plateau {off:+.4f}', flush=True)


which = sys.argv[1]
if which == 'synth':
    cases = [('synth_full', 'synthetic walkers, run until absorbed', 'synth_30k_s50', 'r_full', 'refit.log', r'^full (\d+) it\.'),
             ('synth_cut_all', 'walkers cut at interfaces', 'synth_trunc', 'r_trunc', 'fit_trunc.log', r'^trunc (\d+) it\.'),
             ('synth_cut_even', 'cut walkers, even half', 'synth_trunc_even', 'r_trunc_even', 'fit_trunc.log', r'^trunc_even (\d+) it\.'),
             ('synth_cut_odd', 'cut walkers, odd half', 'synth_trunc_odd', 'r_trunc_odd', 'fit_trunc.log', r'^trunc_odd (\d+) it\.'),
             ('synth_cut_even_lag10', 'cut walkers, even half, lag 10', 'synth_trunc_even', 'r_trunc_even_lag10', 'refit.log', r'^trunc_even_lag10 (\d+) it\.'),
             ('synth_cut_odd_lag10', 'cut walkers, odd half, lag 10', 'synth_trunc_odd', 'r_trunc_odd_lag10', 'refit.log', r'^trunc_odd_lag10 (\d+) it\.')]
    for case, title, data, rfile, log, pat in cases:
        if not os.path.exists(Z + f'synth/{rfile}.npy'):
            print(f'{case}: {rfile}.npy missing, skipped', flush=True); continue
        d = np.load(Z + f'synth/{data}.npz'); y, x, it = d['y'], d['x'], d['i_traj']
        A, B = y <= 0.05, y >= 0.85
        q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=it, t_traj=d['t_traj'].astype(float))
        q.set_fixed_traj_length_trap(A | B, 2 ** 16)
        q.r_traj = cp.asarray(np.load(Z + f'synth/{rfile}.npy'))
        all_plots(case, title, q, y, x, ~(A | B), ref_st, 0.05, True, Z + 'synth/' + log, pat)
        del q; cp.get_default_memory_pool().free_all_blocks()
elif which in ('staple1500', 'staple2250'):
    ppe = '1500' if which == 'staple1500' else '2250'
    p = argparse.ArgumentParser(); tis_data.add_data_arguments(p)
    with contextlib.redirect_stdout(io.StringIO()):
        data = tis_data.prepare_data(p.parse_args([SIM, '--reversed', '--paths-per-ensemble', ppe, '--seed', '0']))
    if ppe == '1500':
        cases = [('staple_both_all', 'staple, both pieces, 1500 paths/ens. (lag 1, 25000 it.)', 'all', R + 'r_continued.npy', R + 'continue.log', r'^(\d+) it\.'),
                 ('staple_both_even', 'staple, both pieces, even half (lag 1)', 'even', R + 'r_half_even.npy', R + 'half_fits.log', r'even (\d+) it\.'),
                 ('staple_both_odd', 'staple, both pieces, odd half (lag 1)', 'odd', R + 'r_half_odd.npy', R + 'half_fits.log', r'odd (\d+) it\.'),
                 ('staple_both_even_lag50', 'staple, both pieces, even half (lag 50)', 'even', R + 'r_lag50_even_ppe1500.npy', R + 'lag_halves50_even.log', r'lag 50 even (\d+) it\.'),
                 ('staple_both_odd_lag50', 'staple, both pieces, odd half (lag 50)', 'odd', R + 'r_lag50_odd.npy', R + 'lag_halves50_odd.log', r'lag 50 odd (\d+) it\.')]
    else:
        cases = [('staple_both_2250_lag50', 'staple, both pieces, 2250 paths/ens. (lag 50)', 'all', R + 'r_lag50_all_ppe2250.npy', R + 'lag_all_2250ppe.log', r'lag 50 all (\d+) it\.')]
    for case, title, half, rfile, log, pat in cases:
        if not os.path.exists(rfile):
            print(f'{case}: {os.path.basename(rfile)} missing, skipped', flush=True); continue
        keep = np.ones(len(data.lam), bool) if half == 'all' else ((data.group_traj % 2) == (0 if half == 'even' else 1))
        y, x = data.X_all[keep, 0], data.X_all[keep, 1]; b0, b1 = data.boundary0[keep], data.boundary1[keep]
        w = data.path_weights[keep]; w = w / w.mean()
        q = optimalrcs.CommittorNE(boundary0=b0, boundary1=b1, i_traj=data.i_traj[keep], t_traj=data.t_traj[keep],
                                   path_weights=w, group_traj=data.group_traj[keep])
        q.set_fixed_traj_length_trap(b0 | b1, 2 ** 16)
        q.r_traj = cp.asarray(np.load(rfile))
        all_plots(case, title, q, y, x, ~(b0 | b1), ref_st, 0.05, True, log, pat)
        del q; cp.get_default_memory_pool().free_all_blocks()
elif which == 'forward':      # the first staple fit: forward pieces only, 3000 paths per ensemble (istarz_br_g10/)
    case = 'staple_forward'
    d = np.load(F + '2d_zpot/brownian_g10/staple/original/committor.npz')
    y, x, b0, b1 = d['X_all'][:, 0], d['X_all'][:, 1], d['boundary0'], d['boundary1']
    q = optimalrcs.CommittorNE(boundary0=b0, boundary1=b1, i_traj=d['i_traj'], t_traj=d['t_traj'], path_weights=d['path_weights'])
    q.set_fixed_traj_length_trap(b0 | b1, 2 ** 16)
    q.r_traj = cp.asarray(d['r'])
    all_plots(case, 'staple, forward pieces only, 3000 paths/ens. (lag 1)', q, y, x, ~(b0 | b1), ref_st, 0.05, True)
elif which == 'md':
    d = np.load(Z + 'npq_md/npq_md.npz')
    y, x, it, row, seed = d['y'], d['x'], d['i_traj'], d['row'].astype(float), d['seed']
    LA = float(d['lambda_a'])
    cases = [('md_npq_all', 'long MD NPq, 16 seeds', np.ones(len(y), bool), d['q_npq'], None, None),
             ('md_npne_s1-8_lag1', 'long MD NPNEq, seeds 1-8, lag 1', seed <= 8, Z + 'lag_md_lag1_env0.01.npy', Z + 'lag_md.log', r'^lag1_env0.01 (\d+) it \('),
             ('md_npne_s1-8_lag10', 'long MD NPNEq, seeds 1-8, lag 10', seed <= 8, Z + 'lag_md_lag10_env0.01.npy', Z + 'lag_md.log', r'^lag10_env0.01 (\d+) it \('),
             ('md_npne_s1-8_lag50', 'long MD NPNEq, seeds 1-8, lag 50', seed <= 8, Z + 'lag_md_lag50_env0.01.npy', Z + 'lag_md.log', r'^lag50_env0.01 (\d+) it \('),
             ('md_npne_s9-16_lag10', 'long MD NPNEq, seeds 9-16, lag 10', seed > 8, Z + 'lag_md_lag10_seeds9-16.npy', None, None),
             ('md_npne_s1-8_envsched', 'long MD NPNEq, seeds 1-8, envelope 1 to 0.001', seed <= 8, Z + 'lag_md_sched_env0.001.npy', Z + 'env_sched.log', 'CUM:' + r'^sched_env\S+ (\d+) it \(')]
    for case, title, m, r, log, pat in cases:
        r = np.load(r) if isinstance(r, str) else r
        A, B = y[m] <= LA, y[m] >= 0.85
        q = optimalrcs.CommittorNE(boundary0=A, boundary1=B, i_traj=it[m], t_traj=row[m])
        q.set_boundary_traps()
        q.r_traj = cp.asarray(r)
        all_plots(case, title, q, y[m], x[m], ~(A | B), ref_md, LA, False, log, pat)
        del q; cp.get_default_memory_pool().free_all_blocks()
print('done', which, flush=True)
